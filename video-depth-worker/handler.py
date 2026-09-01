import os
import sys
import glob
import tempfile
import time
import cv2
import torch
import numpy as np
import OpenEXR
import Imath
from supabase import create_client, Client
import runpod

# --- Blackwell (sm_120) workaround ---
# xformers' bundled Flash-Attention-3 ("Hopper") kernel declares a minimum
# compute capability of sm_90 with no upper bound, so xformers' dispatcher
# selects it on newer architectures like Blackwell (sm_120) too — but the
# kernel binary in this xformers release was only compiled for sm_90, so it
# crashes with "no kernel image is available for execution on the device".
# Disable FA3 so xformers falls back to its more portable CUTLASS kernel.
try:
    from xformers.ops.fmha import _set_use_fa3
    _set_use_fa3(False)
except ImportError:
    pass

# --- CRITICAL: Add repo paths BEFORE importing model modules ---
repo_path = "/app/Video-Depth-Anything"
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)
if "/app" not in sys.path:
    sys.path.insert(0, "/app")

from video_depth_anything.video_depth import VideoDepthAnything

# --- Environment Setup ---
MODEL_NAME = os.environ.get("MODEL_NAME", "Video-Depth-Anything-Base")

# Cap how many frames we hold in memory (raw frames + depth output) at
# once. Holding an entire long video's frames and depths simultaneously was
# crashing real jobs with a silent SIGKILL (exit 137, no Python traceback)
# -- infer_video_depth()'s own np.stack() of the full clip, plus our
# normalization step, each need a full-size copy of the whole clip's depth
# data at once. Processing in bounded chunks keeps peak memory roughly
# constant regardless of video length. Tunable via env var since the right
# chunk size depends on video resolution and available RAM.
CHUNK_SIZE_FRAMES = int(os.environ.get("CHUNK_SIZE_FRAMES", "150"))

# Global variables for model/client caching
MODEL = None
DEVICE = None
SUPABASE = None


def get_supabase() -> Client:
    """Safely initializes and caches the Supabase client."""
    global SUPABASE
    if SUPABASE is None:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError(
                "Missing environment variables! Please ensure 'SUPABASE_URL' "
                "and 'SUPABASE_SERVICE_ROLE_KEY' are set in your RunPod Endpoint settings."
            )
        SUPABASE = create_client(supabase_url, supabase_key)
    return SUPABASE


def load_model() -> tuple[VideoDepthAnything, str]:
    """Safely loads and caches the Video Depth Anything model. Returns (model, device)."""
    global MODEL, DEVICE
    if MODEL is not None:
        return MODEL, DEVICE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Video Depth Anything model ({MODEL_NAME}) on {device}...")

    model_configs = {
        'Video-Depth-Anything-Small': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'Video-Depth-Anything-Base': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'Video-Depth-Anything-Large': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    if MODEL_NAME not in model_configs:
        raise ValueError(
            f"Unknown MODEL_NAME '{MODEL_NAME}'. Must be one of: {', '.join(model_configs)}"
        )
    config = model_configs[MODEL_NAME]
    model = VideoDepthAnything(**config)

    # Matches the filename saved by Dockerfile: /app/checkpoints/video-depth-anything-base.pth
    checkpoint_path = f"/app/checkpoints/{MODEL_NAME.lower()}.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found at {checkpoint_path}. Only "
            "Video-Depth-Anything-Base is pre-downloaded by the Dockerfile — "
            "if you switched MODEL_NAME, add a matching download step there."
        )
    print(f"Found local checkpoint at: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

    MODEL = model.to(device).eval()
    DEVICE = device

    # Ground-truth check, independent of any external dashboard/telemetry:
    # ask PyTorch itself how much GPU memory it's actually holding right
    # after placing the model. If this prints 0 while device == "cuda",
    # something is genuinely wrong with GPU placement, not just a
    # telemetry-reporting quirk on RunPod's end.
    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[GPU CHECK] torch.cuda.memory_allocated() = {allocated:.3f} GB")
        print(f"[GPU CHECK] torch.cuda.memory_reserved()  = {reserved:.3f} GB")

    return MODEL, DEVICE


def save_exr_32bit(depth_map: np.ndarray, output_path: str):
    """Saves a 2D float32 numpy array as a single-channel 32-bit Float EXR image."""
    height, width = depth_map.shape
    depth_float32 = depth_map.astype(np.float32)

    header = OpenEXR.Header(width, height)
    header['channels'] = {'Z': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}

    out = OpenEXR.OutputFile(output_path, header)
    out.writePixels({'Z': depth_float32.tobytes()})
    out.close()


def upload_with_retry(
    supabase: Client,
    bucket: str,
    remote_path: str,
    local_path: str,
    max_attempts: int = 4,
):
    """Uploads a file to Supabase Storage, retrying on transient network
    errors (timeouts, connection resets) with exponential backoff.

    A sequence upload is hundreds to thousands of individual HTTP requests
    -- one flaky read timeout on any single one of them used to kill the
    entire job outright, even after everything before it had succeeded.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            with open(local_path, "rb") as exr_file:
                supabase.storage.from_(bucket).upload(
                    file=exr_file,
                    path=remote_path,
                    file_options={"cache-control": "3600", "upsert": "true"}
                )
            return
        except Exception as e:
            last_error = e
            if attempt == max_attempts:
                break
            backoff = 2 ** (attempt - 1)  # 1s, 2s, 4s, ...
            print(
                f"Upload of '{remote_path}' failed (attempt {attempt}/{max_attempts}): "
                f"{e}. Retrying in {backoff}s..."
            )
            time.sleep(backoff)
    raise last_error


def process_video_depth(
    input_bucket: str,
    video_key: str,
    output_bucket: str,
    output_prefix: str = "depth_sequence",
    davinci_safe: bool = True
):
    supabase = get_supabase()
    model, device = load_model()

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_video_path = os.path.join(tmp_dir, "input.mp4")
        exr_output_dir = os.path.join(tmp_dir, "exr_frames")
        os.makedirs(exr_output_dir, exist_ok=True)

        # 1. Download Video
        print(f"[1/4] Downloading '{video_key}' from bucket '{input_bucket}'...")
        video_bytes = supabase.storage.from_(input_bucket).download(video_key)
        with open(local_video_path, "wb") as f:
            f.write(video_bytes)

        # 2-4. Stream frames from the video in bounded-size chunks, running
        # inference and uploading each chunk's EXRs before moving on to the
        # next one, instead of holding the entire clip's frames and depth
        # output in memory at once. This keeps peak memory roughly constant
        # regardless of video length.
        #
        # Tradeoff: infer_video_depth() does its own internal scale-and-shift
        # alignment ACROSS the frames passed to a single call, so processing
        # independent chunks means there can be a small depth-scale
        # discontinuity at each chunk boundary that wouldn't exist if the
        # whole clip were processed in one call. In practice that's a minor
        # seam every CHUNK_SIZE_FRAMES frames -- a much better tradeoff than
        # the job crashing outright on anything longer than a short clip.
        cap = cv2.VideoCapture(local_video_path)
        target_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        hint_suffix = f", ~{total_frames_hint} frames total" if total_frames_hint else ""
        print(
            f"[2-4/4] Processing video in chunks of {CHUNK_SIZE_FRAMES} frames "
            f"(fps={target_fps}{hint_suffix})..."
        )

        frame_index = 0
        chunk_num = 0

        def flush_chunk(buffer):
            nonlocal frame_index, chunk_num
            if not buffer:
                return
            chunk_num += 1
            chunk_frames = np.stack(buffer, axis=0)
            n = len(chunk_frames)
            print(
                f"[chunk {chunk_num}] Running depth inference on frames "
                f"{frame_index}-{frame_index + n - 1} ({n} frames)..."
            )

            with torch.no_grad():
                chunk_depths, _ = model.infer_video_depth(
                    chunk_frames, target_fps=target_fps, device=device
                )

            # Ground-truth GPU memory check (see load_model()) -- only on the
            # first chunk, so this doesn't spam the log on long videos.
            if chunk_num == 1 and device == "cuda":
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"[GPU CHECK] post-inference memory_allocated() = {allocated:.3f} GB")
                print(f"[GPU CHECK] post-inference memory_reserved()  = {reserved:.3f} GB")

            del chunk_frames
            if device == "cuda":
                torch.cuda.empty_cache()

            if davinci_safe:
                depth_min = float(chunk_depths.min())
                depth_max = float(chunk_depths.max())
                depth_range = max(depth_max - depth_min, 1e-6)
                chunk_depths = (chunk_depths - depth_min) / depth_range
                print(
                    f"[chunk {chunk_num}] Normalized depth range "
                    f"[{depth_min:.4f}, {depth_max:.4f}] -> [0, 1]"
                )
            else:
                print(
                    f"[chunk {chunk_num}] Skipping normalization (davinci_safe=False) "
                    f"— raw depth range [{chunk_depths.min():.4f}, {chunk_depths.max():.4f}]"
                )

            for depth_frame in chunk_depths:
                frame_filename = f"frame_{frame_index:04d}.exr"
                local_exr_path = os.path.join(exr_output_dir, frame_filename)
                remote_upload_path = f"{output_prefix}/{frame_filename}"

                save_exr_32bit(depth_frame, local_exr_path)
                upload_with_retry(supabase, output_bucket, remote_upload_path, local_exr_path)
                # Delete each temp EXR right after upload rather than letting
                # them pile up in tmp_dir for the whole job -- disk on these
                # workers is small (a few GB free) and long videos can mean
                # thousands of frames.
                os.remove(local_exr_path)
                frame_index += 1

            print(f"[chunk {chunk_num}] Uploaded {n} frame(s), {frame_index} total so far.")

        buffer = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buffer.append(frame)
            if len(buffer) >= CHUNK_SIZE_FRAMES:
                flush_chunk(buffer)
                buffer = []
        cap.release()

        flush_chunk(buffer)

        if frame_index == 0:
            raise ValueError("No frames could be extracted from the provided video file.")

        print("Finished sequence generation and upload!")


def handler(job):
    """RunPod Serverless Handler Function wrapped in safety try/except."""
    try:
        job_input = job.get("input", {})

        input_bucket = job_input.get("input_bucket", "depth-outputs")
        video_key = job_input.get("video_key", "sample.mp4")
        output_bucket = job_input.get("output_bucket", "depth-outputs")
        output_prefix = job_input.get("output_prefix", "sequence_001")
        davinci_safe = job_input.get("davinci_safe", True)

        process_video_depth(
            input_bucket=input_bucket,
            video_key=video_key,
            output_bucket=output_bucket,
            output_prefix=output_prefix,
            davinci_safe=davinci_safe
        )

        return {
            "status": "success",
            "output_prefix": output_prefix,
            "message": f"Successfully processed depth sequence for {video_key}"
        }

    except Exception as e:
        print(f"ERROR OCCURRED DURING JOB PROCESSING: {str(e)}")
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e)
        }


if __name__ == "__main__":
    print("Worker starting up and listening for RunPod jobs...")
    runpod.serverless.start({"handler": handler})
