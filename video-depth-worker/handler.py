import os
import sys
import glob
import tempfile
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

    checkpoint_path = f"/app/checkpoints/{MODEL_NAME.lower()}.pth"
    if not os.path.exists(checkpoint_path):
        # Fail loudly instead of silently running inference on randomly
        # initialized weights (which would still report "status": "success"
        # while producing garbage depth maps).
        raise FileNotFoundError(
            f"Checkpoint file not found at {checkpoint_path}. Only "
            "Video-Depth-Anything-Base is pre-downloaded by the Dockerfile — "
            "if you switched MODEL_NAME, add a matching download step there."
        )
    print(f"Found local checkpoint at: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    
    # Move model to device BEFORE eval()
    model = model.to(device)
    model = model.eval()
    
    # CRITICAL: Force ALL submodules to GPU, including pretrained encoder
    for param in model.parameters():
        param.data = param.data.to(device)
    
    # Verify model is on correct device
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Pretrained encoder device: {next(model.pretrained.parameters()).device}")
    
    MODEL = model
    DEVICE = device
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

        # 2. Extract Frames
        print("[2/4] Reading video frames...")
        cap = cv2.VideoCapture(local_video_path)
        target_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # infer_video_depth() expects RGB frames — upstream's own
            # utils/dc_utils.py::read_video_frames() does this exact
            # cv2.cvtColor(..., COLOR_BGR2RGB) before calling it. cv2.VideoCapture
            # yields BGR, so convert here or the model runs on swapped color
            # channels.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if not frames:
            raise ValueError("No frames could be extracted from the provided video file.")

        # infer_video_depth() indexes into `frames` with .shape (video_depth.py
        # does `frame_list = [frames[i] for i in range(frames.shape[0])]`), so
        # it must be a single ndarray, not a plain Python list of per-frame
        # arrays.
        frames = np.stack(frames, axis=0)

        # 3. Run Inference
        print(f"[3/4] Running Depth Inference across {len(frames)} frames (fps={target_fps})...")
        
        # CRITICAL: Verify CUDA is available and device is set correctly
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Device being used: {device}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory before inference: {torch.cuda.memory_allocated() / 1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        print("Starting inference loop...")
        # NOTE: pass `frames` as a plain numpy array here, NOT a torch.Tensor.
        # infer_video_depth() does its own per-frame preprocessing internally
        # (frame_list[i].astype(np.float32) / 255.0, then torch.from_numpy(...)
        # per chunk, moved to `device` right before each forward pass) — it
        # expects raw numpy frames and does its own GPU transfer already. A
        # previous "optimization" pre-converted frames to a GPU tensor before
        # this call, but .astype() doesn't exist on torch.Tensor, so it broke
        # infer_video_depth()'s preprocessing outright, and moving the whole
        # clip to GPU up front bought nothing anyway since the function
        # re-transfers per-chunk regardless.
        with torch.no_grad():
            depths, _ = model.infer_video_depth(
                frames, target_fps=target_fps, device=device
            )

        print("Inference loop completed!")
        
        print(f"GPU Memory after inference: {torch.cuda.memory_allocated() / 1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        del frames
        if device == "cuda":
            torch.cuda.empty_cache()

        if davinci_safe:
            depth_min = float(depths.min())
            depth_max = float(depths.max())
            depth_range = max(depth_max - depth_min, 1e-6)
            depths = (depths - depth_min) / depth_range
            print(f"Normalized depth range [{depth_min:.4f}, {depth_max:.4f}] -> [0, 1]")
        else:
            print(f"Skipping normalization (davinci_safe=False) — raw depth range [{depths.min():.4f}, {depths.max():.4f}]")

        # 4. Save and Upload EXRs
        print(f"[4/4] Writing EXRs and uploading sequence to '{output_bucket}'...")
        for i, depth_frame in enumerate(depths):
            frame_filename = f"frame_{i:04d}.exr"
            local_exr_path = os.path.join(exr_output_dir, frame_filename)
            remote_upload_path = f"{output_prefix}/{frame_filename}"

            save_exr_32bit(depth_frame, local_exr_path)

            with open(local_exr_path, "rb") as exr_file:
                supabase.storage.from_(output_bucket).upload(
                    file=exr_file,
                    path=remote_upload_path,
                    file_options={"cache-control": "3600", "upsert": "true"}
                )

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
