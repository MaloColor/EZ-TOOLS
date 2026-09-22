import os
import sys
import glob
import json
import re
import tempfile
import time
import cv2
import torch
import numpy as np
from supabase import create_client, Client
import runpod

# Prefer decord over cv2.VideoCapture for reading video frames. cv2's FFmpeg
# backend can silently misreport a video's frame count and/or stop reading
# early on certain files (confirmed in production: cv2 reported ~1449 frames
# and stopped there on a video that actually has 8301) -- decord parses the
# container directly and gives an accurate frame count and full access to
# every frame. This matches upstream Video-Depth-Anything's own preference
# (utils/dc_utils.py::read_video_frames() tries decord first, falls back to
# cv2 only if decord isn't installed) -- decord is already in this image
# because it's one of upstream's own dependencies.
try:
    from decord import VideoReader, cpu as decord_cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    print("decord not available, falling back to cv2.VideoCapture for frame reading")

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


def save_depth_png16(depth_map: np.ndarray, output_path: str):
    """Saves a 2D depth array as a 16-bit grayscale PNG.

    Expects values already normalized to [0, 1] (see process_video_depth) --
    PNG has no way to store unbounded float like the EXR output this
    replaced, only a fixed, bounded integer range. 16-bit (65,536 levels)
    keeps gradients smooth (no visible banding) while still being far
    smaller and more broadly compatible than 32-bit float EXR.
    """
    depth_uint16 = np.clip(depth_map, 0.0, 1.0)
    depth_uint16 = (depth_uint16 * 65535.0 + 0.5).astype(np.uint16)
    cv2.imwrite(output_path, depth_uint16)


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
            with open(local_path, "rb") as upload_file:
                supabase.storage.from_(bucket).upload(
                    file=upload_file,
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


FRAME_NAME_RE = re.compile(r"^frame_(\d+)\.png$")


def get_uploaded_frame_indices(supabase: Client, output_bucket: str, output_prefix: str) -> set[int]:
    """Lists frame_NNNN.png files already uploaded under this output_prefix.

    output_prefix is now derived from the input video's content hash (plus
    the davinci_safe flag) rather than a random UUID per job, so a retried
    job for the same input lands on the exact same prefix a prior, possibly
    incomplete, attempt used. That's what makes resuming meaningful -- we
    can tell which frames a previous attempt already finished and skip
    redoing them, instead of either re-running the whole video or (worse)
    mistaking a partial result for a complete one.
    """
    existing: set[int] = set()
    offset = 0
    page_size = 1000
    while True:
        entries = supabase.storage.from_(output_bucket).list(
            output_prefix, {"limit": page_size, "offset": offset}
        )
        if not entries:
            break
        for entry in entries:
            m = FRAME_NAME_RE.match(entry.get("name", ""))
            if m:
                existing.add(int(m.group(1)))
        if len(entries) < page_size:
            break
        offset += page_size
    return existing


def process_video_depth(
    input_bucket: str,
    video_key: str,
    output_bucket: str,
    output_prefix: str = "depth_sequence",
    davinci_safe: bool = True,
    on_progress=None,
):
    """on_progress, if given, is called as on_progress(frame_index, total_frames)
    after total_frames is known and again after every chunk -- lets the caller
    (see handler() below) push real render progress back to RunPod instead of
    the frontend only ever knowing IN_QUEUE / IN_PROGRESS / COMPLETED."""
    supabase = get_supabase()
    model, device = load_model()

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_video_path = os.path.join(tmp_dir, "input.mp4")
        frame_output_dir = os.path.join(tmp_dir, "depth_frames")
        os.makedirs(frame_output_dir, exist_ok=True)

        # 1. Download Video
        print(f"[1/4] Downloading '{video_key}' from bucket '{input_bucket}'...")
        video_bytes = supabase.storage.from_(input_bucket).download(video_key)
        with open(local_video_path, "wb") as f:
            f.write(video_bytes)

        # 2-4. Read frames from the video in bounded-size chunks, running
        # inference and uploading each chunk's PNGs before moving on to the
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
        if DECORD_AVAILABLE:
            vr = VideoReader(local_video_path, ctx=decord_cpu(0))
            total_frames = len(vr)
            target_fps = vr.get_avg_fps() or 30.0
        else:
            cap = cv2.VideoCapture(local_video_path)
            target_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        hint_suffix = f", {total_frames} frames total" if total_frames else ""
        print(
            f"[2-4/4] Processing video in chunks of {CHUNK_SIZE_FRAMES} frames "
            f"(fps={target_fps}{hint_suffix}, reader={'decord' if DECORD_AVAILABLE else 'cv2'})..."
        )
        # Deliberately loud and easy to grep in RunPod logs -- this is the
        # single number everything else in this job depends on. If the
        # implied duration here looks wrong for the source video (e.g. the
        # video is visibly ~10s but this says ~3s), decord/cv2 misread the
        # container -- the same class of bug noted above where cv2 once
        # reported ~1449 frames on a video that actually had 8301 -- and every
        # downstream count (chunks, uploaded frames, the final manifest) will
        # be wrong in exactly the same way, since nothing after this line
        # re-verifies it against the source file.
        if total_frames:
            implied_duration = total_frames / target_fps
            print(
                f"[FRAME COUNT] reader={'decord' if DECORD_AVAILABLE else 'cv2'} detected "
                f"total_frames={total_frames} at fps={target_fps:.3f} "
                f"(implies ~{implied_duration:.2f}s of video) -- this number drives every "
                "chunk boundary and the final frame_count written to _complete.json."
            )
        else:
            print("[FRAME COUNT] WARNING: total_frames could not be determined up front (cv2 fallback, unknown count).")

        def report_progress():
            if on_progress and total_frames:
                on_progress(frame_index, total_frames)

        # output_prefix is content-derived (see get_uploaded_frame_indices),
        # so a retried job for the same video+setting lands here and can
        # pick up where a prior, possibly-killed attempt left off instead of
        # redoing frames that already made it to the output bucket.
        uploaded_frames = get_uploaded_frame_indices(supabase, output_bucket, output_prefix)
        if uploaded_frames:
            print(
                f"Found {len(uploaded_frames)} frame(s) already uploaded for this "
                "output (resuming a prior attempt) -- will skip any chunk that's "
                "already fully present."
            )

        frame_index = 0
        chunk_num = 0
        # Running normalization range from prior chunks, used to smooth the
        # depth range across chunk boundaries (see flush_chunk below) so
        # consecutive chunks don't snap to a different brightness/contrast
        # at the seam. None until the first chunk has been processed.
        prev_depth_min = None
        prev_depth_max = None
        report_progress()  # initial 0/total_frames so the frontend has a real number immediately

        def flush_chunk(buffer):
            nonlocal frame_index, chunk_num, prev_depth_min, prev_depth_max
            if len(buffer) == 0:
                return
            chunk_num += 1
            # `buffer` is a plain list of per-frame arrays from the cv2 path,
            # or an ndarray already shaped [N, H, W, 3] straight from
            # decord's get_batch().asnumpy() -- decord already returns RGB
            # (see utils/dc_utils.py: the decord branch does no color
            # conversion, only the cv2 fallback branch does), so no
            # cv2.cvtColor is needed here either way.
            chunk_frames = np.stack(buffer, axis=0) if isinstance(buffer, list) else buffer
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

            # PNG can only hold a bounded, fixed-precision range -- unlike
            # the EXR output this replaced, there's no way to write
            # unbounded float "raw" depth to a 16-bit PNG. Always normalize
            # per chunk to [0, 1] regardless of davinci_safe; the flag is
            # still accepted (and recorded in the manifest below) for
            # compatibility with existing callers, but it no longer changes
            # what gets written -- there's only one output now.
            #
            # Two adjustments on top of a plain min/max, both aimed at the
            # same failure mode: a single chunk's true min/max is set by
            # whatever pixel in whatever frame happened to be closest/
            # farthest from the camera in that ~150-frame window (e.g. a
            # hand passing close to the lens for a moment). Using that as
            # the normalization range for the *entire* chunk drags every
            # other frame's brightness along with it, producing a visible
            # darken/brighten pulse for the chunk's whole ~6s duration.
            #
            # 1. Percentile-based range instead of true min/max: a handful
            #    of outlier pixels no longer set the range for the whole
            #    chunk -- they just clip to pure black/white themselves
            #    (via the np.clip in save_depth_png16), localized to the
            #    frames/pixels that are actually extreme.
            # 2. Blend with the running range from prior chunks: keeps the
            #    range from snapping to a different value at each chunk
            #    boundary, since consecutive chunks of the same scene
            #    should usually have a similar depth range anyway.
            depth_min = float(np.percentile(chunk_depths, 1))
            depth_max = float(np.percentile(chunk_depths, 99))
            if prev_depth_min is not None:
                carry_forward = 0.7  # weight given to the running range vs. this chunk's own
                depth_min = carry_forward * prev_depth_min + (1 - carry_forward) * depth_min
                depth_max = carry_forward * prev_depth_max + (1 - carry_forward) * depth_max
            prev_depth_min, prev_depth_max = depth_min, depth_max
            depth_range = max(depth_max - depth_min, 1e-6)
            chunk_depths = (chunk_depths - depth_min) / depth_range
            print(
                f"[chunk {chunk_num}] Normalized depth range "
                f"[{depth_min:.4f}, {depth_max:.4f}] -> [0, 1] "
                "(1st/99th percentile, blended with prior chunks)"
            )
            if not davinci_safe:
                print(
                    f"[chunk {chunk_num}] Note: davinci_safe=False was requested, "
                    "but PNG output requires a bounded range -- normalizing anyway."
                )

            for depth_frame in chunk_depths:
                frame_filename = f"frame_{frame_index:04d}.png"
                local_frame_path = os.path.join(frame_output_dir, frame_filename)
                remote_upload_path = f"{output_prefix}/{frame_filename}"

                save_depth_png16(depth_frame, local_frame_path)
                upload_with_retry(supabase, output_bucket, remote_upload_path, local_frame_path)
                # Delete each temp frame right after upload rather than
                # letting them pile up in tmp_dir for the whole job -- disk
                # on these workers is small (a few GB free) and long videos
                # can mean thousands of frames.
                os.remove(local_frame_path)
                frame_index += 1

            print(f"[chunk {chunk_num}] Uploaded {n} frame(s), {frame_index} total so far.")
            report_progress()

        if DECORD_AVAILABLE:
            for start in range(0, total_frames, CHUNK_SIZE_FRAMES):
                end = min(start + CHUNK_SIZE_FRAMES, total_frames)
                if uploaded_frames and all(i in uploaded_frames for i in range(start, end)):
                    print(f"[chunk] frames {start}-{end - 1} already uploaded, skipping.")
                    frame_index = end
                    report_progress()
                    continue
                chunk = vr.get_batch(list(range(start, end))).asnumpy()
                flush_chunk(chunk)
        else:
            # cv2 has no reliable random-access seek here (see the DECORD_AVAILABLE
            # comment at the top of this file), so unlike the decord path this
            # can't skip decoding already-uploaded frames -- it re-decodes
            # everything, but upload_with_retry's upsert makes re-uploading
            # already-present frames a harmless no-op rather than a failure.
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

        # Sanity check against the number this job started with (see
        # [FRAME COUNT] above). These SHOULD always match for the decord path
        # -- the chunk loop always covers range(0, total_frames) in full --
        # so a mismatch here means something upstream silently stopped the
        # loop early rather than raising, and frame_count below is about to
        # be written into _complete.json as if it were correct anyway.
        if total_frames and frame_index != total_frames:
            print(
                f"[FRAME COUNT] WARNING: uploaded {frame_index} frame(s) but total_frames "
                f"was {total_frames} at the start of this job -- _complete.json is about to "
                f"claim frame_count={frame_index}, which will look 'complete' to the frontend "
                "even though it doesn't match what this video was expected to produce."
            )

        # Written only once every frame has actually made it to the output
        # bucket -- the frontend's "already processed, skip straight to
        # download" check looks for this exact marker rather than just any
        # file existing, since a killed job can leave a partial set of
        # frames behind that must NOT be mistaken for a finished result.
        manifest = json.dumps({
            "frame_count": frame_index,
            "format": "png16",
            "davinci_safe": davinci_safe,
            "completed_at": time.time(),
        }).encode("utf-8")
        supabase.storage.from_(output_bucket).upload(
            path=f"{output_prefix}/_complete.json",
            file=manifest,
            file_options={"content-type": "application/json", "upsert": "true"},
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

        # Pushes an intermediate value via RunPod's progress_update() API,
        # which shows up in the regular /status/{id} response's "output"
        # field while the job is still IN_PROGRESS (replaced by the real
        # return value once this function returns below). The frontend reads
        # this to show actual frame-based progress instead of just
        # IN_QUEUE/IN_PROGRESS/COMPLETED -- but progress reporting is
        # inherently best-effort: if this API behaves differently than
        # expected, or the call fails outright, that must never take down
        # the actual video processing, so any error here is only logged.
        def on_progress(frame_index, total_frames):
            try:
                runpod.serverless.progress_update(job, {
                    "frame_index": frame_index,
                    "total_frames": total_frames,
                })
            except Exception as e:
                print(f"[PROGRESS] Could not send progress update ({frame_index}/{total_frames}): {e}")

        process_video_depth(
            input_bucket=input_bucket,
            video_key=video_key,
            output_bucket=output_bucket,
            output_prefix=output_prefix,
            davinci_safe=davinci_safe,
            on_progress=on_progress,
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
