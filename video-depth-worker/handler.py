import os
import sys
import glob
import json
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
# Same helper Video-Depth-Anything uses internally (video_depth.py) to align
# its own 32-frame inference windows onto a shared scale -- reused below to
# align our own, much larger, outer CHUNK_SIZE_FRAMES windows onto each
# other too (see CHUNK_OVERLAP_FRAMES).
from utils.util import compute_scale_and_shift

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

# infer_video_depth() aligns scale internally ACROSS the frames passed to a
# single call (it windows internally at 32 frames with a 10-frame overlap,
# solving a linear scale+shift fit between windows -- see upstream
# video_depth_anything/video_depth.py) -- but that alignment never crosses
# our own outer chunk boundary, since each CHUNK_SIZE_FRAMES chunk is an
# entirely separate call with no knowledge of the previous chunk's scale.
# That produced a real depth-scale jump (not just a normalization artifact)
# at every chunk boundary. Fixed the same way upstream fixes it internally:
# re-run inference on the last CHUNK_OVERLAP_FRAMES frames of the previous
# chunk at the start of the next one, then solve for the scale+shift that
# best maps the new chunk's depth for those frames onto the previous
# chunk's already-output depth for the same frames, and apply that
# transform to the whole new chunk before normalizing. The overlap frames
# themselves are only used for this fit and dropped from the output.
CHUNK_OVERLAP_FRAMES = int(os.environ.get("CHUNK_OVERLAP_FRAMES", "8"))

# How many frame uploads to run at once. Uploads are network-bound (one HTTP
# request per frame to Supabase Storage), so doing them one at a time leaves
# the GPU idle waiting on hundreds of sequential round trips per chunk --
# same fix as DOWNLOAD_CONCURRENCY on the frontend's zip download.
UPLOAD_CONCURRENCY = int(os.environ.get("UPLOAD_CONCURRENCY", "8"))

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


_upload_client_local = threading.local()


def get_supabase_for_upload() -> Client:
    """Per-thread Supabase client, used only by the concurrent frame-upload
    pool in flush_chunk.

    The cached client from get_supabase() holds one underlying HTTP/2
    connection. Sharing that single client (and connection) across
    UPLOAD_CONCURRENCY threads meant every concurrent upload was
    multiplexed onto the same connection -- when the server or an
    intermediary edge dropped it under a burst, every request in flight on
    it failed at once ("Server disconnected"), seen in production as
    whole bursts of frames erroring together, then all succeeding a
    couple seconds later on retry. Giving each worker thread its own
    client (own connection) avoids that shared point of failure.
    """
    if not hasattr(_upload_client_local, "supabase"):
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        _upload_client_local.supabase = create_client(supabase_url, supabase_key)
    return _upload_client_local.supabase


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


def encode_depth_png16(depth_map: np.ndarray) -> bytes:
    """Encodes a 2D depth array as a 16-bit grayscale PNG, in memory.

    Expects values already normalized to [0, 1] (see process_video_depth) --
    PNG has no way to store unbounded float like the EXR output this
    replaced, only a fixed, bounded integer range. 16-bit (65,536 levels)
    keeps gradients smooth (no visible banding) while still being far
    smaller and more broadly compatible than 32-bit float EXR.

    Returns the compressed PNG bytes directly instead of writing to disk --
    the only reason a prior version wrote to a temp file was to hand bytes
    to upload_with_retry, which read the file right back. That write/read/
    delete round trip per frame was pure overhead.
    """
    depth_uint16 = np.clip(depth_map, 0.0, 1.0)
    depth_uint16 = (depth_uint16 * 65535.0 + 0.5).astype(np.uint16)
    ok, encoded = cv2.imencode(".png", depth_uint16)
    if not ok:
        raise RuntimeError("cv2.imencode failed to encode depth frame as PNG")
    return encoded.tobytes()


def upload_with_retry(
    supabase: Client,
    bucket: str,
    remote_path: str,
    data: bytes,
    max_attempts: int = 4,
):
    """Uploads bytes to Supabase Storage, retrying on transient network
    errors (timeouts, connection resets) with exponential backoff.

    A sequence upload is hundreds to thousands of individual HTTP requests
    -- one flaky read timeout on any single one of them used to kill the
    entire job outright, even after everything before it had succeeded.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            supabase.storage.from_(bucket).upload(
                file=data,
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
        # infer_video_depth() does its own internal scale-and-shift alignment
        # ACROSS the frames passed to a single call, so naively processing
        # independent chunks would otherwise mean a real depth-scale
        # discontinuity at each chunk boundary (not just a display/
        # normalization artifact) -- see CHUNK_OVERLAP_FRAMES and flush_chunk
        # below for how each chunk gets aligned onto the previous one's scale
        # before it's normalized and written out.
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
        # Raw (unnormalized) depth for the last CHUNK_OVERLAP_FRAMES frames
        # of the previous chunk's output -- the alignment reference the next
        # chunk's overlap frames get fit against. None until the first chunk
        # has been processed, or after a resumed run skips a chunk outright
        # (see the DECORD_AVAILABLE loop below), in which case the next
        # processed chunk just starts its own fresh scale, same as before
        # this fix, instead of aligning to a chunk we never actually ran.
        prev_tail_depth = None
        report_progress()  # initial 0/total_frames so the frontend has a real number immediately

        # Persistent upload pool + the previous chunk's not-yet-confirmed
        # upload futures. Pipelining (see flush_chunk below): each chunk
        # waits only for the PREVIOUS chunk's uploads before dispatching
        # its own, so the next chunk's decode+inference can start right
        # away while uploads run in the background instead of leaving the
        # GPU idle for the whole upload phase every chunk.
        upload_executor = ThreadPoolExecutor(max_workers=UPLOAD_CONCURRENCY)
        pending_uploads: list = []

        def wait_for_pending_uploads():
            nonlocal pending_uploads
            for future in pending_uploads:
                future.result()
            pending_uploads = []

        def flush_chunk(buffer, overlap=0):
            nonlocal frame_index, chunk_num, prev_depth_min, prev_depth_max, prev_tail_depth, pending_uploads
            if len(buffer) == 0:
                return
            chunk_num += 1
            # `buffer` is a plain list of per-frame arrays from the cv2 path,
            # or an ndarray already shaped [N, H, W, 3] straight from
            # decord's get_batch().asnumpy() -- decord already returns RGB
            # (see utils/dc_utils.py: the decord branch does no color
            # conversion, only the cv2 fallback branch does), so no
            # cv2.cvtColor is needed here either way. The leading `overlap`
            # frames (if any) duplicate the tail of the previous chunk --
            # included here only so inference re-runs on them for alignment,
            # dropped from the output below.
            chunk_frames = np.stack(buffer, axis=0) if isinstance(buffer, list) else buffer
            n_total = len(chunk_frames)
            n_new = n_total - overlap
            print(
                f"[chunk {chunk_num}] Running depth inference on frames "
                f"{frame_index - overlap}-{frame_index + n_new - 1} "
                f"({n_total} frames, {overlap} re-run for cross-chunk alignment)..."
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

            # Not calling torch.cuda.empty_cache() here on purpose. The
            # original SIGKILL/exit-137 crashes that motivated chunking in
            # the first place (see CHUNK_SIZE_FRAMES above) look like a
            # host-RAM OOM, not a CUDA OOM -- chunking itself is what fixed
            # that, not this call. Releasing cached GPU memory every chunk
            # just forces a device sync plus a fresh cudaMalloc on the next
            # chunk instead of letting the caching allocator reuse the
            # blocks it already has reserved, adding a stall between every
            # chunk for no real benefit. Revisit if GPU OOM actually shows
            # up in production logs.
            del chunk_frames

            # Align this chunk's depth scale onto the previous chunk's,
            # using the overlap frames both chunks were run on -- fixes the
            # actual depth-scale discontinuity at the chunk boundary (see
            # CHUNK_OVERLAP_FRAMES above), as opposed to the normalization
            # smoothing below, which only papers over a display-range jump
            # and can't fix the underlying values being on a different
            # scale to begin with.
            if overlap > 0 and prev_tail_depth is not None:
                scale, shift = compute_scale_and_shift(
                    chunk_depths[:overlap].astype(np.float32),
                    prev_tail_depth.astype(np.float32),
                    np.ones_like(prev_tail_depth, dtype=np.float32),
                )
                chunk_depths = chunk_depths * scale + shift
                chunk_depths[chunk_depths < 0] = 0
                print(
                    f"[chunk {chunk_num}] Aligned to previous chunk's depth scale "
                    f"(scale={scale:.4f}, shift={shift:.4f}) using {overlap} "
                    "overlapping frame(s)."
                )
            elif overlap > 0:
                print(
                    f"[chunk {chunk_num}] {overlap} overlapping frame(s) present but no "
                    "prior reference available (first chunk after a resumed skip) -- "
                    "skipping cross-chunk alignment."
                )

            # Drop the overlap frames now that they've served their purpose
            # -- they duplicate frames already accounted for by the
            # previous chunk (or, on a resume, already uploaded).
            chunk_depths = chunk_depths[overlap:]

            # Save this chunk's own tail as the alignment reference for the
            # *next* chunk, in the same (now-aligned) scale as what's about
            # to be written out below.
            tail_len = min(CHUNK_OVERLAP_FRAMES, len(chunk_depths))
            prev_tail_depth = chunk_depths[-tail_len:].copy() if tail_len > 0 else None

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
            #    (via the np.clip in encode_depth_png16), localized to the
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

            # Encode + upload every frame in this chunk in parallel instead
            # of one at a time -- each frame is an independent PNG encode
            # (CPU) plus an independent HTTP request (network-bound).
            # Frame index is computed up front per frame, so completion
            # order doesn't matter.
            chunk_start_index = frame_index

            def encode_and_upload(offset: int):
                idx = chunk_start_index + offset
                frame_filename = f"frame_{idx:04d}.png"
                png_bytes = encode_depth_png16(chunk_depths[offset])
                upload_with_retry(
                    get_supabase_for_upload(), output_bucket, f"{output_prefix}/{frame_filename}", png_bytes
                )

            # Wait for the PREVIOUS chunk's uploads (not this one) before
            # dispatching this chunk's -- bounds in-flight uploads to one
            # chunk's worth (so PNG bytes + chunk_depths for at most two
            # chunks are ever alive at once) and surfaces a permanent
            # upload failure (.result() re-raises) within one chunk of it
            # happening instead of silently deferring it to the end of the
            # job. This chunk's uploads are then dispatched and NOT waited
            # on here -- they run in the background while the caller moves
            # on to decoding/inferring the next chunk, which is the actual
            # pipelining: the GPU is no longer idle during the upload phase.
            wait_for_pending_uploads()
            pending_uploads = [
                upload_executor.submit(encode_and_upload, i) for i in range(n_new)
            ]

            frame_index += n_new
            print(f"[chunk {chunk_num}] Dispatched {n_new} frame(s) for background upload, {frame_index} total so far.")
            report_progress()

        try:
            if DECORD_AVAILABLE:
                for start in range(0, total_frames, CHUNK_SIZE_FRAMES):
                    end = min(start + CHUNK_SIZE_FRAMES, total_frames)
                    if uploaded_frames and all(i in uploaded_frames for i in range(start, end)):
                        print(f"[chunk] frames {start}-{end - 1} already uploaded, skipping.")
                        frame_index = end
                        # A skipped chunk means the next processed chunk has no
                        # prior in-memory depth to align against -- see the
                        # prev_tail_depth comment above.
                        prev_tail_depth = None
                        report_progress()
                        continue
                    # Re-read the previous chunk's last CHUNK_OVERLAP_FRAMES
                    # frames too (decord can seek, so this is just re-decoding,
                    # not re-inferring anything we didn't already infer) so
                    # flush_chunk can align this chunk onto the previous one's
                    # depth scale. None for the very first chunk.
                    overlap = min(CHUNK_OVERLAP_FRAMES, start)
                    chunk = vr.get_batch(list(range(start - overlap, end))).asnumpy()
                    flush_chunk(chunk, overlap=overlap)
            else:
                # cv2 has no reliable random-access seek here (see the DECORD_AVAILABLE
                # comment at the top of this file), so unlike the decord path this
                # can't skip decoding already-uploaded frames -- it re-decodes
                # everything, but upload_with_retry's upsert makes re-uploading
                # already-present frames a harmless no-op rather than a failure.
                # Since there's no seeking, the last CHUNK_OVERLAP_FRAMES raw
                # frames of each chunk are kept around (cheap -- just a handful
                # of images) and prepended to the next chunk's buffer instead,
                # for the same alignment purpose as the decord path's re-seek.
                buffer = []
                overlap_carry = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    buffer.append(frame)
                    if len(buffer) >= CHUNK_SIZE_FRAMES:
                        overlap = len(overlap_carry)
                        full_buffer = overlap_carry + buffer
                        flush_chunk(full_buffer, overlap=overlap)
                        overlap_carry = full_buffer[-CHUNK_OVERLAP_FRAMES:]
                        buffer = []
                cap.release()
                if buffer:
                    overlap = len(overlap_carry)
                    flush_chunk(overlap_carry + buffer, overlap=overlap)

            # Wait for the LAST chunk's uploads too -- the manifest written
            # below must not claim the job complete until every frame has
            # actually finished uploading, not just been dispatched.
            wait_for_pending_uploads()
        finally:
            upload_executor.shutdown(wait=True)

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
