import os
import glob
import tempfile
import cv2
import torch
import numpy as np
import OpenEXR
import Imath
from supabase import create_client, Client
import runpod

# --- Import Video Depth Anything Model ---
# Assumes the repo is cloned in /app/Video-Depth-Anything (configured in Dockerfile)
import sys
sys.path.append("/app/Video-Depth-Anything")

from video_depth_anything.video_depth_anything import VideoDepthAnything


# --- Environment Setup ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "Video-Depth-Anything-Small") # Options: Small, Base, Large


def get_supabase() -> Client:
    """Initializes and returns the Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY environment variables.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def load_model() -> VideoDepthAnything:
    """Loads the Video Depth Anything model onto GPU or CPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Video Depth Anything model ({MODEL_NAME}) on {device}...")
    
    # Model parameters preset
    model_configs = {
        'Video-Depth-Anything-Small': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'Video-Depth-Anything-Base': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'Video-Depth-Anything-Large': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    config = model_configs.get(MODEL_NAME, model_configs['Video-Depth-Anything-Base'])
    model = VideoDepthAnything(**config)
    
    # Load model weights (expects checkpoint in /app/checkpoints or huggingface cache)
    checkpoint_path = f"/app/checkpoints/{MODEL_NAME.lower()}.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    
    model = model.to(device).eval()
    return model


def save_exr_32bit(depth_map: np.ndarray, output_path: str):
    """
    Saves a 2D float32 numpy array as a single-channel 32-bit Float EXR image.
    Ideal for VFX, Nuke, Blender, and Cinema 4D displacement workflows.
    """
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
    output_prefix: str = "depth_sequence"
):
    """
    Main processing pipeline:
    1. Download video from Supabase.
    2. Run Video Depth Anything inference.
    3. Export 32-bit EXR frames.
    4. Upload frame sequence back to Supabase.
    """
    supabase = get_supabase()
    model = load_model()

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_video_path = os.path.join(tmp_dir, "input.mp4")
        exr_output_dir = os.path.join(tmp_dir, "exr_frames")
        os.makedirs(exr_output_dir, exist_ok=True)

        # -------------------------------------------------------------
        # Step 1: Download Video from Supabase
        # -------------------------------------------------------------
        print(f"[1/4] Downloading '{video_key}' from bucket '{input_bucket}'...")
        video_bytes = supabase.storage.from_(input_bucket).download(video_key)
        with open(local_video_path, "wb") as f:
            f.write(video_bytes)

        # -------------------------------------------------------------
        # Step 2: Extract Frames and Run Inference
        # -------------------------------------------------------------
        print("[2/4] Reading video frames...")
        cap = cv2.VideoCapture(local_video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR (OpenCV) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

        if not frames:
            raise ValueError("No frames could be extracted from the provided video file.")

        print(f"[3/4] Running Depth Inference across {len(frames)} frames...")
        # depth_array shape: (N_frames, Height, Width) in normalized float values
        with torch.no_grad():
            depths = model.infer_video(frames)

        # -------------------------------------------------------------
        # Step 3: Write EXR Files & Upload to Supabase
        # -------------------------------------------------------------
        print(f"[4/4] Writing EXRs and uploading sequence to '{output_bucket}'...")
        for i, depth_frame in enumerate(depths):
            frame_filename = f"frame_{i:04d}.exr"
            local_exr_path = os.path.join(exr_output_dir, frame_filename)
            
            # Save local 32-bit EXR
            save_exr_32bit(depth_frame, local_exr_path)

            def handler(job):
    """
    RunPod Serverless Handler Function.
    Extracts parameters from the incoming event payload.
    """
    job_input = job.get("input", {})

    # Extract dynamic payload variables passed from Supabase/WeWeb
    input_bucket = job_input.get("input_bucket", "raw-videos")
    video_key = job_input.get("video_key", "sample.mp4")
    output_bucket = job_input.get("output_bucket", "depth-outputs")
    output_prefix = job_input.get("output_prefix", "sequence_001")

    # Call your processing function
    process_video_depth(
        input_bucket=input_bucket,
        video_key=video_key,
        output_bucket=output_bucket,
        output_prefix=output_prefix
    )

    return {
        "status": "success",
        "output_prefix": output_prefix,
        "message": f"Successfully processed depth sequence for {video_key}"
    }


if __name__ == "__main__":
    # Starts listening for RunPod Serverless jobs
    runpod.serverless.start({"handler": handler})