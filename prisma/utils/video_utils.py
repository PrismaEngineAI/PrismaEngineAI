import cv2
import numpy as np
import torch
from typing import List, Tuple

def extract_frames(
    video_path: str,
    max_frames: int = 32,
    frame_size: Tuple[int, int] = (256, 256)
) -> np.ndarray:
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    # Pad frames if necessary
    while len(frames) < max_frames:
        frames.append(np.zeros((*frame_size, 3), dtype=np.uint8))
    
    return np.stack(frames)

def save_video(
    frames: torch.Tensor,
    output_path: str,
    fps: int = 30
) -> None:
    """Save frames as a video file."""
    frames = (frames.cpu().numpy() * 255).astype(np.uint8)
    height, width = frames.shape[1:3]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()

def normalize_frames(frames: torch.Tensor) -> torch.Tensor:
    """Normalize frames to range [-1, 1]."""
    return (frames - 0.5) * 2

def denormalize_frames(frames: torch.Tensor) -> torch.Tensor:
    """Denormalize frames from range [-1, 1] to [0, 1]."""
    return (frames + 1) / 2 