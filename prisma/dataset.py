import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np
from transformers import AutoTokenizer

class VideoTextDataset(Dataset):
    def __init__(
        self,
        video_paths,
        text_descriptions,
        tokenizer_name="gpt2",
        max_frames=32,
        frame_size=(256, 256)
    ):
        self.video_paths = video_paths
        self.text_descriptions = text_descriptions
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_frames = max_frames
        self.frame_size = frame_size
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        text = self.text_descriptions[idx]
        
        # Load and preprocess video
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        # Pad frames if necessary
        while len(frames) < self.max_frames:
            frames.append(np.zeros((*self.frame_size, 3), dtype=np.uint8))
            
        # Convert frames to tensor
        frames = torch.from_numpy(np.stack(frames)).float() / 255.0
        
        # Tokenize text
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        
        return {
            "frames": frames,
            "input_ids": text_tokens["input_ids"].squeeze(),
            "attention_mask": text_tokens["attention_mask"].squeeze()
        } 