import torch
from transformers import AutoTokenizer
from models.transformer import PrismaConfig, PrismaTransformer
from utils.video_utils import save_video

class PrismaInference:
    def __init__(self, model_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = PrismaConfig()
        self.model = PrismaTransformer(self.config)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def generate_video(self, prompt, max_frames=32, frame_size=(256, 256)):
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            # For demonstration, we treat outputs as video frames (mockup)
            # In a real model, this would be replaced with actual video decoding logic
            frames = outputs[:, :max_frames, :].cpu()  # Mockup: shape [batch, frames, features]
            # Convert to image frames (mockup)
            frames = torch.rand((max_frames, *frame_size, 3))  # Placeholder
        return frames

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prisma Text-to-Video Inference")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model weights")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video file path")
    parser.add_argument("--max-frames", type=int, default=32, help="Number of frames to generate")
    parser.add_argument("--frame-size", type=int, nargs=2, default=[256, 256], help="Frame size (H W)")
    args = parser.parse_args()

    infer = PrismaInference(args.model_path)
    frames = infer.generate_video(args.prompt, max_frames=args.max_frames, frame_size=tuple(args.frame_size))
    save_video(frames, args.output)
    print(f"Video saved to {args.output}") 