# Prisma: Text-to-Video Generation Model

Prisma is an advanced text-to-video generation model that transforms textual descriptions into high-quality video content. Built on state-of-the-art transformer architecture, Prisma enables creative video generation from natural language prompts.

## Features

- High-quality video generation from text descriptions
- Support for various video resolutions and frame rates
- Efficient inference with optimized model architecture
- Easy-to-use API for integration

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from prisma import PrismaModel

model = PrismaModel()
video = model.generate("A beautiful sunset over the ocean")
video.save("output.mp4")
```

## Inference Example

To generate a video from a text prompt, use the inference script as follows:

```bash
python prisma/inference.py --model-path /path/to/model/weights --prompt "A beautiful sunset over the ocean" --output output.mp4 --max-frames 32 --frame-size 256 256
```

This will load the trained model, generate a video based on the provided prompt, and save it as `output.mp4`.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details. 