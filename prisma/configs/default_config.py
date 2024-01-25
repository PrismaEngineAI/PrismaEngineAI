from dataclasses import dataclass

@dataclass
class ModelConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    vocab_size: int = 50257
    max_position_embeddings: int = 1024
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    max_frames: int = 32
    frame_size: tuple = (256, 256)
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

@dataclass
class DatasetConfig:
    train_data_path: str = "data/train"
    val_data_path: str = "data/val"
    tokenizer_name: str = "gpt2"
    max_text_length: int = 77 