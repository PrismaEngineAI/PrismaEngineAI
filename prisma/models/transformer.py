import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class PrismaConfig(PretrainedConfig):
    model_type = "prisma"
    
    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

class PrismaTransformer(PreTrainedModel):
    config_class = PrismaConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size
            ) for _ in range(config.num_hidden_layers)
        ])
        self.final_layer = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embeddings(input_ids)
        
        for layer in self.encoder:
            x = layer(x, src_key_padding_mask=attention_mask)
            
        return self.final_layer(x) 