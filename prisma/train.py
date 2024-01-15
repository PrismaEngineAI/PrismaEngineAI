import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from models.transformer import PrismaConfig, PrismaTransformer

def train(
    model,
    train_dataloader,
    num_epochs,
    learning_rate,
    warmup_steps,
    device
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=len(train_dataloader) * num_epochs
    )
    
    model.train()
    for epoch in range(num_epochs):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = torch.nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1)
            )
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.set_postfix({"loss": loss.item()})

if __name__ == "__main__":
    config = PrismaConfig()
    model = PrismaTransformer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # TODO: Implement dataset and dataloader
    train_dataloader = DataLoader([])
    
    train(
        model=model,
        train_dataloader=train_dataloader,
        num_epochs=10,
        learning_rate=1e-4,
        warmup_steps=1000,
        device=device
    ) 