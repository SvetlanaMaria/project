import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
import os
import torch.nn.functional as F

def clean_lyrics(lyrics):
    if isinstance(lyrics, str):
        lyrics = re.sub(r"[^a-zA-Z0-9\s]+", "", lyrics)  # Remove special characters
        lyrics = lyrics.strip()  # Remove leading/trailing whitespaces
    return lyrics

class LyricsDataset(Dataset):
    def __init__(self, lyrics_data, tokenizer, max_length):
        self.lyrics_data = lyrics_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.lyrics_data)

    def __getitem__(self, idx):
        lyrics = self.lyrics_data[idx]
        if pd.isnull(lyrics) or lyrics == "":
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        else:
            encoding = self.tokenizer.encode_plus(
                lyrics,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()

        return input_ids, attention_mask

# Load the Genius Lyrics dataset from CSV files
folder_path = '/home/allthingsbarcelona/project/archive-4'
file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

# Preprocess the dataset
lyrics_data = []

# Iterate over each file and extract the lyrics
for file_path in file_paths:
    df = pd.read_csv(file_path)
    lyrics = df["lyrics"].tolist()
    
    # Remove special characters and symbols from the lyrics
    lyrics = [clean_lyrics(lyric) for lyric in lyrics]
    lyrics_data.extend(lyrics)
print("Total lyrics:", len(lyrics_data))
print("Sample lyrics:")
for i in range(min(5, len(lyrics_data))):
    print(lyrics_data[i])
# Load the pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
model = AutoModelForCausalLM.from_pretrained("t5-base").to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Hyperparameters
batch_size = 4
max_length = 512
num_epochs = 5
learning_rate = 1e-4

# Prepare the dataset
dataset = LyricsDataset(lyrics_data, tokenizer, max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Fine-tuning
model.train()
print("Training loop started")

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        input_ids, attention_mask = batch
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()

        # Print the loss value for each batch
        print(f"Epoch {epoch+1} - Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item()}")
    
    # Print the average loss value for the epoch
    print(f"Epoch {epoch+1} completed. Average Loss: {loss.item()}")

# Generate lyrics using the fine-tuned model
model.eval()
prompt = "Let it be"
input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt')
output = model.generate(input_ids, max_length=100, num_return_sequences=1)

generated_lyrics = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated lyrics:", generated_lyrics)
