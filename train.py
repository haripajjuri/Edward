import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizer
import json
from model import Edward
from config import modelConfig
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

tokenizer = LlamaTokenizer.from_pretrained("./my_tokenizer")

g = torch.Generator()
g.manual_seed(4241)

current_epoch = 0
total_epochs = 2

current_shard = 0
total_shards = 5

current_chunk = 0
total_chunks = 10

log_file = "logs.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"-> running on {device}")
model_config = modelConfig(vocab_size=tokenizer.vocab_size)

model = Edward(model_config)
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-4 ,weight_decay=0.01)
#scheduler = StepLR(optimizer=optimizer, step_size=5, gamma=0.9)

scheduler = get_linear_schedule_with_warmup(optimizer= optimizer, num_training_steps=100000 , num_warmup_steps = int(0.05 * 100000)  )

if os.path.exists(log_file):
    with open(log_file, 'r', encoding="utf-8") as f:
        logs = json.load(f)
else:
    logs = []


class trainDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_len):
        with open(filepath, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        obj = self.data[index]
        #text = f"<human>{obj['question']} <robot>{obj['answer']}"
        text = obj["text"].replace("\n", "<nl>")

        tokenized = self.tokenizer(
            text,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt",
            max_length = self.max_len,
            add_special_tokens = True
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        lables = input_ids.clone()
        lables[lables == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids[:-1],
            'attention_mask' : attention_mask[:-1], 
            'lables': lables[1:]
        }

try:
    checkpoint = torch.load("checkpoint.pth", map_location='cpu')

    current_epoch = checkpoint['epoch']+1
    current_chunk = checkpoint['chunk']
    current_shard = checkpoint['shard']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"-> loaded from checkpoint ✅")
except:
    print(f"-> no checkpoint found ⚠️")
    print(f"-> starting from scratch 🔥")


model.train()
for shard in range(current_shard, total_shards):
    for chunk in range(current_chunk, total_chunks):


        dataset = trainDataset(filepath=f"train/{shard+1}/chunk_{chunk+1}.json", tokenizer=tokenizer, max_len=512)
        dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, generator= g)

        val_dataset = trainDataset(filepath="val/validation_set.json", tokenizer=tokenizer, max_len=512)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)

        print(f"training shard {shard+1}, chunk {chunk+1}")


        for epoch in range( current_epoch, total_epochs):
            
            total_loss = 0
            for batch in tqdm( dataloader, total=len(dataloader), desc=f"epoch {epoch+1}", unit="batch", colour="green" ):
                optimizer.zero_grad()

                input_tokens = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                lables = batch["lables"].to(device)

                logits, loss = model( inputs = input_tokens, attention_mask=attention_mask, lables = lables)

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_dataloader, total=len(val_dataloader), desc="validation", unit="batch", colour="yellow"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    lables = batch["lables"].to(device)
                    outputs, validation_loss = model(input_ids, attention_mask=attention_mask, lables = lables)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            model.train()

            print(f"training loss -> {avg_loss:.4f}  |  validation loss -> {avg_val_loss:.4f}")

            torch.save({
                'epoch':epoch,
                'shard':shard,
                'chunk':chunk,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict()
            }, "checkpoint.pth" )

            logs.append({
                'shard':shard+1,
                'chunk':chunk+1,
                'epoch':epoch+1,
                'train_loss':avg_loss,
                'val_loss':avg_val_loss,
                'learning_rate':scheduler.get_last_lr()[0]
            })
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=4)
                
            print(f"-> saved to checkpoint 🏁 | output logged.") 
        current_epoch = 0
    current_chunk = 0








'''
dataset = trainDataset(filepath="new_filtered.json", tokenizer=tokenizer, max_len=512)
dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)

try:
    checkpoint = torch.load("checkpoint.pth", map_location='cpu')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"✅ loaded from checkpoint")
    print(f"starting from epoch -> {start_epoch}")
except:
    print(f"❗ no checkpoint found.")
    print(f"⚠️ starting from scratch")



model.train()
for i in range(start_epoch, total_epochs):
    total_loss = 0
    for n,batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_tokens = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lables = batch["lables"].to(device)

        logits, loss = model( inputs = input_tokens, attention_mask=attention_mask, lables = lables)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if n%100 == 0:
            print(f"{n} batches completed")
    scheduler.step()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch -> {i+1} | loss -> {avg_loss:.4f}")

    if (i+1) % 5 == 0:
        torch.save({
            'epoch' : i+1,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'scheduler_state_dict':scheduler.state_dict()
        }, "checkpoint.pth")
        print(f"----- checkpoint saved -----")
    
    if (i+1) == total_epochs:
        torch.save({
            'epoch' : 0,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'scheduler_state_dict':scheduler.state_dict()
        }, "checkpoint.pth")
        print("Training completed ✅")

'''