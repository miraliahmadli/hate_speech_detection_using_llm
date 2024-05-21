import torch
from torch.utils.data import Dataset

class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512, instruct=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        if instruct:
            system_prompt = "This is a hate speech detection model. Please provide a sentence and the model will predict if it is hate speech or not."
            message = {"role": "system", "content": system_prompt}
            self.system = tokenizer.apply_chat_template([message], tokenize=False)
        else:
            self.system = ""

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Format instruction
        message = {"role": "user", "content": text}
        text = self.tokenizer.apply_chat_template([message],
                                                  tokenize=False, 
                                                  add_generation_prompt=True)

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }
