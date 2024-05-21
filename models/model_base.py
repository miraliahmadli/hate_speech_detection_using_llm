import gc
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dataset import HateSpeechDataset


def eval(model, test_dataloader, device):
  total_loss = 0
  all_predictions = []
  all_labels = []

  model.eval()
  with torch.no_grad():
    for batch in test_dataloader:
      input_ids = batch['input_ids'].to(device, dtype = torch.long)
      attention_mask = batch['attention_mask'].to(device, dtype = torch.long)
      labels = batch['labels'].to(device, dtype = torch.long)

      outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
      total_loss += outputs.loss.item()

      _, predictions = torch.max(outputs.logits, dim=1)

      all_predictions.extend(predictions.detach().cpu().tolist())
      all_labels.extend(labels.detach().cpu().tolist())

  eval_loss = total_loss / len(test_dataloader)
  accuracy = accuracy_score(all_labels, all_predictions)
  f1_scores = f1_score(all_labels, all_predictions, average=None)
  return eval_loss, accuracy, f1_scores


def train_epoch(model, train_dataloader, optimizer, device):
  model.train()

  total_loss = 0
  acc = 0
  total_samples = 0

  for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        input_ids = batch['input_ids'].to(device, dtype = torch.long)
        attention_mask = batch['attention_mask'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.long)

        outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        _, predictions = torch.max(outputs.logits, dim=1)
        acc += (predictions == labels).float().sum().item()
        total_samples += len(labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
          print(f"Iteration: {i+1}, Loss: {total_loss / (i+1):.4f}, Accuracy: {acc / total_samples:.4f}")

  epoch_loss = total_loss / len(train_dataloader)
  epoch_accuracy = acc / total_samples
  return epoch_loss, epoch_accuracy


def sft_pipeline(model_name, 
             train_data, 
             test_data, 
             max_len, 
             train_batch_size, 
             eval_batch_size, 
             learning_rate, 
             epochs, 
             num_labels,
             ckpt_path):
    '''
    Args:
        model_name (str): Model name from Huggingface.
        train_data (list): List of tuples containing the training data.
            Format: [(text, label)]
        test_data (list): List of tuples containing the test data.
            Format: [(text, label)]
        max_len (int): Maximum length of the input sequence.
        train_batch_size (int): Batch size for training.
        eval_batch_size (int): Batch size for evaluation.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of epochs for training.
        num_labels (int): Number of labels in the classification task.
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                               num_labels=num_labels)
    model.to(device)

    # Create datasets and dataloaders.
    train_dataset = HateSpeechDataset(train_data, tokenizer, max_len)
    test_dataset = HateSpeechDataset(test_data, tokenizer, max_len)

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=train_batch_size,
                                  shuffle=True, 
                                  worker_init_fn = lambda id: np.random.seed(42))
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=eval_batch_size)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    best_eval_loss = float('inf')
    best_accuracy = 0
    best_f1_scores = 0
    for epoch in range(epochs):
        print(f'-- Epoch {epoch} --')
        epoch_loss, epoch_accuracy = train_epoch(model, train_dataloader, optimizer, device)
        print(f'[Train] Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

        eval_loss, eval_accuracy, eval_f1_scores = eval(model, test_dataloader, device)
        print(f'[Eval] Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}, Macro-F1: {eval_f1_scores.mean():.4f}\n')
        
        if eval_accuracy > best_accuracy:
            best_eval_loss = eval_loss
            best_accuracy = eval_accuracy
            best_f1_scores = eval_f1_scores
            model.save_pretrained(ckpt_path)

    return best_eval_loss, best_accuracy, best_f1_scores

