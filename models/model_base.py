import evaluate
metric = evaluate.load('accuracy')

import gc
import numpy as np
import torch
from transformers import AutoTokenizer, \
  AutoModelForSequenceClassification, DataCollatorForLanguageModeling, \
  Trainer, TrainingArguments

from dataloaders.sft_dataset import HateSpeechDataset

# def compute_metrics(eval_preds):
#     # metric = evaluate.load("accuracy")
#     logits, labels = eval_preds
#     predictions = logits.argmax(dim=-1)
#     # return metric.compute(predictions=predictions, references=labels)
#     return {"accuracy": (predictions == labels).mean().item()}
  

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def sft_pipeline(model_name, 
             train_data, 
             test_data, 
             max_len, 
             train_batch_size, 
             eval_batch_size, 
             learning_rate, 
             epochs, 
             num_labels,
             output_dir,
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
                                                               num_labels=num_labels,)
                                                               # device_map='auto')
    model.to(device)
    print("Loaded Models")

    # Create datasets and dataloaders.
    train_dataset = HateSpeechDataset(train_data, tokenizer, max_len, False)
    test_dataset = HateSpeechDataset(test_data, tokenizer, max_len, False)

    print("Loaded Datasets")

    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
      output_dir=output_dir,
      evaluation_strategy = "steps",
      num_train_epochs = epochs,
      per_device_train_batch_size=train_batch_size,
      per_device_eval_batch_size=eval_batch_size,
      logging_steps=500,
      learning_rate=learning_rate,
      save_steps=1000,
      weight_decay=0.01,
      report_to="tensorboard",
      logging_dir="./tensorboard/sft",)

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        # data_collator=data_collator,
        compute_metrics=compute_metrics,)

    trainer.train()

    trainer.save_model(ckpt_path)

