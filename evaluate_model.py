import evaluate
import numpy as np
import torch
from datasets import concatenate_datasets
from transformers import AutoTokenizer, GPT2Config, GPT2ForSequenceClassification, Trainer, TrainingArguments
from dataloaders.sft_dataset import HateSpeechDataset
from dataloaders.common_utils import *
import os
os.environ['WANDB_DISABLED'] = 'true'

# Load the evaluation metric
metric = evaluate.load('accuracy')

# Define the compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# Define the evaluation pipeline
def evaluate_model(model_checkpoint_path, tokenizer_checkpoint_path, val_set):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2ForSequenceClassification.from_pretrained(model_checkpoint_path, num_labels=2)
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)
    
    print("Loaded Model and Tokenizer")
    
    # Create the evaluation dataset
    test_dataset = HateSpeechDataset(val_set, tokenizer, max_length=256)
    
    print("Loaded Dataset")

    # Define evaluation arguments
    eval_args = TrainingArguments(
        per_device_eval_batch_size=16,  # Adjust as needed
        dataloader_drop_last=False,
        output_dir="./results/eval_results/gpt2_base/test/",  # Adjust as needed
    )

    # Create the Trainer instance
    trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Perform the evaluation
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")

# Assuming you have val_set ready
tg_train, tg_val = load_and_process_toxigen()
tw_train, tw_val = load_and_process_twitter_data()
berkeley_train, berkeley_val = load_and_process_berkeley_data()
gender_train, gender_val = load_and_process_gender_hate_speech_data()
cad_train, cad_val = load_and_process_cad()
# train_datasets = [tg_train, tw_train, berkeley_train, gender_train, cad_train]
# val_datasets = [tg_val, tw_val, berkeley_val, gender_val, cad_val]
val_set = [tw_val]  # Load or define your validation set here
concat_val = concatenate_datasets(val_set)

# Define paths
# model_checkpoint_path = "./checkpoints/final_sft_checkpoint"  # Path to the trained model checkpoint
model_checkpoint_path = "./checkpoints/gpt2_base/checkpoint-34000"  # Path to the trained model checkpoint
tokenizer_checkpoint_path = "gpt2"  # Path to the tokenizer

# Evaluate the model
evaluate_model(model_checkpoint_path, tokenizer_checkpoint_path, concat_val)
