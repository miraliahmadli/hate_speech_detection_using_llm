import gc
import evaluate
import numpy as np
import torch
from datasets import concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2Config, GPT2ForSequenceClassification, Trainer, TrainingArguments

from llama_index.legacy.embeddings import HuggingFaceEmbedding
from RAG.retriever import augment_dataset, VectorDBRetriever
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
def evaluate_model(model_checkpoint_path, tokenizer_checkpoint_path, val_set, with_RAG=False):
    gc.collect()
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the tokenizer and model
    if model_checkpoint_path == "GroNLP/hateBERT":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("GroNLP/hateBERT")
    elif model_checkpoint_path == "tomh/toxigen_hatebert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("tomh/toxigen_hatebert")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2ForSequenceClassification.from_pretrained(model_checkpoint_path, num_labels=2)
        model.config.pad_token_id = model.config.eos_token_id
    model.to(device)
    
    print("Loaded Model and Tokenizer")
    
    # Create the evaluation dataset
    max_length = 256
    if with_RAG:
        max_length += 1024
    test_dataset = HateSpeechDataset(val_set, tokenizer, max_length=max_length)
    
    print("Loaded Dataset")

    # Define evaluation arguments
    cur_batch_size = 16
    if with_RAG:
        cur_batch_size = 8
    eval_args = TrainingArguments(
        per_device_eval_batch_size=cur_batch_size,  # Adjust as needed
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
val_datasets = [tg_val, tw_val, berkeley_val, gender_val, cad_val]
concat_val = concatenate_datasets(val_datasets)
val_datasets += [concat_val]

with_RAG = True

# Define paths
tokenizer_checkpoint_path = "gpt2"  # Path to the tokenizer
model_checkpoint_path = "./checkpoints/final_sft_checkpoint"

model_checkpoint_path = "tomh/toxigen_hatebert"
model_checkpoint_path = "GroNLP/hateBERT"

os.environ['TRANSFORMERS_CACHE'] = '/scratch/izar/ahmadli/.cache/huggingface'

# Evaluate the model
if with_RAG:
    embed_model = HuggingFaceEmbedding(
        model_name="distilbert/distilbert-base-uncased"
    )
    retriever = VectorDBRetriever(embed_model, generate_vector_store=False)
    for data in val_datasets:
        augmented_val_dataset = augment_dataset(data, retriever)
        evaluate_model(model_checkpoint_path, tokenizer_checkpoint_path, augmented_val_dataset, with_RAG)
else:
    for data in val_datasets:
        evaluate_model(model_checkpoint_path, tokenizer_checkpoint_path, data, with_RAG)
