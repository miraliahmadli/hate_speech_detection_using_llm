import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import concatenate_datasets

from models.model_base import sft_pipeline
from dataloaders.common_utils import *

if __name__ == "__main__":
    MODEL_NAME = 'distilbert-base-uncased'
    MAX_LEN = 256
    TRAIN_BATCH_SIZE = 128
    EVAL_BATCH_SIZE = 128
    EPOCHS = 3
    LEARNING_RATE = 1e-5
    NUM_LABELS = 2
    OUTPUT_DIR = './checkpoints'
    CKPT = './checkpoints/finetune_model'

    tg_train, tg_val = load_and_process_toxigen()
    tw_train, tw_val = load_and_process_twitter_data()
    berkeley_train, berkeley_val = load_and_process_berkeley_data()
    gender_train, gender_val = load_and_process_gender_hate_speech_data()
    cad_train, cad_val = load_and_process_cad()
    train_datasets = [tg_train, tw_train, berkeley_train, gender_train, cad_train]
    val_datasets = [tg_val, tw_val, berkeley_val, gender_val, cad_val]

    concat_train = concatenate_datasets(train_datasets)
    concat_val = concatenate_datasets(val_datasets)
    
    sft_pipeline(
        model_name=MODEL_NAME, 
        train_data=concat_train, 
        test_data=concat_val, 
        max_len=MAX_LEN, 
        train_batch_size=TRAIN_BATCH_SIZE, 
        eval_batch_size=EVAL_BATCH_SIZE, 
        learning_rate=LEARNING_RATE, 
        epochs=EPOCHS, 
        num_labels=NUM_LABELS,
        output_dir=OUTPUT_DIR,
        ckpt_path=CKPT,
    )


