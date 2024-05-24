import argparse
import torch
import os
from models.model_base import sft_pipeline
from datasets import load_dataset, Dataset
import pandas as pd


def label_annotations(annotated):
    # Annotations should be the annotated dataset
    label = ((annotated.toxicity_ai + annotated.toxicity_human) > 5.5).astype(int)
    labeled_annotations = pd.DataFrame()
    labeled_annotations["text"] = [i for i in annotated.text.tolist()]
    labeled_annotations["label"] = label
    return labeled_annotations

def get_dataset(name='tweets_hate_speech_detection'):
    if name == 'tweets':
        dataset = load_dataset('tweets_hate_speech_detection')
        trainval_data, test_data = dataset['train'], dataset['test']
        trainval_data = trainval_data.rename_column('tweet', 'text')
        test_data = test_data.rename_column('tweet', 'text')
        splitted = trainval_data.train_test_split(test_size=0.2, stratify_by_column="label")
        # train_data, val_data = splitted['train'], splitted['test']
    elif name == 'toxigen':
        TG = load_dataset("skg/toxigen-data", name="train")
        dataset = pd.DataFrame(TG["train"])
        dataset.rename(columns={"prompt_label": "label", "generation":"text"}, inplace=True)
        train_data = Dataset.from_pandas(dataset)
        test_dateset = load_dataset("skg/toxigen-data", name="annotated")
        human_eval = pd.DataFrame(test_dateset["test"])
        eval_dataset_df = label_annotations(human_eval)
        test_data = Dataset.from_pandas(eval_dataset_df)
    return train_data, test_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=3, type=int, help="Epochs")
    parser.add_argument("--train_bz", default=128, type=int, help="Train batch size")
    parser.add_argument("--test_bz", default=128, type=int, help="Test batch size")
    parser.add_argument("--dataset", default='tweets', type=str, help="dataset")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    NUM_LABELS = 2
    OUTPUT_DIR = './checkpoints'
    CKPT = './checkpoints/finetune_model'
    
    train_data, test_data = get_dataset(args.dataset)
    
    max_length = train_data['text'].apply(len).max()
    
    sft_pipeline(
        model_name='distilbert-base-uncased', 
        train_data=train_data, 
        test_data=test_data, 
        max_len=max_length, 
        train_batch_size=args.train_bz, 
        eval_batch_size=args.test_bz, 
        learning_rate=args.lr, 
        epochs=args.epoch, 
        num_labels=NUM_LABELS,
        output_dir=OUTPUT_DIR,
        ckpt_path=CKPT,
    )