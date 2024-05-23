import argparse
import random
import torch
from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import random_split, DataLoader, RandomSampler, SequentialSampler
import math
import numpy as np
import time
import datetime
import pandas as pd
from toxigen import label_annotations

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# def preprocess_function(examples):
#     max_seq_length = min(TG_max_length, tokenizer.model_max_length)
#     return tokenizer(examples[sentence], truncation=True, padding='max_length', max_length=max_seq_length)

def eval_preprocess_function(examples):
    max_seq_length = min(TG_max_length, tokenizer.model_max_length)
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_seq_length)

def initialize(args, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train(epochs):
    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            b_input_ids = torch.tensor([t['input_ids'] for t in batch]).to(device)
            b_input_mask = torch.tensor([t['attention_mask'] for t in batch]).to(device)
            b_labels = torch.tensor([t['label'] for t in batch]).to(device)

            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            result = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels,
                        return_dict=True)

            loss = result.loss
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            # break
            
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        # break

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

def eval():
    # Prediction on test set
    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions , true_labels = [], []

    # Predict
    for batch in test_dataloader:
        # Add batch to GPU
        
        # dict_keys(['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'])
        b_input_ids = torch.tensor([t['input_ids'] for t in batch]).to(device)
        b_input_mask = torch.tensor([t['attention_mask'] for t in batch]).to(device)
        b_labels = torch.tensor([t['label'] for t in batch]).to(device)

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            result = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            return_dict=True)
            #   print(result)
            #   break

        logits = result.logits

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        #   predictions.append(np.argmax(logits, axis=1).flatten())
        true_labels.append(label_ids)

    print('    DONE.')
    return predictions, true_labels
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=3, type=int, help="Epochs")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # initialize(args, seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # task = 'toxigen'
    # TG_dataset = load_dataset("skg/toxigen-data", name="train")
    # TG = pd.DataFrame(TG_dataset['train'])
    batch_size = 32
    
    num_labels = 2
    # model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    

    
    task_to_keys = {
        "toxigen": ("generation", "prompt_label"),
        "toxigen_human": ("text", "label"),
    }

    TG_human = load_dataset("skg/toxigen-data", name="annotated")
    human_train = pd.DataFrame(TG_human["train"])
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased') #, do_lower_case=True)
    TG_max_length = human_train['text'].apply(len).max()
    
    train_dataset_df = label_annotations(human_train)
    train_dataset = Dataset.from_pandas(train_dataset_df)
    train_sentences = train_dataset['text']
    train_labels = train_dataset['label']
    train_encoded_dataset = train_dataset.map(eval_preprocess_function, batched=False)
    
    train_dataloader = DataLoader(
                train_encoded_dataset, # The validation samples.
                sampler = SequentialSampler(train_encoded_dataset), # Pull out batches sequentially.
                batch_size = batch_size, # Evaluate with this batch size.
                collate_fn=lambda x: x 
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    test_dataloader = DataLoader(
                train_encoded_dataset, # The validation samples.
                sampler = SequentialSampler(train_encoded_dataset), # Pull out batches sequentially.
                batch_size = batch_size, # Evaluate with this batch size.
                collate_fn=lambda x: x 
            )
    
    
    TG_human = load_dataset("skg/toxigen-data", name="annotated")
    human_eval = pd.DataFrame(TG_human["test"])
    eval_dataset_df = label_annotations(human_eval)
    eval_dataset = Dataset.from_pandas(eval_dataset_df)
    eval_sentence, eval_label = task_to_keys['toxigen_human']
    eval_sentences = eval_dataset[eval_sentence]
    eval_labels = eval_dataset[eval_label]
    eval_encoded_dataset = eval_dataset.map(eval_preprocess_function, batched=False)

    # For validation the order doesn't matter, so we'll just read them sequentially.
    test_dataloader = DataLoader(
                eval_encoded_dataset, # The validation samples.
                sampler = SequentialSampler(eval_encoded_dataset), # Pull out batches sequentially.
                batch_size = batch_size, # Evaluate with this batch size.
                collate_fn=lambda x: x 
            )
    
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-cased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = num_labels, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    model.to(device)
    
    base_optimizer = AdamW
    kwargs= {'lr':2e-5}
    optimizer = AdamW(model.parameters(),
                lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
            )
    # Create the learning rate scheduler.
    total_steps = len(train_dataloader) * args.epoch
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    train(args.epoch)
    
    predictions, true_labels = eval()
    
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_true_labels = np.concatenate(true_labels, axis=0)
    acc = flat_accuracy(flat_predictions, flat_true_labels)
    print(acc)
