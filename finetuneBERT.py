import argparse
import random
import torch
from datasets import load_dataset, load_metric
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import random_split, DataLoader, RandomSampler, SequentialSampler
import math
import numpy as np
import time
import datetime


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def preprocess_function(examples):
    max_seq_length = min(128, tokenizer.model_max_length)
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True, padding='max_length', max_length=max_seq_length)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding='max_length', max_length=max_seq_length)


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

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
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
            # dict_keys(['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'])
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
            
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

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
    parser.add_argument("--task", default='cola', type=str, help="GLUE tasks")
    parser.add_argument("--epoch", default=3, type=int, help="Epochs")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    # initialize(args, seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    actual_task = "mnli" if args.task == "mnli-mm" else args.task
    dataset = load_dataset("glue", actual_task)
    metric = load_metric('glue', actual_task)
    batch_size = 32
    
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    sentence1_key, sentence2_key = task_to_keys[args.task]
    sentences1 = dataset['train'][sentence1_key]
    if sentence2_key is not None:
        sentences2 = dataset['train'][sentence2_key]
    labels = dataset['train']['label']
    
    num_labels = 3 if args.task.startswith("mnli") else 1 if args.task=="stsb" else 2
    # model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    metric_name = "pearson" if args.task == "stsb" else "matthews_correlation" if args.task == "cola" else "accuracy"
    validation_key = "validation_mismatched" if args.task == "mnli-mm" else "validation_matched" if args.task == "mnli" else "validation"
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased') #, do_lower_case=True)
    encoded_dataset = dataset.map(preprocess_function, batched=False)
    
    # Calculate the number of samples to include in each set.
    # train_size = int(0.9 * len(encoded_dataset['train']))
    # val_size = len(encoded_dataset['train']) - train_size

    # # Divide the dataset by randomly selecting samples.
    # train_dataset, val_dataset = random_split(encoded_dataset['train'], [train_size, val_size])
    
    train_dataloader = DataLoader(
                encoded_dataset['train'],  # The training samples.
                sampler = RandomSampler(encoded_dataset['train']), # Select batches randomly
                batch_size = batch_size, # Trains with this batch size.
                collate_fn=lambda x: x 
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    # validation_dataloader = DataLoader(
    #             encoded_dataset['train'], # The validation samples.
    #             sampler = SequentialSampler(encoded_dataset['train']), # Pull out batches sequentially.
    #             batch_size = batch_size, # Evaluate with this batch size.
    #             collate_fn=lambda x: x 
    #         )
    
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
    
    test_dataloader = DataLoader(
                encoded_dataset[validation_key], # The validation samples.
                sampler = SequentialSampler(encoded_dataset[validation_key]), # Pull out batches sequentially.
                batch_size = batch_size, # Evaluate with this batch size.
                collate_fn=lambda x: x 
            )
    
    predictions, true_labels = eval()
    
    flat_predictions = np.concatenate(predictions, axis=0)
    # For each sample, pick the label (0 or 1) with the higher score.
    if args.task != 'stsb':
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)

    result = metric.compute(predictions=flat_predictions, references=flat_true_labels)
    print(result)
    
    file_name = './results/result.txt'
    result_str = str(args.task)
    result_str += '_epoch' + str(args.epoch) + ': ' +str(result) + '\n'
    with open(file_name, 'a') as file:
        file.write(result_str)
            
    if args.task == 'mnli':
        test_dataloader = DataLoader(
                    encoded_dataset['validation_mismatched'], # The validation samples.
                    sampler = SequentialSampler(encoded_dataset['validation_mismatched']), # Pull out batches sequentially.
                    batch_size = batch_size, # Evaluate with this batch size.
                    collate_fn=lambda x: x 
                )
        
        predictions, true_labels = eval()
        
        flat_predictions = np.concatenate(predictions, axis=0)
        # For each sample, pick the label (0 or 1) with the higher score.
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        # Combine the correct labels for each batch into a single list.
        flat_true_labels = np.concatenate(true_labels, axis=0)

        result = metric.compute(predictions=flat_predictions, references=flat_true_labels)
        print(result)
        
        result_str = str(args.task) + '-mm_' + str(args.optim) + '_epoch' + str(args.epoch) + ': ' +str(result) + '\n'
        with open(file_name, 'a') as file:
            file.write(result_str)