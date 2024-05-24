import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset

def alignment_dpo_format(tokenizer, samples):
    # Format system
    if len(samples['system']) > 0:
        message = {"role": "system", "content": samples['system']}
        system = tokenizer.apply_chat_template([message], tokenize=False)
    else:
        system = ""

    # Format instruction
    message = {"role": "user", "content": samples['text']}
    prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

    # Format chosen answer
    chosen = samples['chosen'] + "<|im_end|>\n"

    # Format rejected answer
    rejected = samples['rejected'] + "<|im_end|>\n"

    return {
        "prompt": system + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

def alignment_kto_format(tokenizer, samples):
     # Format system
    if len(samples['system']) > 0:
        message = {"role": "system", "content": samples['system']}
        system = tokenizer.apply_chat_template([message], tokenize=False)
    else:
        system = ""

    # Format instruction
    message = {"role": "user", "content": samples['text']}
    prompt = tokenizer.apply_chat_template([message], 
                                           tokenize=False, 
                                           add_generation_prompt=True)

    completion = samples['completion'] + "<|im_end|>\n"
    return {
        "prompt": system + prompt,
        "completion": completion,
        "label": samples["label"],
    }

def load_and_process_toxigen():
    def label_annotations(annotated):
        # Annotations should be the annotated dataset
        label = ((annotated.toxicity_ai + annotated.toxicity_human) > 5.5).astype(int)
        labeled_annotations = pd.DataFrame()
        labeled_annotations["text"] = [i for i in annotated.text.tolist()]
        labeled_annotations["label"] = label
        return labeled_annotations

    TG = load_dataset("skg/toxigen-data", name="train")
    tg_dataset = pd.DataFrame(TG["train"])
    tg_dataset.rename(columns={"prompt_label": "label", "generation":"text"}, inplace=True)
    tg_train_data = Dataset.from_pandas(tg_dataset).map(remove_columns=['prompt', 'generation_method', 'group', 'roberta_prediction'])

    tg_test_dateset = load_dataset("skg/toxigen-data", name="annotated")
    tg_human_eval = pd.DataFrame(tg_test_dateset["test"])
    tg_eval_dataset_df = label_annotations(tg_human_eval)
    tg_test_data = Dataset.from_pandas(tg_eval_dataset_df)
    
    return tg_train_data, tg_test_data

def load_and_process_twitter_data():
    dataset = load_dataset('tweets_hate_speech_detection', name="default")
    trainval_data = pd.DataFrame(dataset['train'])
    trainval_data.rename(columns={'tweet': 'text'}, inplace=True)
    trainval_data['label'] = trainval_data['label'].astype(int)
    train_data, val_data = train_test_split(trainval_data, test_size=0.2)
    train_data = Dataset.from_pandas(train_data)
    val_data = Dataset.from_pandas(val_data)

    return train_data, val_data

