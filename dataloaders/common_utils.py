import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset


def filter_none_text(example, key='text'):
    return example[key] is not None

def load_and_process_toxigen(kto=False, test_size=0.1):
    def label_annotations(annotated):
        # Annotations should be the annotated dataset
        label = ((annotated.toxicity_ai + annotated.toxicity_human) > 5.5).astype(int)
        labeled_annotations = pd.DataFrame()
        labeled_annotations["text"] = [i for i in annotated.text.tolist()]
        labeled_annotations["label"] = label
        return labeled_annotations

    TG = load_dataset("skg/toxigen-data", name="train")
    tg_dataset = TG["train"].to_pandas()
    tg_dataset = tg_dataset.dropna()
    if kto:
        tg_dataset = tg_dataset[["prompt", "prompt_label", "generation"]]
        tg_dataset.rename(columns={"prompt_label": "label", "generation":"completion"}, inplace=True)
        tg_dataset['label'] = tg_dataset['label'].astype(bool)
        tg_dataset = tg_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        train_data, val_data = train_test_split(tg_dataset, test_size=test_size)
        train_data = Dataset.from_pandas(train_data)
        val_data = Dataset.from_pandas(val_data)

        return train_data, val_data

    tg_dataset = tg_dataset[["prompt_label", "generation"]]
    tg_dataset.rename(columns={"prompt_label": "label", "generation":"text"}, inplace=True)
    tg_train_data = Dataset.from_pandas(tg_dataset)

    tg_test_dateset = load_dataset("skg/toxigen-data", name="annotated")
    tg_human_eval = pd.DataFrame(tg_test_dateset["test"])
    tg_eval_dataset_df = label_annotations(tg_human_eval)
    tg_test_data = Dataset.from_pandas(tg_eval_dataset_df)

    return tg_train_data, tg_test_data


def load_and_process_twitter_data(test_size=0.2):
    dataset = load_dataset('tweets_hate_speech_detection', name="default")
    trainval_data = dataset['train'].to_pandas()
    trainval_data.rename(columns={'tweet': 'text'}, inplace=True)
    trainval_data['label'] = trainval_data['label'].astype(int)
    trainval_data = trainval_data.dropna()
    trainval_data = trainval_data.sample(frac=1, random_state=42).reset_index(drop=True)
    train_data, val_data = train_test_split(trainval_data, test_size=test_size)
    train_data = Dataset.from_pandas(train_data)
    val_data = Dataset.from_pandas(val_data)

    return train_data, val_data

def load_and_process_berkeley_data(test_size=0.2):
    dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
    trainval_data = dataset['train'].to_pandas()
    trainval_data = trainval_data[["text", "hate_speech_score"]]
    trainval_data.rename(columns={'hate_speech_score': 'label'}, inplace=True)
    trainval_data['label'] = (trainval_data['label'] > 0.5).astype(int)

    trainval_data = trainval_data.dropna()
    trainval_data = trainval_data.sample(frac=1, random_state=42).reset_index(drop=True)
    train_data, val_data = train_test_split(trainval_data, test_size=test_size)
    train_data = Dataset.from_pandas(train_data)
    val_data = Dataset.from_pandas(val_data)
    return train_data, val_data

def load_and_process_gender_hate_speech_data():
    dataset = load_dataset("ctoraman/gender-hate-speech")
    train = dataset['train'].to_pandas()
    train = train[["Text", "Label"]]
    train.rename(columns={'Text': 'text', 'Label': 'label'}, inplace=True)
    train['label'] = (train['label'] > 0.5).astype(int)

    val = dataset['test'].to_pandas()
    val = val[["Text", "Label"]]
    val.rename(columns={'Text': 'text', 'Label': 'label'}, inplace=True)
    val['label'] = (val['label'] > 0.5).astype(int)

    train = train.dropna()
    val = val.dropna()
    train_data = Dataset.from_pandas(train)
    val_data = Dataset.from_pandas(val)
    return train_data, val_data

def load_and_process_cad(dir_path='data/CAD'):
    train_path = f"{dir_path}/train.tsv"
    dev_path = f"{dir_path}/dev.tsv"
    test_path = f"{dir_path}/test.tsv"
    train_data = pd.read_csv(train_path, sep='\t')
    dev_data = pd.read_csv(dev_path, sep='\t')
    train_data = pd.concat([train_data, dev_data])
    test_data = pd.read_csv(test_path, sep='\t')
    
    train_data.drop(columns=['id'], inplace=True)
    test_data.drop(columns=['id'], inplace=True)
    
    train_data.rename(columns={'labels': 'label'}, inplace=True)
    test_data.rename(columns={'labels': 'label'}, inplace=True)
    
    train_data['label'] = (train_data['label'] != 'Neutral').astype(int)
    test_data['label'] = (test_data['label'] != 'Neutral').astype(int)

    train_data = train_data.dropna()
    test_data = test_data.dropna()

    train_data = Dataset.from_pandas(train_data)
    val_data = Dataset.from_pandas(test_data)
    return train_data, val_data
