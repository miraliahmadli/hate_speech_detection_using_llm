from torch.utils.data import Dataset

from dataloaders.common_utils import filter_none_text

'''
DATASET FORMAT
kto_dataset_dict = {
    "prompt": [
        "Hey, hello",
        "How are you",
        "What is your name?",
        "What is your name?",
        "Which is the best programming language?",
        "Which is the best programming language?",
        "Which is the best programming language?",
    ],
    "completion": [
        "hi nice to meet you",
        "leave me alone",
        "I don't have a name",
        "My name is Mary",
        "Python",
        "C++",
        "Java",
    ],
    "label": [
        True,
        False,
        False,
        True,
        True,
        False,
        False,
    ],
}
'''

def filter_none_text(example):
    return example['text'] is not None


def alignment_kto_format(tokenizer, samples):
     # Format system
    if len(samples.get('system', '')) > 0:
        message = {"role": "system", "content": samples['system']}
        system = tokenizer.apply_chat_template([message], tokenize=False)
    else:
        system = ""

    # Format instruction
    # message = {"role": "user", "content": samples['text']}
    # prompt = tokenizer.apply_chat_template([message], 
    #                                        tokenize=False, 
    #                                        add_generation_prompt=True)
    prompt = samples['text']

    completion = samples['generation']# + "<|im_end|>\n"
    return {
        "prompt": system + prompt,
        "completion": completion,
        "label": bool(samples["label"]),
    }

class HateSpeechKTODataset(Dataset):
    def __init__(self, data):
        self.data = data.filter(filter_none_text)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        return {
            'prompt': data['text'],
            'completion': data['generation'],
            'label': bool(data['label'])
        }
        return alignment_kto_format(self.tokenizer, self.data[idx])
