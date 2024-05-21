from torch.utils.data import Dataset

from dataloaders.common_utils import alignment_kto_format

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

class HateSpeechKTODataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return alignment_kto_format(self.tokenizer, self.data[idx])
