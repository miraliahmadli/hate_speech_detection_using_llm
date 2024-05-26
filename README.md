# EPFL Deep Learning 2024 Project - Robust Hate Speech Detection for LLMs

## Project Description

- For this project, we use SOTA techniques to improve LLMs hate speech detection accuracy.
- First, we fine-tune our baseline LLM for the task. Then, we use KTO to align our model according to human preferences.
- Finally, we use RAG to try to improve our model's performance even further. 

## Codebase File Structure

```txt
.
├── checkpoints
│   ├── checkpoint_name
│   │   ├── config.json
│   │   |── model.safetensor
│   │   └── ...
├── datasets
│   │   ├── dataset_name
│   │   │   └── ...
│   │   └── ...
├── documents (For RAG only)
├── models
│       ├── model_base.py
│       |── model_sft.py
│       └── ...
├── RAG
│       ├── document_scraper.py
│       └── retriever.py
├── evaluate_model.py
├── train_base.py
├── train_kto.py
├── train_sft.py
├── train_sft_kto.py
├── main_config.yaml
├── requirements.txt
└── README.md
```

## Setup

### Setup via Conda Virtual Environment

```bash
# Replace <my-env> with the name of your environment.
conda create --name <my-env> python=3.10.11
conda activate <my-env>

# Install dependencies from a `requirements.txt`
pip install -r requirements.txt
```


## Codebase Introduction

- `model_base.py`: In this file, you will find a wrapper model class `PreTrainedModelWrapper` around a (`transformers.PreTrainedModel`) to be compatible with the (`~transformers.PreTrained`) class in order to keep some attributes and methods of the (`~transformers.PreTrainedModel`) class. You can save a checkpoint through `PreTrainedModelWrapper.save_pretrained(model_name_or_path)` and load a checkpoint locally or from HuggingFace hub through the method `PreTrainedModelWrapper.from_pretrained(model_name_or_path)`.
- `model_dpo.py`: In this file, you will implement your DPO model. Read and complete the TODOs. Note that TODO (Optional) is not required; You only need to do these if you want to add custom modules to your model. If you are working with a causal language model like GPT-2 or LLama2, use the `AutoDPOModelForCausalLM` class. If you are working with a sequence-to-sequence language model like T5 or Bart, use the `AutoDPOModelForSeq2SeqLM` class. The functions that are required for all to implement including `forward`, `prediction_step_reward`, and `prediction_step_mcqa`.
- In addition to a transformer model, you can add custom modules to the `AutoDPOModel` classes. Below is an example custom module. You can follow the `TODO (Optional)` to integrate a custom module into the main model.

### Basic Model functionalities

For `AutoDPOModelForCausalLM` and `AutoDPOModelForSeq2SeqLM`, which both inherit `PreTrainedModelWrapper`, you have the following basic operations:

**Load from a pre-trained model listed in [Huggingface Hub](https://huggingface.co/models)**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.model_dpo import AutoDPOModelForCausalLM

# Download the pre-trained model and tokenizer from the Hub
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize your model class and import the pre-trained model into your class
# Note that if you have a custom module in your class
# You should initialize the weights of this module in the `__init__` function
model_wrapper = AutoDPOModelForCausalLM(pretrained_model=model)
```

**Save your model as a Huggingface transformers compatible checkpoint**

```python
# Save your model and tokenizer to the checkpoint directory `models/dpo_gpt2`
checkpoint_dir = "models/dpo_gpt2"
model_wrapper.save_pretrained(checkpoint_dir)
tokenizer.save_pretrained(checkpoint_dir)
```

**Load from your local checkpoint**

```python
checkpoint_dir = "models/dpo_gpt2"
model_wrapper = AutoDPOModelForCausalLM.from_pretrained(checkpoint_dir)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
```

### Custom Module Example

```python
class CustomModule(nn.Module):
    """
    This is only a dummy example of a custom module. You can replace this with your own custom module.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size

        self.summary = nn.Linear(hidden_size, 1)
        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)
        output = self.summary(output)
        return output
```

- `evaluator.py` is the main evaluation script that we will use to collect your model performance and implementation quality in order to assign you a grade. To execute this script, you first need to specify details in the `main_confi.yaml` configuration file. The details in this config file will be used by the evaluation script to execute the grading properly. Make sure you fill all the important information in the config file.

### Main Configuration Arguments

```yaml
"team_name": "Team 1" # Your team name
"eval_method": ["mcqa", "rag"] # Tells the evaluator which evaluations need to be executed. choices = [mcqa, reward, rag, compression]
"task_type": "causal_lm" # Identifies which model class you use. choices = [causal_lm, seq2seq]
"policy_model_path": "./checkpoints/best_model/" # Your path to the final checkpoint
"reference_model_path": "microsoft/phi-2" # The repo id of your pretrained DPO reference model
"quantized_policy_model_path": "./checkpoints/best_model_quantized/" # Your path to the final quantized checkpoint
"rag_policy_model_path": "./checkpoints/best_model_rag/" # Your path to the final RAG checkpoints
"test_data_path": "./data/test.json" # Your path to the test data. (We will replace it with the official test sets when grading)
"dpo_model_args": null # Any required arguments to load your dpo model using "from_pretrained"
"rag_model_args": # Any required arguments to load your rag model using "from_pretrained" For example:
    "encoder_model_path": "facebook/bart-large"
    "retriever_model_path": "./checkpoints/rag_retriever"
    "document_dir": "./data/documents"
"quantized_model_args": null # Any required arguments to load your quantized model using "from_pretrained"
```

- Note: `eval_method`'s value must be a list object.

- Note: `reward` and `mcqa` cannot co-exist in the `eval_method` list at the same time.

Please review the evaluation script code for detailed evaluation methods and the input and output of each evaluation function.
