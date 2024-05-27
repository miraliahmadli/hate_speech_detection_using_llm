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
├── data
│   │   ├── dataset_name
│   │   │   └── ...
│   │   └── ...
├── dataloaders
│       ├── common_utils.py
│       ├── dpo_dataset.py
│       ├── kto_dataset.py
│       └── sft_dataset.py
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

