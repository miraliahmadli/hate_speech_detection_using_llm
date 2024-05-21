import gc
import torch

from trl import KTOTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

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


'''
SETUP EXAMPLE:

https://colab.research.google.com/drive/1HlH7Ydjcqn0cVghv1n9RcqEjwvCz4a5_?usp=sharing#scrollTo=BbLeGW0E0eAX

# LoRA configuration
# initialize training arguments:
training_args = KTOConfig(
    beta=Config.beta,
    desirable_weight=Config.desirable_weight,
    undesirable_weight=Config.undesirable_weight,
    per_device_train_batch_size=Config.batch_size,
    num_train_epochs=Config.num_train_epochs,
    logging_steps=Config.logging_steps,
    save_steps=Config.save_steps,
    learning_rate=Config.learning_rate,
    evaluation_strategy="steps",
    eval_steps=Config.eval_steps,
    output_dir=Config.output_dir,
    report_to=Config.report_to,
    lr_scheduler_type=Config.lr_scheduler_type,
    optim=Config.optimizer_type,
    bf16=False,
    remove_unused_columns=False,
    run_name="kto_llama_1b",
    seed=Config.seed,
    max_prompt_length=Config.max_prompt_length,
    max_length=Config.max_length,
)

peft_config = LoraConfig(
    r=Config.lora_r,
    lora_alpha=Config.lora_alpha,
    lora_dropout=Config.lora_dropout,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "out_proj",
        "fc_in",
        "fc_out",
        "wte",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

# initialize the DPO trainer
kto_trainer = KTOTrainer(
    model,
    ref_model=None, # use model without lora adapter
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
)
'''

def kto_pipeline(model, 
                 training_args,
                 dataset,
                 tokenizer,
                 peft_config,
                 beta=0.1,
                 max_prompt_length=1024,
                 max_length=1536,
                 ckpt_path="./checkpoints/final_kto_checkpoint",
                 **kwargs):
    # Create KTRO trainer
    kto_trainer = KTOTrainer(
        model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=0.1,
        max_prompt_length=1024,
        max_length=1536,
    )

    # Fine-tune model with KTO
    kto_trainer.train()

    # Save artifacts
    kto_trainer.model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)

    # Flush memory
    del kto_trainer, model
    gc.collect()
    torch.cuda.empty_cache()


def merge_and_push(model_name, kto_path, new_model_path, hf_token):
    # Reload model in FP16 (instead of NF4)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        return_dict=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Merge base model with the adapter
    model = PeftModel.from_pretrained(base_model, kto_path)
    model = model.merge_and_unload()

    # Save model and tokenizer
    model.save_pretrained(new_model_path)
    tokenizer.save_pretrained(new_model_path)

    # OPTIONAL:
    # # Push them to the HF Hub
    # model.push_to_hub(new_model_path, use_temp_dir=False, token=hf_token)
    # tokenizer.push_to_hub(new_model_path, use_temp_dir=False, token=hf_token)