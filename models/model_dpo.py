import gc
import torch

from trl import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

'''
SETUP EXAMPLE:

# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 
                    'gate_proj', 
                    'v_proj', 
                    'up_proj', 
                    'q_proj', 
                    'o_proj', 
                    'down_proj']
)

# Model to fine-tune
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    load_in_4bit=True
)
model.config.use_cache = False

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    max_steps=200,
    save_strategy="no",
    logging_steps=1,
    output_dir=new_model,
    optim="paged_adamw_32bit",
    warmup_steps=100,
    bf16=True,
    report_to="wandb",
)
'''

def dpo_pipeline(model, 
                 training_args,
                 dataset,
                 tokenizer,
                 peft_config,
                 beta=0.1,
                 max_prompt_length=1024,
                 max_length=1536,
                 ckpt_path="final_dpo_checkpoint",
                 **kwargs):
    # Create DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=0.1,
        max_prompt_length=1024,
        max_length=1536,
    )

    # Fine-tune model with DPO
    dpo_trainer.train()

    # Save artifacts
    dpo_trainer.model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)

    # Flush memory
    del dpo_trainer, model
    gc.collect()
    torch.cuda.empty_cache()


def merge_and_push(model_name, new_model_path, hf_token):
    # Reload model in FP16 (instead of NF4)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Merge base model with the adapter
    model = PeftModel.from_pretrained(base_model, "final_checkpoint")
    model = model.merge_and_unload()

    # Save model and tokenizer
    model.save_pretrained(new_model_path)
    tokenizer.save_pretrained(new_model_path)

    # OPTIONAL:
    # # Push them to the HF Hub
    # model.push_to_hub(new_model_path, use_temp_dir=False, token=hf_token)
    # tokenizer.push_to_hub(new_model_path, use_temp_dir=False, token=hf_token)
