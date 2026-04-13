# -*- coding: utf-8 -*-
"""
Fine-tuning script for LLaMA 3.1 using LoRA (PEFT).
This script supports the self-supervised adaptation stage of MONLCD 
by training the model to recognize local topological patterns.
"""

from datasets import Dataset
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
import transformers
import logging
import torch
from peft import LoraConfig, TaskType, get_peft_model

# ===== Logging Setup: Ensure loss is printed in console =====
logging.basicConfig(level=logging.INFO)
transformers.utils.logging.set_verbosity_info()
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

def process_func(example):
    """
    Preprocess data for LLaMA 3.1 using the standard chat template.
    Targets only the model response for loss calculation.
    """
    MAX_LENGTH = 1024
    
    # LLaMA 3.1 Chat Template construction
    # System prompt is consistent with the self-supervised task format [cite: 652]
    instruction = tokenizer(
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{example['instruction'] + example['input']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False
    )
    response = tokenizer(
        f"{example['output']}<|eot_id|>",
        add_special_tokens=False
    )

    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]

    # Labels: Only the assistant response participates in loss calculation (-100 for others)
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]

    # Truncation logic
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# ===== Dataset Loading =====
# Path should point to the adapted datasets (e.g., Zachary Karate Club task samples) [cite: 626]
data_path = '/root/autodl-tmp/ovpa-train-data/train_data_opva_class_Conscientiousness_120.json'
df = pd.read_json(data_path)
ds = Dataset.from_pandas(df)

# ===== Tokenizer Initialization =====
# Using LLaMA 3.1 8B as the base model 
model_id = '/root/autodl-tmp/LLM-Research/Meta-Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

# ===== Model Loading & LoRA Configuration =====
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)
model.enable_input_require_grads()

# LoRA config targeting all linear layers for comprehensive adaptation [cite: 654]
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = get_peft_model(model, config)

# ===== Training Arguments =====
args = TrainingArguments(
    output_dir="./output/llama3_1_monlcd_adaptation",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3, # Convergence is typically reached quickly 
    learning_rate=1e-4,

    # Logging settings
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    report_to=["none"],

    # Saving and Performance
    save_steps=100,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# Start adaptation process
trainer.train()