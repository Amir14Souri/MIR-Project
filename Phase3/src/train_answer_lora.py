import argparse
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load dataset
    dataset = load_dataset("json", data_files=args.train_jsonl, split="train")
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    def format_chatml(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False)
        
    # Load model in 4-bit for memory efficiency on Colab
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Setup LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # Completely eradicate any remaining bfloat16 from the model graph
    # Colab T4 crashes if the optimizer encounters bfloat16 during AMP gradient scaling.
    for param in model.parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
            
    # Setup Trainer
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        logging_steps=1,
        optim="paged_adamw_32bit",
        save_strategy="no",
        fp16=False,
    )
    
    # Use standard language modeling collator
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        formatting_func=format_chatml,
        data_collator=collator,
    )
    
    print("[lora] Starting fine-tuning...")
    trainer.train()
    
    print(f"[lora] Saving adapter to {args.out_dir}...")
    trainer.model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("[lora] Done!")

if __name__ == "__main__":
    main()
