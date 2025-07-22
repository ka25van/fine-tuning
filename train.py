from transformers import AutoTokenizer, AutoModelForCausalLM,TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

model_id = "gpt2"
# meta-llama/Llama-3.1-8B-Instruct
dataset_path = "data/dataset.jsonl"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fastt=True)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files=dataset_path, split="train")

def tokenize(example):
    prompt = f"<s>[INST] {example['instruction']}\n{example['input']} [/INST] {example['output']}</s>"
    return tokenizer(prompt, padding="max_length", truncation=True, max_length=512)

dataset=dataset.map(tokenize)

model=AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

model=prepare_model_for_kbit_training(model)

lora_config=LoraConfig(
     r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn"]
)
model=get_peft_model(model, lora_config)

training_args= TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    logging_dir="logs",
    num_train_epochs=3,
    learning_rate=2e-4,
    save_total_limit=1,
    logging_steps=10,
    save_strategy="epoch"
)

trainer=Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()

model.save_pretrained("outputs/gpt2")
tokenizer.save_pretrained("outputs/gpt2")
