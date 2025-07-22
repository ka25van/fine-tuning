from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = "gpt2"
adapter_path = "outputs/gpt2"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float32,
    
).to('cpu')

model=PeftModel.from_pretrained(model, adapter_path)

def infer(instruction, input_text):
    prompt =  f"<s>[INST] {instruction}\n{input_text} [/INST]"
    inputs= tokenizer(prompt, return_tensors="pt").to('cpu')
    output = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

sample = "I worked on AWS, Terraform, and CI/CD pipelines in Jenkins."
print(infer("List the technical skills in the resume.", sample))