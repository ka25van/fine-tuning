import json
import random


with open('data/resumes_texts.json') as f:
    resumes = json.load(f)

instruction_dataset=[]

for r in resumes:
    text = r["text"[:2000]]
    instruction_dataset.append({
        "instruction": "List the technical skills in the resume.",
        "input": text,
        "output": "Python, Docker, Kubernetes, AWS, Git, LangChain, LangGraph, LLm, GenAI,Angular, Javascript, Nodejs, React, database"
    })
    instruction_dataset.append({
        "instruction": "What is the probable job role for this resume?",
        "input": text,
        "output": "GenAI Engineer, FullStack Developer, DevOps Engineer"
    })
    instruction_dataset.append({
        "instruction": "List companies mentioned in the resume.",
        "input": text,
        "output": "Diya Systems Pvt Ltd, Infosys"
    })

with open('data/dataset.jsonl', 'w') as f:
    for item in instruction_dataset:
        f.write(json.dumps(item)+"\n")