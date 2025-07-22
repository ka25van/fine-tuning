import pdfplumber
import os
import json

INPUT_DIR='data/pdf_files'
OUTPUT_DIR ='data/resumes_texts.json'

def extract_pdf_text(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    

all_texts=[]

for filename in os.listdir(INPUT_DIR):
    if filename.endswith('.pdf'):
        text = extract_pdf_text(os.path.join(INPUT_DIR, filename))
        all_texts.append({"file":filename, "text":text})

with open(OUTPUT_DIR, "w") as f:
    json.dump(all_texts, f, indent=2)



