import os
from pydoc import text

import pdfplumber

def save_text(text:str, file_path:str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

def load_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise ValueError("PDF path cannot be empty.")
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:  # Limit to the first 10 pages
            text += page.extract_text() or ""

    return text

content = load_pdf("datasets/pdfs/Biology-2e_-_WEB.pdf")

save_text(content, "datasets/biology.txt")
print(content)