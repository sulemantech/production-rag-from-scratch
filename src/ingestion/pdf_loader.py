import os
from pydoc import text

import pdfplumber
from sympy import content

def extract_metadata_from_filename(filename:str) -> dict:
    #Input: "11th Class Biology PunjabBoard Year 2026.pdf"
    #Output: {"subject": "Biology", "grade": "11th"", "Board": "PunjabBoard", "Year": "2026", "extension": "pdf"}
    parts = filename.split(" ")
    result = {
        "grade": parts[0],
        "class": parts[1],
        "subject": parts[2],
        "board": parts[3],
        "year": parts[4],
        "title": " ".join(parts[0:4]),
        "extension": parts[5]
    }
    print(f"Extracted metadata from filename: {result}")
    return result

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

if __name__ == "__main__":
    test_filenames = [
        "11th Class Biology PunjabBoard Year 2026.pdf",
        "12th Class Biology PunjabBoard Year 2026.pdf",
        "11th Class Chemistry PunjabBoard Year 2026.pdf",
        "12th Class Chemistry PunjabBoard Year 2026.pdf",
        "11th Class Physics PunjabBoard Year 2026.pdf",
        "12th Class Physics PunjabBoard Year 2026.pdf",
    ]
    for name in test_filenames:
        extract_metadata_from_filename(name)

    # content = load_pdf("datasets/pdfs/Biology-2e_-_WEB.pdf")
    # save_text(content, "datasets/biology.txt")
    # print(content)
