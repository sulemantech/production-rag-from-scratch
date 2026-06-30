
import re


def clean_document(text: str) -> str:
    # remove known footers
    text = text.replace("Access for free at openstax.org", "")
    
    # remove figure captions
    text = re.sub(r'FIGURE \d+\.\d+.*?\n', '', text)
    
    # collapse 3+ blank lines into 2 (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # remove table of contents (pages with just titles and page numbers)
    text = re.sub(r'\n[\w\s,]+\d+\n', '\n', text)

    
    return text.strip()


if __name__ == "__main__":
    with open("datasets/sample_text/biology.txt", "r", encoding="utf-8") as f:
        text = f.read()

    cleaned = clean_document(text)

    print(f"Before: {len(text)} chars")
    print(f"After:  {len(cleaned)} chars")
    print(f"Removed: {len(text) - len(cleaned)} chars")
    print("\n--- Sample cleaned text ---")
    print(cleaned[5000:6000])
