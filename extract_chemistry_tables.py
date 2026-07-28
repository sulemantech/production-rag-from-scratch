"""
Extract clean tabular data (electron configs, oxidation states, periodic
trends etc.) from the 12th grade Chemistry PDF as standalone, per-row text
chunks. The existing ingestion pipeline uses extract_text(), which flattens
tables into unreadable run-on blobs (confirmed: e.g. Cr/Zn electron config
data smeared together with 10+ other elements in one chunk). pdfplumber's
extract_tables() recovers clean per-cell structure for this book -- verified
against a 52-clean/1-scrambled/1-empty sample across all its table pages.

Each table row becomes its own small chunk (e.g. "Cr (24) | [Ar] 3d5 4s1"),
so a specific fact like "what is Cr's electron configuration" becomes
directly retrievable instead of buried in a giant merged blob.

These are ADDITIONAL chunks appended to the corpus -- the original flattened
prose chunks stay, since surrounding paragraph context is still useful.
"""
import re
import pdfplumber

PDF_PATH = "datasets/pdfs/PunjabBoardTextbooks/12th Class Chemistry PunjabBoard Year 2026.pdf"
GRADE = "12th"
SUBJECT = "Chemistry"

SCRAMBLED_RE = re.compile(r"[A-Za-z]\d[A-Za-z]\d?[A-Za-z]")
SUBSHELL_TOKEN_RE = re.compile(r"\d([spdf])")
SUBSHELL_CAPACITY = {"s": 1, "p": 3, "d": 5, "f": 7}  # orbital count per subshell


def looks_scrambled(text: str) -> bool:
    return bool(text) and bool(SCRAMBLED_RE.search(text))


def unpaired_in_subshell(letter: str, electron_count: int) -> int:
    """Hund's rule: electrons fill orbitals singly before pairing."""
    orbitals = SUBSHELL_CAPACITY[letter]
    if electron_count <= orbitals:
        return electron_count
    return 2 * orbitals - electron_count


def parse_subshells(text: str) -> list:
    """
    Finds (letter, electron_count) pairs in a config string. Electron counts
    are 1-2 digits, but PDF extraction sometimes drops the space between
    adjacent subshells (e.g. "2s22p5" instead of "2s2 2p5") -- a naive
    \\d{1,2} match would swallow "22" as one count. Since each subshell type
    has a known max capacity (s<=2, p<=6, d<=10, f<=14), a 2-digit read is
    only accepted when it doesn't exceed that -- otherwise it falls back to
    1 digit, which naturally handles the no-space case correctly.
    """
    tokens = []
    for m in SUBSHELL_TOKEN_RE.finditer(text):
        letter = m.group(1)
        pos = m.end()
        while pos < len(text) and text[pos] == " ":
            pos += 1
        max_electrons = 2 * SUBSHELL_CAPACITY[letter]
        two, one = text[pos:pos + 2], text[pos:pos + 1]
        if two.isdigit() and int(two) <= max_electrons:
            count = int(two)
        elif one.isdigit():
            count = int(one)
        else:
            continue
        tokens.append((letter, count))
    return tokens


def annotate_config(text: str) -> str:
    """
    Appends a Hund's-rule-derived unpaired-electron note to any electron
    configuration string found in the cell. The total count stays on every
    row (needed for "which element has exactly N unpaired electrons?"
    questions -- the number itself varies per row, so it's a real signal,
    not a repeated label). A second, distinctive phrase is added only for
    the two extreme d-subshell cases -- exactly half-filled (d5, every
    orbital singly occupied) or exactly full (d10, every orbital paired) --
    echoing real exam wording ("all ... unpaired" / "all ... paired").

    Earlier version also appended a "d: M unpaired" LABEL to every row
    regardless of value. The label text itself was identical across ~50
    sibling rows, which made BM25 unable to tell them apart: the word exam
    questions actually search on ("unpaired") was diluted by being repeated
    near-verbatim everywhere. Only the two truly distinctive extreme cases
    get special phrasing now -- everything else just gets the plain number.
    """
    tokens = parse_subshells(text)
    if not tokens:
        return text

    total_unpaired = sum(unpaired_in_subshell(letter, count) for letter, count in tokens)
    d_count = next((count for letter, count in tokens if letter == "d"), None)

    suffix = f" ({total_unpaired} unpaired electrons total"
    if d_count == 5:
        suffix += "; all five d orbitals contain unpaired electrons"
    elif d_count == 10:
        suffix += "; all five d orbitals are fully paired"
    suffix += ")"
    return text + suffix


def is_header_like(cells: list) -> bool:
    non_empty = [c for c in cells if c and c.strip()]
    if not non_empty:
        return True
    # header rows are short, label-like text with no digits (real data rows
    # for this book always contain a config string or number)
    return all(len(c) < 30 and not re.search(r"\d", c) for c in non_empty)


def nearest_heading(page_text: str) -> str:
    match = re.search(r"^\s*\d+\.\d*\s+[A-Z][A-Za-z ]{3,40}$", page_text, re.MULTILINE)
    if match:
        return match.group().strip()
    match = re.search(r"^[A-Z][A-Za-z ]{3,40}$", page_text, re.MULTILINE)
    return match.group().strip() if match else "Chemistry"


def extract_table_chunks():
    chunks = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            if not tables:
                continue

            page_text = page.extract_text() or ""
            heading = nearest_heading(page_text)

            for table in tables:
                flat_text = " ".join(str(c) for row in table for c in row if c)
                if not flat_text.strip() or looks_scrambled(flat_text):
                    continue

                for row in table:
                    cells = [c.replace("\n", " ").strip() if c else "" for c in row]
                    if is_header_like(cells):
                        continue
                    non_empty = [annotate_config(c) for c in cells if c]
                    if len(non_empty) < 2:
                        continue
                    row_text = f"{heading} — " + " | ".join(non_empty)
                    chunks.append({
                        "text": row_text,
                        "subject": SUBJECT,
                        "grade": GRADE,
                    })
    return chunks


if __name__ == "__main__":
    chunks = extract_table_chunks()
    print(f"Extracted {len(chunks)} table-row chunks\n")
    for c in chunks[:15]:
        print(repr(c["text"]))
