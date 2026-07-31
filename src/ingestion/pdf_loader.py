import os
import re

import pdfplumber

from src.ingestion.document_cleaner import clean_punjab_board_text

# ---------------------------------------------------------------------
# Chapter/section detection (NOT page numbers -- PDF page position doesn't
# reliably match a printed page number or a different edition a student
# owns; chapter/section titles are stable across printings, so that's
# what gets tracked and later surfaced as a study-pointer citation).
#
# Three detectors, tried in priority order per page. All three were
# validated directly against real sampled pages from all 6 textbooks
# before being written -- reliability varies genuinely by source:
#   - Biology 11th/12th, Chemistry 12th (eLearn.Punjab source): ~90-100%
#     per-page hit rate via the running header.
#   - Chemistry 11th (different source, no running header): numbered
#     section headings only, forward-filled between them.
#   - Physics 12th: mixed -- clean bare chapter titles early on, degrades
#     into OCR noise in later chapters.
#   - Physics 11th (topstudyworld/OCR source): ~1/10 sampled pages had any
#     usable heading signal -- needs a manual override table (see
#     CHAPTER_OVERRIDES below), not more regex-chasing.
# ---------------------------------------------------------------------

_ELEARN_RE = re.compile(
    r'^\s*((?:\d+(?:\.\d+)?\.?\s*)?[A-Za-z][A-Za-z0-9 ,&\-]{2,60}?)\s*eLearn\.Punjab',
    re.MULTILINE,
)
_SECTION_RE = re.compile(
    r'(?m)^\s*(\d+(?:\.\d+){0,2})\s+([A-Z][A-Za-z0-9 ,&\-]{3,80})\s*$'
)
_BARE_TITLE_RE = re.compile(
    r'^\s*([A-Z][a-zA-Z]*(?:\s+[A-Za-z]+){1,5})\s*$',
    re.MULTILINE,
)
# Real headings don't contain runs of common filler words -- this guard
# was added after direct sampling caught the section-regex matching
# garbled OCR body text on a degraded Physics 12th page (falsely captured
# "A tungsten target is struck by electrons that have been accelerated
# from rest" as a "heading").
_STOPWORDS = {"that", "have", "been", "from", "with", "this", "were", "which", "their", "there"}

# End-of-chapter structural labels (exercises, question banks, summaries)
# recur across every book's back matter and are NOT chapter/section
# titles -- confirmed by direct testing: Chemistry 11th's section-regex
# tier was matching "EXERCISE / MULTIPLE CHOICE QUESTIONS", "NUMERICAL
# PROBLEMS", "SHORT ANSWER QUESTIONS" as if they were chapter headings.
# Showing a student "go read the MULTIPLE CHOICE QUESTIONS section" as a
# study pointer would be actively misleading -- worse than no citation.
_STRUCTURAL_BLOCKLIST = {
    "exercise", "exercises", "questions", "problems", "review", "summary",
    "key points", "short answer", "long answer", "multiple choice",
    "numerical", "self assessment", "activity", "activities", "answers",
}


def _is_structural_label(candidate: str) -> bool:
    lowered = candidate.lower()
    return any(term in lowered for term in _STRUCTURAL_BLOCKLIST)


def _looks_garbled(candidate: str) -> bool:
    """Rejects candidates that are mostly single/double-character tokens
    (e.g. "O O\\nO O" -- garbled chemical-structure rendering) or have too
    low an alphabetic-character ratio to plausibly be a real title."""
    tokens = candidate.split()
    if tokens and sum(1 for t in tokens if len(t) <= 2) / len(tokens) > 0.5:
        return True
    alpha = sum(1 for c in candidate if c.isalpha())
    return len(candidate) > 0 and alpha / len(candidate) < 0.6


def _looks_like_prose(candidate: str) -> bool:
    words = candidate.lower().split()
    return sum(1 for w in words if w in _STOPWORDS) >= 2


def _is_valid_heading(candidate: str) -> bool:
    return not (_looks_like_prose(candidate) or _is_structural_label(candidate) or _looks_garbled(candidate))


def detect_page_chapter(page_text: str) -> str | None:
    """Best-effort chapter/section label for one page's RAW extracted text
    (runs before clean_punjab_board_text() -- validated to match reliably
    against raw text directly, no pre-cleaning needed)."""
    m = _ELEARN_RE.search(page_text)
    if m and _is_valid_heading(m.group(1)):
        return m.group(1).strip()

    m = _SECTION_RE.search(page_text)
    if m and _is_valid_heading(m.group(2)):
        return f"{m.group(1)}. {m.group(2).strip()}"

    m = _BARE_TITLE_RE.match(page_text)
    if m and _is_valid_heading(m.group(1)):
        return m.group(1).strip()

    return None


# Manual (page_index, chapter_label) overrides for books where automatic
# detection is unreliable, sourced directly from each book's own printed
# table of contents (verified via pdfplumber page dumps, not guessed):
#
# - Chemistry 11th: has a clean, complete 16-chapter TOC on pages 2-3.
#   Printed-page -> PDF-index offset of +3 confirmed exactly at 6
#   independently-checked chapter starts across the whole book (1, 2, 4,
#   8, 11, 16) -- high confidence.
# - Physics 12th: has a 9-chapter TOC on page 2, but extraction is
#   OCR-degraded (watermark bleed). Offset of +4 lands within 1-2 pages
#   of the true chapter start at every checked point -- good enough for
#   chapter-level (not page-level) attribution.
# - Physics 11th: confirmed via a full-book scan (not just the front
#   matter) to have NO extractable table of contents anywhere -- this
#   book has no override table; it falls back to automatic detection
#   (which found ~1/10 sampled pages usable) and will show "Front
#   Matter"/stale forward-fill for most of its content. Flagged
#   explicitly rather than guessed at.
CHAPTER_OVERRIDES: dict[str, list[tuple[int, str]]] = {
    "11th Class Chemistry PunjabBoard Year 2025.pdf": [
        (4, "1. Periodic Table and Periodic Properties"),
        (23, "2. Atomic Structure"),
        (46, "3. Chemical Bonding"),
        (73, "4. Stoichiometry"),
        (94, "5. States and Phases of Matter"),
        (115, "6. Chemical Energetics"),
        (143, "7. Reaction Kinetics"),
        (166, "8. Chemical Equilibrium"),
        (188, "9. Acid-Base Chemistry"),
        (210, "10. Electrochemistry"),
        (239, "11. Hydrocarbons"),
        (268, "12. Nitrogen and Sulfur"),
        (286, "13. Halogens"),
        (300, "14. Atmosphere"),
        (317, "15. Basic Separation Techniques"),
        (329, "16. Lab Safety and Practical Skills"),
    ],
    "12th Class Physics PunjabBoard Year 2026.pdf": [
        (5, "1. Thermal Physics"),
        (19, "2. Simple Harmonic Motion"),
        (46, "3. Physical Optics"),
        (61, "4. Electrostatics"),
        (85, "5. Alternating Current"),
        (118, "6. Quantum Physics"),
        (137, "7. Nuclear and Particle Physics"),
        (156, "8. Medical Physics"),
        (169, "9. Space and Environment"),
    ],
}


def detect_page_chapters(page_texts: list[str], filename: str) -> list[str]:
    """Per-page best-effort chapter label for a whole book, forward-filled
    across pages with no detectable heading. Falls back to
    CHAPTER_OVERRIDES for books flagged as unreliable for auto-detection."""
    if filename in CHAPTER_OVERRIDES:
        overrides = CHAPTER_OVERRIDES[filename]
        labels, current, idx = [], "Front Matter", 0
        for i in range(len(page_texts)):
            while idx < len(overrides) and overrides[idx][0] == i:
                current = overrides[idx][1]
                idx += 1
            labels.append(current)
        return labels

    raw_labels = [detect_page_chapter(t) for t in page_texts]
    filled, current = [], None
    for label in raw_labels:
        if label is not None:
            current = label
        filled.append(current if current is not None else "Front Matter")
    return filled


def chapter_transitions(page_chapters: list[str]) -> list[tuple[int, str]]:
    """Collapse a per-page chapter list to (page_index, chapter) only where
    the chapter changes from the previous page."""
    transitions, last = [], None
    for i, chapter in enumerate(page_chapters):
        if chapter != last:
            transitions.append((i, chapter))
            last = chapter
    return transitions


def locate_chapter_boundaries(cleaned_text: str, transitions: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """
    Re-finds each transition's chapter label inside the final CLEANED
    whole-book text, returning (offset, chapter) pairs in ascending offset
    order. Headings are never deleted by clean_punjab_board_text()'s rules,
    so they reliably survive and can be re-located -- a monotonic search
    cursor keeps ordering correct even if a short label could otherwise
    coincidentally match earlier in the document. If a label can't be
    re-found at all (rare), that transition is skipped and the previous
    chapter's range simply extends to cover it -- a safe degrade, not a
    crash, matching how the project has handled similar edge cases before.
    """
    boundaries, cursor = [], 0
    for _, chapter in transitions:
        if chapter == "Front Matter":
            boundaries.append((0, chapter))
            continue
        words = chapter.split()
        if not words:
            continue
        pattern = r"\s+".join(re.escape(w) for w in words)
        match = re.search(pattern, cleaned_text[cursor:])
        if match:
            offset = cursor + match.start()
            boundaries.append((offset, chapter))
            cursor = offset

    # Collapse consecutive entries that resolved to the same offset --
    # happens when forward-fill captures slightly different substrings of
    # the same physical heading on nearby pages (e.g. "13. Gaseous" then
    # "13. Gaseous Exchange" both point at the same text). Keep the later,
    # typically fuller, label.
    deduped = []
    for offset, chapter in boundaries:
        if deduped and deduped[-1][0] == offset:
            deduped[-1] = (offset, chapter)
        else:
            deduped.append((offset, chapter))
    return deduped


def locate_override_boundaries(
    page_texts: list[str], cleaned_text: str, overrides: list[tuple[int, str]]
) -> list[tuple[int, str]]:
    """
    Locates manually-overridden chapter boundaries in the cleaned text.

    Unlike locate_chapter_boundaries(), this does NOT search for the
    override's hand-written display label (e.g. "1. Periodic Table and
    Periodic Properties") -- that label was never literally extracted
    from the PDF, so it won't match verbatim against source text that's
    often differently cased/laid out (e.g. "PERIODIC TABLE AND\\n1\\n
    PERIODIC PROPERTIES"). Instead, each override's own page's ACTUAL raw
    text supplies the search anchor (its first substantial line, which is
    typically the chapter title as it really appears, and reliably
    repeats as a running header throughout that chapter -- so the first
    match found is its start), while the hand-written label is kept only
    for display.
    """
    boundaries, cursor = [], 0
    for page_index, label in overrides:
        if page_index >= len(page_texts):
            continue
        anchor = None
        for line in page_texts[page_index].splitlines():
            candidate = line.strip()
            if sum(1 for c in candidate if c.isalpha()) >= 10:
                anchor = candidate
                break
        if not anchor:
            continue
        words = anchor.split()[:8]
        if not words:
            continue
        pattern = r"\s+".join(re.escape(w) for w in words)
        match = re.search(pattern, cleaned_text[cursor:])
        if match:
            offset = cursor + match.start()
            boundaries.append((offset, label))
            cursor = offset
    return boundaries


def load_all_textbooks(directory: str) -> list[dict]:
    """
    Loads and cleans every real Punjab Board textbook PDF in `directory`.

    Returns:
        List of dicts: {"text": cleaned_full_text, "subject": ..., "grade": ...,
        "chapter_boundaries": [(offset, chapter), ...]} one entry per book
        (chunking happens later, in a separate step).
    """
    all_textbooks = []

    if not os.path.exists(directory):
        raise FileNotFoundError(f"The system cannot find the path specified: '{directory}'")

    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]

    for each_file in pdf_files:
        metadata = extract_metadata_from_filename(each_file)
        full_path = os.path.join(directory, each_file)

        # Single pdfplumber pass: extract per-page text for both the
        # existing concatenation (unchanged behavior -- byte-identical
        # to before) and the new chapter-detection pass, so we don't
        # open/parse the same PDF twice.
        with pdfplumber.open(full_path) as pdf:
            page_texts = [page.extract_text() or "" for page in pdf.pages]
        raw_text = "".join(page_texts)
        cleaned_text = clean_punjab_board_text(raw_text)

        if each_file in CHAPTER_OVERRIDES:
            # Override page indices are known-good (sourced from the
            # book's own printed table of contents), but need their own
            # anchor-based location logic -- see locate_override_boundaries().
            chapter_boundaries = locate_override_boundaries(page_texts, cleaned_text, CHAPTER_OVERRIDES[each_file])
        else:
            page_chapters = detect_page_chapters(page_texts, each_file)
            transitions = chapter_transitions(page_chapters)
            chapter_boundaries = locate_chapter_boundaries(cleaned_text, transitions)

        result_dict = {
            "text": cleaned_text,
            "subject": metadata["subject"],
            "grade": metadata["grade"],
            "board": metadata["board"],
            "year": metadata["year"],
            "title": metadata["title"],
            "extension": metadata["extension"],
            "chapter_boundaries": chapter_boundaries,
        }
        print(f"Loaded and cleaned textbook: {result_dict['title']} ({len(chapter_boundaries)} chapter transitions detected)")

        all_textbooks.append(result_dict)

    return all_textbooks


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
        "11th Class Biology PunjabBoard Year 2018.pdf",
        "12th Class Biology PunjabBoard Year 2026.pdf",
        "11th Class Chemistry PunjabBoard Year 2025.pdf",
        "12th Class Chemistry PunjabBoard Year 2026.pdf",
        "11th Class Physics PunjabBoard Year 2026.pdf",
        "12th Class Physics PunjabBoard Year 2026.pdf",
    ]
    for name in test_filenames:
        extract_metadata_from_filename(name)

    # content = load_pdf("datasets/pdfs/Biology-2e_-_WEB.pdf")
    # save_text(content, "datasets/biology.txt")
    # print(content)
