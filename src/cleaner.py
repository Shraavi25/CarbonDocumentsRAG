"""Clean text extracted by loader.py.

Handles PDF-extraction artifacts:
- words split across line breaks with a trailing hyphen
- GHG gas formulas extracted with a stray space before the subscript digit
  (e.g. "CO 2" -> "CO2", "CH 4" -> "CH4")
- GHG drop-cap initials, which pdfplumber extracts as their own line
  separated from the rest of the word (e.g. "T\\nhe Greenhouse..." -> "The Greenhouse...")
- standalone page-number lines
- irregular whitespace

Running headers/footers (chapter/appendix titles, page numbers) are already
excluded by loader.py based on their position on the page, not handled here.
"""

import re

_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_GAS_SUBSCRIPT_RE = re.compile(r"\b(CO|CH|SF|NF|PFC|HFC|N)\s+(\d)")
# The subscript digit of a gas formula (e.g. the "2" in "CO2") sits much
# closer to the next character than body text does, so the word-spacing
# heuristic that produced this text often missed the space that should
# follow it (e.g. "25 tonnes CO2and the total" -> "...CO2 and the total").
_GAS_FORMULA_RE = re.compile(r"\b(CO2|CH4|N2O|SF6|NF3|HFCs|PFCs)(?=[A-Za-z])")
# A lone capital letter on its own line, immediately followed (possibly after
# one or more "## Heading" lines, when the drop-cap's paragraph starts with a
# subsection heading) by the rest of the word in lowercase, is a drop-cap
# initial split from its word.
_DROPCAP_RE = re.compile(r"^([A-Z])\n((?:##[^\n]*\n)*)(?=[a-z])", re.MULTILINE)
_PAGE_NUM_LINE_RE = re.compile(r"^\d{1,4}$")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def dehyphenate(text):
    return _HYPHEN_BREAK_RE.sub(r"\1\2", text)


def fix_gas_formulas(text):
    text = _GAS_SUBSCRIPT_RE.sub(r"\1\2", text)
    return _GAS_FORMULA_RE.sub(r"\1 ", text)


def merge_dropcaps(text):
    return _DROPCAP_RE.sub(r"\2\1", text)


def strip_page_number_lines(text):
    lines = [l for l in text.split("\n") if not _PAGE_NUM_LINE_RE.match(l.strip())]
    return "\n".join(lines)


def normalize_whitespace(text):
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def clean_text(text, doc_id=None):
    text = dehyphenate(text)
    text = fix_gas_formulas(text)
    if doc_id == "ghg_protocol":
        text = merge_dropcaps(text)
    text = strip_page_number_lines(text)
    text = normalize_whitespace(text)
    return text


def clean_record(record):
    cleaned = dict(record)
    cleaned["text"] = clean_text(record["text"], record.get("doc_id"))
    cleaned["tables"] = [
        {**t, "text": normalize_whitespace(fix_gas_formulas(dehyphenate(t["text"])))}
        for t in record.get("tables", [])
    ]
    return cleaned


def clean_documents(records):
    return [clean_record(r) for r in records]


if __name__ == "__main__":
    from loader import load_document, DOCUMENTS

    for doc_id in DOCUMENTS:
        recs = clean_documents(load_document(doc_id))
        sample = recs[len(recs) // 2]
        print(f"{doc_id} | page {sample['page']} | section: {sample['section']!r}")
        print(sample["text"][:400])
        if sample["tables"]:
            print("--- table ---")
            print(sample["tables"][0]["text"][:400])
        print()
