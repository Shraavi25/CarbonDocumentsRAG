"""Load BRSR 2023 and GHG Protocol PDFs into per-page records.

Each record contains the page's narrative text (with table regions removed)
plus its tables, converted to a representation suited to that document:
- BRSR tables are mostly empty disclosure-format schemas, so they are
  linearized into descriptive sentences ("field -- reported as: columns").
- GHG Protocol tables carry real data, so they are rendered as markdown.

GHG pages also get a lot of false-positive "tables" from decorative page
furniture (vertical sidebar text, chapter-number graphics); these are
filtered out before extraction.
"""

import re
from collections import Counter, defaultdict
from pathlib import Path

import pdfplumber

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DOCUMENTS = {
    "brsr_2023": {
        "path": DATA_DIR / "brsr_2023.pdf",
        "skip_front": 0,
        "skip_back": 0,
    },
    "ghg_protocol": {
        "path": DATA_DIR / "ghg_protocol.pdf",
        "skip_front": 3,   # cover, contributor list, table of contents
        "skip_back": 11,   # contributor lists, disclaimer, about WBCSD/WRI, back cover
    },
}

# Headers are set in caps ("SECTION B: ...", "PRINCIPLE 2 ..."); in-sentence
# references use title case, so matching case-sensitively avoids false positives.
_BRSR_SECTION_RE = re.compile(r"^(SECTION [A-C]\s*:.*|PRINCIPLE\s+\d+\b.*)")

# Chapter/appendix titles appear in two styles:
# - divider pages: letters spaced out, e.g. "C H A P T E R  3  Setting Organizational Boundaries"
# - running headers on regular pages: "CHAPTER 3 Setting Organizational Boundaries 23" (trailing page no.)
_GHG_CHAPTER_SPACED_RE = re.compile(r"^C\s+H\s+A\s+P\s+T\s+E\s+R\s+([\d\s]+?)\s{2,}(.*)$")
_GHG_CHAPTER_RUNNING_RE = re.compile(r"^CHAPTER\s+(\d+):?\s+(.+?)\s+\d{1,3}$")
_GHG_APPENDIX_SPACED_RE = re.compile(r"^A\s+P\s+P\s+E\s+N\s+D\s+I\s+X\s+([A-Z])\b\s*(.*)$")
_GHG_APPENDIX_RUNNING_RE = re.compile(r"^APPENDIX\s+([A-Z])\s+(.+?)\s+\d{1,3}$")
# Divider page with no title, e.g. "APPENDIX A87" (letter + page no, no space)
_GHG_APPENDIX_DIVIDER_RE = re.compile(r"^APPENDIX\s+([A-Z])\s*\d{0,3}$")
_GHG_INTRO_RE = re.compile(r"^(?:Introduction|INTRODUCTION(?:\s+\d{1,3})?)$")
# Back-matter divider pages
_GHG_REFERENCE_RE = re.compile(r"^(Acronyms|Glossary|References|Bibliography)$")

# --- Layout-aware extraction for GHG's two-column pages ---------------------
# Running headers/chapter-divider titles/chapter-number graphics all sit in a
# narrow band at the very top of the page (top < ~30pt); footers (running
# chapter+page-number) sit in a band at the bottom (top > ~810pt on an
# 842pt-tall page). Body content never falls in either band.
_GHG_HEADER_MAX_TOP = 35
_GHG_FOOTER_MIN_TOP = 800

# A gap narrower than this between adjacent x-ranges on a row is just normal
# word/character spacing, not a column break.
_GHG_MIN_GUTTER = 15

# Right-column text is left-justified, so its left edge sits at a consistent
# x across the page even when no single row contains both columns. Candidate
# x's are the start of any second-or-later x-range on a row, or the start of
# any single x-range that begins past this fraction of the page width.
_GHG_LEFT_MARGIN = 37
_GHG_RIGHT_START_MIN_FRAC = 0.45
_GHG_MIN_SPLIT_CANDIDATES = 3
_GHG_SPLIT_BUFFER = 5
# A narrow left "column" relative to the right (e.g. a Glossary/Acronyms
# term-definition layout) is a paired-row table, not two body-text columns.
_GHG_MIN_COLUMN_RATIO = 0.6
# A row spans the gutter as one full-width line only if it starts near the
# left margin; sidebar rows (e.g. a callout question) can also start left of
# split_x while still belonging entirely to the right column.
_GHG_FULL_WIDTH_MAX_X0 = _GHG_LEFT_MARGIN + 50

# Subsection headings render in a distinct bold-condensed font, 12-20pt, in the
# document's green accent color (a single-component "spot color" of 1.0).
# The same font/color is also used for ~90pt drop-cap initials, which the
# upper size bound excludes.
_GHG_HEADING_FONT_RE = re.compile(r"BoldCondTwenty")
_GHG_HEADING_MIN_SIZE = 12
_GHG_HEADING_MAX_SIZE = 20


def _clean_cell(cell):
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", cell).strip()


def _is_decorative_cell(cell):
    """Detect single-character-per-line vertical sidebar text (e.g. 'S\\nT\\nA\\nN...')."""
    if not cell:
        return True
    lines = cell.split("\n")
    return len(lines) > 1 and all(len(l.strip()) <= 1 for l in lines)


def is_real_table(table_rows, min_substantive_cells=4, min_cell_chars=3):
    """Filter out pdfplumber false positives from decorative page layout."""
    if len(table_rows) < 2 or len(table_rows[0]) < 2:
        return False
    count = 0
    for row in table_rows:
        for cell in row:
            if cell and not _is_decorative_cell(cell) and len(cell.strip()) >= min_cell_chars:
                count += 1
    return count >= min_substantive_cells


# A table whose first row is one cell spanning (almost) the full table width
# and much taller than the next row is a multi-line, multi-column header that
# pdfplumber couldn't grid into per-column cells (and which often overlaps the
# first data row too, dumping their text into one jumbled blob). Its real
# column layout only emerges in the data rows below it.
_GHG_HEADER_ROW_HEIGHT_RATIO = 1.5
_GHG_MIN_COLUMN_WIDTH = 12
_GHG_HEADER_UPPERCASE_RATIO = 0.8
# Acronyms to keep uppercase when humanizing an ALL-CAPS header fragment.
_GHG_HEADER_ACRONYMS = ("ghg",)


def _humanize_header(text):
    """Turn an ALL-CAPS header fragment into a readable phrase, collapsing
    its line breaks and keeping known acronyms (e.g. "GHG") uppercase."""
    text = re.sub(r"\s+", " ", text).strip().lower()
    for acronym in _GHG_HEADER_ACRONYMS:
        text = re.sub(rf"\b{acronym}\b", acronym.upper(), text)
    return text


def _capitalize(text):
    return text[:1].upper() + text[1:] if text else text


def _reconstruct_merged_header(table, page):
    """Rebuild the header of a table whose top row is one oversized cell.

    The real column boundaries are recovered from the data rows below: for
    each column index, its true width is whichever width recurs across
    multiple rows (one-off larger/smaller widths come from anomalous merged
    cells in a single row, e.g. a footnote, or border-line slivers, and are
    excluded). For each real column, its header text is re-extracted by
    cropping the page to that column's x-range within the header band.

    Columns that share a second-row "sub-header" (e.g. "BASED ON EQUITY
    SHARE" / "BASED ON FINANCIAL CONTROL" under one combined super-header)
    are grouped so their shared super-header text is cropped as one
    contiguous string, rather than split mid-word at the column boundary.

    Returns a new list of rows (header + data, real columns only), or None if
    `table` doesn't match this "oversized merged header cell" pattern.
    """
    trows = table.rows
    if len(trows) < 3:
        return None

    row0_cells = trows[0].cells
    non_none0 = [c for c in row0_cells if c is not None]
    if len(non_none0) != 1:
        return None
    row0_height = non_none0[0][3] - non_none0[0][1]
    row1_height = trows[1].bbox[3] - trows[1].bbox[1]
    if row0_height <= row1_height * _GHG_HEADER_ROW_HEIGHT_RATIO:
        return None

    # Real table headers in this document are set in caps; an oversized row-0
    # cell over body-text prose (e.g. a case-study sidebar box with its own
    # small embedded table) is mixed-case and isn't a header to reconstruct.
    x0, _, x1, _ = non_none0[0]
    band_text = page.crop((x0, trows[0].bbox[1], x1, trows[1].bbox[1])).extract_text() or ""
    letters = [ch for ch in band_text if ch.isalpha()]
    if not letters or sum(1 for ch in letters if ch.isupper()) / len(letters) < _GHG_HEADER_UPPERCASE_RATIO:
        return None

    widths_by_col = defaultdict(list)
    for row in trows[1:]:
        for i, cell in enumerate(row.cells):
            if cell is not None:
                widths_by_col[i].append((round(cell[2] - cell[0], 1), cell))

    col_bounds = {}
    for i, entries in widths_by_col.items():
        mode_width, count = Counter(w for w, _ in entries).most_common(1)[0]
        if count >= 2 and mode_width >= _GHG_MIN_COLUMN_WIDTH:
            cell = next(c for w, c in entries if w == mode_width)
            col_bounds[i] = (cell[0], cell[2])

    col_indices = sorted(col_bounds)
    if len(col_indices) < 2:
        return None

    # The first row where every real column is populated is the first data row.
    data_idx = next(
        (ri for ri in range(1, len(trows))
         if all(trows[ri].cells[i] is not None for i in col_indices)),
        None,
    )
    if data_idx is None:
        return None

    # Rows between the merged header and the data grid hold sub-headers.
    sub_header = {i: "" for i in col_indices}
    for row in trows[1:data_idx]:
        for i in col_indices:
            cell = row.cells[i]
            if cell is not None:
                text = (page.crop(cell).extract_text() or "").strip()
                if text:
                    sub_header[i] = text

    header_top = trows[0].bbox[1]
    subheader_top = trows[1].bbox[1]
    data_top = trows[data_idx].bbox[1]

    # Group adjacent columns that share a sub-header -- their row-0 text is
    # one combined super-header, cropped as a single contiguous string so it
    # isn't split mid-word at the inter-column boundary.
    groups = []
    i = 0
    while i < len(col_indices):
        c = col_indices[i]
        group = [c]
        if sub_header[c]:
            while i + 1 < len(col_indices) and sub_header[col_indices[i + 1]]:
                i += 1
                group.append(col_indices[i])
        groups.append(group)
        i += 1

    header = {}
    for group in groups:
        band_bottom = subheader_top if sub_header[group[0]] else data_top
        x0, _ = col_bounds[group[0]]
        _, x1 = col_bounds[group[-1]]
        text = _humanize_header(page.crop((x0, header_top, x1, band_bottom)).extract_text() or "")
        for c in group:
            if sub_header[c]:
                header[c] = f"{_capitalize(text)} — {_humanize_header(sub_header[c])}"
            else:
                header[c] = _capitalize(text)

    extracted = table.extract()
    out_rows = [[header[i] for i in col_indices]]
    for row in extracted[data_idx:]:
        values = [row[i] for i in col_indices]
        if any((v or "").strip() for v in values):
            out_rows.append(values)

    return out_rows


def table_to_markdown(table_rows):
    rows = [[_clean_cell(c) for c in row] for row in table_rows]
    if not rows:
        return ""
    width = len(rows[0])
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join(["---"] * width) + "|"]
    for row in rows[1:]:
        row = (row + [""] * width)[:width]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _is_prose_table(table_rows):
    """Small boxes (case-study sidebars) get detected as tables because of their
    border lines, but really hold 1-2 paragraphs of prose rather than a grid."""
    non_empty = [_clean_cell(c) for row in table_rows for c in row if _clean_cell(c)]
    return len(non_empty) <= 6


def render_prose_table(table_rows):
    """Flatten a sidebar-box 'table' back into deduplicated prose paragraphs.

    Border lines inside the box often cause the same text (or fragments of it)
    to be detected as multiple overlapping cells, so we drop exact duplicates
    and any cell that is wholly contained in another, longer cell.
    """
    seen = []
    for row in table_rows:
        for cell in row:
            cell = _clean_cell(cell)
            if cell and cell not in seen:
                seen.append(cell)

    kept = []
    for cell in sorted(seen, key=len, reverse=True):
        if not any(cell in k for k in kept):
            kept.append(cell)

    order = {cell: i for i, cell in enumerate(seen)}
    kept.sort(key=lambda c: order[c])
    return "\n\n".join(kept)


# A "S. No." / "S.No" column holds a row's serial number, not its label; the
# real label for each row is in the next column instead (e.g. "Particulars").
_SERIAL_COL_RE = re.compile(r"^S\.?\s*No\.?$", re.IGNORECASE)
# Column-header codes like "(B / A)" extracted with spaces around the slash;
# tighten to "(B/A)" to match how they're written in the prose.
_SLASH_IN_PARENS_RE = re.compile(r"\(\s*([A-Za-z0-9]+)\s*/\s*([A-Za-z0-9]+)\s*\)")
# Tables wider than this (NGRBC principle compliance grids, P1-P9) don't
# reduce to a clean per-column label; render as markdown instead.
_LINEARIZE_MAX_COLS = 12


def _normalize_header_fragment(text):
    return _SLASH_IN_PARENS_RE.sub(r"(\1/\2)", text)


def _fill_forward_row(raw_row, cleaned_row):
    """Replace each merged-cell continuation (raw cell is None) with the
    cleaned text of the cell that started the span, to its left."""
    filled = []
    last = ""
    for raw, cell in zip(raw_row, cleaned_row):
        if raw is None:
            filled.append(last)
        else:
            filled.append(cell)
            last = cell
    return filled


def _is_header_like(raw_row, cleaned_row):
    """A row that still has merged-cell gaps but names >=2 columns is a
    (super-)header row, not a data row or single-cell section divider."""
    return any(c is None for c in raw_row) and sum(1 for c in cleaned_row if c) >= 2


def _is_header_continuation(cleaned_row, reportable_cols):
    """A row with no merged-cell gaps of its own can still be a second header
    row if it densely labels every reportable column (e.g. "Gender" /
    "Return to work rate" / "Retention rate" stacked under a "Permanent
    employees" / "Permanent workers" super-header) -- unlike a data row,
    whose reportable cells are blank in this (unfilled) BRSR template."""
    return bool(reportable_cols) and all(cleaned_row[c] for c in reportable_cols)


def _is_divider_row(raw_row, cleaned_row):
    """A row holding one cell merged across the full width is a section
    divider (e.g. "EMPLOYEES" / "Water withdrawal by source (in kilolitres)"),
    introducing the rows that follow rather than being a row of its own."""
    non_none = [i for i, c in enumerate(raw_row) if c is not None]
    return len(non_none) == 1 and bool(cleaned_row[non_none[0]])


def _build_combined_labels(header_raw_rows, header_cleaned_rows, reportable_cols):
    """Merge a run of header rows into one combined label per reportable column.

    Each header row is filled-forward so a merged super-header's text applies
    to every column it spans. A header row whose filled-forward value is
    identical across every reportable column describes the table as a whole
    (e.g. "% of employees covered by") rather than distinguishing columns
    from each other, and is dropped -- unless every row is like that, in
    which case the top header row is kept anyway so columns still get labels.
    """
    filled_rows = [
        _fill_forward_row(raw, cleaned)
        for raw, cleaned in zip(header_raw_rows, header_cleaned_rows)
    ]
    discriminating = [
        filled
        for filled in filled_rows
        if len(reportable_cols) <= 1
        or len({filled[c] for c in reportable_cols if filled[c]}) > 1
    ]
    if not discriminating:
        discriminating = filled_rows[:1]

    labels = {}
    for c in reportable_cols:
        parts = []
        for filled in discriminating:
            v = filled[c]
            if v and (not parts or parts[-1] != v):
                parts.append(v)
        if parts:
            labels[c] = _normalize_header_fragment(" ".join(parts))
    return labels


def linearize_schema_table(table_rows):
    """Turn a (mostly-empty) BRSR disclosure table into descriptive sentences.

    Handles multi-row headers (merging e.g. "Male" + "No. (B)" into
    "Male No. (B)"), tables whose row label lives in a "Particulars" column
    after a serial-number column, and section-divider rows ("EMPLOYEES" /
    "WORKERS") that provide context for the rows beneath them.
    """
    if len(table_rows) < 2:
        return ""
    cleaned = [[_clean_cell(c) for c in row] for row in table_rows]
    ncols = len(cleaned[0])

    name_col, serial_col = 0, None
    if _SERIAL_COL_RE.match(cleaned[0][0]):
        serial_col, name_col = 0, 1
    reportable_cols = [c for c in range(ncols) if c not in (name_col, serial_col)]

    # Header block: rows that either have merged-cell gaps spanning >=2
    # columns (a super-header, e.g. "Permanent employees" over two columns)
    # or, lacking gaps of their own, still densely label every reportable
    # column (e.g. "Gender"/"Return to work rate"/"Retention rate" stacked
    # underneath it). A row failing both -- including row 0, for a
    # headerless continuation fragment whose first row is really data --
    # is left for the main loop below.
    header_end = 0
    while header_end < len(cleaned) and (
        _is_header_like(table_rows[header_end], cleaned[header_end])
        or _is_header_continuation(cleaned[header_end], reportable_cols)
    ):
        header_end += 1

    combined_label = _build_combined_labels(table_rows[:header_end], cleaned[:header_end], reportable_cols)

    lines = []
    context = None
    prev_was_divider = False
    for i in range(header_end, len(cleaned)):
        raw_row, row = table_rows[i], cleaned[i]

        if _is_header_like(raw_row, row) or _is_header_continuation(row, reportable_cols):
            # A (sub-)header reappears mid-table -- e.g. a second grid
            # ("Non-Monetary") stacked under the first ("Monetary") within
            # the same pdfplumber table, or a second label row (e.g.
            # "Gender"/"Return to work rate") that the header_end scan
            # above didn't reach. Refresh the labels it covers.
            combined_label.update(_build_combined_labels([raw_row], [row], reportable_cols))
            prev_was_divider = False
            continue

        if _is_divider_row(raw_row, row):
            divider = row[next(j for j, c in enumerate(raw_row) if c is not None)]
            if divider.isupper():
                divider = divider.capitalize()
            context = f"{context} > {divider}" if prev_was_divider and context else divider
            prev_was_divider = True
            continue

        prev_was_divider = False
        label = row[name_col]
        if not label:
            continue

        fields = []
        for c in reportable_cols:
            field_label = combined_label.get(c)
            if not field_label:
                continue
            text = f"{field_label}: {row[c]}" if row[c] else field_label
            if not fields or fields[-1] != text:
                fields.append(text)

        prefix = f"{context}: {label}" if context else label
        lines.append(f"{prefix} -- reported as: {', '.join(fields)}." if fields else prefix)

    if not lines and combined_label:
        name_header = next((cleaned[r][name_col] for r in range(header_end) if cleaned[r][name_col]), "")
        fields = [combined_label[c] for c in reportable_cols if c in combined_label]
        if name_header:
            target = f"each {name_header}"
        elif context:
            target = f"each {context} row"
        else:
            target = "each row"
        lines.append(f"For {target}, report: {', '.join(fields)}.")

    return "\n".join(lines)


_TOP_ITEM_RE = re.compile(r"^(\d{1,2})\.\s+(.*)")
_SUB_ITEM_RE = re.compile(r"^([a-z])\.\s+(.*)")


def _table_item_label(page, table_bbox, bboxes):
    """Find the question a table answers, e.g. "18.a Employees and workers
    (including differently abled):" for the employees/workers table on page
    2 -- the nearest numbered ("18.") and, if any, lettered ("a.") item
    immediately above the table's top edge. Lines that fall inside any
    table's own bbox are skipped, since those are table content (e.g. a
    "1."/"2." serial-number column), not document headings.
    """
    table_top = table_bbox[1]
    top_item = sub_item = None
    for line in page.extract_text_lines():
        top = line["top"]
        if top > table_top or any(b[1] <= top <= b[3] for b in bboxes):
            continue
        text = line["text"].strip()
        m = _TOP_ITEM_RE.match(text)
        if m:
            top_item = (top, m.group(1), text)
            sub_item = None
            continue
        m = _SUB_ITEM_RE.match(text)
        if m and top_item and top > top_item[0]:
            sub_item = (m.group(1), m.group(2))

    if not top_item:
        return None
    if sub_item:
        return f"{top_item[1]}.{sub_item[0]} {sub_item[1]}"
    return top_item[2]


def _detect_brsr_section(text):
    """Return the last SECTION/PRINCIPLE header on the page (most specific /
    most representative of what continues onto the next page)."""
    section = None
    for line in text.split("\n"):
        m = _BRSR_SECTION_RE.match(line.strip())
        if m:
            section = m.group(0).strip()
    return section


def _detect_ghg_section(furniture_text):
    """Scan header/footer band text (one or more lines) for a section marker."""
    for line in furniture_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _GHG_INTRO_RE.match(line):
            return "Introduction"
        if _GHG_REFERENCE_RE.match(line):
            return line
        m = _GHG_CHAPTER_SPACED_RE.match(line) or _GHG_CHAPTER_RUNNING_RE.match(line)
        if m:
            number = re.sub(r"\s+", "", m.group(1))
            return f"Chapter {number}: {m.group(2).strip()}"
        m = _GHG_APPENDIX_SPACED_RE.match(line) or _GHG_APPENDIX_RUNNING_RE.match(line)
        if m:
            title = m.group(2).strip()
            return f"Appendix {m.group(1)}" + (f": {title}" if title else "")
        m = _GHG_APPENDIX_DIVIDER_RE.match(line)
        if m:
            return f"Appendix {m.group(1)}"
    return None


def _merge_x_ranges(chars):
    """Merge overlapping/adjacent (x0, x1) character spans into solid blocks."""
    intervals = sorted((c["x0"], c["x1"]) for c in chars)
    merged = []
    for x0, x1 in intervals:
        if merged and x0 <= merged[-1][1] + 2:
            merged[-1] = (merged[-1][0], max(merged[-1][1], x1))
        else:
            merged.append((x0, x1))
    return merged


# A gas-formula subscript digit (e.g. the "2" in "CO2") is rendered far
# smaller than body text and a few points below the baseline -- enough to
# land in its own row group between two lines of text once tops are rounded,
# severing it from "CO"/"CH"/etc. and corrupting whatever word follows on the
# next line. Snap it back onto the preceding character's line so it stays
# attached to its gas formula.
_GHG_SUBSCRIPT_MAX_SIZE = 7
_GHG_SUBSCRIPT_MAX_DROP = 8


def _fix_subscript_chars(chars):
    out = []
    prev = None
    for c in chars:
        if (
            prev is not None
            and c.get("size", 0) <= _GHG_SUBSCRIPT_MAX_SIZE
            and c["text"].isdigit()
            and 0 < c["top"] - prev["top"] <= _GHG_SUBSCRIPT_MAX_DROP
            and c["x0"] >= prev["x1"] - 1
        ):
            c = {**c, "top": prev["top"]}
        out.append(c)
        prev = c
    return out


def _group_rows(chars):
    rows = defaultdict(list)
    for c in chars:
        rows[round(c["top"])].append(c)
    return rows


def _find_column_split(rows, page_width):
    """Find the page's column gutter as the right column's left margin.

    Right-column text is left-justified, so its left edge sits at a
    consistent x across the page even when no single row's content spans
    both columns (e.g. one column has more/taller lines than the other).
    Candidates for that margin are: the start of any second-or-later merged
    x-range on a row (a real gap within that row), or the start of a row's
    single x-range when it begins past the page's midpoint. The most common
    candidate value is the column's left margin; indented sub-items (e.g.
    nested bullets) produce a secondary, larger value with comparable
    frequency, so ties for "most common" are broken toward the smaller value
    -- text is left-justified, so indentation only ever pushes a line's start
    to the right of the true margin, never left of it. This is robust to the
    handful of full-width or off-grid (figure/diagram) rows that don't fit
    the pattern.

    A narrow left "column" relative to the right indicates a paired
    term/definition row layout (e.g. Glossary, Acronyms) rather than two
    body-text columns, and is left unsplit. Returns None for single-column
    pages and these paired-row pages alike.
    """
    right_starts = []
    for row in rows.values():
        merged = _merge_x_ranges(row)
        if len(merged) >= 2:
            for (_, a1), (b0, _) in zip(merged, merged[1:]):
                if b0 - a1 >= _GHG_MIN_GUTTER:
                    right_starts.append(b0)
        elif len(merged) == 1:
            x0, _ = merged[0]
            if x0 >= page_width * _GHG_RIGHT_START_MIN_FRAC:
                right_starts.append(x0)

    if len(right_starts) < _GHG_MIN_SPLIT_CANDIDATES:
        return None

    # Round to absorb sub-point floating-point jitter between glyphs that
    # share the same nominal left edge, without merging genuinely different
    # column positions.
    counts = Counter(round(x, 1) for x in right_starts)
    max_count = max(counts.values())
    right_margin = min(x for x, n in counts.items() if n == max_count)

    left_width = right_margin - _GHG_LEFT_MARGIN
    right_width = (page_width - _GHG_LEFT_MARGIN) - right_margin
    if right_width <= 0 or left_width / right_width < _GHG_MIN_COLUMN_RATIO:
        return None

    split_x = right_margin - _GHG_SPLIT_BUFFER

    left_only = sum(
        1 for row in rows.values()
        if len(_merge_x_ranges(row)) == 1 and _merge_x_ranges(row)[0][1] <= split_x
    )
    if left_only < _GHG_MIN_SPLIT_CANDIDATES:
        return None

    return split_x


def _is_ghg_heading_char(c):
    color = c.get("non_stroking_color")
    return (
        bool(_GHG_HEADING_FONT_RE.search(c.get("fontname") or ""))
        and _GHG_HEADING_MIN_SIZE <= c.get("size", 0) <= _GHG_HEADING_MAX_SIZE
        and isinstance(color, (tuple, list))
        and len(color) == 1
        and round(color[0], 2) == 1.0
    )


def _is_white_char(c):
    """White-on-colored-bar decorative text (vertical 'STANDARD'/'GUIDANCE'/
    appendix-letter tabs spelled one character per line down the page edge).
    These run down the side of nearly every page and aren't real content."""
    color = c.get("non_stroking_color")
    return isinstance(color, (tuple, list)) and len(color) == 3 and all(round(v, 2) == 1.0 for v in color)


def _extract_ghg_page(page, table_bboxes):
    """Layout-aware extraction for a GHG Protocol page.

    Strips the top-of-page running header / chapter-divider title and the
    bottom-of-page running footer (both returned separately, for section
    detection -- never included in body text). The remaining body band is
    split into left/right columns if a central gutter is found, and
    subsection headings (green bold-condensed text) are marked with "## ".
    """
    not_in_table = _table_bbox_filter(table_bboxes)

    def in_body_band(c):
        return (
            not_in_table(c)
            and _GHG_HEADER_MAX_TOP <= c["top"] < _GHG_FOOTER_MIN_TOP
            and not _is_white_char(c)
        )

    header_text = page.filter(lambda c: not_in_table(c) and c["top"] < _GHG_HEADER_MAX_TOP).extract_text() or ""
    footer_text = page.filter(lambda c: not_in_table(c) and c["top"] >= _GHG_FOOTER_MIN_TOP).extract_text() or ""

    body_chars = [c for c in page.chars if in_body_band(c)]
    if not body_chars:
        return header_text, footer_text, ""

    body_chars = _fix_subscript_chars(body_chars)
    rows = _group_rows(body_chars)
    split_x = _find_column_split(rows, page.width)
    if split_x is None:
        body_text = pdfplumber.utils.extract_text(body_chars) or ""
    else:
        left_chars, right_chars = [], []
        for top in sorted(rows):
            row_chars = rows[top]
            # A row that has a character straddling split_x and starts near
            # the left margin spans the gutter as one full-width line (a lead
            # paragraph or wide heading); keep it together on the left so its
            # reading order is preserved. A row that straddles split_x but
            # starts well to the right of the left margin is a right-column
            # row (e.g. a sidebar callout) that merely begins left of split_x;
            # keep it together on the right instead of splitting it mid-line.
            # Otherwise, a row with content on only one side falls entirely
            # into that side via the normal x0/x1 split below.
            straddled = any(c["x0"] < split_x < c["x1"] for c in row_chars)
            if straddled and min(c["x0"] for c in row_chars) <= _GHG_FULL_WIDTH_MAX_X0:
                left_chars.extend(row_chars)
            elif straddled:
                right_chars.extend(row_chars)
            else:
                left_chars.extend(c for c in row_chars if c["x1"] <= split_x)
                right_chars.extend(c for c in row_chars if c["x0"] > split_x)
        left = pdfplumber.utils.extract_text(left_chars) or ""
        right = pdfplumber.utils.extract_text(right_chars) or ""
        body_text = "\n".join(t for t in (left, right) if t)

    heading_raw = pdfplumber.utils.extract_text([c for c in body_chars if _is_ghg_heading_char(c)]) or ""
    # A heading that wraps onto a second line (e.g. "Base year emissions" /
    # "recalculation for structural changes") produces two heading_raw lines
    # whose first line, taken alone, can collide with unrelated body text that
    # happens to start the same way (e.g. a figure caption beginning "Base
    # year emissions ..."). Try matching the longest run of consecutive
    # heading_raw lines as a single contiguous block in body_text first, and
    # collapse it into one "## "-prefixed line; only fall back to matching a
    # single line on its own once no longer block matches.
    heading_lines = [l.strip() for l in heading_raw.split("\n") if l.strip()]
    i = 0
    while i < len(heading_lines):
        for j in range(len(heading_lines), i + 1, -1):
            block = "\n".join(heading_lines[i:j])
            marked = f"## {' '.join(heading_lines[i:j])}"
            if block in body_text and marked not in body_text:
                body_text = body_text.replace(block, marked, 1)
                i = j
                break
        else:
            heading = heading_lines[i]
            marked = f"## {heading}"
            if heading in body_text and marked not in body_text:
                body_text = body_text.replace(heading, marked, 1)
            i += 1

    return header_text, footer_text, body_text


def _table_bbox_filter(bboxes):
    def keep(obj):
        for (x0, top, x1, bottom) in bboxes:
            if x0 - 1 <= obj["x0"] and obj["x1"] <= x1 + 1 and top - 1 <= obj["top"] and obj["bottom"] <= bottom + 1:
                return False
        return True

    return keep


def load_document(doc_id):
    cfg = DOCUMENTS[doc_id]
    records = []
    current_section = None

    with pdfplumber.open(cfg["path"]) as pdf:
        start = cfg["skip_front"]
        end = len(pdf.pages) - cfg["skip_back"]

        for page in pdf.pages[start:end]:
            found_tables = page.find_tables()
            real_tables = [t for t in found_tables if is_real_table(t.extract())]
            bboxes = [t.bbox for t in real_tables]

            if doc_id == "ghg_protocol":
                header_text, footer_text, text = _extract_ghg_page(page, bboxes)
                section = _detect_ghg_section(header_text + "\n" + footer_text) or current_section
            else:
                if bboxes:
                    text = page.filter(_table_bbox_filter(bboxes)).extract_text() or ""
                else:
                    text = page.extract_text() or ""
                section = _detect_brsr_section(text) or current_section
            current_section = section

            tables_out = []
            for table in real_tables:
                rows = table.extract()
                if doc_id == "ghg_protocol":
                    rows = _reconstruct_merged_header(table, page) or rows
                if doc_id == "brsr_2023":
                    if len(rows[0]) <= _LINEARIZE_MAX_COLS:
                        rendered = linearize_schema_table(rows)
                        table_type = "schema"
                    else:
                        rendered = table_to_markdown(rows)
                        table_type = "markdown"
                elif _is_prose_table(rows):
                    rendered = render_prose_table(rows)
                    table_type = "text"
                else:
                    rendered = table_to_markdown(rows)
                    table_type = "markdown"
                if rendered:
                    if doc_id == "brsr_2023":
                        item_label = _table_item_label(page, table.bbox, bboxes)
                        if item_label:
                            rendered = f"[{item_label}]\n{rendered}"
                    tables_out.append({"type": table_type, "text": rendered})

            records.append({
                "doc_id": doc_id,
                "page": page.page_number,
                "section": section,
                "text": text,
                "tables": tables_out,
            })

    return records


def load_all():
    records = []
    for doc_id in DOCUMENTS:
        records.extend(load_document(doc_id))
    return records


if __name__ == "__main__":
    for doc_id in DOCUMENTS:
        recs = load_document(doc_id)
        n_tables = sum(len(r["tables"]) for r in recs)
        print(f"{doc_id}: {len(recs)} pages loaded, {n_tables} tables extracted")
        sample = recs[len(recs) // 2]
        print(f"  sample page {sample['page']} | section: {sample['section']!r}")
        print(f"  text preview: {sample['text'][:150]!r}")
        if sample["tables"]:
            print(f"  table preview:\n{sample['tables'][0]['text'][:300]}")
        print()
