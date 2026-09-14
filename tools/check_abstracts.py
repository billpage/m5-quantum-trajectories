"""
Vendored from https://github.com/billpage/wpmw (src/wpmwlib/check_abstracts.py),
generalized to also parse unordered-bullet indexes, not just numbered
ladders -- wpmw's own version anticipated this extension but left it as a
follow-up (its docs/algorithm and docs/supplement use bullets, not a
ladder, and aren't yet covered by any linter there either). If wpmw picks
up the same generalization, prefer pulling its version over hand-merging
this copy.

Abstract-consistency linter for the M5 project.

Each entry in a docs/<dir>/README.md index -- numbered ladder rung or
plain bullet -- carries its own description: what the note establishes,
what it depends on, what it corrects or supersedes. That same text is
also repeated as a blockquote directly under the note's own `# Title`, so
a reader who lands on the file itself (a raw link, a search hit, a folder
listing) gets the same orientation a reader of the README would have had,
without a detour.

Two copies of the same text drift the moment one is hand-edited and the
other isn't. This linter is the guard against that: it re-parses the
index entries out of `<dir>/README.md` (numbered `1. **[...]** ...` or
bulleted `- **[...]** ...`, either form, per directory), and for every
entry whose target file exists, checks that the file's own title-blockquote
is present and matches -- word for word, modulo whitespace -- the README
description. It also flags an entry whose target file has no
title-blockquote at all, and an entry whose target file does not exist
(a broken link into the index).

This is a text-equality check, not a math or link-target check, and is
kept as a separate tool from check_md_math for the same reason wpmw keeps
its math and notebook-math linters separate: unrelated failure modes want
unrelated tools, not one script with mixed concerns.

Run as:

    python3 tools/check_abstracts.py docs/algorithm docs/supplement docs/analysis

Exits 0 if every index entry's abstract matches its file, 1 if any
mismatch, missing blockquote, or broken link is found, 2 on bad input.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Start of an index entry, either a numbered ladder rung or a plain bullet:
#   14. **[`compensated_liouville_splitting.md`](compensated_liouville_splitting.md)** -- ...
#   -   **[`phase_space_crystal_lattice_algorithm.md`](phase_space_crystal_lattice_algorithm.md)** -- ...
# The filename and link target are usually identical (same-directory
# link); only the filename is used to locate the target file.
_ITEM_START = re.compile(
    r"^(?:(?P<num>\d+)\.|-)\s+\*\*\[`(?P<file>[^`]+)`\]\((?P<link>[^)]+)\)\*\*\s*(?P<rest>.*)$"
)
_HEADING = re.compile(r"^#{1,6}\s")
_EM_DASH_PREFIX = re.compile(r"^[\u2014-]\s*")


@dataclass
class IndexEntry:
    number: int | None  # None for a bullet entry (no ladder number)
    filename: str
    abstract: str
    line: int

    @property
    def label(self) -> str:
        return f"item {self.number}" if self.number is not None else f"entry ({self.filename})"


@dataclass
class Issue:
    file: str
    severity: str
    message: str


def _clean_paragraph(lines: list[str]) -> str:
    """Join wrapped list-item lines into one paragraph: strip list
    indentation, collapse internal whitespace, strip the entry's
    leading em dash or hyphen separator."""
    text = " ".join(line.strip() for line in lines if line.strip())
    text = _EM_DASH_PREFIX.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_index(readme_path: Path) -> tuple[list[IndexEntry], list[str]]:
    """Parse numbered-ladder or bulleted index entries out of a
    docs/<dir>/README.md. A single README is expected to use one style
    consistently, but entries are matched per-line, so a mixed file
    would still parse (each line stands on its own).

    An entry's `abstract` preserves blank-line-separated paragraphs
    (joined with a blank line); within a paragraph, wrapped source lines
    are collapsed to single spaces, matching how the abstract is meant
    to be reproduced as a single-line-per-paragraph blockquote.
    """
    lines = readme_path.read_text().splitlines()
    entries: list[IndexEntry] = []
    warnings: list[str] = []

    i, n = 0, len(lines)
    while i < n:
        m = _ITEM_START.match(lines[i])
        if not m:
            i += 1
            continue
        start_line = i + 1
        num = int(m.group("num")) if m.group("num") else None
        filename = m.group("file")
        para_lines: list[str] = [m.group("rest")]
        paragraphs: list[str] = []
        i += 1
        while i < n and not _ITEM_START.match(lines[i]) and not _HEADING.match(lines[i]):
            if lines[i].strip() == "":
                if para_lines:
                    cleaned = _clean_paragraph(para_lines)
                    if cleaned:
                        paragraphs.append(cleaned)
                    para_lines = []
            else:
                para_lines.append(lines[i])
            i += 1
        if para_lines:
            cleaned = _clean_paragraph(para_lines)
            if cleaned:
                paragraphs.append(cleaned)
        if not paragraphs:
            label = f"item {num}" if num is not None else f"entry"
            warnings.append(f"{readme_path}:{start_line}: {label} "
                             f"({filename}) has no description text")
            continue
        entries.append(IndexEntry(num, filename, "\n\n".join(paragraphs), start_line))
    return entries, warnings


def render_blockquote(abstract: str) -> str:
    """Render a parsed abstract as a GitHub-markdown blockquote: one
    `> ` line per paragraph, a bare `>` separating paragraphs."""
    paragraphs = abstract.split("\n\n")
    blocks = [f"> {p}" for p in paragraphs]
    return "\n>\n".join(blocks)


def _doc_abstract(doc_path: Path) -> str | None:
    """Extract the blockquote immediately following the file's `#`
    title (and the one blank line after it), normalised the same way
    as parse_index normalises the README side. Returns None if there
    is no title, or no blockquote right after it."""
    lines = doc_path.read_text().splitlines()
    if not lines or not lines[0].startswith("# "):
        return None
    i = 1
    if i < len(lines) and lines[i].strip() == "":
        i += 1
    if i >= len(lines) or not lines[i].startswith(">"):
        return None
    para_lines: list[str] = []
    paragraphs: list[str] = []
    while i < len(lines) and lines[i].startswith(">"):
        content = lines[i][1:].strip()
        if content == "":
            if para_lines:
                paragraphs.append(" ".join(para_lines))
                para_lines = []
        else:
            para_lines.append(content)
        i += 1
    if para_lines:
        paragraphs.append(" ".join(para_lines))
    return "\n\n".join(paragraphs) if paragraphs else None


def check_directory(dir_path: Path) -> tuple[list[Issue], int]:
    """Check one docs subdirectory's README index against its files.
    Returns (issues, number of index entries checked)."""
    readme_path = dir_path / "README.md"
    issues: list[Issue] = []
    if not readme_path.exists():
        return [Issue(str(dir_path), "NOREADME",
                       "no README.md in this directory")], 0

    entries, warnings = parse_index(readme_path)
    for w in warnings:
        issues.append(Issue(str(readme_path), "PARSE", w))

    for entry in entries:
        doc_path = dir_path / entry.filename
        if not doc_path.exists():
            issues.append(Issue(
                str(readme_path), "BROKENLINK",
                f"{entry.label} points at {entry.filename}, which "
                "does not exist in this directory"))
            continue
        doc_abstract = _doc_abstract(doc_path)
        if doc_abstract is None:
            issues.append(Issue(
                str(doc_path), "MISSING",
                f"no blockquote abstract found right after the title "
                f"({entry.label} in {readme_path.name})"))
            continue
        if doc_abstract != entry.abstract:
            issues.append(Issue(
                str(doc_path), "MISMATCH",
                f"title-blockquote does not match {entry.label} "
                f"in {readme_path.name}:\n"
                f"    README : {entry.abstract[:120]!r}...\n"
                f"    {doc_path.name} : {doc_abstract[:120]!r}..."))

    return issues, len(entries)


def _format_report(issues: list[Issue], checked_dirs: list[Path],
                    total_entries: int) -> str:
    lines: list[str] = []
    if not issues:
        for d in checked_dirs:
            lines.append(f"OK   {d}/README.md ({total_entries} index "
                          "entr" + ("y" if total_entries == 1 else "ies") +
                          ", all matched)")
    else:
        by_file: dict[str, list[Issue]] = {}
        for it in issues:
            by_file.setdefault(it.file, []).append(it)
        for file, file_issues in by_file.items():
            lines.append(f"=== {file} ({len(file_issues)} issue(s)) ===")
            for it in file_issues:
                lines.append(f"  [{it.severity:10s}] {it.message}")
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 tools/check_abstracts.py",
        description="Check that each README.md index entry (numbered "
                    "ladder or bulleted) in a docs subdirectory matches "
                    "the blockquote abstract under the corresponding "
                    "file's title.",
    )
    parser.add_argument("paths", nargs="*",
                        default=["docs/algorithm", "docs/supplement", "docs/analysis"],
                        help="Directories to check, each expected to "
                             "contain its own README.md with an index "
                             "(default: docs/algorithm docs/supplement "
                             "docs/analysis).")
    args = parser.parse_args(argv)

    dirs = [Path(p) for p in args.paths]
    missing = [d for d in dirs if not d.is_dir()]
    if missing:
        print(f"error: not a directory: {', '.join(map(str, missing))}",
              file=sys.stderr)
        return 2

    all_issues: list[Issue] = []
    total_entries = 0
    for d in dirs:
        issues, n = check_directory(d)
        all_issues.extend(issues)
        total_entries += n

    report = _format_report(all_issues, dirs, total_entries)
    if report:
        print(report)
    print(f"Summary: {len(all_issues)} issue(s) across {total_entries} "
          f"index entr{'y' if total_entries == 1 else 'ies'} in "
          f"{len(dirs)} director{'y' if len(dirs) == 1 else 'ies'}.")
    return 0 if not all_issues else 1


if __name__ == "__main__":
    sys.exit(main())
