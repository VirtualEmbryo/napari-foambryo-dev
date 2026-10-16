"""T1: the repository's documentation is for users: no internal design record, no decision labels, no dead link.

Up to 0.1.0.dev0 the README and the package docstring pointed to `docs/SPECIFICATION.md`, the plugin's internal
design record (approvals, decision log, evidence labels), and comments cited its decisions as "(decision Qn)". The
record moved to the plugin's research branch; `docs/how-it-works.md` says what each dock does instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
PUBLIC = sorted(
    [
        REPOSITORY / "README.md",
        REPOSITORY / "pyproject.toml",
        *(REPOSITORY / "docs").rglob("*.md"),
        *(REPOSITORY / "src").rglob("*.py"),
        *(REPOSITORY / "src").rglob("*.yaml"),
        *(REPOSITORY / "examples").rglob("*.py"),
    ],
)
LABELS = re.compile(r"decision Q\d|tier T4|SPECIFICATION\.md")
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def test_the_design_record_is_not_in_the_repository():
    assert not (REPOSITORY / "docs" / "SPECIFICATION.md").exists()
    assert (REPOSITORY / "docs" / "how-it-works.md").is_file()
    assert "docs/how-it-works.md" in (REPOSITORY / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("path", PUBLIC, ids=lambda p: p.relative_to(REPOSITORY).as_posix())
def test_no_design_record_label(path):
    found = LABELS.search(path.read_text(encoding="utf-8"))
    assert found is None, f"{path.name}: {found.group(0)!r}"


@pytest.mark.parametrize(
    "page",
    [REPOSITORY / "README.md", *sorted((REPOSITORY / "docs").rglob("*.md"))],
    ids=lambda p: p.relative_to(REPOSITORY).as_posix(),
)
def test_every_relative_link_resolves(page):
    for target in LINK.findall(page.read_text(encoding="utf-8")):
        if "://" in target or target.startswith("mailto:"):
            continue
        assert (page.parent / target).exists(), f"{page.name} links to {target}, which does not exist"
