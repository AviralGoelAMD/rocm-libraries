#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# clone_check.py -- report Python code copies that a change ADDS (advisory).
#
# Compares a base commit with either the working tree (default) or a head
# commit. Every run parses all in-scope Python files on both sides, fingerprints
# every window of WINDOW consecutive statements, and reports a fingerprint only
# when the change adds a genuine new occurrence of code that now exists in two
# or more places. Python only: C++ duplication is not measured.
#
# Usage:
#   python tools/clone_check.py                   # base = merge-base with origin/develop, head = working tree
#   python tools/clone_check.py --base <commit>   # other base
#   python tools/clone_check.py --head <commit>   # compare two commits (PR mode)
#   python tools/clone_check.py --format json     # machine-readable
#   python tools/clone_check.py --fail-on-new     # exit 1 when a copy is found (off by default)
#
# Exit codes: 0 ran (advisory), 1 copies found with --fail-on-new, 2 tool error.

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROCKE = Path(__file__).resolve().parents[1]  # tools -> rocke
SCOPE_DIRS = ("library", "platform/python/rocke")
# Test, benchmark, verify and example code is out of scope: its duplication is
# harness scaffolding, not kernel-authoring debt.
_OUT_OF_SCOPE = re.compile(
    r"(^|/)tests/"
    r"|(^|/)(examples|experiments)/"
    r"|^library/benchmarks/"
    r"|^platform/python/rocke/benchmark/"
    r"|/[^/]*_(verify|sweep)\.py$"
    r"|/(bench|benchmark|reproduce)_[^/]*\.py$"
)
# Host-side command-line and test-input helpers inside builder scripts. They are
# skipped together with `if __name__ == "__main__":` blocks. `check` and `bench`
# stay in scope: copies of them were judged real duplication.
_HARNESS_FUNCTIONS = frozenset({"main", "make_inputs", "parse_args", "_parser"})
GROUPS_FILE = "tools/clone_groups.toml"
DEFAULT_UPSTREAM = "origin/develop"
WINDOW = 5
WORKTREE = "WORKTREE"

NEW_COPY = "NEW COPY"
LIKELY_COPY = "LIKELY COPY"
SIBLING_COPY = "SIBLING COPY"

_LOCAL = "\x00"
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class ToolError(Exception):
    """A git or configuration failure; reported with exit code 2."""


# --------------------------------------------------------------------------- git


def _git(root: Path, *args: str, stdin: bytes | None = None) -> bytes:
    proc = subprocess.run(["git", *args], cwd=root, input=stdin, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip()
        raise ToolError(f"git {' '.join(args)} failed: {err}")
    return proc.stdout


def in_scope(path: str) -> bool:
    return (
        path.endswith(".py")
        and any(path.startswith(d + "/") for d in SCOPE_DIRS)
        and not _OUT_OF_SCOPE.search(path)
    )


def resolve(root: Path, rev: str) -> str:
    return _git(root, "rev-parse", "--verify", rev + "^{commit}").decode().strip()


def commit_sources(root: Path, commit: str) -> dict[str, bytes]:
    """All in-scope files at ``commit``, read in one ``git cat-file --batch`` call."""
    listing = _git(root, "ls-tree", "-r", "--name-only", commit, "--", *SCOPE_DIRS)
    paths = sorted(p for p in listing.decode().splitlines() if in_scope(p))
    if not paths:
        return {}
    request = "".join(f"{commit}:./{p}\n" for p in paths).encode()
    out = _git(root, "cat-file", "--batch", stdin=request)
    sources: dict[str, bytes] = {}
    pos = 0
    for path in paths:
        newline = out.index(b"\n", pos)
        header = out[pos:newline].split()
        if header[-1] == b"missing":
            raise ToolError(f"git cat-file could not read {commit}:{path}")
        size = int(header[2])
        sources[path] = out[newline + 1 : newline + 1 + size]
        pos = newline + 1 + size + 1
    return sources


def worktree_sources(root: Path) -> tuple[dict[str, bytes], set[str]]:
    """In-scope files on disk (tracked and untracked) plus the untracked subset."""
    tracked = _git(root, "ls-files", "--", *SCOPE_DIRS).decode().splitlines()
    untracked = set(
        _git(root, "ls-files", "--others", "--exclude-standard", "--", *SCOPE_DIRS)
        .decode()
        .splitlines()
    )
    sources = {
        p: (root / p).read_bytes()
        for p in sorted(set(tracked) | untracked)
        if in_scope(p) and (root / p).is_file()
    }
    return sources, {p for p in untracked if p in sources}


# -------------------------------------------------------------------------- diff


@dataclass
class FileDiff:
    """Line correspondence for one file between base and head (``git diff -U0``)."""

    base_path: str | None  # None: the file does not exist at base
    head_path: str | None  # None: the file was deleted
    hunks: list[tuple[int, int, int, int]] = field(default_factory=list)

    def head_kind(self, line: int) -> str:
        """``unchanged``, ``added`` (no base line behind it) or ``modified``."""
        if self.base_path is None:
            return "added"
        for old_start, old_count, new_start, new_count in self.hunks:
            if new_count and new_start <= line < new_start + new_count:
                return "added" if old_count == 0 else "modified"
        return "unchanged"

    def to_base(self, line: int) -> int | None:
        if self.base_path is None:
            return None
        delta = 0
        for old_start, old_count, new_start, new_count in self.hunks:
            if new_count and new_start <= line < new_start + new_count:
                if old_count == 0:
                    return None
                return old_start + min(line - new_start, old_count - 1)
            if new_start + max(new_count, 1) - 1 < line:
                delta += new_count - old_count
        return line - delta

    def to_head(self, line: int) -> int | None:
        if self.head_path is None:
            return None
        delta = 0
        for old_start, old_count, new_start, new_count in self.hunks:
            if old_count and old_start <= line < old_start + old_count:
                if new_count == 0:
                    return None
                return new_start + min(line - old_start, new_count - 1)
            if old_start + max(old_count, 1) - 1 < line:
                delta += new_count - old_count
        return line + delta

    def base_changed(self, start: int, end: int) -> bool:
        """True when the diff deletes, replaces or inserts inside base lines start..end."""
        if self.head_path is None:
            return True
        for old_start, old_count, _new_start, _new_count in self.hunks:
            if old_count and old_start <= end and start < old_start + old_count:
                return True
            if old_count == 0 and start <= old_start < end:
                return True
        return False


@dataclass
class Diffs:
    by_head: dict[str, FileDiff]
    by_base: dict[str, FileDiff]
    base_files: set[str]
    head_files: set[str]

    def for_head(self, path: str) -> FileDiff:
        if path in self.by_head:
            return self.by_head[path]
        return FileDiff(path if path in self.base_files else None, path)

    def for_base(self, path: str) -> FileDiff:
        if path in self.by_base:
            return self.by_base[path]
        return FileDiff(path, path if path in self.head_files else None)


def parse_diff(text: str) -> list[FileDiff]:
    diffs: list[FileDiff] = []
    current: FileDiff | None = None
    for line in text.splitlines():
        if line.startswith("diff --git "):
            a, b = line[len("diff --git ") :].split(" b/", 1)
            current = FileDiff(a[len("a/") :], b)
            diffs.append(current)
        elif current is None:
            continue
        elif line.startswith("rename from "):
            current.base_path = line[len("rename from ") :]
        elif line.startswith("rename to "):
            current.head_path = line[len("rename to ") :]
        elif line == "--- /dev/null":
            current.base_path = None
        elif line == "+++ /dev/null":
            current.head_path = None
        elif match := _HUNK.match(line):
            old_start, old_count, new_start, new_count = match.groups()
            current.hunks.append(
                (
                    int(old_start),
                    1 if old_count is None else int(old_count),
                    int(new_start),
                    1 if new_count is None else int(new_count),
                )
            )
    return diffs


def collect_diffs(
    root: Path,
    base: str,
    head: str | None,
    base_files: set[str],
    head_files: set[str],
    untracked: set[str],
) -> Diffs:
    # Pin every option that user git config could change, so hunks (and
    # therefore occurrence classes) are the same on every machine.
    args = [
        "diff",
        "-M",
        "-U0",
        "--diff-algorithm=myers",
        "--indent-heuristic",
        "--relative",
        "--no-color",
        "--no-ext-diff",
        base,
    ]
    if head is not None:
        args.append(head)
    text = _git(root, *args, "--", *SCOPE_DIRS).decode(errors="replace")
    by_head: dict[str, FileDiff] = {}
    by_base: dict[str, FileDiff] = {}
    for fd in parse_diff(text):
        if fd.head_path is not None:
            by_head[fd.head_path] = fd
        if fd.base_path is not None:
            by_base[fd.base_path] = fd
    for path in untracked:
        by_head[path] = FileDiff(None, path)
    return Diffs(by_head, by_base, base_files, head_files)


# ------------------------------------------------------------------ fingerprints


@dataclass(frozen=True)
class Window:
    path: str
    block: int  # block id, unique within the file
    first: int  # index of the window's first statement in its block
    start: int  # first source line
    end: int  # last source line
    exact: str
    renamed: str


def _scope_locals(scope: ast.AST) -> frozenset[str]:
    """Parameters and names bound in this scope.

    Nested functions, classes and lambdas are their own scopes and are skipped.
    Names declared ``global`` or ``nonlocal`` bind an outer scope, so they are
    not local even when assigned here.
    """
    names: set[str] = set()
    declared_outer: set[str] = set()
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = scope.args
        for arg in (
            *args.posonlyargs,
            *args.args,
            *args.kwonlyargs,
            args.vararg,
            args.kwarg,
        ):
            if arg is not None:
                names.add(arg.arg)
    stack: list[ast.AST] = list(scope.body)
    while stack:
        node = stack.pop()
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            declared_outer.update(node.names)
        stack.extend(ast.iter_child_nodes(node))
    return frozenset(names - declared_outer)


def _tokens(node: object, local_names: frozenset[str], out: list[str]) -> None:
    """Serialise an AST subtree; local variable names become ``_LOCAL + name``."""
    if isinstance(node, ast.Name) and node.id in local_names:
        out.append(_LOCAL + node.id)
    elif isinstance(node, ast.AST):
        out.append(type(node).__name__)
        for name, value in ast.iter_fields(node):
            if name in ("ctx", "type_comment"):
                continue
            out.append("." + name)
            _tokens(value, local_names, out)
        out.append(")")
    elif isinstance(node, list):
        out.append("[")
        for item in node:
            _tokens(item, local_names, out)
        out.append("]")
    else:
        out.append(f"{type(node).__name__}:{node!r}")


def _window_keys(statements: list[list[str]]) -> tuple[str, str]:
    """Exact and renamed keys; ONE rename map is shared by the whole window."""
    exact: list[str] = []
    renamed: list[str] = []
    mapping: dict[str, str] = {}
    for tokens in statements:
        for token in tokens:
            if token.startswith(_LOCAL):
                name = token[1:]
                mapping.setdefault(name, f"V{len(mapping) + 1}")
                exact.append("local:" + name)
                renamed.append(mapping[name])
            else:
                exact.append(token)
                renamed.append(token)
        exact.append("\n")
        renamed.append("\n")
    digest = lambda parts: hashlib.sha256(
        "\x1f".join(parts).encode()
    ).hexdigest()  # noqa: E731
    return digest(exact), digest(renamed)


def _dotted(node: ast.AST) -> str:
    """``"sys.path.insert"`` for a Name/Attribute chain, ``""`` otherwise."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _is_bootstrap(value: ast.AST) -> bool:
    """Script path set-up: an ``os.path.*`` call or anything built from ``__file__``."""
    if any(isinstance(n, ast.Name) and n.id == "__file__" for n in ast.walk(value)):
        return True
    return isinstance(value, ast.Call) and _dotted(value.func).startswith("os.path.")


def _is_setup(stmt: ast.stmt, in_class: bool = False) -> bool:
    """Recognised setup: constants, plain values, imports, field declarations,
    script path bootstrap, field-only classes, pass and docstrings."""
    if isinstance(stmt, (ast.Pass, ast.Import, ast.ImportFrom)):
        return True
    if isinstance(stmt, ast.AnnAssign) and (stmt.value is None or in_class):
        return True  # `x: int`, or any typed field directly in a class body
    if isinstance(stmt, ast.ClassDef):
        return all(_is_setup(s, in_class=True) for s in stmt.body)
    if isinstance(stmt, ast.Expr):
        if isinstance(stmt.value, ast.Call):
            return _dotted(stmt.value.func).startswith("sys.path.")
        return isinstance(stmt.value, ast.Constant) and isinstance(
            stmt.value.value, str
        )
    if isinstance(stmt, ast.Assign) or (isinstance(stmt, ast.AnnAssign) and stmt.value):
        value = stmt.value
        if _is_bootstrap(value):
            return True
        if isinstance(value, ast.Call):
            return isinstance(value.func, ast.Attribute) and value.func.attr.startswith(
                "const_"
            )
        if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
            value = value.operand
        while isinstance(value, ast.Attribute):
            value = value.value
        return isinstance(value, (ast.Constant, ast.Name))
    return False


def _is_harness(stmt: ast.stmt) -> bool:
    """A CLI/test-input helper function or an ``if __name__ == "__main__":`` block."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return stmt.name in _HARNESS_FUNCTIONS
    if isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Compare):
        test = stmt.test
        return (
            isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        )
    return False


def _child_blocks(stmt: ast.stmt):
    for name in ("body", "orelse", "finalbody"):
        block = getattr(stmt, name, None)
        if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
            yield block
    for handler in getattr(stmt, "handlers", ()):
        yield handler.body
    for case in getattr(stmt, "cases", ()):
        yield case.body


def file_windows(path: str, source: bytes, window: int = WINDOW) -> list[Window]:
    """Every non-trivial window of ``window`` consecutive statements in one file."""
    tree = ast.parse(source, filename=path)
    windows: list[Window] = []
    next_block = 0

    def visit_block(
        stmts: list[ast.stmt], local_names: frozenset[str], in_class: bool = False
    ) -> None:
        nonlocal next_block
        block = next_block
        next_block += 1
        tokens = []
        for stmt in stmts:
            out: list[str] = []
            _tokens(stmt, local_names, out)
            tokens.append(out)
        setup = [_is_setup(stmt, in_class) for stmt in stmts]
        # Harness code is skipped entirely: no window may contain it, so it can
        # neither match nor complete a match with its neighbours.
        harness = [_is_harness(stmt) for stmt in stmts]
        for i in range(len(stmts) - window + 1):
            if all(setup[i : i + window]) or any(harness[i : i + window]):
                continue
            exact, renamed = _window_keys(tokens[i : i + window])
            last = stmts[i + window - 1]
            windows.append(
                Window(
                    path,
                    block,
                    i,
                    stmts[i].lineno,
                    last.end_lineno or last.lineno,
                    exact,
                    renamed,
                )
            )
        for stmt, skip in zip(stmts, harness):
            if skip:
                continue
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit_block(stmt.body, _scope_locals(stmt))
            elif isinstance(stmt, ast.ClassDef):
                visit_block(stmt.body, _scope_locals(stmt), in_class=True)
            else:
                for child in _child_blocks(stmt):
                    visit_block(child, local_names)

    visit_block(tree.body, _scope_locals(tree))
    return windows


def occurrence_table(
    windows: list[Window], attr: str, window: int = WINDOW
) -> dict[str, list[Window]]:
    """key -> non-overlapping occurrences (overlapping windows in one block count once)."""
    grouped: dict[tuple[str, str, int], list[Window]] = {}
    for w in windows:
        grouped.setdefault((getattr(w, attr), w.path, w.block), []).append(w)
    table: dict[str, list[Window]] = {}
    for (key, _path, _block), ws in grouped.items():
        last = -1
        for w in sorted(ws, key=lambda w: w.first):
            if w.first > last:
                table.setdefault(key, []).append(w)
                last = w.first + window - 1
    for occurrences in table.values():
        occurrences.sort(key=lambda w: (w.path, w.start))
    return table


@dataclass
class Side:
    windows: list[Window]
    renamed: dict[str, list[Window]]
    exact: dict[str, list[Window]]
    by_path: dict[str, list[Window]]


def fingerprint(
    sources: dict[str, bytes], side: str, skipped: list[dict], window: int = WINDOW
) -> Side:
    windows: list[Window] = []
    for path, source in sources.items():
        try:
            windows.extend(file_windows(path, source, window))
        except (SyntaxError, ValueError) as exc:
            skipped.append(
                {"side": side, "path": path, "error": f"{type(exc).__name__}: {exc}"}
            )
    renamed = occurrence_table(windows, "renamed", window)
    by_path: dict[str, list[Window]] = {}
    for occurrences in renamed.values():
        for w in occurrences:
            by_path.setdefault(w.path, []).append(w)
    return Side(windows, renamed, occurrence_table(windows, "exact", window), by_path)


# ------------------------------------------------------------------------ groups


@dataclass(frozen=True)
class Group:
    name: str
    files: frozenset[str]
    reason: str


def parse_groups(text: str) -> list[Group]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ToolError(f"{GROUPS_FILE}: {exc}") from exc
    groups = []
    for i, entry in enumerate(data.get("group", []), start=1):
        name = entry.get("name")
        files = entry.get("files")
        reason = entry.get("reason")
        if not isinstance(name, str) or not name.strip():
            raise ToolError(f"{GROUPS_FILE}: group #{i} has no 'name'")
        if (
            not isinstance(files, list)
            or len(files) < 2
            or not all(isinstance(f, str) for f in files)
        ):
            raise ToolError(
                f"{GROUPS_FILE}: group {name!r} must list at least 2 'files'"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ToolError(f"{GROUPS_FILE}: group {name!r} has no 'reason'")
        groups.append(Group(name, frozenset(files), reason))
    return groups


def load_groups(root: Path, head: str | None) -> list[Group]:
    if head is None:
        path = root / GROUPS_FILE
        return parse_groups(path.read_text()) if path.is_file() else []
    proc = subprocess.run(
        ["git", "show", f"{head}:./{GROUPS_FILE}"], cwd=root, capture_output=True
    )
    return parse_groups(proc.stdout.decode()) if proc.returncode == 0 else []


# ---------------------------------------------------------------------- analysis


@dataclass
class Location:
    path: str
    start: int
    end: int
    cls: str  # unchanged | maintained | new | replaced


@dataclass
class Finding:
    kind: str
    match: str
    group: str | None
    before: int
    after: int
    locations: list[Location]


@dataclass
class SiblingNote:
    group: str
    changed: tuple[str | None, int | None, int | None]
    sibling: tuple[str | None, int | None, int | None]


@dataclass
class Report:
    base: str
    head: str
    findings: list[Finding]
    notes: list[SiblingNote]
    skipped: list[dict]


def _base_span(w: Window, fd: FileDiff) -> tuple[int, int] | None:
    lines = [
        fd.to_base(n) for n in range(w.start, w.end + 1) if fd.head_kind(n) != "added"
    ]
    lines = [n for n in lines if n is not None]
    return (min(lines), max(lines)) if lines else None


def classify(w: Window, peers: list[Window], diffs: Diffs, base: Side) -> str:
    fd = diffs.for_head(w.path)
    kinds = {fd.head_kind(n) for n in range(w.start, w.end + 1)}
    if kinds == {"unchanged"}:
        return "unchanged"
    if kinds == {"added"}:
        return "new"
    span = _base_span(w, fd)
    if span is not None and fd.base_path is not None:
        lo, hi = span
        for old in base.by_path.get(fd.base_path, []):
            if not (old.start <= hi and lo <= old.end):
                continue
            partners = [o for o in base.renamed.get(old.renamed, []) if o is not old]
            for peer in peers:
                if peer is w:
                    continue
                pfd = diffs.for_head(peer.path)
                pspan = _base_span(peer, pfd)
                if pspan is None or pfd.base_path is None:
                    continue
                if any(
                    p.path == pfd.base_path
                    and p.start <= pspan[1]
                    and pspan[0] <= p.end
                    for p in partners
                ):
                    return "maintained"
    return "replaced"


def _merge_findings(findings: list[Finding]) -> list[Finding]:
    """Join findings from overlapping windows of one copied region.

    Two findings merge only when their added ranges overlap AND every location
    of the newer one overlaps a location of the merged one: overlapping windows
    copied from different sources stay separate findings.
    """

    def contributing(f: Finding) -> list[Location]:
        return [loc for loc in f.locations if loc.cls in ("new", "replaced")]

    def touches(x: Location, locations: list[Location]) -> bool:
        return any(
            x.path == y.path and x.start <= y.end and y.start <= x.end
            for y in locations
        )

    def overlaps(a: Finding, b: Finding) -> bool:
        return any(touches(x, contributing(a)) for x in contributing(b)) and all(
            touches(x, a.locations) for x in b.locations
        )

    rank = {"new": 3, "replaced": 2, "maintained": 1, "unchanged": 0}
    order = sorted(
        findings,
        key=lambda f: (contributing(f)[0].path, contributing(f)[0].start, f.kind),
    )
    merged: list[Finding] = []
    for f in order:
        target = next(
            (
                m
                for m in merged
                if m.kind == f.kind and m.group == f.group and overlaps(m, f)
            ),
            None,
        )
        if target is None:
            merged.append(
                Finding(f.kind, f.match, f.group, f.before, f.after, list(f.locations))
            )
            continue
        if f.match == "renamed":
            target.match = "renamed"
        if f.after > target.after:
            target.before, target.after = f.before, f.after
        locations = sorted(
            target.locations + f.locations, key=lambda loc: (loc.path, loc.start)
        )
        joined: list[Location] = []
        for loc in locations:
            prev = joined[-1] if joined else None
            if prev and prev.path == loc.path and loc.start <= prev.end:
                prev.end = max(prev.end, loc.end)
                if rank[loc.cls] > rank[prev.cls]:
                    prev.cls = loc.cls
            else:
                joined.append(Location(loc.path, loc.start, loc.end, loc.cls))
        target.locations = joined
    merged.sort(key=lambda f: (contributing(f)[0].path, contributing(f)[0].start))
    return merged


def find_copies(
    base: Side, head: Side, diffs: Diffs, groups: list[Group]
) -> list[Finding]:
    findings = []
    for key, occurrences in head.renamed.items():
        before = len(base.renamed.get(key, []))
        if len(occurrences) < 2 or len(occurrences) <= before:
            continue
        locations = [
            Location(w.path, w.start, w.end, classify(w, occurrences, diffs, base))
            for w in occurrences
        ]
        added = [
            (w, loc)
            for w, loc in zip(occurrences, locations)
            if loc.cls in ("new", "replaced")
        ]
        if not added:
            continue
        exact = any(
            len(head.exact.get(w.exact, [])) >= 2
            and len(head.exact.get(w.exact, [])) > len(base.exact.get(w.exact, []))
            for w, _loc in added
        )
        paths = [w.path for w in occurrences]
        # Approved forks cover copies BETWEEN sibling files: one occurrence per file.
        group = next(
            (
                g
                for g in groups
                if set(paths) <= g.files and len(set(paths)) == len(paths)
            ),
            None,
        )
        if group is not None:
            kind = SIBLING_COPY
        elif any(loc.cls == "new" for _w, loc in added):
            kind = NEW_COPY
        else:
            kind = LIKELY_COPY
        findings.append(
            Finding(
                kind,
                "exact" if exact else "renamed",
                group.name if group else None,
                before,
                len(occurrences),
                locations,
            )
        )
    return _merge_findings(findings)


def find_sibling_notes(
    base: Side, diffs: Diffs, groups: list[Group]
) -> list[SiblingNote]:
    notes: dict[tuple, SiblingNote] = {}
    for group in groups:
        for occurrences in base.renamed.values():
            members = [w for w in occurrences if w.path in group.files]
            if len({w.path for w in members}) < 2:
                continue
            for changed in members:
                cfd = diffs.for_base(changed.path)
                if not cfd.base_changed(changed.start, changed.end):
                    continue
                for sibling in members:
                    if sibling.path == changed.path:
                        continue
                    sfd = diffs.for_base(sibling.path)
                    if sfd.base_changed(sibling.start, sibling.end):
                        continue
                    note = SiblingNote(
                        group.name,
                        (
                            cfd.head_path,
                            cfd.to_head(changed.start),
                            cfd.to_head(changed.end),
                        ),
                        (
                            sfd.head_path,
                            sfd.to_head(sibling.start),
                            sfd.to_head(sibling.end),
                        ),
                    )
                    _add_note(notes, note)
    return sorted(notes.values(), key=lambda n: (str(n.changed), str(n.sibling)))


def _add_note(notes: dict[tuple, SiblingNote], note: SiblingNote) -> None:
    """Merge notes whose changed and sibling ranges both overlap an existing note."""

    def touch(a: tuple, b: tuple) -> bool:
        return (
            a[0] == b[0]
            and None not in (a[1], a[2], b[1], b[2])
            and a[1] <= b[2] + 1
            and b[1] <= a[2] + 1
        )

    for key, old in notes.items():
        if (
            old.group == note.group
            and touch(old.changed, note.changed)
            and touch(old.sibling, note.sibling)
        ):
            old.changed = (
                old.changed[0],
                min(old.changed[1], note.changed[1]),
                max(old.changed[2], note.changed[2]),
            )
            old.sibling = (
                old.sibling[0],
                min(old.sibling[1], note.sibling[1]),
                max(old.sibling[2], note.sibling[2]),
            )
            return
    notes[(note.group, note.changed, note.sibling)] = note


def analyze(
    root: Path, base_rev: str | None, head_rev: str | None, window: int = WINDOW
) -> Report:
    head = resolve(root, head_rev) if head_rev else None
    if base_rev is None:
        base_rev = (
            _git(root, "merge-base", head or "HEAD", DEFAULT_UPSTREAM).decode().strip()
        )
    base = resolve(root, base_rev)
    skipped: list[dict] = []
    base_sources = commit_sources(root, base)
    if head is None:
        head_sources, untracked = worktree_sources(root)
    else:
        head_sources, untracked = commit_sources(root, head), set()
    base_side = fingerprint(base_sources, "base", skipped, window)
    head_side = fingerprint(head_sources, "head", skipped, window)
    diffs = collect_diffs(
        root, base, head, set(base_sources), set(head_sources), untracked
    )
    groups = load_groups(root, head)
    return Report(
        base,
        head or WORKTREE,
        find_copies(base_side, head_side, diffs, groups),
        find_sibling_notes(base_side, diffs, groups),
        sorted(skipped, key=lambda s: (s["side"], s["path"])),
    )


# ------------------------------------------------------------------------ output


_FIX = {
    NEW_COPY: "Fix: extract one helper and call it from every copy, or explain in the PR why it must stay a copy.",
    LIKELY_COPY: "Fix: call the existing code instead of repeating it, or explain in the PR why it must stay a copy.",
    SIBLING_COPY: "Note: copy between approved fork files; no action needed if intended.",
}


def _span(path: str | None, start: int | None, end: int | None) -> str:
    if path is None or start is None or end is None:
        return f"{path or '?'} (deleted)"
    return f"{path}:{start}-{end}"


def render_text(report: Report) -> str:
    head = "<working tree>" if report.head == WORKTREE else report.head[:11]
    lines = [
        f"rocKE clone check  base {report.base[:11]}  head {head}",
        "Python only; C++ duplication not measured.",
        "",
    ]
    for f in report.findings:
        added = [loc for loc in f.locations if loc.cls in ("new", "replaced")]
        others = [loc for loc in f.locations if loc.cls not in ("new", "replaced")]
        label = (
            f"{f.kind} ({'replaced code, ' if f.kind == LIKELY_COPY else ''}{f.match})"
        )
        lines.append(
            f"{label}  "
            + ", ".join(f"{_span(l.path, l.start, l.end)} [{l.cls}]" for l in added)
        )
        if others:
            lines.append(
                "  same code at "
                + ", ".join(_span(l.path, l.start, l.end) for l in others)
            )
        lines.append(
            f"  copies {f.before} → {f.after}"
            + (f"   (group: {f.group})" if f.group else "")
        )
        lines.append("  " + _FIX[f.kind])
        lines.append("")
    for n in report.notes:
        lines.append(f"SIBLING NOTE  {_span(*n.changed)} changed")
        lines.append(
            f"  same block in {_span(*n.sibling)} was not changed   (group: {n.group})"
        )
        lines.append("")
    for s in report.skipped:
        lines.append(f"SKIPPED ({s['side']})  {s['path']}: {s['error']}")
    count = lambda kind: sum(f.kind == kind for f in report.findings)  # noqa: E731
    lines.append(
        f"Summary: new copies {count(NEW_COPY)} · likely copies {count(LIKELY_COPY)} · "
        f"sibling copies {count(SIBLING_COPY)} · sibling notes {len(report.notes)} · "
        f"skipped files {len(report.skipped)}"
    )
    return "\n".join(lines)


def render_json(report: Report) -> str:
    def span(t: tuple) -> dict:
        return {"path": t[0], "start": t[1], "end": t[2]}

    data = {
        "base": report.base,
        "head": report.head,
        "scope": "python",
        "findings": [
            {
                "kind": f.kind,
                "match": f.match,
                "group": f.group,
                "copies_before": f.before,
                "copies_after": f.after,
                "locations": [
                    {"path": l.path, "start": l.start, "end": l.end, "class": l.cls}
                    for l in f.locations
                ],
            }
            for f in report.findings
        ],
        "sibling_notes": [
            {"group": n.group, "changed": span(n.changed), "sibling": span(n.sibling)}
            for n in report.notes
        ],
        "skipped": report.skipped,
    }
    return json.dumps(data, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="report Python code copies a change adds (advisory)"
    )
    ap.add_argument(
        "--base", help=f"base commit (default: merge-base with {DEFAULT_UPSTREAM})"
    )
    ap.add_argument("--head", help="head commit (default: the working tree)")
    ap.add_argument("--format", choices=("text", "json"), default="text")
    ap.add_argument(
        "--fail-on-new",
        action="store_true",
        help="exit 1 when a NEW or LIKELY copy is found",
    )
    ap.add_argument(
        "--root", default=str(ROCKE), help="rocKE root (default: this tool's parent)"
    )
    args = ap.parse_args(argv)
    try:
        report = analyze(Path(args.root), args.base, args.head)
    except ToolError as exc:
        print(f"clone_check: error: {exc}", file=sys.stderr)
        return 2
    print(render_json(report) if args.format == "json" else render_text(report))
    blocking = any(f.kind in (NEW_COPY, LIKELY_COPY) for f in report.findings)
    return 1 if args.fail_on_new and blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
