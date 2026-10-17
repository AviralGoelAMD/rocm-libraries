#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# clone_check.py -- report Python code copies that a change ADDS (advisory).
#
# Compares a base commit with either the working tree (default) or a head
# commit. Every run tokenises all in-scope Python files on both sides,
# fingerprints every window of MIN_TOKENS consecutive code tokens (across
# statement boundaries, local variable names normalised), and reports a
# fingerprint only when the change adds a genuine new occurrence of code that
# now exists in two or more places. Python only: C++ duplication is not measured.
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
import bisect
import hashlib
import inspect
import io
import json
import re
import subprocess
import sys
import tokenize
import tomllib
from array import array
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
# Smallest copy reported, in code tokens (names, operators, literals; comments
# and layout excluded). Measured on rocKE history: 30 doubles the noise, 100
# finds little more than whole-statement windows.
MIN_TOKENS = 50
WORKTREE = "WORKTREE"

NEW_COPY = "NEW COPY"
LIKELY_COPY = "LIKELY COPY"
SIBLING_COPY = "SIBLING COPY"
EXISTING_EDITED = "EXISTING COPIES EDITED"
SIBLING_NOTE = "SIBLING NOTE"
MAX_EXISTING_SHOWN = 10  # text report; JSON always lists every entry

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
    base_text: dict[str, bytes]  # in-scope base files, for comparing old code
    head_files: set[str]

    def for_head(self, path: str) -> FileDiff:
        if path in self.by_head:
            return self.by_head[path]
        return FileDiff(path if path in self.base_text else None, path)

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
    base_sources: dict[str, bytes],
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
    return Diffs(by_head, by_base, base_sources, head_files)


# ------------------------------------------------------------------ fingerprints


@dataclass(frozen=True)
class Window:
    path: str
    first: int  # index of the window's first token in its file
    start: int  # first source line
    end: int  # last source line
    exact: int
    renamed: int


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


_SKIPPED_TOKENS = frozenset(
    {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
    }
)
# Polynomial rolling hash modulo a Mersenne prime: window keys are 61-bit.
_MOD = (1 << 61) - 1
_BASE = 1_000_003
_values: dict[str, int] = {}


def _value(text: str) -> int:
    """A fixed pseudo-random value per token text (stable across runs)."""
    value = _values.get(text)
    if value is None:
        digest = hashlib.blake2b(text.encode(), digest_size=8).digest()
        value = _values[text] = int.from_bytes(digest, "little") % _MOD
    return value


# A local name's renamed value is this plus the distance (in tokens) to the
# previous use of the same name inside the window, or plus 0 for its first use.
# One rename map is therefore shared by the whole window: two windows match
# only when their variables flow the same way.
_LOCAL_VALUE = _value("\x00local")


Pos = tuple[int, int]  # (line, column) of a token or node start


@dataclass
class _Marks:
    """What the AST says about individual tokens of one file."""

    local: set[Pos] = field(default_factory=set)  # local variable names
    docstrings: dict[Pos, str] = field(default_factory=dict)  # -> cleandoc text
    # (start, end, is_setup) for every statement, outer before inner
    statements: list[tuple[Pos, Pos, bool]] = field(default_factory=list)
    harness: list[tuple[Pos, Pos]] = field(default_factory=list)


def _statement_start(stmt: ast.stmt) -> Pos:
    decorators = getattr(stmt, "decorator_list", None) or []
    return min(
        [(stmt.lineno, stmt.col_offset)]
        + [(d.lineno, d.col_offset - 1) for d in decorators]  # include the "@"
    )


def _mark(tree: ast.Module) -> _Marks:
    marks = _Marks()

    def docstring(body: list[ast.stmt]) -> None:
        if body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                # Re-indenting copied code re-indents its docstrings, so they
                # are compared without their common indentation. Every other
                # string keeps its exact text.
                text = "docstring:" + repr(inspect.cleandoc(value.value))
                marks.docstrings[(value.lineno, value.col_offset)] = text

    def locals_of(body: list[ast.stmt], local_names: frozenset[str]) -> None:
        stack: list[ast.AST] = list(body)
        while stack:
            node = stack.pop()
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
            ):
                continue  # its own scope
            if isinstance(node, ast.Name) and node.id in local_names:
                marks.local.add((node.lineno, node.col_offset))
            stack.extend(ast.iter_child_nodes(node))

    def visit(stmts: list[ast.stmt], in_class: bool = False) -> None:
        for stmt in stmts:
            span = (_statement_start(stmt), (stmt.end_lineno, stmt.end_col_offset))
            # Harness code is skipped entirely: no window may contain it, so it
            # can neither match nor complete a match with its neighbours.
            if _is_harness(stmt):
                marks.harness.append(span)
                continue
            # A def/class header (decorators, signature) may be part of a copy
            # but cannot make one: its body statements override this mark.
            header = isinstance(
                stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
            marks.statements.append((*span, header or _is_setup(stmt, in_class)))
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = stmt.args
                for arg in (
                    *args.posonlyargs,
                    *args.args,
                    *args.kwonlyargs,
                    args.vararg,
                    args.kwarg,
                ):
                    if arg is not None:
                        marks.local.add((arg.lineno, arg.col_offset))
                docstring(stmt.body)
                locals_of(stmt.body, _scope_locals(stmt))
                visit(stmt.body)
            elif isinstance(stmt, ast.ClassDef):
                docstring(stmt.body)
                locals_of(stmt.body, _scope_locals(stmt))
                visit(stmt.body, in_class=True)
            else:
                for child in _child_blocks(stmt):
                    visit(child)

    docstring(tree.body)
    locals_of(tree.body, _scope_locals(tree))
    visit(tree.body)
    return marks


@dataclass
class FileKeys:
    """Every non-trivial token window of one file, in compact arrays."""

    first: array = field(default_factory=lambda: array("l"))
    start: array = field(default_factory=lambda: array("l"))
    end: array = field(default_factory=lambda: array("l"))
    exact: array = field(default_factory=lambda: array("q"))
    renamed: array = field(default_factory=lambda: array("q"))


def file_windows(path: str, source: bytes, size: int = MIN_TOKENS) -> FileKeys:
    """Every window of ``size`` consecutive code tokens that is neither all
    setup nor touches harness code."""
    marks = _mark(ast.parse(source, filename=path))
    tokens = [
        t
        for t in tokenize.tokenize(io.BytesIO(source).readline)
        if t.type not in _SKIPPED_TOKENS
    ]
    keys = FileKeys()
    count = len(tokens)
    if count < size:
        return keys
    starts = [t.start for t in tokens]

    def token_range(span_start: Pos, span_end: Pos) -> range:
        return range(
            bisect.bisect_left(starts, span_start), bisect.bisect_left(starts, span_end)
        )

    # Tokens outside every statement (none in valid code) count as setup.
    setup = [True] * count
    for span_start, span_end, is_setup in marks.statements:  # inner ones override
        for i in token_range(span_start, span_end):
            setup[i] = is_setup
    harness = [False] * count
    for span_start, span_end in marks.harness:
        for i in token_range(span_start, span_end):
            harness[i] = True

    text = [_value(marks.docstrings.get(t.start, t.string)) for t in tokens]
    local = [t.type == tokenize.NAME and t.start in marks.local for t in tokens]
    prev = [-1] * count  # previous use of the same local name
    nxt = [-1] * count  # next use of the same local name
    last_use: dict[str, int] = {}
    for i, t in enumerate(tokens):
        if local[i]:
            p = last_use.get(t.string, -1)
            prev[i] = p
            if p >= 0:
                nxt[p] = i
            last_use[t.string] = i

    def renamed_value(i: int, window_start: int) -> int:
        if not local[i]:
            return text[i]
        return _LOCAL_VALUE + (i - prev[i] if prev[i] >= window_start else 0)

    power = [1] * size
    for i in range(1, size):
        power[i] = power[i - 1] * _BASE % _MOD
    exact = renamed = 0
    for i in range(size):
        exact = (exact * _BASE + text[i]) % _MOD
        renamed = (renamed * _BASE + renamed_value(i, 0)) % _MOD
    setup_sum = [0]
    harness_sum = [0]
    for i in range(count):
        setup_sum.append(setup_sum[-1] + setup[i])
        harness_sum.append(harness_sum[-1] + harness[i])

    s = 0
    while True:
        e = s + size - 1
        if (
            harness_sum[e + 1] == harness_sum[s]
            and setup_sum[e + 1] - setup_sum[s] < size
        ):
            keys.first.append(s)
            keys.start.append(tokens[s].start[0])
            keys.end.append(tokens[e].end[0])
            keys.exact.append(exact)
            keys.renamed.append(renamed)
        if e + 1 == count:
            return keys
        # Slide by one token. When the outgoing token is a local name, its next
        # use inside the window becomes a first use: its value drops to 0.
        exact = ((exact - text[s] * power[-1]) * _BASE + text[e + 1]) % _MOD
        renamed = (renamed - renamed_value(s, s) * power[-1]) % _MOD
        k = nxt[s]
        if k != -1 and k <= e:
            renamed = (renamed - (k - s) * power[e - k]) % _MOD
        renamed = (renamed * _BASE + renamed_value(e + 1, s + 1)) % _MOD
        s += 1


def occurrence_table(windows: list[Window], attr: str) -> dict[int, list[Window]]:
    """key -> occurrences, one per run: same-key windows in one file that
    overlap one another (a block, or a repetitive stretch such as an unrolled
    loop) count once."""
    grouped: dict[tuple[int, str], list[Window]] = {}
    for w in windows:
        grouped.setdefault((getattr(w, attr), w.path), []).append(w)
    table: dict[int, list[Window]] = {}
    for (key, _path), ws in grouped.items():
        last = -1
        for w in sorted(ws, key=lambda w: w.first):
            if w.first > last:
                table.setdefault(key, []).append(w)
            last = w.first + MIN_TOKENS - 1
    for occurrences in table.values():
        occurrences.sort(key=lambda w: (w.path, w.start))
    return table


@dataclass
class Side:
    renamed: dict[int, list[Window]]
    exact: dict[int, list[Window]]
    by_path: dict[str, list[Window]]


def fingerprint(
    sources: dict[str, bytes],
    side: str,
    skipped: list[dict],
    cache: dict[tuple[str, bytes], FileKeys | str],
) -> dict[str, FileKeys]:
    """Token windows per file; files identical on both sides are keyed once."""
    keys: dict[str, FileKeys] = {}
    for path, source in sources.items():
        entry = cache.get((path, source))
        if entry is None:
            try:
                entry = file_windows(path, source)
            except (SyntaxError, ValueError, tokenize.TokenError) as exc:
                entry = f"{type(exc).__name__}: {exc}"
            cache[(path, source)] = entry
        if isinstance(entry, str):
            skipped.append({"side": side, "path": path, "error": entry})
        else:
            keys[path] = entry
    return keys


def build_sides(
    base: dict[str, FileKeys], head: dict[str, FileKeys]
) -> tuple[Side, Side]:
    """Window objects only for keys that repeat on either side: a key seen once
    on both sides can never be a copy. Its count on the other side is kept."""

    def repeated(files: dict[str, FileKeys]) -> set[int]:
        ordered = sorted(k for f in files.values() for k in f.renamed)
        return {a for a, b in zip(ordered, ordered[1:]) if a == b}

    relevant = repeated(base) | repeated(head)

    def side(files: dict[str, FileKeys]) -> Side:
        windows = [
            Window(path, *row)
            for path, f in files.items()
            for row in zip(f.first, f.start, f.end, f.exact, f.renamed)
            if row[4] in relevant
        ]
        renamed = occurrence_table(windows, "renamed")
        by_path: dict[str, list[Window]] = {}
        for occurrences in renamed.values():
            for w in occurrences:
                by_path.setdefault(w.path, []).append(w)
        return Side(renamed, occurrence_table(windows, "exact"), by_path)

    return side(base), side(head)


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
class ExistingCopies:
    """Copies that already existed and that this change edited."""

    kind: str  # EXISTING_EDITED, or SIBLING_NOTE for an approved fork group
    group: str | None
    locations: list[Location]  # cls: edited | unchanged


@dataclass
class Report:
    base: str
    head: str
    findings: list[Finding]
    existing: list[ExistingCopies]
    skipped: list[dict]


def _base_span(w: Window, fd: FileDiff) -> tuple[int, int] | None:
    lines = [
        fd.to_base(n) for n in range(w.start, w.end + 1) if fd.head_kind(n) != "added"
    ]
    lines = [n for n in lines if n is not None]
    return (min(lines), max(lines)) if lines else None


def _code_lines(source: bytes, start: int, end: int) -> tuple[str, ...]:
    """Non-blank, non-comment source lines ``start..end`` (1-based), with the
    common leading indent removed and relative indentation kept, so that
    different control flow never compares equal."""
    text = source.decode(errors="replace").splitlines()[start - 1 : end]
    lines = [t.rstrip() for t in text if t.strip() and not t.lstrip().startswith("#")]
    indent = min((len(t) - len(t.lstrip()) for t in lines), default=0)
    return tuple(t[indent:] for t in lines)


def _old_code(w: Window, diffs: Diffs) -> tuple[str, ...] | None:
    """The base-side code behind a head window (at least two lines), else None."""
    fd = diffs.for_head(w.path)
    span = _base_span(w, fd)
    if span is None or fd.base_path not in diffs.base_text:
        return None
    lines = _code_lines(diffs.base_text[fd.base_path], *span)
    return lines if len(lines) >= 2 else None


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
    # Copies too short for a window in the base: identical old code behind two
    # occurrences means they already existed and were edited, not pasted.
    mine = _old_code(w, diffs)
    if mine is not None and any(
        peer is not w and _old_code(peer, diffs) == mine for peer in peers
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


def _fork_group(paths: list[str], groups: list[Group]) -> Group | None:
    """The approved fork group holding these locations, one per sibling file."""
    return next(
        (g for g in groups if set(paths) <= g.files and len(set(paths)) == len(paths)),
        None,
    )


def find_copies(
    base: Side, head: Side, diffs: Diffs, groups: list[Group]
) -> tuple[list[Finding], list[ExistingCopies]]:
    """Copies the change added, plus existing copies it edited that only the head
    side can see (copies too short for a window in the base)."""
    findings = []
    edited: list[ExistingCopies] = []
    for key, occurrences in head.renamed.items():
        before = len(base.renamed.get(key, []))
        if len(occurrences) < 2 or len(occurrences) <= before:
            continue
        locations = [
            Location(w.path, w.start, w.end, classify(w, occurrences, diffs, base))
            for w in occurrences
        ]
        existing = [
            Location(
                l.path,
                l.start,
                l.end,
                "edited" if l.cls == "maintained" else "unchanged",
            )
            for l in locations
            if l.cls in ("maintained", "unchanged")
        ]
        if len(existing) >= 2 and any(l.cls == "edited" for l in existing):
            edited.append(ExistingCopies(EXISTING_EDITED, None, existing))
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
        # Approved forks cover copies BETWEEN sibling files: one occurrence per file.
        group = _fork_group([w.path for w in occurrences], groups)
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
    return _merge_findings(findings), edited


def find_existing_copies(
    base: Side,
    diffs: Diffs,
    groups: list[Group],
    from_head: list[ExistingCopies],
    head_text: dict[str, bytes],
) -> list[ExistingCopies]:
    """Copy groups that already existed in the base and that the change edited.

    Each still-existing member is listed as ``edited`` or ``unchanged``; deleted
    members are dropped, so a moved copy is not an edit. Groups inside an
    approved fork become SIBLING NOTEs when a sibling was left unchanged or the
    edited siblings now differ; a fork whose sides end up identical is in sync.
    """
    raw = list(from_head)
    for occurrences in base.renamed.values():
        if len(occurrences) < 2:
            continue
        locations = []
        for w in occurrences:
            fd = diffs.for_base(w.path)
            if fd.head_path is None:
                continue
            mapped = [fd.to_head(n) for n in range(w.start, w.end + 1)]
            mapped = [n for n in mapped if n is not None]
            if not mapped:
                continue
            edited = fd.base_changed(w.start, w.end)
            locations.append(
                Location(
                    fd.head_path,
                    min(mapped),
                    max(mapped),
                    "edited" if edited else "unchanged",
                )
            )
        if len(locations) >= 2 and any(l.cls == "edited" for l in locations):
            raw.append(ExistingCopies(EXISTING_EDITED, None, locations))
    items = []
    for item in raw:
        group = _fork_group([l.path for l in item.locations], groups)
        if group is None:
            items.append(item)
            continue
        codes = {
            _code_lines(head_text[l.path], l.start, l.end)
            for l in item.locations
            if l.path in head_text
        }
        if any(l.cls == "unchanged" for l in item.locations) or len(codes) > 1:
            items.append(ExistingCopies(SIBLING_NOTE, group.name, item.locations))
    return _merge_existing(items)


def _merge_existing(items: list[ExistingCopies]) -> list[ExistingCopies]:
    """Join entries from overlapping windows of one edited copy group."""

    def touches(x: Location, locations: list[Location]) -> bool:
        return any(
            x.path == y.path and x.start <= y.end + 1 and y.start <= x.end + 1
            for y in locations
        )

    def edited(e: ExistingCopies) -> list[Location]:
        return [l for l in e.locations if l.cls == "edited"]

    merged: list[ExistingCopies] = []
    for item in sorted(items, key=lambda e: (edited(e)[0].path, edited(e)[0].start)):
        target = next(
            (
                m
                for m in merged
                if m.kind == item.kind
                and m.group == item.group
                and any(touches(x, edited(m)) for x in edited(item))
                and all(touches(x, m.locations) for x in item.locations)
            ),
            None,
        )
        if target is None:
            merged.append(ExistingCopies(item.kind, item.group, list(item.locations)))
            continue
        joined: list[Location] = []
        for loc in sorted(
            target.locations + item.locations, key=lambda l: (l.path, l.start)
        ):
            prev = joined[-1] if joined else None
            if prev and prev.path == loc.path and loc.start <= prev.end + 1:
                prev.end = max(prev.end, loc.end)
                if loc.cls == "edited":
                    prev.cls = "edited"
            else:
                joined.append(Location(loc.path, loc.start, loc.end, loc.cls))
        target.locations = joined
    return merged


def analyze(root: Path, base_rev: str | None, head_rev: str | None) -> Report:
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
    cache: dict[tuple[str, bytes], FileKeys | str] = {}
    base_side, head_side = build_sides(
        fingerprint(base_sources, "base", skipped, cache),
        fingerprint(head_sources, "head", skipped, cache),
    )
    diffs = collect_diffs(root, base, head, base_sources, set(head_sources), untracked)
    groups = load_groups(root, head)
    findings, from_head = find_copies(base_side, head_side, diffs, groups)
    return Report(
        base,
        head or WORKTREE,
        findings,
        find_existing_copies(base_side, diffs, groups, from_head, head_sources),
        sorted(skipped, key=lambda s: (s["side"], s["path"])),
    )


# ------------------------------------------------------------------------ output


_FIX = {
    NEW_COPY: "Fix: extract one helper and call it from every copy, or explain in the PR why it must stay a copy.",
    LIKELY_COPY: "Fix: call the existing code instead of repeating it, or explain in the PR why it must stay a copy.",
    SIBLING_COPY: "Note: copy between approved fork files; no action needed if intended.",
}


def _span(path: str, start: int, end: int) -> str:
    return f"{path}:{start}-{end}"


def _spans(locations: list[Location]) -> str:
    return ", ".join(_span(l.path, l.start, l.end) for l in locations)


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
    shown_edits = 0
    for e in report.existing:
        edited = [l for l in e.locations if l.cls == "edited"]
        unchanged = [l for l in e.locations if l.cls == "unchanged"]
        if e.kind == SIBLING_NOTE:
            lines.append(f"SIBLING NOTE  {_spans(edited)} changed")
            if unchanged:
                lines.append(
                    f"  same block in {_spans(unchanged)} was not changed   (group: {e.group})"
                )
            else:
                lines.append(
                    f"  the sides were changed differently; check they should differ   (group: {e.group})"
                )
        elif shown_edits < MAX_EXISTING_SHOWN:
            shown_edits += 1
            lines.append(f"EXISTING COPIES EDITED  {_spans(edited)}")
            if unchanged:
                lines.append(f"  same code, not changed here, at {_spans(unchanged)}")
            lines.append(
                f"  This code already existed in {len(e.locations)} places before this change."
            )
            lines.append(
                "  Fix: extract one helper now, so the next change touches one place."
            )
        else:
            continue
        lines.append("")
    hidden = sum(e.kind == EXISTING_EDITED for e in report.existing) - shown_edits
    if hidden:
        lines.append(
            f"+{hidden} more existing copy groups edited; run with --format json for the full list"
        )
        lines.append("")
    for s in report.skipped:
        lines.append(f"SKIPPED ({s['side']})  {s['path']}: {s['error']}")
    count = lambda kind: sum(f.kind == kind for f in report.findings)  # noqa: E731
    edits = sum(e.kind == EXISTING_EDITED for e in report.existing)
    lines.append(
        f"Summary: new copies {count(NEW_COPY)} · likely copies {count(LIKELY_COPY)} · "
        f"sibling copies {count(SIBLING_COPY)} · existing copies edited {edits} · "
        f"sibling notes {len(report.existing) - edits} · "
        f"skipped files {len(report.skipped)}"
    )
    return "\n".join(lines)


def render_json(report: Report) -> str:
    def locations(items: list[Location]) -> list[dict]:
        return [
            {"path": l.path, "start": l.start, "end": l.end, "class": l.cls}
            for l in items
        ]

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
                "locations": locations(f.locations),
            }
            for f in report.findings
        ],
        "existing_copies": [
            {"kind": e.kind, "group": e.group, "locations": locations(e.locations)}
            for e in report.existing
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
