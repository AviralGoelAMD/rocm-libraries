# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Behaviour of tools/clone_check.py on small synthetic git repositories."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[3] / "tools" / "clone_check.py"

_GIT_ENV = dict(
    os.environ,
    GIT_AUTHOR_NAME="t",
    GIT_AUTHOR_EMAIL="t@example.com",
    GIT_COMMITTER_NAME="t",
    GIT_COMMITTER_EMAIL="t@example.com",
)

# One copied region: five non-trivial emitter statements, 66 tokens. {load}
# and {k} sit mid-block: a change there leaves fewer than 50 matching tokens on
# either side, even counting the shared end of the signature.
BLOCK = """\
    v1 = b.add(base, b.mul(row, stride))
    v2 = b.{load}(v1, b.const_f32({k}), ptr)
    v3 = b.select(ok, v2, zero)
    v4 = b.fmul(v3, v1)
    b.store(out, v3, v4)
"""


def func(name: str, body: str = BLOCK, load: str = "load", k: str = "32.0") -> str:
    head = f"def {name}(b, base, row, stride, ok, zero, ptr, out):\n"
    return head + body.format(load=load, k=k) + "\n\n"


UNRELATED = """\
    a1 = b.sub(base, row)
    a2 = b.cmp_lt(a1, stride)
    a3 = b.land(a2, ok)
    a4 = b.mod(a1, stride)
    b.barrier(a3, a4)
"""


class Repo:
    def __init__(self, path: Path):
        self.path = path
        self.git("init", "-q", "-b", "develop")

    def git(self, *args: str) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=self.path,
            env=_GIT_ENV,
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()

    def write(self, rel: str, text: str) -> None:
        target = self.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def commit(self) -> str:
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", "c")
        return self.git("rev-parse", "HEAD")

    def check(self, *args: str, root: Path | None = None) -> tuple[int, dict]:
        proc = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--root",
                str(root or self.path),
                "--format",
                "json",
                *args,
            ],
            capture_output=True,
            text=True,
        )
        return proc.returncode, (json.loads(proc.stdout) if proc.stdout else {})


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    r = Repo(tmp_path)
    r.write("library/kernels/keep.py", "X = 1\n")
    return r


def kinds(report: dict) -> list[str]:
    return [f["kind"] for f in report["findings"]]


def test_code_written_once_is_not_a_copy(repo):
    base = repo.commit()
    repo.write("library/kernels/a.py", func("f1"))
    head = repo.commit()
    rc, report = repo.check("--base", base, "--head", head)
    assert rc == 0 and report["findings"] == []


def test_new_code_written_twice_is_a_new_copy(repo):
    base = repo.commit()
    repo.write("library/kernels/a.py", func("f1") + func("f2"))
    head = repo.commit()
    rc, report = repo.check("--base", base, "--head", head)
    assert rc == 0
    assert kinds(report) == ["NEW COPY"]
    assert (
        report["findings"][0]["copies_before"],
        report["findings"][0]["copies_after"],
    ) == (0, 2)
    rc, _ = repo.check("--base", base, "--head", head, "--fail-on-new")
    assert rc == 1


def test_pasting_an_existing_block_is_an_exact_new_copy(repo):
    repo.write("library/kernels/a.py", func("f1") + func("f2"))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("f3"))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    (finding,) = report["findings"]
    assert (finding["kind"], finding["match"]) == ("NEW COPY", "exact")
    assert (finding["copies_before"], finding["copies_after"]) == (2, 3)
    new = [loc["path"] for loc in finding["locations"] if loc["class"] == "new"]
    assert new == ["library/kernels/b.py"]


def test_pasted_block_with_renamed_variables_is_a_renamed_copy(repo):
    repo.write("library/kernels/a.py", func("f1"))
    base = repo.commit()
    renamed = (
        BLOCK.replace("v1", "w1")
        .replace("v2", "w2")
        .replace("v3", "w3")
        .replace("v4", "w4")
    )
    repo.write("library/kernels/b.py", func("g", renamed))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert [(f["kind"], f["match"]) for f in report["findings"]] == [
        ("NEW COPY", "renamed")
    ]


@pytest.mark.parametrize("variant", [{"load": "store"}, {"k": "16.0"}])
def test_different_method_or_constant_is_not_a_copy(repo, variant):
    repo.write("library/kernels/a.py", func("f1"))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("g", **variant))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


def test_rename_map_spans_the_whole_window(repo):
    repo.write("library/kernels/a.py", func("f1"))
    base = repo.commit()
    # Same statements, but the second one reads v2 instead of v1: a different data flow.
    repo.write(
        "library/kernels/b.py",
        func("g", BLOCK.replace("{load}(v1,", "{load}(v2,")),
    )
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


# Two statements, well over the 50-token minimum.
LONG = """\
    acc = b.mfma(b.load(a_ptr, b.add(base, b.mul(row, stride))), b.load(q_ptr, b.add(base, b.mul(stride, row))), acc)
    b.store(out, b.add(base, b.mul(row, stride)), b.select(ok, acc, zero))
"""


def test_copy_of_two_long_statements_is_a_copy(repo):
    repo.write("library/kernels/a.py", func("f1", LONG))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("g", LONG))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


ARGS = (
    "    {callee}(kernel, grid=(n_blocks, 1, 1), block=(256, 1, 1), shared=lds_bytes,"
    " args=[a_ptr, b_ptr, c_ptr, n_elems, stride_a, stride_b, stride_c, alpha, beta, gamma])\n"
)


def test_argument_list_copied_into_a_different_call_is_a_copy(repo):
    repo.write("library/kernels/a.py", func("f1", ARGS.format(callee="launch")))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("g", ARGS.format(callee="run_kernel")))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


# Five statements, but under 50 tokens even with the shared end of the signature.
TINY = """\
    x1 = b.f(base)
    x2 = b.g(x1)
    x3 = b.h(x2)
    x4 = b.k(x3)
    b.st(x4)
"""


def test_copy_below_the_token_minimum_is_not_a_copy(repo):
    repo.write("library/kernels/a.py", "def f1(b, base):\n" + TINY)
    base = repo.commit()
    repo.write("library/kernels/b.py", "def g(b, base, extra):\n" + TINY)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


def test_moving_a_block_or_renaming_a_file_is_not_a_copy(repo):
    repo.write("library/kernels/a.py", func("f1"))
    repo.write("library/kernels/b.py", "Y = 2\n\n\n" + func("f2"))
    base = repo.commit()
    repo.write("library/kernels/b.py", "Y = 2\n")
    repo.write("library/kernels/c.py", func("f2"))
    repo.git("mv", "library/kernels/a.py", "library/kernels/d.py")
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []
    assert report["existing_copies"] == []  # a moved copy was not edited


def existing(report: dict) -> list[tuple]:
    return [
        (e["kind"], sorted((loc["path"], loc["class"]) for loc in e["locations"]))
        for e in report["existing_copies"]
    ]


def test_same_fix_in_both_existing_copies_is_reported_as_existing_copies(repo):
    repo.write("library/kernels/a.py", func("f1"))
    repo.write("library/kernels/b.py", func("f2"))
    base = repo.commit()
    fixed = BLOCK.replace("b.add(", "b.sub(")
    repo.write("library/kernels/a.py", func("f1", fixed))
    repo.write("library/kernels/b.py", func("f2", fixed))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []
    assert existing(report) == [
        (
            "EXISTING COPIES EDITED",
            [("library/kernels/a.py", "edited"), ("library/kernels/b.py", "edited")],
        )
    ]


def test_editing_one_of_three_existing_copies_lists_the_others(repo):
    for name in "abc":
        repo.write(f"library/kernels/{name}.py", func(f"f_{name}"))
    base = repo.commit()
    repo.write("library/kernels/a.py", func("f_a", BLOCK.replace("b.add(", "b.mul(")))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []
    assert existing(report) == [
        (
            "EXISTING COPIES EDITED",
            [
                ("library/kernels/a.py", "edited"),
                ("library/kernels/b.py", "unchanged"),
                ("library/kernels/c.py", "unchanged"),
            ],
        )
    ]


SHORT = """\
    v1 = b.add(base, row)
    v2 = b.select(ok, v1, zero)
    b.store(out, v2, v1)
"""


def test_short_existing_copies_edited_the_same_way_are_not_a_new_copy(repo):
    # Three shared statements are below the window, so the base shows no copy.
    # Adding the same two lines to both makes them visible; they already existed.
    repo.write("library/kernels/a.py", func("f1", SHORT))
    repo.write("library/kernels/b.py", func("f2", SHORT))
    base = repo.commit()
    grown = SHORT.replace(
        "    b.store(",
        "    v3 = b.load(ptr, v2)\n    v4 = b.fmul(v3, v1)\n    b.store(",
    )
    repo.write("library/kernels/a.py", func("f1", grown))
    repo.write("library/kernels/b.py", func("f2", grown))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []
    assert existing(report) == [
        (
            "EXISTING COPIES EDITED",
            [("library/kernels/a.py", "edited"), ("library/kernels/b.py", "edited")],
        )
    ]


def test_replacing_unrelated_code_with_a_pasted_block_is_a_likely_copy(repo):
    repo.write("library/kernels/a.py", func("f1"))
    repo.write("library/kernels/b.py", func("g", UNRELATED))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("g"))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    (finding,) = report["findings"]
    assert finding["kind"] == "LIKELY COPY"
    assert [
        loc["class"] for loc in finding["locations"] if loc["path"].endswith("b.py")
    ] == ["replaced"]


GROUPS = """\
[[group]]
name = "per-arch"
files = ["library/kernels/gfx942/k.py", "library/kernels/gfx950/k.py"]
reason = "arch-specific schedules"
"""


def test_editing_one_side_of_an_approved_fork_names_the_unchanged_sibling(repo):
    repo.write("tools/clone_groups.toml", GROUPS)
    repo.write("library/kernels/gfx942/k.py", func("f"))
    repo.write("library/kernels/gfx950/k.py", func("f"))
    base = repo.commit()
    repo.write(
        "library/kernels/gfx942/k.py", func("f", BLOCK.replace("b.add(", "b.mul("))
    )
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []
    assert existing(report) == [
        (
            "SIBLING NOTE",
            [
                ("library/kernels/gfx942/k.py", "edited"),
                ("library/kernels/gfx950/k.py", "unchanged"),
            ],
        )
    ]
    assert report["existing_copies"][0]["group"] == "per-arch"


def test_different_edits_on_both_sides_of_an_approved_fork_get_a_sibling_note(repo):
    repo.write("tools/clone_groups.toml", GROUPS)
    repo.write("library/kernels/gfx942/k.py", func("f"))
    repo.write("library/kernels/gfx950/k.py", func("f"))
    base = repo.commit()
    repo.write(
        "library/kernels/gfx942/k.py", func("f", BLOCK.replace("b.add(", "b.mul("))
    )
    repo.write(
        "library/kernels/gfx950/k.py", func("f", BLOCK.replace("b.add(", "b.sub("))
    )
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert existing(report) == [
        (
            "SIBLING NOTE",
            [
                ("library/kernels/gfx942/k.py", "edited"),
                ("library/kernels/gfx950/k.py", "edited"),
            ],
        )
    ]


def test_text_report_lists_ten_existing_copy_groups_then_counts_the_rest(repo):
    # Twelve separate copy pairs, one function per file, so no larger window
    # joins them into a single group.
    for i in range(12):
        repo.write(f"library/kernels/a{i}.py", func("f", k=f"{i}.0"))
        repo.write(f"library/kernels/b{i}.py", func("f", k=f"{i}.0"))
    base = repo.commit()
    for i in range(12):
        edited = func("f", BLOCK.replace("b.add(", "b.mul("), k=f"{i}.0")
        repo.write(f"library/kernels/a{i}.py", edited)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert len(report["existing_copies"]) == 12  # JSON keeps every entry
    text = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--root",
            str(repo.path),
            "--base",
            base,
            "--head",
            head,
        ],
        capture_output=True,
        text=True,
    ).stdout
    assert text.count("EXISTING COPIES EDITED  ") == 10
    assert "+2 more existing copy groups edited" in text


IF_BOTH = """\
    v1 = b.add(base, row)
    v2 = b.select(ok, v1, zero)
    if ok:
        v3 = b.load(ptr, v2)
        v4 = b.fmul(v3, b.const_f32(32.0))
    b.store(out, v2, v4)
"""
IF_ONE = IF_BOTH.replace(
    "        v4 = b.fmul(", "    v4 = b.fmul("
)  # same lines, different control flow


def test_forks_that_differ_only_in_indentation_get_a_sibling_note(repo):
    repo.write("tools/clone_groups.toml", GROUPS)
    repo.write("library/kernels/gfx942/k.py", func("f"))
    repo.write("library/kernels/gfx950/k.py", func("f"))
    base = repo.commit()
    repo.write("library/kernels/gfx942/k.py", func("f", IF_BOTH))
    repo.write("library/kernels/gfx950/k.py", func("f", IF_ONE))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert [e["kind"] for e in report["existing_copies"]] == ["SIBLING NOTE"]


def test_same_edit_on_both_sides_of_an_approved_fork_reports_nothing(repo):
    repo.write("tools/clone_groups.toml", GROUPS)
    repo.write("library/kernels/gfx942/k.py", func("f"))
    repo.write("library/kernels/gfx950/k.py", func("f"))
    base = repo.commit()
    fixed = func("f", BLOCK.replace("b.add(", "b.sub("))
    repo.write("library/kernels/gfx942/k.py", fixed)
    repo.write("library/kernels/gfx950/k.py", fixed)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []
    assert report["existing_copies"] == []  # the fork was kept in sync


def test_copy_between_approved_fork_files_is_a_sibling_copy(repo):
    repo.write("tools/clone_groups.toml", GROUPS)
    repo.write("library/kernels/gfx942/k.py", func("f"))
    repo.write("library/kernels/gfx950/k.py", "Z = 0\n")
    base = repo.commit()
    repo.write("library/kernels/gfx950/k.py", func("f"))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert [(f["kind"], f["group"]) for f in report["findings"]] == [
        ("SIBLING COPY", "per-arch")
    ]


def test_copy_into_a_file_outside_an_approved_group_is_a_new_copy(repo):
    repo.write("tools/clone_groups.toml", GROUPS)
    repo.write("library/kernels/gfx942/k.py", func("f"))
    repo.write("library/kernels/gfx950/k.py", func("f"))
    base = repo.commit()
    repo.write("library/kernels/gfx1250/k.py", func("f"))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


def test_second_copy_inside_one_approved_file_is_a_new_copy(repo):
    repo.write("tools/clone_groups.toml", GROUPS)
    repo.write("library/kernels/gfx942/k.py", func("f"))
    repo.write("library/kernels/gfx950/k.py", func("f"))
    base = repo.commit()
    repo.write("library/kernels/gfx942/k.py", func("f") + func("g"))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


SIX = [
    "    t1 = b.add(x, y)\n",
    "    t2 = b.mul(t1, y)\n",
    "    t3 = b.sub(t2, x)\n",
    "    t4 = b.select(t3, t1, t2)\n",
    "    t5 = b.load(t4, x)\n",
    "    b.store(t5, t4, t3)\n",
]


PIECE_P = [
    "p1 = b.add(base, b.mul(row, stride))",
    "p2 = b.select(ok, p1, b.const_f32(1.0))",
    "p3 = b.fmul(p2, b.load(ptr, p1))",
    "b.store(out, p3, p2)",
]
PIECE_Q = [
    "q1 = b.sub(stride, b.mod(row, base))",
    "q2 = b.cmp_lt(q1, b.const_i32(64))",
    "q3 = b.land(q2, ok)",
    "b.barrier(q3, b.load(ptr, q1))",
]


def two_pieces(name: str, first: list[str], second: list[str]) -> str:
    """Two pieces whose junction shares one line, so their windows overlap in lines."""
    lines = first[:-1] + [first[-1] + "; " + second[0]] + second[1:]
    head = f"def {name}(b, base, row, stride, ok, ptr, out):\n"
    return head + "".join(f"    {line}\n" for line in lines)


def test_reordered_pieces_from_one_source_stay_separate_findings(repo):
    repo.write("library/kernels/a.py", two_pieces("fa", PIECE_P, PIECE_Q))
    base = repo.commit()
    # Q then P: the copy advances while its source jumps backwards, so the two
    # pieces must not merge into one region that claims a:2-6 was copied as b:2-6.
    repo.write("library/kernels/b.py", two_pieces("fb", PIECE_Q, PIECE_P))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    spans = sorted(
        tuple((loc["path"][-4:], loc["start"], loc["end"]) for loc in f["locations"])
        for f in report["findings"]
    )
    assert spans == [
        (("a.py", 2, 5), ("b.py", 5, 8)),  # P
        (("a.py", 5, 8), ("b.py", 2, 5)),  # Q
    ]


def test_overlapping_fragments_from_two_sources_stay_separate(repo):
    repo.write("library/kernels/a.py", "def fa(b, x, y):\n" + "".join(SIX[:5]))
    # t1 is a parameter here, so it is a local variable like in fa and fc.
    repo.write("library/kernels/b.py", "def fb(b, x, y, t1):\n" + "".join(SIX[1:]))
    base = repo.commit()
    repo.write("library/kernels/c.py", "def fc(b, x, y):\n" + "".join(SIX))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    sources = sorted(
        tuple(loc["path"] for loc in f["locations"] if loc["class"] == "unchanged")
        for f in report["findings"]
    )
    assert sources == [("library/kernels/a.py",), ("library/kernels/b.py",)]


NESTED = """\
    v1 = b.add(base, row)
    v2 = {helper}(b, v1)
    v3 = b.select(ok, v2, zero)
    v4 = b.load(ptr, v3)
    b.store(out, v3, v4)

    def inner():
        {helper} = 0
        return {helper}
"""


def test_nested_function_locals_do_not_rename_outer_helpers(repo):
    repo.write("library/kernels/a.py", func("f1", NESTED.replace("{helper}", "emit_a")))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("f2", NESTED.replace("{helper}", "emit_b")))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


GLOBAL = """\
    global {mod}
    {mod} = load_module()
    v1 = {mod}.add(base, row)
    v2 = {mod}.select(ok, v1, zero)
    v3 = {mod}.load(ptr, v2)
    b.store(out, v2, v3)
"""


def test_declared_global_names_are_not_renamed(repo):
    # Assigning a `global` name binds the module-level name, not a local one.
    repo.write("library/kernels/a.py", func("f1", GLOBAL.replace("{mod}", "torch")))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("f2", GLOBAL.replace("{mod}", "np")))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


def test_bare_field_declarations_are_skipped(repo):
    base = repo.commit()
    fields = "".join(f"    f{i}: int\n" for i in range(5))
    repo.write("library/kernels/a.py", f"class A:\n{fields}\n\nclass B:\n{fields}")
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


IMPORTS = """\
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
"""


def test_import_blocks_are_skipped(repo):
    base = repo.commit()
    repo.write("library/kernels/a.py", IMPORTS + "A = 1\n")
    repo.write("library/kernels/b.py", IMPORTS + "B = 2\n")
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


@pytest.mark.parametrize(
    "path",
    [
        "library/benchmarks/common/bench.py",
        "platform/python/rocke/examples/demo.py",
        "platform/python/rocke/benchmark/gemm/run.py",
        "library/builders/gfx950/attention/thing_verify.py",
        "library/builders/gfx950/attention/thing_sweep.py",
        "library/builders/gfx950/attention/bench_thing.py",
        "library/builders/gfx950/attention/benchmark_thing.py",
        "library/builders/gfx950/attention/reproduce_thing.py",
        "platform/python/rocke/portable_ir/examples/demo.py",
        "library/kernels/gfx942/experiments/d256/p1_prefetch.py",
    ],
)
def test_benchmark_verify_and_example_scripts_are_out_of_scope(repo, path):
    repo.write("library/kernels/a.py", func("f1"))
    base = repo.commit()
    repo.write(path, func("f2"))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


@pytest.mark.parametrize("name", ["main", "make_inputs", "parse_args", "_parser"])
def test_cli_and_input_helper_functions_are_skipped(repo, name):
    repo.write("library/builders/gfx950/x/a.py", func(name))
    base = repo.commit()
    repo.write("library/builders/gfx950/x/b.py", func(name))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


AROUND_MAIN = """\
A1 = b0.add(x, y)
A2 = b0.mul(A1, y)


{main}
A3 = b0.sub(A2, x)
A4 = b0.select(A3, A1, A2)
"""


def test_skipped_function_does_not_complete_a_module_window(repo):
    # Four shared top-level statements plus an excluded main() must not form a
    # five-statement match.
    text = AROUND_MAIN.format(main=func("main"))
    repo.write("library/builders/gfx950/x/a.py", text)
    base = repo.commit()
    repo.write("library/builders/gfx950/x/b.py", text)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


def test_dunder_main_block_is_skipped(repo):
    script = 'if __name__ == "__main__":\n' + BLOCK.format(load="load", k="32.0")
    repo.write("library/builders/gfx950/x/a.py", script)
    base = repo.commit()
    repo.write("library/builders/gfx950/x/b.py", script)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


@pytest.mark.parametrize("name", ["check", "bench"])
def test_check_and_bench_functions_are_still_checked(repo, name):
    repo.write("library/kernels/common/a.py", func(name))
    base = repo.commit()
    repo.write("library/kernels/common/b.py", func(name))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


CONFIG = """\
@dataclass
class Cfg:
    data: Spec = Spec(1)
    tile_m: int = field(default=64)
    tile_n: int = make_tile(64)
    warp_m: int = pick(2)
    warp_n: int = pick(4)
"""


def test_class_fields_with_defaults_are_skipped(repo):
    repo.write("library/kernels/a.py", CONFIG)
    base = repo.commit()
    repo.write("library/kernels/b.py", CONFIG)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


def test_field_only_class_does_not_complete_a_module_window(repo):
    module = IMPORTS + "\n\n" + CONFIG
    repo.write("library/kernels/a.py", module)
    base = repo.commit()
    repo.write("library/kernels/b.py", module)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


BOOTSTRAP = """\
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.append(_HERE)
ROOT = Path(__file__).resolve().parents[2]
"""


def test_script_bootstrap_lines_are_setup(repo):
    repo.write("library/builders/gfx950/x/a.py", BOOTSTRAP)
    base = repo.commit()
    repo.write("library/builders/gfx950/x/b.py", BOOTSTRAP)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


KERNEL_SIGNATURE = """\
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    A = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    D = b.param("D", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    N = b.param("N", I32)
"""
SIGNATURE_CHAIN = """\
    sig = (
        SignatureBuilder("kernel")
        .ptr("q_ptr", io_dtype)
        .ptr("k_ptr", io_dtype)
        .ptr("v_ptr", io_dtype)
        .ptr("o_ptr", io_dtype)
        .scalar("scale", "f32")
        .scalar("num_seqs", "i32")
        .build()
    )
"""
# Braces are doubled: func() passes the body through str.format.
GUARDS = """\
    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    if spec.dtype not in ("f16", "bf16"):
        return False, f"unsupported dtype {{spec.dtype!r}}"
    if spec.head_size % 16 != 0:
        raise ValueError(f"head_size must be a multiple of 16, got {{spec.head_size}}")
    if spec.mask_mode not in ("none", "causal"):
        errors.append(f"unsupported mask_mode {{spec.mask_mode!r}}")
"""
PASS_THROUGH = """\
    return GemmSpec(
        tile_m=self.tile_m,
        tile_n=self.tile_n,
        tile_k=self.tile_k,
        warp_m=self.warp_m,
        warp_n=self.warp_n,
        wave_size=self.wave_size,
        pipeline=pipeline,
        epilogue=epilogue,
    )
"""


@pytest.mark.parametrize(
    "body", [KERNEL_SIGNATURE, SIGNATURE_CHAIN, GUARDS, PASS_THROUGH]
)
def test_declaration_validation_and_pass_through_boilerplate_is_not_a_copy(repo, body):
    repo.write("library/kernels/a.py", func("f1", body))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("f2", body))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


def test_guard_with_logic_in_its_body_is_still_compared(repo):
    # One statement, so only what it does decides whether the `if` is a guard.
    body = GUARDS.replace(
        '        errors.append(f"unsupported mask_mode {{spec.mask_mode!r}}")\n',
        "        b.store(out, b.add(base, b.mul(row, stride)),"
        " b.select(ok, b.fmul(row, stride), b.const_f32(0.0)))\n",
    )
    repo.write("library/kernels/a.py", func("f1", body))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("f2", body))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


CONDITIONAL_PARTS = "".join(
    f"    if {flag}:\n        parts.append({flag})\n"
    for flag in "prologue smem_block launch_bounds extern_c debug_hooks epilogue".split()
)
DISPATCH = "".join(
    f'    if spec.kind == "{kind}":\n        return {kind}_kernel\n'
    for kind in "gemm conv norm moe attn gdn".split()
)


@pytest.mark.parametrize("body", [CONDITIONAL_PARTS, DISPATCH])
def test_conditional_appends_and_returns_of_values_are_compared(repo, body):
    # Only messages make a guard: appending or returning a value is logic.
    repo.write("library/kernels/a.py", func("f1", body))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("f2", body))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


FORWARDED = "".join(
    f"        {name}=self.{name},\n"
    for name in (
        "tile_m tile_n tile_k warp_m warp_n warp_tile_m warp_tile_n warp_tile_k"
        " wave_size pipeline epilogue"
    ).split()
)


def test_pass_through_keywords_inside_a_different_call_are_not_a_copy(repo):
    repo.write(
        "library/kernels/a.py",
        func(
            "f1",
            '    return ConvSpec(\n        problem=p.with_layout("nhwc"),\n'
            + FORWARDED
            + "    )\n",
        ),
    )
    base = repo.commit()
    repo.write(
        "library/kernels/b.py",
        func("f2", "    return GemmSpec(\n        layout=q,\n" + FORWARDED + "    )\n"),
    )
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


def test_constant_setup_windows_are_skipped(repo):
    base = repo.commit()
    setup = "".join(f"    c{i} = b.const_i32({i})\n" for i in range(5))
    repo.write("library/kernels/a.py", func("f1", setup) + func("f2", setup))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


STORE_BLOCK = """\
    d_base_bytes = b.mul(d_base, c_half_bytes)
    safe_d_off = b.select(store_ok, d_base_bytes, oob_sentinel)
    acc_h = _trunc_f32(b, p.dtype, acc_to_flush)
    _buf_store_vN(b, p.dtype, d_rsrc, safe_d_off, c0, acc_h, 2)
    accs_flat[flat] = zero_acc
"""


def test_copied_straight_line_store_block_is_detected(repo):
    repo.write("library/kernels/a.py", func("f1", STORE_BLOCK))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("f2", STORE_BLOCK))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


def test_working_tree_mode_sees_uncommitted_and_untracked_files(repo):
    repo.write("library/kernels/a.py", func("f1"))
    repo.write("library/kernels/b.py", "Y = 2\n")
    base = repo.commit()
    repo.write(
        "library/kernels/b.py", "Y = 2\n\n\n" + func("f2")
    )  # tracked, not committed
    repo.write("library/kernels/new.py", func("f3"))  # untracked
    _, report = repo.check("--base", base)
    assert report["head"] == "WORKTREE"
    (finding,) = report["findings"]
    new = sorted(loc["path"] for loc in finding["locations"] if loc["class"] == "new")
    assert new == ["library/kernels/b.py", "library/kernels/new.py"]


def test_linked_worktree_gives_the_same_result(repo, tmp_path):
    repo.write("library/kernels/a.py", func("f1"))
    base = repo.commit()
    linked = tmp_path.parent / (tmp_path.name + "-linked")
    repo.git("worktree", "add", "-q", str(linked), "HEAD")
    assert (linked / ".git").is_file()
    (linked / "library/kernels/new.py").write_text(func("f2"))
    _, report = repo.check("--base", base, root=linked)
    assert kinds(report) == ["NEW COPY"]


def test_unparsable_file_is_listed_and_the_run_continues(repo):
    base = repo.commit()
    repo.write("library/kernels/bad.py", "def broken(:\n")
    repo.write("library/kernels/a.py", func("f1") + func("f2"))
    head = repo.commit()
    rc, report = repo.check("--base", base, "--head", head)
    assert rc == 0
    assert [(s["side"], s["path"]) for s in report["skipped"]] == [
        ("head", "library/kernels/bad.py")
    ]
    assert kinds(report) == ["NEW COPY"]


def test_group_without_reason_is_a_tool_error(repo):
    repo.write(
        "tools/clone_groups.toml",
        GROUPS.replace('reason = "arch-specific schedules"\n', ""),
    )
    base = repo.commit()
    rc, _ = repo.check("--base", base)
    assert rc == 2


def backlog_groups(report: dict) -> list[tuple]:
    return [
        (g["kind"], g["copies"], sorted(loc["path"] for loc in g["locations"]))
        for g in report["groups"]
    ]


def test_backlog_lists_copies_that_already_exist_in_the_working_tree(repo):
    repo.write("library/kernels/a.py", func("f1"))
    repo.write("library/kernels/b.py", func("f2"))
    repo.commit()
    rc, report = repo.check("--backlog")
    assert rc == 0
    assert report["head"] == "WORKTREE"
    assert backlog_groups(report) == [
        ("COPY GROUP", 2, ["library/kernels/a.py", "library/kernels/b.py"])
    ]
    assert report["groups"][0]["duplicated_lines"] > 0


def test_backlog_of_a_commit_ignores_the_working_tree(repo):
    repo.write("library/kernels/a.py", func("f1"))
    repo.write("library/kernels/b.py", func("f2"))
    head = repo.commit()
    (repo.path / "library/kernels/b.py").unlink()
    _, at_commit = repo.check("--backlog", "--head", head)
    _, in_tree = repo.check("--backlog")
    assert len(at_commit["groups"]) == 1
    assert in_tree["groups"] == []


def test_backlog_puts_the_most_duplicated_lines_first(repo):
    repo.write("library/kernels/a.py", func("f1"))
    repo.write("library/kernels/b.py", func("f2"))
    for name in "cde":
        repo.write(f"library/kernels/{name}.py", func("g", k="16.0"))
    repo.commit()
    _, report = repo.check("--backlog")
    assert [g["copies"] for g in report["groups"]] == [3, 2]
    sizes = [g["duplicated_lines"] for g in report["groups"]]
    assert sizes == sorted(sizes, reverse=True)


def test_backlog_labels_approved_forks_separately(repo):
    repo.write("tools/clone_groups.toml", GROUPS)
    repo.write("library/kernels/gfx942/k.py", func("f"))
    repo.write("library/kernels/gfx950/k.py", func("f"))
    repo.commit()
    _, report = repo.check("--backlog")
    assert [(g["kind"], g["group"]) for g in report["groups"]] == [
        ("APPROVED FORK", "per-arch")
    ]


@pytest.mark.parametrize("extra", [["--base", "HEAD"], ["--fail-on-new"]])
def test_backlog_rejects_options_that_compare_two_revisions(repo, extra):
    repo.commit()
    rc, _ = repo.check("--backlog", *extra)
    assert rc == 2


def test_backlog_text_lists_fifty_groups_then_counts_the_rest(repo):
    for i in range(52):
        repo.write(f"library/kernels/a{i}.py", func("f", k=f"{i}.0"))
        repo.write(f"library/kernels/b{i}.py", func("f", k=f"{i}.0"))
    repo.commit()
    text = subprocess.run(
        [sys.executable, str(TOOL), "--root", str(repo.path), "--backlog"],
        capture_output=True,
        text=True,
    ).stdout
    assert text.count("COPY GROUP ") == 50
    assert "+2 more copy groups" in text


ONE_LINE_TWICE = "    v1 = b.add(base, b.mul(row, b.const_i32(7)), b.add(b.const_i32(1), b.const_i32(2)), b.add(b.const_i32(3), b.const_i32(4)), b.add(b.const_i32(5), b.const_i32(6))); b.store(out, v1, v1)\n"


def test_two_adjacent_copies_on_one_line_count_twice(repo):
    repo.write("library/kernels/a.py", "def f(b, base, row, out):\n" + ONE_LINE_TWICE)
    base = repo.commit()
    repo.write("library/kernels/b.py", "def g(b, base, row, out):\n" + ONE_LINE_TWICE)
    repo.commit()
    _, report = repo.check("--backlog")
    (group,) = report["groups"]
    assert group["copies"] == 2


def test_backlog_regions_come_from_token_runs_not_line_ranges(repo):
    # One copy of the shared block: the backlog must not split one copied
    # region into a small overlapping-window group plus a separate 1-copy group.
    repo.write("library/kernels/a.py", func("f1"))
    repo.commit()
    _, report = repo.check("--backlog")
    assert all(g["copies"] >= 2 for g in report["groups"])


def test_a_region_shared_by_fewer_files_is_a_separate_group(repo):
    # X is in a, b and c; the X+Y stretch only in a and b. The three-copy
    # group must cover X in all three files, not grow to X+Y in a and b only.
    y = BLOCK.replace("b.add(", "b.sub(")
    repo.write("library/kernels/a.py", func("fa", BLOCK + y))
    repo.write("library/kernels/b.py", func("fb", BLOCK + y))
    repo.write("library/kernels/c.py", func("fc", BLOCK))
    repo.commit()
    _, report = repo.check("--backlog")
    assert all(g["copies"] >= 2 for g in report["groups"])
    (triple,) = [g for g in report["groups"] if g["copies"] == 3]
    spans = {loc["end"] - loc["start"] for loc in triple["locations"]}
    assert len(spans) == 1
    assert any(g["copies"] == 2 for g in report["groups"])


def test_same_inputs_give_identical_output(repo):
    repo.write("library/kernels/a.py", func("f1"))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("f2") + func("f3"))
    head = repo.commit()
    run = lambda: subprocess.run(  # noqa: E731
        [
            sys.executable,
            str(TOOL),
            "--root",
            str(repo.path),
            "--format",
            "json",
            "--base",
            base,
            "--head",
            head,
        ],
        capture_output=True,
    ).stdout
    first = run()
    assert first and first == run()


def test_overlapping_windows_in_one_run_are_not_a_copy(repo):
    base = repo.commit()
    repeated = "".join("    acc = b.fadd(acc, b.fmul(x, y))\n" for _ in range(6))
    repo.write("library/kernels/a.py", "def f(b, acc, x, y):\n" + repeated)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


DOC_BLOCK = '''\
    v1 = b.add(base, row)
    v2 = b.select(ok, v1, zero)

    def inner():
        """Line one.

        Line two.
        """
        return v2

    v3 = b.load(ptr, inner())
    b.store(out, v2, v3)
'''


def test_copy_at_a_deeper_nesting_level_is_still_a_copy(repo):
    # Re-indenting the copy also re-indents the nested docstring's text.
    repo.write("library/kernels/a.py", func("f1", DOC_BLOCK))
    base = repo.commit()
    deeper = "    if ok:\n" + "".join(
        "    " + line if line.strip() else line
        for line in DOC_BLOCK.splitlines(keepends=True)
    )
    repo.write("library/kernels/b.py", func("f2", deeper))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


STRING_BLOCK = """\
    v0 = {text}
    v1 = b.add(base, row)
    v2 = b.select(ok, v1, zero)
    v3 = b.load(ptr, v2)
    b.store(out, v0, v3)
"""


def test_string_values_that_differ_only_in_indentation_are_not_a_copy(repo):
    # Only docstrings ignore indentation; a string value keeps its exact text.
    repo.write(
        "library/kernels/a.py", func("f1", STRING_BLOCK.replace("{text}", '"a\\n  b"'))
    )
    base = repo.commit()
    repo.write(
        "library/kernels/b.py", func("f2", STRING_BLOCK.replace("{text}", '"a\\nb"'))
    )
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


def test_two_separated_copies_in_one_block_are_a_copy(repo):
    base = repo.commit()
    body = BLOCK + "    marker = b.barrier()\n" + BLOCK
    repo.write("library/kernels/a.py", func("f", body))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]
