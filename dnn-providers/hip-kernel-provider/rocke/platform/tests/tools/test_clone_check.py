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

# One copied region: five non-trivial emitter statements.
BLOCK = """\
    v1 = b.add(base, b.mul(row, stride))
    v2 = b.select(ok, v1, zero)
    v3 = b.{load}(ptr, v2)
    v4 = b.fmul(v3, b.const_f32({k}))
    b.store(out, v2, v4)
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
    # Same statements, but the last one stores v1 instead of v2: a different data flow.
    repo.write(
        "library/kernels/b.py",
        func("g", BLOCK.replace("b.store(out, v2, v4)", "b.store(out, v1, v4)")),
    )
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


def test_same_fix_in_both_existing_copies_is_not_a_copy(repo):
    repo.write("library/kernels/a.py", func("f1"))
    repo.write("library/kernels/b.py", func("f2"))
    base = repo.commit()
    fixed = BLOCK.replace("b.add(", "b.sub(")
    repo.write("library/kernels/a.py", func("f1", fixed))
    repo.write("library/kernels/b.py", func("f2", fixed))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


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
    (note,) = report["sibling_notes"]
    assert note["changed"]["path"] == "library/kernels/gfx942/k.py"
    assert note["sibling"]["path"] == "library/kernels/gfx950/k.py"


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


def test_two_separated_copies_in_one_block_are_a_copy(repo):
    base = repo.commit()
    body = BLOCK + "    marker = b.barrier()\n" + BLOCK
    repo.write("library/kernels/a.py", func("f", body))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]
