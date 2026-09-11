"""Check Chapter 01 source mirrors; optionally compile/run its CPU examples.

Usage: python3 scripts/verify_chapter01.py --compiler g++ [--sanitize]
       python3 scripts/verify_chapter01.py --static-only
Only temporary build outputs are created; tutorial sources are not modified.
"""

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile


BOOK = pathlib.Path(__file__).resolve().parents[1]
SOURCE = BOOK / "code" / "01_cpp_basics"
CHAPTER = BOOK / "01-你需要的C++与Linux基础.md"
ANSWERS = BOOK / "附录-C-练习答案.md"

CASES = [
    ("hello", ["01_hello.cpp"], "Hello, CUDA learner!\n"),
    ("values", ["02_values.cpp"],
     "n=8\nfactor=2\nreference=0.333333\nready=1\n"
     "integer division=2\nfloating division=2.5\nelements=8\n"
     "float bytes=4\narray bytes=32\n"),
    ("loop_array", ["03_loop_array.cpp"],
     "i=0 value=11\ni=1 value=21\ni=2 value=31\ni=3 value=41\n"),
    ("functions", ["04_functions.cpp"],
     "square=9\ninside copy=99\nafter copy=3\nafter reference=99\n"),
    ("pointer_alias", ["05_pointer_alias.cpp"],
     "p[1]=99\nvalues[1]=7\nvalues=11,8,31\nfirst=11\n"),
    ("vector", ["06_vector.cpp"],
     "values[0]=9\ncopied[0]=2\nafter value loop=9\n"
     "after reference loop=7\nsize=4\n"),
    ("vector_add", ["07_vector_add_cpu.cpp"],
     "".join(f"c[{i}]={3*i}\n" for i in range(8)) + "PASS n=8\n"),
    ("multi_file", ["08_multi_file/main.cpp", "08_multi_file/add.cpp"], "5\n"),
    ("scale_solution", ["solutions/scale_cpu.cpp"],
     "3\n6\n9\n12\n15\nPASS scale\n"),
]


def normalized(text):
    return text.replace("\r\n", "\n").strip()


def check_static():
    chapter = CHAPTER.read_text(encoding="utf-8")
    answers = ANSWERS.read_text(encoding="utf-8")
    blocks = {
        normalized(block)
        for block in re.findall(r"```cpp\n(.*?)\n```", chapter + "\n" + answers, re.S)
    }
    sources = sorted(SOURCE.rglob("*.cpp")) + sorted(SOURCE.rglob("*.h"))
    if len(sources) != 11:
        raise RuntimeError(f"Expected 11 source/header files, found {len(sources)}")
    for path in sources:
        if normalized(path.read_text(encoding="utf-8")) not in blocks:
            raise RuntimeError(f"Source not mirrored exactly in chapter/answers: {path}")
    if "@@code:" in chapter:
        raise RuntimeError("Unresolved code placeholder in chapter")
    for path in (CHAPTER, ANSWERS, BOOK / "00-教材修订进度.md", SOURCE / "README.md"):
        content = path.read_text(encoding="utf-8")
        fence_count = sum(line.startswith("```") for line in content.splitlines())
        if fence_count % 2:
            raise RuntimeError(f"Unbalanced code fences: {path}")
    note_names = {path.stem for path in BOOK.rglob("*.md")}
    for target in re.findall(r"\[\[([^\]]+)\]\]", chapter):
        note = target.split("|", 1)[0].split("#", 1)[0]
        if note and note not in note_names:
            raise RuntimeError(f"Unresolved chapter wikilink: {target}")
    for relative in (
        "code/01_cpp_basics/README.md",
        "01-你需要的C++与Linux基础.md",
        "00-教材修订进度.md",
    ):
        if not (BOOK / relative).is_file():
            raise RuntimeError(f"Missing local link target: {relative}")
    print("PASS static: 11 source mirrors, code fences, chapter links")


def invoke(command, cwd):
    result = subprocess.run(
        command, cwd=cwd, text=True, encoding="utf-8", errors="replace",
        capture_output=True, timeout=90,
    )
    return result


def build_and_run(compiler, sanitize):
    flags = ["-std=c++17", "-Wall", "-Wextra", "-pedantic-errors", "-O0", "-g"]
    if sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    version = invoke([compiler, "--version"], SOURCE)
    if version.returncode:
        raise RuntimeError(version.stderr or "Compiler --version failed")
    print(version.stdout.splitlines()[0])
    with tempfile.TemporaryDirectory(prefix="cuda-book-ch01-") as temporary:
        build = pathlib.Path(temporary)
        for name, sources, expected in CASES:
            executable = build / (name + (".exe" if sys.platform == "win32" else ""))
            command = [compiler, *flags, *(str(SOURCE / s) for s in sources),
                       "-o", str(executable)]
            compiled = invoke(command, build)
            if compiled.returncode:
                raise RuntimeError(f"Build failed: {name}\n{compiled.stdout}\n{compiled.stderr}")
            if compiled.stderr:
                print(compiled.stderr, file=sys.stderr)
            run = invoke([str(executable)], build)
            if run.returncode or normalized(run.stdout) != normalized(expected) or run.stderr:
                raise RuntimeError(
                    f"Run failed: {name}, exit={run.returncode}\n"
                    f"stdout:\n{run.stdout}\nstderr:\n{run.stderr}\nexpected:\n{expected}"
                )
            print(f"PASS run: {name}")
        # A deliberate link failure, not an unsafe executable.
        missing = invoke(
            [compiler, *flags, str(SOURCE / "08_multi_file/main.cpp"),
             "-o", str(build / "missing_definition")], build,
        )
        if missing.returncode == 0 or "add" not in (missing.stdout + missing.stderr):
            raise RuntimeError("Missing-definition test did not show the expected add link failure")
        print("PASS negative build: missing add definition rejected")
    print("PASS compiled suite: 9 programs + 1 negative build")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--compiler", help="g++/clang++ name or executable path (no extra flags)")
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    if args.static_only and args.sanitize:
        parser.error("--sanitize requires compiling; omit --static-only")
    try:
        check_static()
        if args.static_only:
            print("NOT RUN: C++ compilation/execution (static-only mode)")
            return 0
        compiler = shutil.which(args.compiler) if args.compiler else (
            shutil.which("g++") or shutil.which("clang++")
        )
        if not compiler:
            print("NOT RUN: no g++/clang++ compiler; use --compiler on a configured host",
                  file=sys.stderr)
            return 2
        build_and_run(compiler, args.sanitize)
        return 0
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
