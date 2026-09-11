"""Chapter 05 source mirrors and optional compiler/device acceptance tests."""
import argparse
import math
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

BOOK = pathlib.Path(__file__).resolve().parents[1]
SOURCES = ("index_map.cpp", "memory_roundtrip.cu", "vector_add.cu")


def static_checks():
    note = (BOOK / "05-一维索引与向量运算.md").read_text(encoding="utf-8")
    fence = chr(96) * 3
    blocks = {x.strip() for x in re.findall(fence + r"cpp\n(.*?)\n" + fence, note, re.S)}
    for name in SOURCES:
        source = (BOOK / "code/05_vector_add" / name).read_text(encoding="utf-8").strip()
        if source not in blocks:
            raise RuntimeError(f"Source mirror missing: {name}")
    if "@@code:" in note or sum(x.startswith(fence) for x in note.splitlines()) % 2:
        raise RuntimeError("Unresolved placeholder or unbalanced code fence")
    notes = {p.stem for p in BOOK.rglob("*.md")}
    for link in re.findall(r"\[\[([^\]]+)\]\]", note):
        if link.split("|")[0].split("#")[0] not in notes:
            raise RuntimeError(f"Broken chapter link: {link}")
    print("PASS static chapter 05: 3 source mirrors, fences, links")


def validate_index(stdout):
    expected = []
    for block in range(3):
        for tid in range(4):
            i = block * 4 + tid
            expected.append(f"block={block} tid={tid} i={i} " + ("work" if i < 10 else "skip"))
    expected.append("CPU model: n=10 blocks=3 slots=12")
    if stdout.strip().splitlines() != expected:
        raise RuntimeError("CPU index model output differs from expected 12 positions")


def validate_roundtrip(stdout):
    expected = [f"i={i} input={i+1} output={i+1}" for i in range(4)]
    expected.append("PASS memory roundtrip: 4 elements; no kernel launched")
    if stdout.strip().splitlines() != expected:
        raise RuntimeError("Memory roundtrip records differ")


def validate_vector(stdout, n, threads):
    lines = stdout.strip().splitlines()
    if n == 0:
        if lines != ["PASS: 0 elements (no GPU work)"]:
            raise RuntimeError("Empty task did not explicitly skip GPU work")
        return
    count = n // threads + (n % threads != 0)
    header = (f"n={n} threads={threads} blocks={count} slots={count*threads}"
              f" bytes_per_array={n*4}")
    if not lines or lines[0] != header or lines[-1] != f"PASS: {n} elements":
        raise RuntimeError("Incorrect launch metadata or completion marker")
    rows = lines[1:-1]
    if len(rows) != (n if n <= 16 else 0):
        raise RuntimeError("Unexpected number of printed result rows")
    for i, row in enumerate(rows):
        match = re.fullmatch(r"i=(\d+) a=(\S+) b=(\S+) c=(\S+)", row)
        if not match or int(match[1]) != i:
            raise RuntimeError("Incorrect result row identity")
        values = [float(match[j]) for j in (2, 3, 4)]
        expected = [0.5*i, 2.0*i, 2.5*i]
        if any(not math.isfinite(a) or abs(a-b) > 1e-5 for a, b in zip(values, expected)):
            raise RuntimeError("Incorrect printed result")


def invoke(command, cwd):
    return subprocess.run(command, cwd=cwd, text=True, encoding="utf-8",
                          errors="replace", capture_output=True, timeout=120)


def require(result, code=0):
    if result.returncode != code:
        raise RuntimeError(f"Unexpected exit {result.returncode}, expected {code}\n"
                           f"{result.stdout}\n{result.stderr}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--static-only", action="store_true")
    mode.add_argument("--cpu-only", action="store_true")
    mode.add_argument("--build-only", action="store_true")
    parser.add_argument("--cxx", default="g++", help="GCC/Clang-compatible C++ compiler for cpu-only")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--arch", help="Supported CUDA architecture such as sm_80")
    args = parser.parse_args()
    try:
        static_checks()
        if args.static_only:
            print("NOT RUN: compilation and CPU/GPU execution (static-only)")
            return 0
        compiler = shutil.which(args.cxx if args.cpu_only else args.nvcc)
        if compiler is None:
            print("NOT RUN: requested compiler not found", file=sys.stderr)
            return 2
        with tempfile.TemporaryDirectory(prefix="cuda-book-ch05-") as directory:
            build = pathlib.Path(directory)
            programs = {}
            for name in (SOURCES[:1] if args.cpu_only else SOURCES):
                target = build / (pathlib.Path(name).stem + (".exe" if sys.platform == "win32" else ""))
                command = [compiler, "-std=c++17"]
                if args.arch and name.endswith(".cu") and not args.cpu_only:
                    command.append("-arch=" + args.arch)
                command += [str(BOOK / "code/05_vector_add" / name), "-o", str(target)]
                result = invoke(command, build)
                require(result)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
                programs[name] = target
                print(f"PASS build: {name}")
            if args.build_only:
                print("NOT RUN: CPU/GPU execution (build-only)")
                return 0
            result = invoke([str(programs["index_map.cpp"])], build)
            require(result)
            validate_index(result.stdout)
            print("PASS CPU index model")
            if args.cpu_only:
                print("NOT RUN: CUDA compilation and GPU tests (cpu-only)")
                return 0
            result = invoke([str(programs["memory_roundtrip.cu"])], build)
            require(result)
            validate_roundtrip(result.stdout)
            print("PASS device memory roundtrip")
            exe = str(programs["vector_add.cu"])
            cases = [(0, 256), (1, 32), (10, 4), (31, 32), (32, 32),
                     (33, 32), (1000, 64), (1000, 128), (1000, 256), (1048576, 256)]
            for n, threads in cases:
                result = invoke([exe, str(n), str(threads)], build)
                require(result)
                validate_vector(result.stdout, n, threads)
                print(f"PASS vector_add n={n} threads={threads}")
            result = invoke([exe], build)
            require(result)
            validate_vector(result.stdout, 1048576, 256)
            invalid = [["-1"], ["10x"], ["1048577"], ["10", "0"], ["10", "1025"],
                       ["10", "4", "--unknown"], ["10", "4", "--inject-nan", "extra"],
                       ["999999999999999999999"], [""], ["1.0"], ["0", "4", "--inject-nan"]]
            for values in invalid:
                result = invoke([exe, *values], build)
                require(result, 2)
                if "PASS" in result.stdout or not result.stderr.strip():
                    raise RuntimeError("Invalid input did not fail clearly")
            result = invoke([exe, "10", "4", "--inject-nan"], build)
            require(result, 1)
            if ("Injected NaN into host validation input" not in result.stderr
                or "mismatch at 5:" not in result.stderr or "PASS:" in result.stdout):
                raise RuntimeError("NaN injection did not trigger the intended validation failure")
            print("PASS invalid-input and injected-NaN paths")
            print("PASS chapter 05 acceptance; no timing/performance conclusion")
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
