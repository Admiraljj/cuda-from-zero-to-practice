"""Chapter 06 static mirrors, CPU model, and optional real CUDA tests."""
import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from verify_chapter05 import invoke, require

BOOK = pathlib.Path(__file__).resolve().parents[1]
SOURCES = ("grid_map.cpp", "invert.cu", "rgb_invert.cu", "pitched_scale.cu")


def static_checks():
    note = (BOOK / "06-二维索引与图像网格.md").read_text(encoding="utf-8")
    fence = chr(96) * 3
    blocks = {s.strip() for s in re.findall(fence + r"cpp\n(.*?)\n" + fence, note, re.S)}
    for name in SOURCES:
        content = (BOOK / "code/06_image_grid" / name).read_text(encoding="utf-8").strip()
        if content not in blocks:
            raise RuntimeError(f"Missing source mirror: {name}")
    if "@@code:" in note or sum(line.startswith(fence) for line in note.splitlines()) % 2:
        raise RuntimeError("Placeholder or unbalanced fence")
    names = {p.stem for p in BOOK.rglob("*.md")}
    for target in re.findall(r"\[\[([^\]]+)\]\]", note):
        if target.split("|")[0].split("#")[0] not in names:
            raise RuntimeError(f"Broken link: {target}")
    print("PASS static chapter 06: 4 source mirrors, fences, links")


def validate_grid(stdout):
    lines = stdout.strip().splitlines()
    if len(lines) != 33 or lines[-1] != "CPU grid model: valid=15 skipped=17":
        raise RuntimeError("Incorrect CPU grid record count or summary")
    seen = set()
    indices = []
    for line in lines[:-1]:
        m = re.fullmatch(r"block=\((\d+),(\d+)\) thread=\((\d+),(\d+)\) "
                         r"row=(\d+) col=(\d+) (?:index=(\d+)|(skip))", line)
        if not m:
            raise RuntimeError("Malformed grid record")
        bx, by, tx, ty, row, col = (int(m[i]) for i in range(1, 7))
        identity = (bx, by, tx, ty)
        if not (bx < 2 and by < 2 and tx < 4 and ty < 2) or identity in seen:
            raise RuntimeError("Duplicate or invalid thread identity")
        seen.add(identity)
        if (col, row) != (bx*4+tx, by*2+ty):
            raise RuntimeError("Incorrect global coordinate")
        if row < 3 and col < 5:
            if m[7] is None or int(m[7]) != row*5+col:
                raise RuntimeError("Missing or incorrect valid index")
            indices.append(int(m[7]))
        elif m[8] != "skip":
            raise RuntimeError("Invalid coordinate was not skipped")
    if sorted(indices) != list(range(15)):
        raise RuntimeError("Valid pixels must be covered exactly once")


def validate_invert(stdout):
    expected = ["PASS empty: 0 x 3 (no GPU work)"]
    for w, h, bx, by in [(5, 3, 4, 2), (1, 1, 16, 16), (17, 1, 16, 16),
                          (1, 17, 16, 16), (16, 16, 16, 16),
                          (37, 23, 16, 16), (37, 23, 32, 8)]:
        if (w, h) == (5, 3):
            expected += [f"row={r}: " + " ".join(str(255-(17*r+3*c)) for c in range(5))
                         for r in range(3)]
        gx, gy = (w+bx-1)//bx, (h+by-1)//by
        expected.append(f"PASS: {w} x {h} image; block={bx}x{by} grid={gx}x{gy}")
    if stdout.strip().splitlines() != expected:
        raise RuntimeError("Gray-image output or case metadata incorrect")


def validate_rgb(stdout):
    expected = [f"pixel={i} R={255-i} G={205-i} B={155-i}" for i in range(15)]
    expected.append("PASS RGB: 5 x 3 pixels, 45 components")
    if stdout.strip().splitlines() != expected:
        raise RuntimeError("RGB components or identities incorrect")


def validate_pitch(stdout):
    lines = stdout.strip().splitlines()
    if len(lines) != 5:
        raise RuntimeError("Incorrect pitched-output line count")
    m = re.fullmatch(r"row_bytes=20 input_pitch=(\d+) output_pitch=(\d+)", lines[0])
    if not m or any(int(m[i]) < 20 or int(m[i]) % 4 != 0 for i in (1, 2)):
        raise RuntimeError("Invalid reported float row pitches")
    expected = [f"row={r}: " + " ".join(str(20*r+2*c) for c in range(5)) for r in range(3)]
    expected.append("PASS pitched scale: 5 x 3")
    if lines[1:] != expected:
        raise RuntimeError("Pitched result rows incorrect")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--static-only", action="store_true")
    modes.add_argument("--cpu-only", action="store_true")
    modes.add_argument("--build-only", action="store_true")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--arch")
    args = parser.parse_args()
    try:
        static_checks()
        if args.static_only:
            print("NOT RUN: compilation or CPU/GPU execution")
            return 0
        compiler = shutil.which(args.cxx if args.cpu_only else args.nvcc)
        if not compiler:
            print("NOT RUN: requested compiler not available", file=sys.stderr)
            return 2
        validators = dict(zip(SOURCES, [validate_grid, validate_invert, validate_rgb, validate_pitch]))
        with tempfile.TemporaryDirectory(prefix="cuda-book-ch06-") as directory:
            build = pathlib.Path(directory)
            programs = {}
            for name in (SOURCES[:1] if args.cpu_only else SOURCES):
                exe = build / (pathlib.Path(name).stem + (".exe" if sys.platform == "win32" else ""))
                command = [compiler, "-std=c++17"]
                if args.arch and name.endswith(".cu") and not args.cpu_only:
                    command += ["-arch=" + args.arch]
                command += [str(BOOK / "code/06_image_grid" / name), "-o", str(exe)]
                result = invoke(command, build)
                require(result)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
                programs[name] = exe
                print(f"PASS build: {name}")
            if args.build_only:
                print("NOT RUN: CPU/GPU execution (build-only)")
                return 0
            for name, exe in programs.items():
                result = invoke([str(exe)], build)
                require(result)
                validators[name](result.stdout)
                print(f"PASS run and output validation: {name}")
            if args.cpu_only:
                print("NOT RUN: CUDA compilation and GPU execution")
            else:
                print("PASS chapter 06 acceptance; not a performance measurement")
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
