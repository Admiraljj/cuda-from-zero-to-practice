"""Static mirrors and optional NVCC tests for revised CUDA introduction chapters.

No driver installation, GPU reset, or persistent environment change is performed.
"""

import argparse
from collections import Counter
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

BOOK = pathlib.Path(__file__).resolve().parents[1]
CHAPTERS = {
    "02": {
        "note": "02-检查GPU服务器与CUDA环境.md",
        "directory": "code/02_environment",
        "sources": ["device_count.cu", "device_info.cu", "cuda_versions.cu"],
    },
    "03": {
        "note": "03-第一段CUDA程序与NVCC.md",
        "directory": "code/03_hello",
        "sources": ["hello_one.cu", "hello.cu", "launch_order.cu", "scalar_tasks.cu"],
    },
    "04": {
        "note": "04-线程-Block-Grid-Warp与SIMT.md",
        "directory": "code/04_execution_model",
        "sources": ["inspect_layout.cu", "branch_paths.cu"],
    },
}


def normalized(value):
    return value.replace("\r\n", "\n").strip()


def check_static(chapter):
    spec = CHAPTERS[chapter]
    note = (BOOK / spec["note"]).read_text(encoding="utf-8")
    snippets = {
        normalized(block) for block in re.findall(r"```cpp\n(.*?)\n```", note, re.S)
    }
    for source in spec["sources"]:
        path = BOOK / spec["directory"] / source
        if normalized(path.read_text(encoding="utf-8")) not in snippets:
            raise RuntimeError(f"Source is not mirrored in chapter: {path}")
    if "@@code:" in note:
        raise RuntimeError("Unresolved code placeholder")
    if sum(line.startswith("```") for line in note.splitlines()) % 2:
        raise RuntimeError("Unbalanced code fences")
    names = {path.stem for path in BOOK.rglob("*.md")}
    for target in re.findall(r"\[\[([^\]]+)\]\]", note):
        name = target.split("|", 1)[0].split("#", 1)[0]
        if name and name not in names:
            raise RuntimeError(f"Unresolved wikilink: {target}")
    for relative in ("README.md",):
        if not (BOOK / spec["directory"] / relative).is_file():
            raise RuntimeError(f"Missing companion file: {relative}")
    print(f"PASS static chapter {chapter}: {len(spec['sources'])} source mirrors and note checks")


def invoke(command, cwd, env=None):
    return subprocess.run(
        command, cwd=cwd, env=env, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=90,
    )


def require_success(result, context):
    if result.returncode != 0:
        raise RuntimeError(
            f"{context}: exit={result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def test_chapter02(executables, build):
    count = invoke([str(executables["device_count.cu"])], build)
    require_success(count, "device enumeration")
    match = re.search(r"visible CUDA devices: (\d+)", count.stdout)
    if not match or int(match.group(1)) < 1:
        raise RuntimeError("Enumeration did not report a positive device count")
    number = int(match.group(1))
    print(count.stdout.strip())

    info = invoke([str(executables["device_info.cu"])], build)
    require_success(info, "device properties")
    indices = re.findall(r"^\[(\d+)\] ", info.stdout, re.M)
    if indices != [str(i) for i in range(number)]:
        raise RuntimeError("Property records do not match the enumerated device count")
    if "PASS device query (no kernel executed)" not in info.stdout:
        raise RuntimeError("Missing successful property-query marker")
    print(info.stdout.strip())

    versions = invoke([str(executables["cuda_versions.cu"])], build)
    require_success(versions, "version report")
    records = re.findall(r": (\d+)\.(\d+) \(raw=(\d+)\)", versions.stdout)
    if len(records) != 2:
        raise RuntimeError("Expected two CUDA version records")
    for major, minor, raw in records:
        if int(raw) != int(major) * 1000 + int(minor) * 10:
            raise RuntimeError("Invalid version decoding")
    print(versions.stdout.strip())

    hidden_env = dict(os.environ)
    hidden_env["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("device_count.cu", "device_info.cu"):
        hidden = invoke([str(executables[name])], build, hidden_env)
        # Depending on driver/runtime, this can be a CUDA error or successful zero count.
        if hidden.returncode not in (1, 2):
            raise RuntimeError(f"Expected a controlled no-device failure for {name}")
        if "PASS" in hidden.stdout:
            raise RuntimeError(f"Hidden-device run incorrectly reported PASS: {name}")
        expected_diagnostic = (
            "cudaGetDeviceCount failed:" in hidden.stderr
            or "No visible CUDA device" in hidden.stderr
        )
        if not expected_diagnostic:
            raise RuntimeError(f"No-device test failed for an unrelated reason: {hidden.stderr}")
        print(f"PASS negative path: {name} rejects no visible devices")
    print("PASS chapter 02 queries; NO kernel execution has been tested")


def validate_chapter03_outputs(outputs):
    """Validate captured text without claiming that fixtures executed on a GPU."""
    expectations = {
        "hello_one.cu": (
            ["CPU: before launch", "CPU: GPU work finished"],
            ["GPU: hello from one thread"],
        ),
        "hello.cu": (
            ["CPU: before launch", "CPU: GPU work finished"],
            [f"GPU: block={b} thread={t} block_size=4 grid_size=2"
             for b in range(2) for t in range(4)],
        ),
        "launch_order.cu": (
            ["CPU: A before launch", "CPU: B launch call returned",
             "CPU: C synchronization succeeded"],
            [f"GPU: thread={t}" for t in range(4)],
        ),
        "scalar_tasks.cu": (
            ["CPU: scalar tasks finished"],
            [f"GPU: thread={t} input={10+t} output={2*(10+t)}" for t in range(4)],
        ),
    }
    if set(outputs) != set(expectations):
        raise RuntimeError("Expected output from all four Chapter 03 programs")
    for name, (cpu_expected, gpu_expected) in expectations.items():
        lines = [line for line in normalized(outputs[name]).splitlines() if line.strip()]
        cpu_lines = [line for line in lines if line.startswith("CPU:")]
        gpu_lines = [line for line in lines if line.startswith("GPU:")]
        if len(lines) != len(cpu_lines) + len(gpu_lines):
            raise RuntimeError(f"Unexpected output line: {name}")
        if cpu_lines != cpu_expected:
            raise RuntimeError(f"Incorrect host progress markers: {name}")
        if Counter(gpu_lines) != Counter(gpu_expected):
            raise RuntimeError(f"Missing, duplicate, or incorrect GPU records: {name}")
        if lines[-1] != cpu_expected[-1]:
            raise RuntimeError(f"Completion marker preceded device output: {name}")
        if name != "scalar_tasks.cu" and lines[0] != cpu_expected[0]:
            raise RuntimeError(f"Device output preceded the initial host marker: {name}")
        # Do not assume B occurs before the GPU records in launch_order.cu.


def test_chapter03(executables, build):
    outputs = {}
    for name, executable in executables.items():
        result = invoke([str(executable)], build)
        require_success(result, f"GPU run: {name}")
        outputs[name] = result.stdout
        print(result.stdout.strip())
    validate_chapter03_outputs(outputs)
    print("PASS chapter 03: four kernel examples, unordered thread records checked")


def validate_chapter04_outputs(outputs):
    """Check logical records, not hardware scheduling or performance."""
    layout = [
        f"block={b} tid={t} global={b*40+t} warp={t//32} lane={t%32}"
        for b in range(2) for t in range(40)
    ]
    branches = []
    for mode in range(2):
        for tid in range(64):
            choose_a = tid < 32 if mode == 0 else tid % 2 == 0
            path = "A" if choose_a else "B"
            value = tid + (100 if choose_a else 200)
            branches.append(
                f"mode={mode} tid={tid} warp={tid//32} lane={tid%32} "
                f"path={path} value={value}"
            )
    expected = {
        "inspect_layout.cu": (layout, "PASS launch: 2 blocks, 40 threads each; inspect 80 records."),
        "branch_paths.cu": (branches, "PASS launch: two modes; inspect 128 records."),
    }
    if set(outputs) != set(expected):
        raise RuntimeError("Expected output from both Chapter 04 programs")
    for name, (records, marker) in expected.items():
        lines = [line for line in normalized(outputs[name]).splitlines() if line.strip()]
        if not lines or lines[-1] != marker:
            raise RuntimeError(f"Missing final completion marker: {name}")
        if Counter(lines[:-1]) != Counter(records):
            raise RuntimeError(f"Missing, duplicate, or incorrect thread records: {name}")
        if name == "branch_paths.cu":
            modes = [int(line[5]) for line in lines[:-1]]
            if modes != sorted(modes):
                raise RuntimeError("Modes must respect per-launch host synchronization")


def test_chapter04(executables, build):
    outputs = {}
    for name, executable in executables.items():
        result = invoke([str(executable)], build)
        require_success(result, f"GPU run: {name}")
        outputs[name] = result.stdout
    validate_chapter04_outputs(outputs)
    print("PASS chapter 04: 80 layout records and 128 branch records checked")
    print("NOT TESTED: physical scheduling, instruction branching, or performance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapter", choices=CHAPTERS, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--static-only", action="store_true")
    modes.add_argument("--build-only", action="store_true")
    parser.add_argument("--nvcc", default="nvcc", help="NVCC executable name or path")
    parser.add_argument("--arch", help="Optional supported architecture, e.g. sm_80")
    args = parser.parse_args()
    try:
        check_static(args.chapter)
        if args.static_only:
            print("NOT RUN: NVCC build and device tests (static-only mode)")
            return 0
        nvcc = shutil.which(args.nvcc)
        if not nvcc:
            print("NOT RUN: nvcc not available in this environment", file=sys.stderr)
            return 2
        spec = CHAPTERS[args.chapter]
        with tempfile.TemporaryDirectory(prefix=f"cuda-book-ch{args.chapter}-") as directory:
            build = pathlib.Path(directory)
            executables = {}
            for source in spec["sources"]:
                executable = build / (pathlib.Path(source).stem + (".exe" if os.name == "nt" else ""))
                command = [nvcc, "-std=c++17"]
                if args.arch:
                    command += ["-arch=" + args.arch]
                command += [str(BOOK / spec["directory"] / source), "-o", str(executable)]
                compiled = invoke(command, build)
                require_success(compiled, f"NVCC build: {source}")
                if compiled.stderr:
                    print(compiled.stderr, file=sys.stderr)
                executables[source] = executable
                print(f"PASS build: {source}")
            if args.build_only:
                print("NOT RUN: device tests (build-only mode)")
                return 0
            if args.chapter == "02":
                test_chapter02(executables, build)
            elif args.chapter == "03":
                test_chapter03(executables, build)
            elif args.chapter == "04":
                test_chapter04(executables, build)
        return 0
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
