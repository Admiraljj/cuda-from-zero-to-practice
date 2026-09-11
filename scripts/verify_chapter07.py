"""Chapter 07 checks. Intentional CUDA faults require an explicit option."""
import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from verify_chapter05 import invoke, require

BOOK = pathlib.Path(__file__).resolve().parents[1]
SOURCES = ("numeric_check.cpp", "debug_cases.cu", "broken_oob.cu")
GOOD_MODES = ("oob-good", "init-good", "race-good", "sync-good", "launch-good")


def static_checks():
    note = (BOOK / "07-错误检查与Compute-Sanitizer.md").read_text(encoding="utf-8")
    fence = chr(96) * 3
    snippets = {s.strip() for s in re.findall(fence + r"cpp\n(.*?)\n" + fence, note, re.S)}
    for name in SOURCES:
        source = (BOOK / "code/07_debug" / name).read_text(encoding="utf-8").strip()
        if source not in snippets:
            raise RuntimeError(f"Missing source mirror: {name}")
    if "@@code:" in note or sum(line.startswith(fence) for line in note.splitlines()) % 2:
        raise RuntimeError("Placeholder or unbalanced fence")
    names = {p.stem for p in BOOK.rglob("*.md")}
    for link in re.findall(r"\[\[([^\]]+)\]\]", note):
        if link.split("|")[0].split("#")[0] not in names:
            raise RuntimeError(f"Broken link: {link}")
    print("PASS static chapter 07: 3 source mirrors, fences, links")


def validate_numeric(stdout):
    names = ("exact", "near_zero", "relative", "too_far", "nan", "infinity",
             "negative_tolerance", "invalid_tolerance")
    expected = [f"{name}: accepted={int(i < 3)}" for i, name in enumerate(names)]
    expected.append("PASS numeric checker: 8 CPU cases")
    if stdout.strip().splitlines() != expected:
        raise RuntimeError("Incorrect numeric-checker output")


def validate_good(stdout, mode):
    count = {"oob-good": 33, "init-good": 1, "race-good": 64, "sync-good": 64}
    marker = "PASS launch-good" if mode == "launch-good" else f"PASS {mode}: {count[mode]} checked elements"
    if marker not in stdout.splitlines() or "BAD CASE" in stdout:
        raise RuntimeError(f"Missing actual result-validation marker: {mode}")


def validate_sanitizer(log, returncode, tool, bad_kind=None):
    """Use diagnostic class AND nonzero summary; any arbitrary failure is not enough."""
    if "COMPUTE-SANITIZER" not in log:
        raise RuntimeError("No evidence that Compute Sanitizer ran")
    if re.search(r"Failed to (?:connect|initialize)|No attachable process|Unable to launch", log, re.I):
        raise RuntimeError("Tool failed before a trustworthy test")
    if tool == "racecheck":
        summaries = re.findall(r"RACECHECK SUMMARY:\s*(\d+)\s+hazards?", log, re.I)
    else:
        summaries = re.findall(r"ERROR SUMMARY:\s*(\d+)\s+errors?", log, re.I)
    if not summaries:
        raise RuntimeError("Tool summary missing or unsupported report format")
    counts = [int(n) for n in summaries]
    if bad_kind is None:
        if returncode != 0 or any(counts):
            raise RuntimeError("Fixed case failed or sanitizer found problems")
        return
    patterns = {
        "oob": r"Invalid\s+__global__\s+write",
        "init": r"Uninitialized\s+__global__\s+memory read",
        "race": r"(?:Potential\s+\w+\s+hazard|Race reported between)",
        "sync": r"Barrier error detected",
    }
    if returncode == 0 or not any(counts) or not re.search(patterns[bad_kind], log, re.I):
        raise RuntimeError(f"Expected {bad_kind} diagnosis not demonstrated")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--static-only", action="store_true")
    modes.add_argument("--cpu-only", action="store_true")
    modes.add_argument("--build-only", action="store_true")
    parser.add_argument("--with-intentional-errors", action="store_true")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--sanitizer", default="compute-sanitizer")
    parser.add_argument("--arch")
    args = parser.parse_args()
    try:
        static_checks()
        if args.static_only:
            print("NOT RUN: C++/CUDA compilation and execution")
            return 0
        compiler = shutil.which(args.cxx if args.cpu_only else args.nvcc)
        if not compiler:
            print("NOT RUN: requested compiler not available", file=sys.stderr)
            return 2
        sanitizer = shutil.which(args.sanitizer)
        if not args.cpu_only and not args.build_only and not sanitizer:
            print("NOT RUN: Compute Sanitizer not available", file=sys.stderr)
            return 2
        with tempfile.TemporaryDirectory(prefix="cuda-book-ch07-") as directory:
            build = pathlib.Path(directory)
            exes = {}
            for name in (SOURCES[:1] if args.cpu_only else SOURCES):
                exe = build / (pathlib.Path(name).stem + (".exe" if sys.platform == "win32" else ""))
                command = [compiler, "-std=c++17"]
                if name.endswith(".cu"):
                    command.append("-lineinfo")
                    if args.arch:
                        command.append("-arch=" + args.arch)
                command += [str(BOOK / "code/07_debug" / name), "-o", str(exe)]
                result = invoke(command, build)
                require(result)
                exes[name] = str(exe)
                print(f"PASS build: {name}")
            if args.build_only:
                print("NOT RUN: executions (build-only)")
                return 0
            result = invoke([exes["numeric_check.cpp"]], build)
            require(result)
            validate_numeric(result.stdout)
            print("PASS numeric CPU cases")
            if args.cpu_only:
                print("NOT RUN: CUDA and sanitizer tests")
                return 0
            exe = exes["debug_cases.cu"]
            for values in (["oob-bad"], ["unknown"], ["race-good", "--run-broken"]):
                result = invoke([exe, *values], build)
                require(result, 2)
                if "Use:" not in result.stderr:
                    raise RuntimeError("Safety/argument guard failed")
            result = invoke([exes["broken_oob.cu"]], build)
            require(result, 2)
            if "Refusing" not in result.stderr:
                raise RuntimeError("Legacy intentional example did not refuse default execution")

            def run_tool(mode, tool, bad_kind=None, target=None):
                command = [sanitizer, "--tool", tool, "--error-exitcode", "99"]
                if tool == "memcheck":
                    command += ["--padding", "32"]
                if tool == "racecheck":
                    command += ["--racecheck-report", "all"]
                command += [target or exe]
                if target is None:
                    command.append(mode)
                if bad_kind:
                    command.append("--run-broken")
                result = invoke(command, build)
                log = result.stdout + "\n" + result.stderr
                try:
                    validate_sanitizer(log, result.returncode, tool, bad_kind)
                    if bad_kind is None:
                        validate_good(result.stdout, mode)
                except RuntimeError as error:
                    raise RuntimeError(f"{mode}/{tool}: {error}\n{log}") from error
                print(f"PASS {'expected diagnosis' if bad_kind else 'fixed case'}: {mode}/{tool}")

            for mode in GOOD_MODES:
                result = invoke([exe, mode], build)
                require(result)
                validate_good(result.stdout, mode)
                run_tool(mode, "memcheck")
            for prefix, tool in (("init", "initcheck"), ("race", "racecheck"), ("sync", "synccheck")):
                run_tool(prefix + "-good", tool)
            if args.with_intentional_errors:
                for prefix, tool in (("oob", "memcheck"), ("init", "initcheck"),
                                     ("race", "racecheck"), ("sync", "synccheck")):
                    run_tool(prefix + "-bad", tool, prefix)
                run_tool("broken_oob", "memcheck", "oob", exes["broken_oob.cu"])
                result = invoke([exe, "launch-bad", "--run-broken"], build)
                require(result)
                if "OBSERVED expected launch rejection; not a successful kernel" not in result.stdout:
                    raise RuntimeError("Expected launch rejection not demonstrated")
                print("PASS launch-error-state demonstration")
            else:
                print("NOT RUN: intentional fault execution (requires --with-intentional-errors)")
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
