"""Synthetic sanitizer logs only; never claim these executed on a GPU."""
import unittest
from verify_chapter07 import validate_sanitizer, validate_numeric, validate_good


class LogChecks(unittest.TestCase):
    def test_clean_tools(self):
        for tool, line in [("memcheck", "ERROR SUMMARY: 0 errors"),
                           ("initcheck", "ERROR SUMMARY: 0 errors"),
                           ("synccheck", "ERROR SUMMARY: 0 errors"),
                           ("racecheck", "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)")]:
            validate_sanitizer("COMPUTE-SANITIZER\n" + line, 0, tool)

    def test_intentional_diagnostics(self):
        records = [
            ("oob", "memcheck", "Invalid __global__ write of size 4 bytes", "ERROR SUMMARY: 31 errors"),
            ("init", "initcheck", "Uninitialized __global__ memory read of size 4 bytes", "ERROR SUMMARY: 1 error"),
            ("race", "racecheck", "ERROR: Potential RAW hazard detected", "RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)"),
            ("sync", "synccheck", "Barrier error detected. Divergent threads", "ERROR SUMMARY: 16 errors"),
        ]
        for kind, tool, diagnostic, summary in records:
            with self.subTest(kind=kind):
                validate_sanitizer("COMPUTE-SANITIZER\n" + diagnostic + "\n" + summary, 99, tool, kind)

    def test_arbitrary_failure_is_not_a_detected_bug(self):
        with self.assertRaises(RuntimeError):
            validate_sanitizer("no CUDA device", 1, "memcheck", "oob")

    def test_wrong_diagnostic_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_sanitizer("COMPUTE-SANITIZER\nAPI error\nERROR SUMMARY: 1 error", 99, "memcheck", "oob")

    def test_zero_summary_not_reproduction(self):
        with self.assertRaises(RuntimeError):
            validate_sanitizer("COMPUTE-SANITIZER\nInvalid __global__ write\nERROR SUMMARY: 0 errors",
                               99, "memcheck", "oob")

    def test_clean_summary_but_failed_app(self):
        with self.assertRaises(RuntimeError):
            validate_sanitizer("COMPUTE-SANITIZER\nERROR SUMMARY: 0 errors", 1, "memcheck")

    def test_nonzero_summary_on_good_case(self):
        with self.assertRaises(RuntimeError):
            validate_sanitizer("COMPUTE-SANITIZER\nERROR SUMMARY: 1 error", 0, "memcheck")

    def test_missing_summary(self):
        with self.assertRaises(RuntimeError):
            validate_sanitizer("COMPUTE-SANITIZER\nInvalid __global__ write", 99, "memcheck", "oob")

    def test_good_marker(self):
        validate_good("PASS race-good: 64 checked elements\n", "race-good")
        with self.assertRaises(RuntimeError):
            validate_good("BAD CASE completed\n", "race-good")

    def test_numeric_fixture(self):
        output = ("exact: accepted=1\nnear_zero: accepted=1\nrelative: accepted=1\n"
                  "too_far: accepted=0\nnan: accepted=0\ninfinity: accepted=0\n"
                  "negative_tolerance: accepted=0\ninvalid_tolerance: accepted=0\n"
                  "PASS numeric checker: 8 CPU cases")
        validate_numeric(output)
        with self.assertRaises(RuntimeError):
            validate_numeric(output.replace("nan: accepted=0", "nan: accepted=1"))


if __name__ == "__main__":
    print("Offline synthetic logs only; no sanitizer or CUDA execution.")
    unittest.main(verbosity=2)
