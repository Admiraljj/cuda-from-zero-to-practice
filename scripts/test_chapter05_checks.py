"""Offline fixtures for Chapter 05 checkers; no CUDA execution."""
import unittest
from verify_chapter05 import validate_index, validate_roundtrip, validate_vector


def vector_fixture(n=10, threads=4):
    if n == 0:
        return "PASS: 0 elements (no GPU work)\n"
    blocks = (n + threads - 1) // threads
    rows = [f"n={n} threads={threads} blocks={blocks} slots={blocks*threads} bytes_per_array={n*4}"]
    if n <= 16:
        rows += [f"i={i} a={i/2:g} b={i*2:g} c={i*2.5:g}" for i in range(n)]
    rows += [f"PASS: {n} elements"]
    return "\n".join(rows)


class Checkers(unittest.TestCase):
    def test_index(self):
        rows = [f"block={i//4} tid={i%4} i={i} " + ("work" if i < 10 else "skip")
                for i in range(12)]
        rows.append("CPU model: n=10 blocks=3 slots=12")
        validate_index("\n".join(rows))
        with self.assertRaises(RuntimeError):
            validate_index("\n".join(rows).replace("i=10 skip", "i=10 work"))

    def test_roundtrip(self):
        rows = [f"i={i} input={i+1} output={i+1}" for i in range(4)]
        rows.append("PASS memory roundtrip: 4 elements; no kernel launched")
        validate_roundtrip("\n".join(rows))
        with self.assertRaises(RuntimeError):
            validate_roundtrip("\n".join(rows).replace("output=4", "output=3"))

    def test_valid_sizes(self):
        for n, threads in [(0, 256), (1, 32), (10, 4), (16, 4), (17, 4),
                           (31, 32), (32, 32), (33, 32), (1000, 256), (1048576, 256)]:
            with self.subTest(n=n):
                validate_vector(vector_fixture(n, threads), n, threads)

    def test_bad_metadata(self):
        for old, new in [("blocks=3", "blocks=2"), ("slots=12", "slots=10"),
                         ("bytes_per_array=40", "bytes_per_array=10"),
                         ("PASS: 10", "PASS: 9")]:
            with self.subTest(new=new), self.assertRaises(RuntimeError):
                validate_vector(vector_fixture().replace(old, new), 10, 4)

    def test_nonfinite_and_wrong_values(self):
        for bad in ["nan", "inf", "-inf", "23"]:
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                validate_vector(vector_fixture().replace("c=22.5", f"c={bad}"), 10, 4)

    def test_missing_and_duplicate(self):
        rows = vector_fixture().splitlines()
        with self.assertRaises(RuntimeError):
            validate_vector("\n".join(rows[:5] + rows[6:]), 10, 4)
        rows[5] = rows[4]
        with self.assertRaises(RuntimeError):
            validate_vector("\n".join(rows), 10, 4)

    def test_empty_must_state_no_work(self):
        with self.assertRaises(RuntimeError):
            validate_vector("PASS: 0 elements", 0, 256)

    def test_extra_output(self):
        with self.assertRaises(RuntimeError):
            validate_vector(vector_fixture() + "\nERROR", 10, 4)


if __name__ == "__main__":
    print("Offline checker fixtures only; no C++ or CUDA execution.")
    unittest.main(verbosity=2)
