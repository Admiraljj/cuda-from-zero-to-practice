"""Offline synthetic-output and mapping tests; no GPU execution."""
import unittest
from verify_chapter06 import validate_grid, validate_invert, validate_rgb, validate_pitch


def grid_fixture():
    rows = []
    for row in range(4):
        for col in range(8):
            bx, tx = divmod(col, 4)
            by, ty = divmod(row, 2)
            tail = f"index={row*5+col}" if row < 3 and col < 5 else "skip"
            rows.append(f"block=({bx},{by}) thread=({tx},{ty}) row={row} col={col} {tail}")
    return "\n".join(rows + ["CPU grid model: valid=15 skipped=17"])


class Checkers(unittest.TestCase):
    def test_grid_all_identities(self):
        validate_grid(grid_fixture())

    def test_grid_invalid_column_not_rescued_by_linear_index(self):
        output = grid_fixture().replace("row=0 col=5 skip", "row=0 col=5 index=5")
        with self.assertRaises(RuntimeError):
            validate_grid(output)

    def test_grid_duplicate(self):
        lines = grid_fixture().splitlines()
        lines[1] = lines[0]
        with self.assertRaises(RuntimeError):
            validate_grid("\n".join(lines))

    def test_grid_wrong_stride(self):
        output = grid_fixture().replace("row=2 col=3 index=13", "row=2 col=3 index=9")
        with self.assertRaises(RuntimeError):
            validate_grid(output)

    def test_invert_cases(self):
        lines = [
            "PASS empty: 0 x 3 (no GPU work)",
            "row=0: 255 252 249 246 243",
            "row=1: 238 235 232 229 226",
            "row=2: 221 218 215 212 209",
            "PASS: 5 x 3 image; block=4x2 grid=2x2",
            "PASS: 1 x 1 image; block=16x16 grid=1x1",
            "PASS: 17 x 1 image; block=16x16 grid=2x1",
            "PASS: 1 x 17 image; block=16x16 grid=1x2",
            "PASS: 16 x 16 image; block=16x16 grid=1x1",
            "PASS: 37 x 23 image; block=16x16 grid=3x2",
            "PASS: 37 x 23 image; block=32x8 grid=2x3",
        ]
        validate_invert("\n".join(lines))
        with self.assertRaises(RuntimeError):
            validate_invert("\n".join(lines).replace("grid=2x3", "grid=3x2"))

    def test_rgb_channels(self):
        lines = [f"pixel={i} R={255-i} G={255-(50+i)} B={255-(100+i)}" for i in range(15)]
        lines += ["PASS RGB: 5 x 3 pixels, 45 components"]
        validate_rgb("\n".join(lines))
        with self.assertRaises(RuntimeError):
            validate_rgb("\n".join(lines).replace("G=205 B=155", "G=155 B=205"))

    def test_pitch_independent_and_not_fixed(self):
        rows = "\nrow=0: 0 2 4 6 8\nrow=1: 20 22 24 26 28\nrow=2: 40 42 44 46 48\nPASS pitched scale: 5 x 3"
        for a, b in [(20, 20), (32, 64), (128, 128), (512, 256)]:
            validate_pitch(f"row_bytes=20 input_pitch={a} output_pitch={b}" + rows)
        for a, b in [(16, 64), (32, 0), (21, 32)]:
            with self.assertRaises(RuntimeError):
                validate_pitch(f"row_bytes=20 input_pitch={a} output_pitch={b}" + rows)

    def test_mapping_property_many_shapes(self):
        for w, h in [(1, 1), (5, 3), (17, 1), (1, 17), (16, 16), (37, 23)]:
            for bx, by in [(4, 2), (16, 16), (32, 8)]:
                gx, gy = (w+bx-1)//bx, (h+by-1)//by
                indices = [r*w+c for r in range(gy*by) for c in range(gx*bx) if r<h and c<w]
                self.assertEqual(indices, list(range(w*h)))
                self.assertEqual(len(set(indices)), w*h)

    def test_warp_uses_block_width(self):
        self.assertEqual((1*16+3)//32, 0)
        self.assertEqual((1*16+3)%32, 19)
        self.assertNotEqual((1*37+3)%32, 19)


if __name__ == "__main__":
    print("Offline fixtures and mathematical models only; no CUDA execution.")
    unittest.main(verbosity=2)
