"""Offline tests of the output checker. These are NOT CUDA execution tests."""

import unittest

from verify_cuda_intro import validate_chapter03_outputs, validate_chapter04_outputs


def valid_outputs():
    return {
        "hello_one.cu": "CPU: before launch\nGPU: hello from one thread\nCPU: GPU work finished\n",
        "hello.cu": "CPU: before launch\n" + "\n".join(
            f"GPU: block={b} thread={t} block_size=4 grid_size=2"
            for b in range(2) for t in range(4)
        ) + "\nCPU: GPU work finished\n",
        "launch_order.cu": "CPU: A before launch\nCPU: B launch call returned\n"
            + "\n".join(f"GPU: thread={t}" for t in range(4))
            + "\nCPU: C synchronization succeeded\n",
        "scalar_tasks.cu": "\n".join(
            f"GPU: thread={t} input={10+t} output={2*(10+t)}" for t in range(4)
        ) + "\nCPU: scalar tasks finished\n",
    }


class OutputCheckerTests(unittest.TestCase):
    def test_valid(self):
        validate_chapter03_outputs(valid_outputs())

    def test_gpu_records_may_reverse(self):
        outputs = valid_outputs()
        lines = outputs["hello.cu"].splitlines()
        outputs["hello.cu"] = "\n".join([lines[0], *reversed(lines[1:-1]), lines[-1]])
        validate_chapter03_outputs(outputs)

    def test_gpu_records_may_precede_host_b(self):
        outputs = valid_outputs()
        outputs["launch_order.cu"] = (
            "CPU: A before launch\nGPU: thread=3\nGPU: thread=1\n"
            "CPU: B launch call returned\nGPU: thread=0\nGPU: thread=2\n"
            "CPU: C synchronization succeeded\n"
        )
        validate_chapter03_outputs(outputs)

    def test_duplicate_thread_rejected(self):
        outputs = valid_outputs()
        outputs["launch_order.cu"] = outputs["launch_order.cu"].replace(
            "GPU: thread=3", "GPU: thread=2"
        )
        with self.assertRaises(RuntimeError):
            validate_chapter03_outputs(outputs)

    def test_missing_thread_rejected(self):
        outputs = valid_outputs()
        outputs["launch_order.cu"] = outputs["launch_order.cu"].replace("GPU: thread=3\n", "")
        with self.assertRaises(RuntimeError):
            validate_chapter03_outputs(outputs)

    def test_wrong_numeric_result_rejected(self):
        outputs = valid_outputs()
        outputs["scalar_tasks.cu"] = outputs["scalar_tasks.cu"].replace("output=26", "output=25")
        with self.assertRaises(RuntimeError):
            validate_chapter03_outputs(outputs)

    def test_completion_must_follow_device_output(self):
        outputs = valid_outputs()
        outputs["hello_one.cu"] = (
            "CPU: before launch\nCPU: GPU work finished\nGPU: hello from one thread\n"
        )
        with self.assertRaises(RuntimeError):
            validate_chapter03_outputs(outputs)

    def test_missing_program_rejected(self):
        outputs = valid_outputs()
        del outputs["hello.cu"]
        with self.assertRaises(RuntimeError):
            validate_chapter03_outputs(outputs)

    def test_unexpected_output_rejected(self):
        outputs = valid_outputs()
        outputs["hello_one.cu"] += "UNEXPECTED ERROR\n"
        with self.assertRaises(RuntimeError):
            validate_chapter03_outputs(outputs)


def valid_chapter04_outputs():
    layout = []
    for block in range(2):
        for warp, count in ((0, 32), (1, 8)):
            for lane in range(count):
                tid = warp * 32 + lane
                layout.append(
                    f"block={block} tid={tid} global={block*40+tid} warp={warp} lane={lane}"
                )
    branch = []
    for mode in range(2):
        for warp in range(2):
            for lane in range(32):
                tid = warp * 32 + lane
                path = ("A" if warp == 0 else "B") if mode == 0 else ("AB"[lane % 2])
                value = tid + {"A": 100, "B": 200}[path]
                branch.append(f"mode={mode} tid={tid} warp={warp} lane={lane} path={path} value={value}")
    return {
        "inspect_layout.cu": "\n".join(layout)
            + "\nPASS launch: 2 blocks, 40 threads each; inspect 80 records.\n",
        "branch_paths.cu": "\n".join(branch)
            + "\nPASS launch: two modes; inspect 128 records.\n",
    }


class Chapter04CheckerTests(unittest.TestCase):
    def test_valid(self):
        validate_chapter04_outputs(valid_chapter04_outputs())

    def test_reordered_records_valid(self):
        outputs = valid_chapter04_outputs()
        lines = outputs["inspect_layout.cu"].splitlines()
        outputs["inspect_layout.cu"] = "\n".join([*reversed(lines[:-1]), lines[-1]])
        lines = outputs["branch_paths.cu"].splitlines()
        outputs["branch_paths.cu"] = "\n".join(
            [*reversed(lines[:64]), *reversed(lines[64:128]), lines[-1]]
        )
        validate_chapter04_outputs(outputs)

    def test_invalid_layout_records(self):
        for original, replacement in (
            ("global=40 warp=0 lane=0", "global=40 warp=0 lane=8"),
            ("global=40 warp=0 lane=0", "global=40 warp=1 lane=0"),
            ("block=1 tid=39 global=79 warp=1 lane=7\n", ""),
            ("global=79", "global=80"),
        ):
            with self.subTest(replacement=replacement):
                outputs = valid_chapter04_outputs()
                outputs["inspect_layout.cu"] = outputs["inspect_layout.cu"].replace(original, replacement)
                with self.assertRaises(RuntimeError):
                    validate_chapter04_outputs(outputs)

    def test_duplicate_rejected(self):
        outputs = valid_chapter04_outputs()
        outputs["inspect_layout.cu"] = "block=0 tid=0 global=0 warp=0 lane=0\n" + outputs["inspect_layout.cu"]
        with self.assertRaises(RuntimeError):
            validate_chapter04_outputs(outputs)

    def test_wrong_branch_and_value_rejected(self):
        for replacement in ("path=A value=101", "path=B value=200"):
            with self.subTest(replacement=replacement):
                outputs = valid_chapter04_outputs()
                outputs["branch_paths.cu"] = outputs["branch_paths.cu"].replace("path=B value=201", replacement)
                with self.assertRaises(RuntimeError):
                    validate_chapter04_outputs(outputs)

    def test_reversed_modes_rejected(self):
        outputs = valid_chapter04_outputs()
        lines = outputs["branch_paths.cu"].splitlines()
        outputs["branch_paths.cu"] = "\n".join([*lines[64:128], *lines[:64], lines[-1]])
        with self.assertRaises(RuntimeError):
            validate_chapter04_outputs(outputs)

    def test_missing_program_rejected(self):
        outputs = valid_chapter04_outputs()
        del outputs["inspect_layout.cu"]
        with self.assertRaises(RuntimeError):
            validate_chapter04_outputs(outputs)

    def test_missing_marker_rejected(self):
        outputs = valid_chapter04_outputs()
        outputs["inspect_layout.cu"] = "\n".join(outputs["inspect_layout.cu"].splitlines()[:-1])
        with self.assertRaises(RuntimeError):
            validate_chapter04_outputs(outputs)

    def test_warp_partition_math(self):
        for threads, warps, last in ((1, 1, 1), (32, 1, 32), (40, 2, 8),
                                     (80, 3, 16), (100, 4, 4), (256, 8, 32)):
            groups = [list(range(threads))[i:i+32] for i in range(0, threads, 32)]
            self.assertEqual(len(groups), warps)
            self.assertEqual(len(groups[-1]), last)
        identities = {(block, tid // 32) for block in range(2) for tid in range(40)}
        self.assertEqual(len(identities), 4)


if __name__ == "__main__":
    print("Offline checker tests only; no CUDA code is compiled or executed.")
    unittest.main(verbosity=2)
