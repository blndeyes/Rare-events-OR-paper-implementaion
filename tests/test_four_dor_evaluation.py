import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from or_video_reproduction.evaluation.four_dor import build_four_dor_evaluation


class FakeSam2Runner:
    def __init__(self) -> None:
        self.calls = []

    def run_points(self, video: Path, prompts: Path, output: Path) -> None:
        self.calls.append((video, prompts, output))


class FourDorEvaluationTests(unittest.TestCase):
    def test_pairs_reference_and_generated_masks_by_id(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inference = root / "inference.json"
            pairs = root / "pairs.json"
            inference.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "samples": [
                            {
                                "id": "clip-a",
                                "reference_video": "target.mp4",
                                "generated_video": "generated.mp4",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            pairs.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "samples": [
                            {
                                "id": "clip-a",
                                "labels": "reference.npz",
                                "prompt_manifest": "prompt.json",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            runner = FakeSam2Runner()

            report = build_four_dor_evaluation(
                inference,
                pairs,
                output_root=root / "evaluation",
                sam2_runner=runner,
                require_six=False,
            )

        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(report["paper_table1_metrics"], ["fvd", "ssim", "psnr", "lpips"])
        self.assertEqual(report["samples"][0]["reference_masks"], str(root / "reference.npz"))

    def test_table1_contract_requires_exactly_six(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {"schema_version": 1, "samples": []}
            (root / "a.json").write_text(json.dumps(payload), encoding="utf-8")
            (root / "b.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly six"):
                build_four_dor_evaluation(
                    root / "a.json",
                    root / "b.json",
                    output_root=root / "out",
                    sam2_runner=FakeSam2Runner(),
                )


if __name__ == "__main__":
    unittest.main()
