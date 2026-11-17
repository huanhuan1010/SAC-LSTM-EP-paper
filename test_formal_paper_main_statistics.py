import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_formal_paper_main import analyze, paired_summary


def sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class FormalPaperMainStatisticsTests(unittest.TestCase):
    def test_paired_summary_uses_five_block_differences(self):
        summary = paired_summary(
            [-0.270818, -0.080199, -0.095379, -0.042229, -0.365920],
            favorable_direction="lower",
        )
        self.assertEqual(summary["n_pairs"], 5)
        self.assertEqual(summary["wins"], 5)
        self.assertEqual(summary["ties"], 0)
        self.assertAlmostEqual(summary["mean_difference"], -0.170909, places=6)
        self.assertLess(summary["ci95_low"], 0.0)
        self.assertGreater(summary["ci95_high"], 0.0)
        self.assertTrue(summary["ci95_includes_zero"])

    def _fixture(self, root):
        train_seeds = [10, 20, 30, 40, 50]
        eval_seeds = [110, 120, 130, 140, 150]
        algorithms = ("Full", "Control")
        results = {}
        for offset, algorithm in enumerate(algorithms):
            per_seed = {}
            summaries = {}
            for index, seed in enumerate(train_seeds):
                per_seed[str(seed)] = {"evaluation_seed": eval_seeds[index]}
                summaries[str(seed)] = {
                    "tasks": 10.0 + index + offset,
                    "system_energy": 1000.0 + 10.0 * offset + index,
                    "physical_constraint_violation_rate": 5.0 - offset,
                    "mission_success_rate": 80.0 + offset,
                }
            results[algorithm] = {
                "per_seed": per_seed,
                "seed_summaries": summaries,
                "summary": {},
            }
        comparison = {
            "status": "complete",
            "source_fingerprint": "source",
            "environment_fingerprint": "environment",
            "evaluation_seed_design": "paired_independent_blocks",
            "evaluation_seeds": eval_seeds,
            "results": results,
        }
        comparison_path = root / "eval_results.json"
        comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
        formal = {
            "status": "complete",
            "protocol_status": "frozen_formal",
            "formal_estimates_available": True,
            "source_fingerprint": "source",
            "environment_fingerprint": "environment",
            "comparison_result_sha256": sha256(comparison_path),
            "training_seeds": train_seeds,
            "evaluation_seeds": eval_seeds,
            "planned_contrasts": [{
                "id": "full_vs_control",
                "treatment": "Full",
                "control": "Control",
                "purpose": "fixture",
            }],
        }
        spec = {
            "protocol_status": "post_result_paired_estimation_v1",
            "expected_seed_count": 5,
            "contrast_ids": ["full_vs_control"],
            "claim_boundary": "descriptive only",
            "metrics": [
                {
                    "id": "tasks",
                    "label": "Tasks",
                    "unit": "tasks/episode",
                    "favorable_direction": "higher",
                    "display_scale": 1.0,
                },
                {
                    "id": "system_energy",
                    "label": "Energy",
                    "unit": "kJ/episode",
                    "favorable_direction": "lower",
                    "display_scale": 0.001,
                },
                {
                    "id": "physical_constraint_violation_rate",
                    "label": "Violation",
                    "unit": "percentage points",
                    "favorable_direction": "lower",
                    "display_scale": 1.0,
                },
                {
                    "id": "mission_success_rate",
                    "label": "Success",
                    "unit": "percentage points",
                    "favorable_direction": "higher",
                    "display_scale": 1.0,
                },
            ],
        }
        return formal, comparison, comparison_path, spec

    def test_analysis_pairs_by_training_seed_and_scales_energy(self):
        with tempfile.TemporaryDirectory() as directory:
            formal, comparison, path, spec = self._fixture(Path(directory))
            result, rows = analyze(formal, comparison, path, spec)
        self.assertFalse(result["formal_claims_authorized"])
        self.assertEqual(result["sampling_unit"], "paired training-seed/evaluation-seed block")
        self.assertEqual(len(rows), 4)
        energy = next(row for row in rows if row["metric"] == "system_energy")
        self.assertEqual(energy["wins"], 5)
        self.assertTrue(math.isclose(energy["mean_difference"], -0.01))

    def test_analysis_rejects_mismatched_evaluation_block(self):
        with tempfile.TemporaryDirectory() as directory:
            formal, comparison, path, spec = self._fixture(Path(directory))
            comparison["results"]["Full"]["per_seed"]["10"]["evaluation_seed"] = 999
            path.write_text(json.dumps(comparison), encoding="utf-8")
            formal["comparison_result_sha256"] = sha256(path)
            with self.assertRaisesRegex(ValueError, "evaluation block mismatch"):
                analyze(formal, comparison, path, spec)


if __name__ == "__main__":
    unittest.main()
