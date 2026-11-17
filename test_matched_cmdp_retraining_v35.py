import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_matched_cmdp_retraining import (
    build_plan,
    load_protocol_spec,
    validate_prior_development_handoff,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'configs' / 'matched_cmdp_retraining_v3_5.yaml'


class MatchedCmdpRetrainingV35Tests(unittest.TestCase):
    def prior_payload(self, **overrides):
        payload = {
            'status': 'needs_revision',
            'passed': False,
            'gates': {
                'full_cmdp_training_mechanism_active': True,
                'full_directionally_improves_matched_nocmdp': False,
                'paired_seed_improvement_gate': True,
            },
            'formal_claims_authorized': False,
        }
        payload.update(overrides)
        return payload

    def test_amendment_preserves_gates_and_uses_new_seed_blocks(self):
        spec = load_protocol_spec(SPEC)
        budget = spec['budget_reference']
        pilot = spec['pilot']
        self.assertEqual(budget['training_episodes'], 500)
        self.assertEqual(budget['evaluation_episodes'], 150)
        self.assertEqual(budget['budget_quantile'], 0.60)
        self.assertEqual(
            budget['reference_gates']['min_budget_exceedance_rate'], 30.0
        )
        self.assertEqual(
            budget['reference_gates']['max_budget_exceedance_rate'], 50.0
        )
        self.assertEqual(pilot['training_episodes'], 800)
        self.assertEqual(pilot['evaluation_episodes'], 150)
        self.assertEqual(
            pilot['directional_gates'][
                'minimum_cmdp_discounted_cost_reduction'
            ],
            0.10,
        )
        self.assertEqual(
            pilot['directional_gates'][
                'minimum_budget_exceedance_reduction_percentage_points'
            ],
            2.0,
        )
        ranges = spec['seed_ranges']
        for left, right in zip(ranges, ranges[1:]):
            self.assertLess(left['end'], right['start'])
        self.assertGreaterEqual(ranges[0]['start'], 710000)

    def test_prior_handoff_accepts_only_declared_v34_failure_pattern(self):
        spec = load_protocol_spec(SPEC)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pilot_result.json'
            path.write_text(
                json.dumps(self.prior_payload()), encoding='utf-8'
            )
            spec['prior_development_result_path'] = str(path)
            handoff = validate_prior_development_handoff(spec)
        self.assertEqual(handoff['status'], 'needs_revision')
        self.assertFalse(handoff['passed'])
        self.assertFalse(handoff['formal_claims_authorized'])

    def test_prior_handoff_rejects_changed_directional_gate(self):
        spec = load_protocol_spec(SPEC)
        payload = self.prior_payload()
        payload['gates'][
            'full_directionally_improves_matched_nocmdp'
        ] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pilot_result.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            spec['prior_development_result_path'] = str(path)
            with self.assertRaisesRegex(
                RuntimeError, 'prior development gate mismatch'
            ):
                validate_prior_development_handoff(spec)

    def test_plan_records_prior_result_without_authorizing_claims(self):
        spec = load_protocol_spec(SPEC)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            prior = directory / 'pilot_result.json'
            prior.write_text(
                json.dumps(self.prior_payload()), encoding='utf-8'
            )
            calibration = directory / 'calibration_result.json'
            calibration.write_text(json.dumps({
                'status': 'candidate_selected',
                'passed': True,
                'selected_candidate_id': 'relaxed_y006_o003',
                'selected_parameters': {
                    'threshold_yellow': 0.06,
                    'threshold_orange': 0.03,
                },
                'formal_claims_authorized': False,
                'retraining_required_before_confirmatory_use': True,
            }), encoding='utf-8')
            spec['prior_development_result_path'] = str(prior)
            spec['calibration_result_path'] = str(calibration)
            plan = build_plan(spec)
        self.assertFalse(plan['formal_claims_authorized'])
        self.assertEqual(
            plan['prior_development_handoff']['status'], 'needs_revision'
        )
        self.assertFalse(
            plan['prior_development_handoff']['formal_claims_authorized']
        )


if __name__ == '__main__':
    unittest.main()
