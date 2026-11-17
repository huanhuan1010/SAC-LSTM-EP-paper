import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.run_formal_multiseed_pareto import (
    build_plan,
    inspect_prerequisites,
    load_formal_protocol,
    validate_formal_seed_protocol,
)
from utils.config import load_config
from utils.provenance import source_fingerprint


class FormalMultiseedPlanTests(unittest.TestCase):
    @staticmethod
    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def test_old_adjacent_seed_example_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_formal_seed_protocol(
                [42, 43, 44],
                training_episodes=2000,
                evaluation_seed=1042,
                evaluation_episodes=100,
                minimum_training_seeds=3,
            )

    def test_formal_protocol_uses_disjoint_episode_ranges(self):
        protocol = load_formal_protocol(
            'configs/formal_multiseed_pareto_v2.yaml'
        )
        ranges = protocol['seed_ranges']
        for index, first in enumerate(ranges):
            for second in ranges[index + 1:]:
                self.assertTrue(
                    first['end'] < second['start']
                    or second['end'] < first['start']
                )
        self.assertEqual(len(protocol['training_seeds']), 5)
        self.assertEqual(protocol['evaluation_episodes'], 100)
        self.assertIn('SAC-LSTM-EP', protocol['algorithms'])
        self.assertIn('SAC-LSTM-EP-SOC', protocol['algorithms'])
        self.assertIn('SAC-LSTM-EP-NoCMDP', protocol['algorithms'])
        self.assertFalse(protocol['weighted_composite_score_used'])
        self.assertEqual(protocol['run_name'], 'paper-formal-v2')
        self.assertEqual(
            Path(protocol['calibration_spec_path']).name,
            'safety_cmdp_calibration_v2.yaml',
        )
        self.assertEqual(
            Path(protocol['lstm_validation_spec_path']).name,
            'lstm_prediction_validation_formal_v2.yaml',
        )

    def test_plan_commands_follow_v2_protocol_and_are_linux_safe(self):
        protocol = load_formal_protocol(
            'configs/formal_multiseed_pareto_v2.yaml'
        )
        base_config = load_config(protocol['base_config_path'])
        prerequisites = {
            'ready': False,
            'checks': {},
            'details': {},
            'calibrated_config_path': None,
        }
        plan = build_plan(protocol, base_config, prerequisites)
        commands = plan['exact_commands']
        self.assertIn('configs/safety_cmdp_calibration_v2.yaml', commands['calibration'])
        self.assertIn(
            'configs/lstm_prediction_validation_formal_v2.yaml',
            commands['lstm_validation'],
        )
        self.assertIn('configs/formal_multiseed_pareto_v2.yaml', commands['formal_run'])
        for command in commands.values():
            self.assertNotIn('\\', command)

    def test_v2_preregistered_seed_ranges_are_disjoint(self):
        config_root = Path('configs')
        with (config_root / 'lstm_prediction_validation_formal_v2.yaml').open(
            'r', encoding='utf-8'
        ) as stream:
            lstm = yaml.safe_load(stream)
        with (config_root / 'safety_cmdp_calibration_v2.yaml').open(
            'r', encoding='utf-8'
        ) as stream:
            calibration = yaml.safe_load(stream)

        ranges = []

        def register(label, start, count):
            ranges.append((label, int(start), int(start) + int(count) - 1))

        for split in ('train', 'validation', 'test'):
            register(
                f'lstm_{split}',
                lstm['split_environment_seed_starts'][split],
                lstm['split_episode_counts'][split],
            )
        for model_seed in lstm['model_seeds']:
            register(f'lstm_model_{model_seed}', model_seed, 1)
        for start in calibration['training_seed_bases']:
            register(
                f'calibration_training_{start}',
                start,
                calibration['training_episodes'],
            )
        register(
            'calibration_selection',
            calibration['calibration_seed'],
            calibration['calibration_episodes'],
        )
        register(
            'calibration_feasibility',
            calibration['cmdp_feasibility_seed'],
            calibration['cmdp_feasibility_episodes'],
        )
        for formal_range in load_formal_protocol(
            config_root / 'formal_multiseed_pareto_v2.yaml'
        )['seed_ranges']:
            if formal_range['label'] != 'final_evaluation':
                register(
                    f"formal_{formal_range['label']}",
                    formal_range['start'],
                    formal_range['count'],
                )

        for index, (first_label, first_start, first_end) in enumerate(ranges):
            for second_label, second_start, second_end in ranges[index + 1:]:
                self.assertTrue(
                    first_end < second_start or second_end < first_start,
                    f'{first_label} overlaps {second_label}',
                )

        self.assertEqual(
            (
                calibration['final_evaluation_seed'],
                calibration['final_evaluation_episodes'],
            ),
            (260000, 100),
        )
        formal = load_formal_protocol(
            config_root / 'formal_multiseed_pareto_v2.yaml'
        )
        self.assertEqual(
            (formal['evaluation_seed'], formal['evaluation_episodes']),
            (
                calibration['final_evaluation_seed'],
                calibration['final_evaluation_episodes'],
            ),
        )

    def test_v1_protocol_defaults_remain_loadable(self):
        protocol = load_formal_protocol(
            'configs/formal_multiseed_pareto.yaml'
        )
        self.assertEqual(protocol['run_name'], 'paper-formal-v1')
        self.assertEqual(
            Path(protocol['calibration_spec_path']).name,
            'safety_cmdp_calibration.yaml',
        )

    def test_formal_v3_draft_is_fail_closed(self):
        protocol = load_formal_protocol(
            'configs/formal_multiseed_pareto_v3_draft.yaml'
        )
        self.assertEqual(protocol['protocol_status'], 'draft')
        base_config = load_config(protocol['base_config_path'])
        prerequisites = inspect_prerequisites(protocol, base_config)
        self.assertFalse(prerequisites['checks']['protocol_frozen'])
        self.assertFalse(prerequisites['ready'])
        plan = build_plan(protocol, base_config, prerequisites)
        self.assertEqual(plan['status'], 'draft_protocol')
        self.assertEqual(
            protocol['evaluation_seed_design'],
            'paired_independent_blocks',
        )
        self.assertEqual(len(protocol['evaluation_seeds']), 5)
        self.assertFalse(protocol['training_seed_is_inferential_unit'])
        self.assertTrue(
            protocol['paired_training_evaluation_block_is_inferential_unit']
        )

    def test_formal_v3_uses_disjoint_paired_evaluation_blocks(self):
        protocol = load_formal_protocol(
            'configs/formal_multiseed_pareto_v3_draft.yaml'
        )
        evaluation_ranges = [
            item for item in protocol['seed_ranges']
            if item['label'].startswith('final_evaluation')
        ]
        self.assertEqual(len(evaluation_ranges), 5)
        self.assertEqual(
            [item['start'] for item in evaluation_ranges],
            protocol['evaluation_seeds'],
        )

    def test_terminal_preflight_can_start_formal_with_lstm_report_only(self):
        base_path = Path('configs/wind_mission_energy_stress_socA.yaml').resolve()
        spec_path = Path('configs/formal_v3_preflight_v3g.yaml').resolve()
        base_config = load_config(str(base_path))
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / 'freeze.json'
            manifest_path.write_text(json.dumps({
                'status': 'frozen',
                'source_fingerprint_lock': source_fingerprint(),
                'protocol_files': {
                    'environment': {'sha256': self.sha256(base_path)},
                    'formal': {'sha256': self.sha256(spec_path)},
                },
            }), encoding='utf-8')
            protocol = {
                'protocol_status': 'frozen',
                'source_fingerprint_lock': source_fingerprint(),
                'calibration_mode': 'preflight_frozen',
                'lstm_validation_mode': 'report_only',
                'preflight_freeze_manifest_path': str(manifest_path),
                'confirmatory_direction_was_not_used_for_entry': True,
                'base_config_path': str(base_path),
                'spec_file': str(spec_path),
                'spec_sha256': self.sha256(spec_path),
                'calibration_run_name': 'unused-terminal-calibration',
                'lstm_validation_run_name': 'missing-report-only-lstm',
                'evaluation_seeds': [600000],
                'evaluation_seed': None,
                'evaluation_episodes': 10,
            }
            prerequisites = inspect_prerequisites(protocol, base_config)

        self.assertTrue(prerequisites['ready'])
        self.assertEqual(
            prerequisites['calibrated_config_path'], str(base_path)
        )
        self.assertFalse(
            prerequisites['checks']['lstm_validation_result_exists']
        )
        self.assertNotIn(
            'lstm_validation_result_exists',
            prerequisites['blocking_checks'],
        )

    def test_formal_v3_seed_ranges_do_not_overlap_v3_prerequisites(self):
        protocol = load_formal_protocol(
            'configs/formal_multiseed_pareto_v3_draft.yaml'
        )
        with Path(
            'configs/safety_cmdp_calibration_v3a_draft.yaml'
        ).open('r', encoding='utf-8') as stream:
            calibration = yaml.safe_load(stream)
        with Path(
            'configs/lstm_prediction_validation_formal_v3_draft.yaml'
        ).open('r', encoding='utf-8') as stream:
            lstm = yaml.safe_load(stream)

        ranges = list(protocol['seed_ranges'])
        for start in calibration['training_seed_bases']:
            ranges.append({
                'label': f'calibration_training_{start}',
                'start': int(start),
                'end': int(start) + int(calibration['training_episodes']) - 1,
            })
        for label, start, count in (
            ('calibration_selection', calibration['calibration_seed'],
             calibration['calibration_episodes']),
            ('calibration_feasibility', calibration['cmdp_feasibility_seed'],
             calibration['cmdp_feasibility_episodes']),
        ):
            ranges.append({
                'label': label,
                'start': int(start),
                'end': int(start) + int(count) - 1,
            })
        for split, start in lstm['split_environment_seed_starts'].items():
            count = lstm['split_episode_counts'][split]
            ranges.append({
                'label': f'lstm_{split}',
                'start': int(start),
                'end': int(start) + int(count) - 1,
            })
        for seed in lstm['model_seeds']:
            ranges.append({
                'label': f'lstm_model_{seed}',
                'start': int(seed),
                'end': int(seed),
            })

        for index, first in enumerate(ranges):
            for second in ranges[index + 1:]:
                self.assertTrue(
                    first['end'] < second['start']
                    or second['end'] < first['start'],
                    f"{first['label']} overlaps {second['label']}",
                )


if __name__ == '__main__':
    unittest.main()
