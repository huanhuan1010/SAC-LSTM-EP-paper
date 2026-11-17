import tempfile
import unittest
from pathlib import Path

from utils.checkpoints import (
    load_checkpoint_pair,
    pair_manifest_path,
    paired_lstm_checkpoint,
    save_checkpoint_pair,
)


class DummyAgent:
    def __init__(self, payload=b'policy-v1'):
        self.payload = payload
        self.loaded = None

    def save(self, path):
        Path(path).write_bytes(self.payload)

    def load(self, path):
        self.loaded = Path(path).read_bytes()


class DummyPredictor:
    def __init__(self, payload=b'lstm-v1'):
        self.payload = payload
        self.loaded = None
        self.eval_called = False

    def save(self, path):
        Path(path).write_bytes(self.payload)

    def load(self, path):
        self.loaded = Path(path).read_bytes()
        return self

    def eval(self):
        self.eval_called = True
        return self


class CheckpointPairTests(unittest.TestCase):
    def test_pair_round_trip_loads_matching_policy_and_lstm(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / 'agent.pt'
            saved = save_checkpoint_pair(
                DummyAgent(), DummyPredictor(), policy,
                algorithm='SAC-LSTM-EP',
                metadata={'state_dim': 52, 'horizon': 5},
            )

            self.assertEqual(Path(saved['lstm']), paired_lstm_checkpoint(policy))
            self.assertTrue(pair_manifest_path(policy).is_file())

            agent = DummyAgent(payload=b'unused')
            predictor = DummyPredictor(payload=b'unused')
            loaded = load_checkpoint_pair(
                agent,
                predictor,
                policy,
                expected_algorithm='SAC-LSTM-EP',
            )
            self.assertEqual(agent.loaded, b'policy-v1')
            self.assertEqual(predictor.loaded, b'lstm-v1')
            self.assertTrue(predictor.eval_called)
            self.assertIsNotNone(loaded['manifest'])

    def test_tampered_lstm_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / 'agent.pt'
            save_checkpoint_pair(
                DummyAgent(), DummyPredictor(), policy,
                algorithm='SAC-LSTM-EP',
            )
            paired_lstm_checkpoint(policy).write_bytes(b'tampered')

            with self.assertRaisesRegex(ValueError, 'LSTM checkpoint hash'):
                load_checkpoint_pair(
                    DummyAgent(), DummyPredictor(), policy,
                    expected_algorithm='SAC-LSTM-EP',
                )

    def test_missing_lstm_is_rejected_instead_of_using_random_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / 'agent.pt'
            save_checkpoint_pair(
                DummyAgent(), DummyPredictor(), policy,
                algorithm='SAC-LSTM-EP',
            )
            paired_lstm_checkpoint(policy).unlink()

            with self.assertRaises(FileNotFoundError):
                load_checkpoint_pair(
                    DummyAgent(), DummyPredictor(), policy,
                    expected_algorithm='SAC-LSTM-EP',
                )

    def test_algorithm_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / 'agent.pt'
            save_checkpoint_pair(
                DummyAgent(), DummyPredictor(), policy,
                algorithm='DDPG',
            )

            with self.assertRaisesRegex(ValueError, 'algorithm mismatch'):
                load_checkpoint_pair(
                    DummyAgent(), DummyPredictor(), policy,
                    expected_algorithm='SAC-LSTM-EP',
                )

    def test_environment_metadata_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / 'agent.pt'
            save_checkpoint_pair(
                DummyAgent(), DummyPredictor(), policy,
                algorithm='SAC-LSTM-EP',
                metadata={'state_dim': 52, 'horizon': 5},
            )

            with self.assertRaisesRegex(ValueError, 'state_dim'):
                load_checkpoint_pair(
                    DummyAgent(), DummyPredictor(), policy,
                    expected_algorithm='SAC-LSTM-EP',
                    expected_metadata={'state_dim': 27, 'horizon': 5},
                )

    def test_exact_source_fingerprint_waiver_is_recorded(self):
        old_source = 'a' * 64
        current_source = 'b' * 64
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / 'agent.pt'
            save_checkpoint_pair(
                DummyAgent(), DummyPredictor(), policy,
                algorithm='SAC-LSTM-EP',
                metadata={
                    'source_fingerprint': old_source,
                    'state_dim': 52,
                    'horizon': 5,
                },
            )

            loaded = load_checkpoint_pair(
                DummyAgent(), DummyPredictor(), policy,
                expected_algorithm='SAC-LSTM-EP',
                expected_metadata={
                    'source_fingerprint': current_source,
                    'state_dim': 52,
                    'horizon': 5,
                },
                allowed_source_fingerprints=[old_source],
            )

            self.assertEqual(len(loaded['metadata_waivers']), 1)
            self.assertEqual(
                loaded['metadata_waivers'][0]['saved'], old_source
            )
            self.assertEqual(
                loaded['metadata_waivers'][0]['expected'], current_source
            )

    def test_source_waiver_does_not_bypass_other_metadata_mismatch(self):
        old_source = 'a' * 64
        current_source = 'b' * 64
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / 'agent.pt'
            save_checkpoint_pair(
                DummyAgent(), DummyPredictor(), policy,
                algorithm='SAC-LSTM-EP',
                metadata={
                    'source_fingerprint': old_source,
                    'state_dim': 52,
                    'horizon': 5,
                },
            )

            with self.assertRaisesRegex(ValueError, 'state_dim'):
                load_checkpoint_pair(
                    DummyAgent(), DummyPredictor(), policy,
                    expected_algorithm='SAC-LSTM-EP',
                    expected_metadata={
                        'source_fingerprint': current_source,
                        'state_dim': 27,
                        'horizon': 5,
                    },
                    allowed_source_fingerprints=[old_source],
                )


if __name__ == '__main__':
    unittest.main()
