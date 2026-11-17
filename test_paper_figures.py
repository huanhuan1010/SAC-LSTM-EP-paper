import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.paper_figures as paper_figures


class PaperFigureLogTests(unittest.TestCase):
    def test_single_episode_log_is_loaded_as_one_dimensional_array(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_dir = root / 'SAC-LSTM-EP' / 'seed_42'
            log_dir.mkdir(parents=True)
            (log_dir / 'log.csv').write_text(
                'episode,reward,energy,tasks_completed,battery_depleted\n'
                '0,-1.0,100.0,2,0\n',
                encoding='utf-8',
            )
            with patch.object(paper_figures, 'RUN_ROOT', str(root)), \
                    patch.object(
                        paper_figures,
                        'EVAL_PACKAGE',
                        {'train_seeds': [42]},
                    ):
                data = paper_figures.load_log('SAC-LSTM-EP')

            self.assertIsNotNone(data)
            self.assertEqual(data.shape, (1,))
            self.assertEqual(float(data[0]['reward']), -1.0)


if __name__ == '__main__':
    unittest.main()
