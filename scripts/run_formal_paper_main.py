"""Run the frozen confirmatory comparison for the paper-main method.

The development smoke is an eligibility prerequisite only.  Its effect sizes
are never copied into this confirmatory result and the final evaluation set is
not used for model, threshold, or checkpoint selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.compare import (  # noqa: E402
    run_multiseed_comparison,
    validate_multiseed_episode_ranges,
)
from utils.algorithm_profiles import (  # noqa: E402
    algorithm_profile,
    paper_algorithm_label,
)
from utils.config import load_config  # noqa: E402
from utils.provenance import (  # noqa: E402
    environment_fingerprint,
    source_fingerprint,
    write_json_atomic,
)


REQUIRED_SMOKE_ALGORITHMS = {
    'SAC-LSTM-EP-PaperMain',
    'SAC-LSTM-EP-NoPrediction',
    'SAC-LSTM-EP-FixedSOC',
    'SAC-LSTM-EP-CMDP',
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_formal_spec(path):
    spec_path = Path(path).resolve()
    with spec_path.open('r', encoding='utf-8') as stream:
        spec = yaml.safe_load(stream)
    if not isinstance(spec, dict):
        raise ValueError('formal paper-main spec must be a mapping')
    if spec.get('protocol_status') != 'frozen_formal':
        raise ValueError('protocol_status must be frozen_formal')
    if bool(spec.get('development_only', True)):
        raise ValueError('formal protocol must set development_only=false')

    def resolve(value):
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = spec_path.parent / candidate
        return candidate.resolve()

    algorithms = [str(value) for value in spec['algorithms']]
    if len(algorithms) != len(set(algorithms)):
        raise ValueError('formal algorithm list contains duplicates')
    for algorithm in algorithms:
        algorithm_profile(algorithm)

    minimum_training_seeds = int(spec.get('minimum_training_seeds', 5))
    if minimum_training_seeds < 5:
        raise ValueError('formal comparison requires at least five seeds')
    training_seeds = validate_multiseed_episode_ranges(
        spec['training_seeds'],
        spec['training_episodes'],
        None,
        spec['evaluation_episodes'],
        evaluation_seeds=spec['evaluation_seeds'],
        minimum_unique_seeds=minimum_training_seeds,
    )
    resolved = dict(spec)
    resolved.update({
        'spec_path': str(spec_path),
        'base_config_path': str(resolve(spec['base_config'])),
        'smoke_prerequisite_path': str(resolve(spec['smoke_prerequisite'])),
        'algorithms': algorithms,
        'training_seeds': training_seeds,
        'evaluation_seeds': [int(x) for x in spec['evaluation_seeds']],
        'minimum_training_seeds': minimum_training_seeds,
    })
    return resolved


def validate_smoke_prerequisite(spec, base_fingerprint, prerequisite=None):
    path = Path(spec['smoke_prerequisite_path'])
    if prerequisite is None:
        if not path.is_file():
            raise FileNotFoundError(f'missing completed smoke result: {path}')
        with path.open('r', encoding='utf-8') as stream:
            prerequisite = json.load(stream)
    checks = {
        'smoke_completed': prerequisite.get('status') == 'complete',
        'smoke_remains_development_only': (
            prerequisite.get('formal_claims_authorized') is False
        ),
        'environment_matches': (
            prerequisite.get('environment_fingerprint') == base_fingerprint
        ),
        'required_controls_present': REQUIRED_SMOKE_ALGORITHMS.issubset(
            set(prerequisite.get('algorithms', []))
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            'formal smoke prerequisite failed: ' + ', '.join(failed)
        )
    return {
        'path': str(path),
        'sha256': file_sha256(path) if path.is_file() else 'test-fixture',
        'checks': checks,
        'development_effects_are_not_formal_evidence': True,
    }


def build_plan(spec, prerequisite=None):
    base_config = load_config(spec['base_config_path'])
    if base_config.get('policy_action_mode') != 'offloading_only':
        raise ValueError('formal paper-main study requires offloading_only mode')
    base_fingerprint = environment_fingerprint(base_config)
    smoke = validate_smoke_prerequisite(
        spec, base_fingerprint, prerequisite=prerequisite
    )
    return {
        'status': 'planned',
        'protocol_status': 'frozen_formal',
        'development_only': False,
        'formal_claims_authorized': False,
        'final_evaluation_is_used_for_selection': False,
        'run_name': str(spec['run_name']),
        'spec_path': spec['spec_path'],
        'spec_sha256': file_sha256(spec['spec_path']),
        'runner_sha256': file_sha256(__file__),
        'base_config_path': spec['base_config_path'],
        'environment_fingerprint': base_fingerprint,
        'source_fingerprint': source_fingerprint(),
        'smoke_prerequisite': smoke,
        'algorithms': list(spec['algorithms']),
        'paper_labels': {
            name: paper_algorithm_label(name) for name in spec['algorithms']
        },
        'profile_contracts': {
            name: asdict(algorithm_profile(name)) for name in spec['algorithms']
        },
        'training_seeds': list(spec['training_seeds']),
        'training_episodes': int(spec['training_episodes']),
        'evaluation_seeds': list(spec['evaluation_seeds']),
        'evaluation_episodes': int(spec['evaluation_episodes']),
        'minimum_training_seeds': int(spec['minimum_training_seeds']),
        'training_runs': len(spec['algorithms']) * len(spec['training_seeds']),
        'planned_contrasts': list(spec.get('planned_contrasts', [])),
        'confirmatory_interpretation': (
            'Report all registered algorithms and contrasts. Superiority '
            'claims require the preregistered multi-seed statistical analysis.'
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Plan or execute the frozen paper-main comparison'
    )
    parser.add_argument('--spec', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()

    spec = load_formal_spec(args.spec)
    plan = build_plan(spec)
    root = (
        PROJECT_ROOT / 'results' / 'formal_paper_main'
        / plan['environment_fingerprint'][:12] / plan['run_name']
    )
    plan_file = root / 'plan.json'
    if not args.execute:
        if plan_file.exists():
            registered = json.loads(plan_file.read_text(encoding='utf-8'))
            if registered != plan:
                raise RuntimeError(
                    'a different formal plan is already registered at this path'
                )
        else:
            write_json_atomic(plan_file, plan)
        print(json.dumps({
            'status': 'planned',
            'plan_file': str(plan_file),
            'training_runs': plan['training_runs'],
            'execute_command': (
                'python scripts/run_formal_paper_main.py --spec '
                f'{args.spec} --execute'
            ),
        }, ensure_ascii=False, indent=2))
        return

    if not plan_file.is_file():
        raise RuntimeError('run plan-only once before formal execution')
    registered = json.loads(plan_file.read_text(encoding='utf-8'))
    if registered != plan:
        raise RuntimeError(
            'formal protocol drift detected; source, spec, environment, or '
            'smoke prerequisite changed after registration'
        )

    result = run_multiseed_comparison(
        config_path=spec['base_config_path'],
        num_train_episodes=int(spec['training_episodes']),
        num_eval_episodes=int(spec['evaluation_episodes']),
        algos=spec['algorithms'],
        seeds=spec['training_seeds'],
        eval_seed=None,
        eval_seeds=spec['evaluation_seeds'],
        run_name=spec['run_name'],
        publish_latest=False,
        resume=not args.no_resume,
        minimum_unique_seeds=int(spec['minimum_training_seeds']),
    )
    comparison_file = (
        PROJECT_ROOT / 'results' / 'experiments'
        / plan['environment_fingerprint'][:12] / plan['run_name']
        / 'eval_results.json'
    )
    if not comparison_file.is_file():
        raise RuntimeError('comparison completed without eval_results.json')
    formal_result = {
        **registered,
        'status': 'complete',
        'comparison_status': result.get('status'),
        'comparison_result_file': str(comparison_file),
        'comparison_result_sha256': file_sha256(comparison_file),
        'formal_estimates_available': True,
        'formal_claims_authorized': False,
        'next_step': 'run preregistered multi-seed statistical analysis',
    }
    result_file = root / 'formal_result.json'
    write_json_atomic(result_file, formal_result)
    print(json.dumps({
        'status': 'complete',
        'formal_result_file': str(result_file),
        'comparison_result_file': str(comparison_file),
        'formal_estimates_available': True,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
