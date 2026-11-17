"""Evaluate completed frozen policies under SOC and task-load scenarios."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.soc_sweep import (  # noqa: E402
    aggregate_seed_summaries,
    evaluate_frozen_policy_with_overrides,
    load_comparison,
)
from utils.provenance import (  # noqa: E402
    environment_fingerprint,
    source_fingerprint,
    write_json_atomic,
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _is_sha256(value):
    value = str(value)
    try:
        return len(value) == 64 and int(value, 16) >= 0
    except ValueError:
        return False


def validate_comparison_handoff(spec):
    """Bind a frozen sweep to one completed development comparison.

    The handoff authorizes only the exact source fingerprint and comparison
    file registered by the upstream pilot.  It is intentionally narrower than
    a user-supplied wildcard waiver and is recorded in both plan and result.
    """
    rule = spec.get('comparison_handoff')
    if rule is None:
        return None
    if not isinstance(rule, dict) or not rule.get('result_file'):
        raise ValueError('comparison_handoff.result_file is required')
    result_path = _project_path(rule['result_file'])
    if not result_path.is_file():
        raise FileNotFoundError(
            f'comparison handoff result not found: {result_path}'
        )
    with result_path.open('r', encoding='utf-8') as stream:
        payload = json.load(stream)

    required_status = str(rule.get('required_status', 'needs_revision'))
    if payload.get('status') != required_status:
        raise RuntimeError(
            'comparison handoff status mismatch: expected '
            f'{required_status}, got {payload.get("status")}'
        )
    if 'required_passed' in rule:
        required_passed = bool(rule['required_passed'])
        if payload.get('passed') is not required_passed:
            raise RuntimeError('comparison handoff pass flag mismatch')
    for gate, expected in dict(rule.get('required_gates', {})).items():
        actual = payload.get('gates', {}).get(gate)
        if actual is not bool(expected):
            raise RuntimeError(
                f'comparison handoff gate mismatch for {gate}: '
                f'expected {bool(expected)}, got {actual}'
            )
    if 'required_formal_claims_authorized' in rule:
        expected = bool(rule['required_formal_claims_authorized'])
        if payload.get('formal_claims_authorized') is not expected:
            raise RuntimeError(
                'comparison handoff formal-claims flag mismatch'
            )

    comparison_value = payload.get('comparison_result_file')
    if not comparison_value:
        raise RuntimeError(
            'comparison handoff does not register comparison_result_file'
        )
    comparison_path = _project_path(comparison_value)
    if not comparison_path.is_file():
        raise FileNotFoundError(
            f'comparison file registered by handoff is missing: '
            f'{comparison_path}'
        )
    configured = spec.get('comparison_file')
    if configured and _project_path(configured) != comparison_path:
        raise RuntimeError(
            'scenario comparison_file differs from comparison handoff'
        )
    with comparison_path.open('r', encoding='utf-8') as stream:
        comparison = json.load(stream)
    if comparison.get('status') != 'complete':
        raise RuntimeError('handoff comparison is not complete')
    saved_source = str(payload.get('source_fingerprint', ''))
    if not _is_sha256(saved_source):
        raise RuntimeError('comparison handoff has invalid source fingerprint')
    if str(comparison.get('source_fingerprint', '')) != saved_source:
        raise RuntimeError(
            'handoff and comparison source fingerprints do not match'
        )
    return {
        'path': str(result_path),
        'sha256': _sha256_file(result_path),
        'status': payload['status'],
        'passed': payload.get('passed'),
        'gates': dict(payload.get('gates', {})),
        'formal_claims_authorized': payload.get(
            'formal_claims_authorized'
        ),
        'authorized_checkpoint_source_fingerprint': saved_source,
        'comparison_file': str(comparison_path),
        'comparison_sha256': _sha256_file(comparison_path),
        'comparison_environment_fingerprint': comparison.get(
            'environment_fingerprint'
        ),
        'comparison_train_seeds': list(comparison.get('train_seeds', ())),
    }


def _frozen_plan_payload(spec_path, spec, prerequisite, handoff):
    return {
        'status': 'planned',
        'protocol_status': str(spec.get(
            'protocol_status', 'development'
        )),
        'development_only': bool(spec.get('development_only', False)),
        'formal_claims_authorized': False,
        'spec_file': str(spec_path),
        'spec_sha256': _sha256_file(spec_path),
        'runner_sha256': _sha256_file(Path(__file__).resolve()),
        'evaluator_sha256': _sha256_file(
            PROJECT_ROOT / 'scripts' / 'soc_sweep.py'
        ),
        'runtime_source_fingerprint': source_fingerprint(),
        'comparison_file': (
            handoff['comparison_file'] if handoff is not None
            else str(spec.get('comparison_file', ''))
        ),
        'algorithms': list(spec['algorithms']),
        'scenarios': dict(spec['scenarios']),
        'contrasts': list(spec.get('contrasts', ())),
        'assessment_mode': spec.get('assessment_mode'),
        'acceptance_criteria': dict(spec.get(
            'acceptance_criteria', {}
        )),
        'locked_control_keys': list(spec.get(
            'locked_control_keys', ()
        )),
        'allowed_checkpoint_source_fingerprints': list(spec.get(
            'allowed_checkpoint_source_fingerprints', ()
        )),
        'evaluation_seed': int(spec['evaluation_seed']),
        'evaluation_episodes_per_seed': int(spec['evaluation_episodes']),
        'prerequisite_assessment': prerequisite,
        'comparison_handoff': handoff,
        'no_training_or_parameter_selection_performed': True,
    }


def _frozen_plan_path(spec):
    root = _project_path(spec.get(
        'output_root', 'results/scenario_sweep/task-offloading-v1'
    ))
    return root / 'plan.json'


def _validate_registered_plan(spec_path, spec, current_plan):
    path = _frozen_plan_path(spec)
    if not path.is_file():
        raise RuntimeError(
            'frozen plan is missing; run once without --execute and review '
            f'{path}'
        )
    with path.open('r', encoding='utf-8') as stream:
        registered = json.load(stream)
    checks = {
        'spec_sha256': _sha256_file(spec_path),
        'runner_sha256': _sha256_file(Path(__file__).resolve()),
        'evaluator_sha256': _sha256_file(
            PROJECT_ROOT / 'scripts' / 'soc_sweep.py'
        ),
        'runtime_source_fingerprint': source_fingerprint(),
    }
    for key, expected in checks.items():
        if registered.get(key) != expected:
            raise RuntimeError(f'frozen plan mismatch for {key}')
    for key in (
        'comparison_file', 'algorithms', 'scenarios', 'contrasts',
        'assessment_mode', 'acceptance_criteria', 'evaluation_seed',
        'evaluation_episodes_per_seed', 'comparison_handoff',
    ):
        if registered.get(key) != current_plan.get(key):
            raise RuntimeError(f'frozen plan content changed for {key}')
    return {'path': str(path), 'sha256': _sha256_file(path)}


def validate_prerequisite_assessment(spec):
    """Validate an optional gate artifact before a frozen-policy sweep."""
    prerequisite = spec.get('prerequisite_assessment')
    if prerequisite is None:
        return None
    if not isinstance(prerequisite, dict) or not prerequisite.get('path'):
        raise ValueError('prerequisite_assessment.path is required')
    path = Path(prerequisite['path'])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f'prerequisite assessment not found: {path}'
        )
    with path.open('r', encoding='utf-8') as stream:
        package = json.load(stream)
    required_status = str(prerequisite.get(
        'required_status', 'ready_for_energy_stress'
    ))
    if package.get('status') != required_status:
        raise RuntimeError(
            'prerequisite assessment status mismatch: expected '
            f'{required_status}, got {package.get("status")}'
        )
    required_amendment = prerequisite.get('required_amendment_id')
    actual_amendment = package.get('protocol_amendment', {}).get('id')
    if (
        required_amendment is not None
        and actual_amendment != required_amendment
    ):
        raise RuntimeError('prerequisite assessment amendment id mismatch')
    return {
        'path': str(path),
        'sha256': _sha256_file(path),
        'status': package['status'],
        'amendment_id': actual_amendment,
        'source_comparison_sha256': package.get(
            'source_artifacts', {}
        ).get('comparison_file', {}).get('sha256'),
    }


def load_scenario_spec(path):
    spec_path = Path(path).resolve()
    with spec_path.open('r', encoding='utf-8') as stream:
        spec = yaml.safe_load(stream)
    if not isinstance(spec, dict) or not spec.get('scenarios'):
        raise ValueError('scenario spec must contain a nonempty scenarios map')
    algorithms = [str(value) for value in spec.get('algorithms', ())]
    if not algorithms or len(algorithms) != len(set(algorithms)):
        raise ValueError('scenario algorithms must be nonempty and unique')
    allowed_sources = [
        str(value) for value in spec.get(
            'allowed_checkpoint_source_fingerprints', ()
        )
    ]
    if len(allowed_sources) != len(set(allowed_sources)):
        raise ValueError('allowed source fingerprints must be unique')
    for value in allowed_sources:
        try:
            valid = len(value) == 64 and int(value, 16) >= 0
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(
                'allowed source fingerprints must be 64-character hex '
                'digests'
            )
    spec['allowed_checkpoint_source_fingerprints'] = allowed_sources
    locked_keys = {str(value) for value in spec.get(
        'locked_control_keys', ()
    )}
    for name, scenario in spec['scenarios'].items():
        if not isinstance(scenario, dict):
            raise ValueError(f'scenario {name!r} must be a mapping')
        overrides = scenario.get('overrides', {})
        if not isinstance(overrides, dict):
            raise ValueError(f'scenario {name!r} overrides must be a mapping')
        changed_controls = sorted(locked_keys.intersection(overrides))
        if changed_controls:
            raise ValueError(
                f'scenario {name!r} changes locked control parameters: '
                f'{changed_controls}'
            )
    contrast_ids = set()
    for contrast in spec.get('contrasts', ()):
        contrast_id = str(contrast['id'])
        if contrast_id in contrast_ids:
            raise ValueError(f'duplicate contrast id: {contrast_id}')
        contrast_ids.add(contrast_id)
        for role in ('treatment', 'control'):
            if str(contrast[role]) not in algorithms:
                raise ValueError(
                    f'contrast {contrast_id!r} {role} is not an algorithm'
                )
    return spec_path, spec


def paired_seed_contrasts(results, contrast_specs, train_seeds):
    """Compute treatment-minus-control effects on matched policy seeds.

    These are descriptive development diagnostics.  Keeping the raw paired
    differences prevents the across-seed standard deviations in the summary
    table from being mistaken for uncertainty on an unpaired comparison.
    """
    contrasts = {}
    contrast_rows = []
    seed_keys = [str(int(value)) for value in train_seeds]
    for scenario_name, scenario in results.items():
        scenario_contrasts = {}
        for contrast in contrast_specs:
            contrast_id = str(contrast['id'])
            treatment = str(contrast['treatment'])
            control = str(contrast['control'])
            treatment_rows = scenario['algorithms'][treatment]['per_seed']
            control_rows = scenario['algorithms'][control]['per_seed']
            if set(treatment_rows) != set(control_rows):
                raise ValueError(
                    f'unpaired policy seeds for contrast {contrast_id!r}'
                )
            first_seed = seed_keys[0]
            metric_names = sorted(
                set(treatment_rows[first_seed]).intersection(
                    control_rows[first_seed]
                )
            )
            metric_results = {}
            for metric in metric_names:
                paired_values = []
                for seed in seed_keys:
                    treatment_value = treatment_rows[seed][metric]
                    control_value = control_rows[seed][metric]
                    if treatment_value is None or control_value is None:
                        difference = None
                    else:
                        treatment_value = float(treatment_value)
                        control_value = float(control_value)
                        difference = (
                            treatment_value - control_value
                            if math.isfinite(treatment_value)
                            and math.isfinite(control_value)
                            else None
                        )
                    paired_values.append(difference)
                    contrast_rows.append({
                        'scenario': scenario_name,
                        'contrast': contrast_id,
                        'treatment': treatment,
                        'control': control,
                        'train_seed': int(seed),
                        'metric': metric,
                        'difference': difference,
                    })
                finite = [
                    float(value) for value in paired_values
                    if value is not None and math.isfinite(float(value))
                ]
                metric_results[metric] = {
                    'mean_difference': (
                        float(np.mean(finite)) if finite else None
                    ),
                    'std_difference': (
                        float(np.std(finite, ddof=1))
                        if len(finite) > 1 else 0.0
                    ) if finite else None,
                    'paired_values': paired_values,
                    'finite_pair_count': len(finite),
                }
            scenario_contrasts[contrast_id] = {
                'treatment': treatment,
                'control': control,
                'difference_definition': 'treatment_minus_control',
                'metrics': metric_results,
            }
        contrasts[scenario_name] = scenario_contrasts
    return contrasts, contrast_rows


def _summary_mean(results, scenario, algorithm, metric):
    value = results[scenario]['algorithms'][algorithm][
        'summary'
    ][metric]['mean']
    if value is None or not math.isfinite(float(value)):
        raise ValueError(
            f'non-finite summary metric: {scenario}/{algorithm}/{metric}'
        )
    return float(value)


def assess_cmdp_binding_stress(results, contrasts, criteria, train_seeds):
    """Assess whether CMDP changes frozen-policy behavior when risk binds.

    This is a preregistered development gate.  It does not select a policy,
    tune a threshold, or authorize confirmatory effect sizes.
    """
    contrast_id = str(criteria['contrast'])
    treatment = str(criteria['treatment'])
    control = str(criteria['control'])
    binding_thresholds = {
        str(key): float(value) for key, value in dict(
            criteria['binding_minimums']
        ).items()
    }
    primary_maximums = {
        str(key): float(value) for key, value in dict(
            criteria['primary_maximum_treatment_minus_control']
        ).items()
    }
    noninferiority = dict(criteria['noninferiority'])
    minimum_seed_wins = int(criteria.get(
        'minimum_primary_improvement_seed_wins', 2
    ))
    scenario_records = {}
    binding_count = 0
    improved_count = 0
    all_noninferior = True

    for scenario_name in results:
        if contrast_id not in contrasts.get(scenario_name, {}):
            raise ValueError(
                f'missing contrast {contrast_id!r} for {scenario_name}'
            )
        contrast = contrasts[scenario_name][contrast_id]
        if (
            contrast['treatment'] != treatment
            or contrast['control'] != control
        ):
            raise ValueError('CMDP stress contrast roles do not match criteria')

        binding_checks = {
            metric: _summary_mean(
                results, scenario_name, control, metric
            ) >= threshold
            for metric, threshold in binding_thresholds.items()
        }
        binding = any(binding_checks.values())
        binding_count += int(binding)

        metric_records = {}
        per_seed_any_win = [False] * len(train_seeds)
        for metric, maximum in primary_maximums.items():
            metric_result = contrast['metrics'][metric]
            mean_difference = metric_result['mean_difference']
            mean_passed = bool(
                mean_difference is not None
                and float(mean_difference) <= maximum
            )
            seed_passes = []
            for index, value in enumerate(metric_result['paired_values']):
                passed = bool(
                    value is not None and float(value) <= maximum
                )
                seed_passes.append(passed)
                per_seed_any_win[index] = per_seed_any_win[index] or passed
            metric_seed_win_count = int(sum(seed_passes))
            metric_records[metric] = {
                'maximum_treatment_minus_control': maximum,
                'mean_difference': mean_difference,
                'mean_passed': mean_passed,
                'per_seed_passed': seed_passes,
                'seed_win_count': metric_seed_win_count,
                'passed': bool(
                    mean_passed
                    and metric_seed_win_count >= minimum_seed_wins
                ),
            }
        seed_win_count = int(sum(per_seed_any_win))
        passing_metrics = [
            metric for metric, record in metric_records.items()
            if record['passed']
        ]
        improved = bool(
            binding
            and passing_metrics
        )
        improved_count += int(improved)

        physical_difference = float(
            contrast['metrics'][
                'physical_constraint_violation_rate'
            ]['mean_difference']
        )
        safe_return_difference = float(
            contrast['metrics']['safe_return_rate']['mean_difference']
        )
        mission_difference = float(
            contrast['metrics']['mission_success_rate']['mean_difference']
        )
        treatment_completion = _summary_mean(
            results, scenario_name, treatment, 'task_completion_rate'
        )
        control_completion = _summary_mean(
            results, scenario_name, control, 'task_completion_rate'
        )
        completion_retention = (
            treatment_completion / control_completion
            if control_completion > 1e-12
            else (1.0 if treatment_completion >= control_completion else 0.0)
        )
        noninferiority_checks = {
            'physical_constraint_violation': physical_difference <= float(
                noninferiority[
                    'maximum_physical_violation_increase_percentage_points'
                ]
            ),
            'safe_return': safe_return_difference >= float(
                noninferiority[
                    'minimum_safe_return_difference_percentage_points'
                ]
            ),
            'mission_success': mission_difference >= float(
                noninferiority[
                    'minimum_mission_success_difference_percentage_points'
                ]
            ),
            'task_completion_retention': completion_retention >= float(
                noninferiority['minimum_task_completion_retention_ratio']
            ),
        }
        scenario_noninferior = all(noninferiority_checks.values())
        all_noninferior = all_noninferior and scenario_noninferior
        scenario_records[scenario_name] = {
            'binding': binding,
            'binding_checks': binding_checks,
            'primary_metric_checks': metric_records,
            'primary_improvement_seed_wins': seed_win_count,
            'minimum_primary_improvement_seed_wins': minimum_seed_wins,
            'passing_primary_metrics': passing_metrics,
            'improved': improved,
            'noninferiority': {
                'passed': scenario_noninferior,
                'checks': noninferiority_checks,
                'physical_violation_difference_percentage_points': (
                    physical_difference
                ),
                'safe_return_difference_percentage_points': (
                    safe_return_difference
                ),
                'mission_success_difference_percentage_points': (
                    mission_difference
                ),
                'task_completion_retention_ratio': completion_retention,
            },
        }

    minimum_binding = int(criteria['minimum_binding_scenarios'])
    minimum_improved = int(criteria['minimum_improved_scenarios'])
    gates = {
        'stress_is_binding': binding_count >= minimum_binding,
        'full_improves_cmdp_risk_under_binding_stress': (
            improved_count >= minimum_improved
        ),
        'full_retains_safety_and_service': bool(all_noninferior),
    }
    passed = all(gates.values())
    if not gates['stress_is_binding']:
        classification = 'stress_not_binding'
    elif not gates['full_improves_cmdp_risk_under_binding_stress']:
        classification = 'cmdp_policy_effect_not_identified'
    elif not gates['full_retains_safety_and_service']:
        classification = 'risk_reduction_with_unacceptable_tradeoff'
    else:
        classification = 'binding_constraint_robustness_identified'
    return {
        'status': 'passed' if passed else 'needs_revision',
        'passed': passed,
        'development_only': True,
        'formal_claims_authorized': False,
        'classification': classification,
        'gates': gates,
        'binding_scenario_count': binding_count,
        'minimum_binding_scenarios': minimum_binding,
        'improved_binding_scenario_count': improved_count,
        'minimum_improved_scenarios': minimum_improved,
        'scenario_assessments': scenario_records,
        'next_step': (
            'Freeze a separate confirmatory protocol without reusing these '
            'stress episodes for tuning.'
            if passed else
            'Do not start confirmatory training. Use the classification to '
            'decide whether the stress design or CMDP action sensitivity '
            'requires a separately preregistered revision.'
        ),
    }


def run_scenarios(spec_path, comparison_file=None, execute=False):
    resolved_spec_path, spec = load_scenario_spec(spec_path)
    prerequisite = validate_prerequisite_assessment(spec)
    handoff = validate_comparison_handoff(spec)
    plan = _frozen_plan_payload(
        resolved_spec_path, spec, prerequisite, handoff
    )
    if not execute:
        if bool(spec.get('require_frozen_plan', False)):
            plan_path = _frozen_plan_path(spec)
            write_json_atomic(plan_path, plan)
            plan['plan_file'] = str(plan_path)
        return plan

    registered_plan = None
    if bool(spec.get('require_frozen_plan', False)):
        registered_plan = _validate_registered_plan(
            resolved_spec_path, spec, plan
        )

    selected_comparison = (
        comparison_file
        or spec.get('comparison_file')
        or (handoff or {}).get('comparison_file')
    )
    if not selected_comparison:
        raise ValueError(
            'an explicit comparison_file is required for stress evaluation'
        )
    selected_comparison = _project_path(selected_comparison)
    if (
        handoff is not None
        and selected_comparison != Path(handoff['comparison_file'])
    ):
        raise RuntimeError(
            'selected comparison differs from the frozen comparison handoff'
        )
    allowed_sources = list(spec.get(
        'allowed_checkpoint_source_fingerprints', ()
    ))
    if handoff is not None:
        allowed_sources.append(
            handoff['authorized_checkpoint_source_fingerprint']
        )
    allowed_sources = list(dict.fromkeys(allowed_sources))
    source_path, package = load_comparison(
        selected_comparison,
        allowed_source_fingerprints=allowed_sources,
    )
    if (
        handoff is not None
        and _sha256_file(source_path) != handoff['comparison_sha256']
    ):
        raise RuntimeError('comparison changed after the frozen handoff')
    if prerequisite is not None:
        expected_hash = prerequisite['source_comparison_sha256']
        if not expected_hash:
            raise RuntimeError(
                'prerequisite assessment does not register its source '
                'comparison hash'
            )
        actual_hash = _sha256_file(source_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                'comparison file differs from the artifact authorized by '
                'the prerequisite assessment'
            )
    algorithms = [str(value) for value in spec['algorithms']]
    missing = sorted(set(algorithms) - set(package['results']))
    if missing:
        raise ValueError(f'algorithms absent from comparison: {missing}')
    train_seeds = [int(value) for value in package['train_seeds']]
    evaluation_seed = int(spec['evaluation_seed'])
    episodes = int(spec['evaluation_episodes'])

    results = {}
    rows = []
    checkpoint_waivers = []
    for scenario_name, scenario in spec['scenarios'].items():
        overrides = dict(scenario.get('overrides', {}))
        results[scenario_name] = {
            'description': str(scenario.get('description', '')),
            'protocol_role': str(scenario.get(
                'protocol_role', 'diagnostic'
            )),
            'overrides': overrides,
            'environment_fingerprint': environment_fingerprint({
                **package['config'], **overrides,
            }),
            'algorithms': {},
        }
        for algorithm in algorithms:
            per_seed = {}
            for train_seed in train_seeds:
                print(
                    '[scenario-sweep] '
                    f'scenario={scenario_name} algorithm={algorithm} '
                    f'train_seed={train_seed} episodes={episodes}',
                    flush=True,
                )
                summary = evaluate_frozen_policy_with_overrides(
                    package, algorithm, train_seed, overrides, episodes,
                    evaluation_seed,
                    allowed_checkpoint_source_fingerprints=allowed_sources,
                    checkpoint_waiver_collector=checkpoint_waivers,
                )
                print(
                    '[scenario-sweep] completed '
                    f'scenario={scenario_name} algorithm={algorithm} '
                    f'train_seed={train_seed} '
                    'energy_infeasible_rate='
                    f'{summary["short_horizon_energy_infeasible_rate"]:.3f}% '
                    'mission_success_rate='
                    f'{summary["mission_success_rate"]:.3f}%',
                    flush=True,
                )
                per_seed[str(train_seed)] = summary
                rows.append({
                    'scenario': scenario_name,
                    'algorithm': algorithm,
                    'train_seed': train_seed,
                    **summary,
                })
            results[scenario_name]['algorithms'][algorithm] = {
                'per_seed': per_seed,
                'summary': aggregate_seed_summaries(per_seed),
            }

    contrasts, contrast_rows = paired_seed_contrasts(
        results, spec.get('contrasts', ()), train_seeds
    )

    assessment = None
    if spec.get('assessment_mode') == 'cmdp_binding_stress':
        assessment = assess_cmdp_binding_stress(
            results,
            contrasts,
            dict(spec.get('acceptance_criteria', {})),
            train_seeds,
        )

    output_root = _project_path(spec.get(
        'output_root', 'results/scenario_sweep/task-offloading-v1'
    ))
    output_root.mkdir(parents=True, exist_ok=True)
    output = {
        'status': 'complete',
        'protocol_status': str(spec.get('protocol_status', 'development')),
        'development_only': bool(spec.get('development_only', False)),
        'formal_claims_authorized': False,
        'experiment': 'frozen_policy_scenario_sweep',
        'spec_file': str(resolved_spec_path.as_posix()),
        'source_result_file': str(source_path.as_posix()),
        'source_fingerprint': source_fingerprint(),
        'algorithms': algorithms,
        'train_seeds': train_seeds,
        'evaluation_seed': evaluation_seed,
        'evaluation_episodes_per_seed': episodes,
        'paired_episode_seeds_across_algorithms_and_scenarios': True,
        'no_training_or_parameter_selection_performed': True,
        'locked_control_keys': list(spec.get('locked_control_keys', ())),
        'allowed_checkpoint_source_fingerprints': allowed_sources,
        'comparison_source_fingerprint_waivers': list(package.get(
            'runtime_source_fingerprint_waivers', ()
        )),
        'checkpoint_metadata_waivers': checkpoint_waivers,
        'acceptance_criteria': dict(spec.get('acceptance_criteria', {})),
        'prerequisite_assessment': prerequisite,
        'comparison_handoff': handoff,
        'registered_plan': registered_plan,
        'scenarios': results,
        'paired_contrasts': contrasts,
        'hypothesis_assessment': assessment,
        'hypothesis_gate_passed': (
            assessment.get('passed') if assessment is not None else None
        ),
    }
    write_json_atomic(output_root / 'scenario_sweep_results.json', output)
    pd.DataFrame(rows).to_csv(
        output_root / 'scenario_sweep.csv', index=False
    )
    pd.DataFrame(contrast_rows).to_csv(
        output_root / 'paired_contrasts.csv', index=False
    )
    return output


def main():
    parser = argparse.ArgumentParser(
        description='Plan or run frozen-policy SOC/task-load sweeps'
    )
    parser.add_argument('--spec', required=True)
    parser.add_argument('--comparison-file', default=None)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    result = run_scenarios(
        args.spec, comparison_file=args.comparison_file,
        execute=args.execute,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
