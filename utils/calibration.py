"""Deterministic selection rules for safety and CMDP calibration."""
from __future__ import annotations

import math
from statistics import NormalDist

from utils.safety_constraints import validate_safety_thresholds


DEFAULT_SAFETY_GATES = {
    'max_physical_constraint_violation_rate': 5.0,
    'max_battery_depletion_rate': 1.0,
    'max_mission_deadline_miss_rate': 10.0,
    'min_safe_return_rate': 90.0,
    # Service-participation floor (v3).  Prevents a candidate from passing
    # every safety gate by returning home immediately: a feasible candidate
    # must actually serve the window.  0.0 disables the gate (v2 behaviour).
    'min_task_completion_rate': 0.0,
    # The permissive default preserves historical protocols.  v3d supplies
    # a tighter value so the hard shield cannot replace the learned policy
    # on nearly every step.
    'max_safety_override_rate': 100.0,
}


DEFAULT_IDENTIFIABILITY_GATES = {
    # At least half of the graded candidates must have meaningfully different
    # outcome signatures.  Metrics are quantized by the tolerance below so
    # floating-point noise cannot make identical controllers look distinct.
    'minimum_distinct_graded_outcome_fraction': 0.50,
    'outcome_tolerance_percentage_points': 1.0,
    # A useful smoke test needs at least one non-degenerate mission outcome.
    'mission_success_boundary_tolerance': 1e-9,
    # SOC sweeps use few evaluation episodes, so permit bounded sampling noise
    # while still requiring the expected monotone safety direction.
    'soc_monotonic_tolerance_percentage_points': 10.0,
}


IDENTIFIABILITY_OUTCOME_METRICS = (
    'mission_success_rate',
    'task_completion_rate',
    'safe_return_rate',
    'physical_constraint_violation_rate',
    'safety_override_rate',
)


DEFAULT_CMDP_IDENTIFIABILITY_GATES = {
    'minimum_cmdp_discounted_cost_difference': 0.25,
    'minimum_budget_exceedance_difference_percentage_points': 2.0,
    'minimum_physical_violation_difference_percentage_points': 2.0,
    'minimum_safety_override_difference_percentage_points': 2.0,
}


DEFAULT_CMDP_DIRECTIONAL_GATES = {
    'minimum_cmdp_discounted_cost_reduction': 0.25,
    'minimum_budget_exceedance_reduction_percentage_points': 2.0,
    'minimum_raw_action_energy_infeasible_reduction_percentage_points': 0.0,
    'minimum_safety_override_reduction_percentage_points': 2.0,
    'minimum_physical_violation_reduction_percentage_points': 2.0,
    'maximum_physical_violation_increase_percentage_points': 1.0,
    'minimum_task_completion_retention_ratio': 0.85,
}


def _percentile(values, probability):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError('at least one value is required')
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError('percentile probability must lie in [0, 1]')
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def derive_nosafety_service_gates(task_completion_rate,
                                  core_task_completion_rate=None,
                                  target_fraction=0.90,
                                  rounding_grid=1.0):
    """Derive service floors from a separately evaluated NoSafety ceiling."""
    task = float(task_completion_rate)
    core = (
        None if core_task_completion_rate is None
        else float(core_task_completion_rate)
    )
    fraction = float(target_fraction)
    grid = float(rounding_grid)
    values = [task] + ([] if core is None else [core])
    if any(not math.isfinite(value) or not 0.0 <= value <= 100.0
           for value in values):
        raise ValueError('completion ceilings must lie in [0, 100]')
    if not 0.0 < fraction <= 1.0:
        raise ValueError('target_fraction must lie in (0, 1]')
    if grid <= 0.0:
        raise ValueError('rounding_grid must be positive')

    def floor_to_grid(value):
        return math.floor((value + 1e-12) / grid) * grid

    return {
        'target_fraction': fraction,
        'rounding_grid': grid,
        'nosafety_task_completion_ceiling': task,
        'recommended_min_task_completion_rate': floor_to_grid(
            fraction * task
        ),
        'nosafety_core_task_completion_ceiling': core,
        'recommended_min_core_task_completion_rate': (
            None if core is None else floor_to_grid(fraction * core)
        ),
        'interpretation': (
            'Development-only gate derivation; this is not a formal result.'
        ),
    }


def derive_margin_scan_recommendations(episodes, precision=3):
    """Derive controller grids from operational trajectories.

    Step-0 SOC and margin only describe initial-condition spread.  They do not
    identify where a return controller will actually switch during service.
    Candidate return thresholds are therefore centred on the last observed
    pre-service-window state of each trajectory produced by a trained
    reference policy.
    """
    records = list(episodes)
    if len(records) < 30:
        raise ValueError('margin/SOC derivation requires at least 30 episodes')
    margins0 = [float(item['margin0']) for item in records]
    required = [
        float(item['ret0']) + float(item['safe0']) for item in records
    ]

    trajectories = []
    service_margins = []
    service_socs = []
    for item in records:
        samples = list(item.get('trajectory', ()))
        service_steps = int(item.get('service_steps', 0))
        eligible = [
            sample for sample in samples
            if int(sample.get('step', 0)) < service_steps
        ]
        if not eligible:
            raise ValueError(
                'margin scan requires a non-empty operational trajectory '
                'and service_steps for every episode'
            )
        trajectories.extend(eligible)
        service_margins.append(float(eligible[-1]['margin']))
        service_socs.append(float(eligible[-1]['soc']))

    operational_margins = [
        float(sample['margin']) for sample in trajectories
    ]
    operational_socs = [float(sample['soc']) for sample in trajectories]
    finite_values = (
        margins0 + required + operational_margins + operational_socs
        + service_margins + service_socs
    )
    if any(not math.isfinite(value) for value in finite_values):
        raise ValueError('margin scan contains non-finite values')

    def rounded(value):
        return round(min(max(value, 0.0), 1.0), int(precision))

    def quantile_grid(values, label, upper=1.0):
        bounded = [min(max(float(value), 0.0), upper) for value in values]
        result = [
            rounded(_percentile(bounded, probability))
            for probability in (0.10, 0.30, 0.50, 0.70, 0.90)
        ]
        if len(set(result)) < 5:
            low = max(0.0, _percentile(bounded, 0.05))
            high = min(upper, _percentile(bounded, 0.95))
            quantum = 10.0 ** (-int(precision))
            if high - low < 4.0 * quantum:
                raise ValueError(
                    f'operational {label} range is too narrow for five '
                    'distinct candidates'
                )
            result = [
                rounded(low + (high - low) * index / 4.0)
                for index in range(5)
            ]
        if len(set(result)) < 5:
            raise ValueError(
                f'rounded operational {label} candidates are not distinct'
            )
        return result

    margin_iqr = max(
        0.0,
        _percentile(operational_margins, 0.75)
        - _percentile(operational_margins, 0.25),
    )
    zone_gap = rounded(max(0.03, min(0.08, margin_iqr / 2.0)))
    red_values = quantile_grid(
        service_margins, 'service-boundary margin',
        upper=max(0.0, 1.0 - 2.0 * zone_gap),
    )
    paired_candidates = [
        {
            'threshold_yellow': rounded(red + 2.0 * zone_gap),
            'threshold_orange': rounded(red + zone_gap),
            'threshold_red': red,
        }
        for red in red_values
    ]
    for candidate in paired_candidates:
        validate_safety_thresholds(candidate)
    grid = {
        'red': [item['threshold_red'] for item in paired_candidates],
        'orange': [item['threshold_orange'] for item in paired_candidates],
        'yellow': [item['threshold_yellow'] for item in paired_candidates],
    }

    soc_candidates = quantile_grid(
        service_socs, 'service-boundary SOC', upper=0.99
    )
    return {
        'episode_count': len(records),
        'margin0': {
            'p05': _percentile(margins0, 0.05),
            'p50': _percentile(margins0, 0.50),
            'p95': _percentile(margins0, 0.95),
            'p05_p95_span': (
                _percentile(margins0, 0.95)
                - _percentile(margins0, 0.05)
            ),
        },
        'service_boundary_margin': {
            'p05': _percentile(service_margins, 0.05),
            'p50': _percentile(service_margins, 0.50),
            'p95': _percentile(service_margins, 0.95),
        },
        'service_boundary_soc': {
            'p05': _percentile(service_socs, 0.05),
            'p50': _percentile(service_socs, 0.50),
            'p95': _percentile(service_socs, 0.95),
        },
        'required0_frac_of_capacity': {
            'p05': _percentile(required, 0.05),
            'p50': _percentile(required, 0.50),
            'p95': _percentile(required, 0.95),
        },
        'zone_gap': zone_gap,
        'threshold_grid': grid,
        'threshold_candidates': paired_candidates,
        'soc_threshold_candidates': soc_candidates,
        'interpretation': (
            'Development-only operational-trajectory grid derivation; freeze '
            'before confirmatory runs.'
        ),
    }


def assess_cmdp_ablation_identifiability(full_metrics, no_cmdp_metrics,
                                         gates=None):
    """Check that Full and No-CMDP differ measurably without requiring a win."""
    resolved = dict(DEFAULT_CMDP_IDENTIFIABILITY_GATES)
    if gates:
        unknown = set(gates) - set(resolved)
        if unknown:
            raise ValueError(
                f'unknown CMDP identifiability gates: {sorted(unknown)}'
            )
        resolved.update({key: float(value) for key, value in gates.items()})
    if any(value < 0.0 for value in resolved.values()):
        raise ValueError('CMDP identifiability gates must be nonnegative')
    pairs = (
        ('cmdp_discounted_cost',
         'minimum_cmdp_discounted_cost_difference'),
        ('cmdp_budget_exceedance_rate',
         'minimum_budget_exceedance_difference_percentage_points'),
        ('physical_constraint_violation_rate',
         'minimum_physical_violation_difference_percentage_points'),
        ('safety_override_rate',
         'minimum_safety_override_difference_percentage_points'),
    )
    differences = {}
    checks = {}
    for metric, gate in pairs:
        full = _finite_metric(full_metrics, metric, math.nan)
        no_cmdp = _finite_metric(no_cmdp_metrics, metric, math.nan)
        difference = (
            abs(full - no_cmdp)
            if math.isfinite(full) and math.isfinite(no_cmdp)
            else math.nan
        )
        differences[metric] = difference
        checks[metric] = (
            math.isfinite(difference) and difference >= resolved[gate]
        )
    return {
        'passed': any(checks.values()),
        'checks': checks,
        'absolute_differences': differences,
        'gates': resolved,
        'interpretation': (
            'Identifiability only; direction and superiority are not tested.'
        ),
    }


def assess_cmdp_directional_effect(full_metrics, no_cmdp_metrics,
                                   gates=None):
    """Require a measurable CMDP improvement in the intended direction.

    A development pilot passes only when Full improves at least one primary
    CMDP outcome over the matched No-CMDP controller, while preserving the
    common physical constraint and a declared fraction of task completion.
    This is an engineering gate before confirmatory experiments, not a
    significance test or paper result.
    """
    resolved = dict(DEFAULT_CMDP_DIRECTIONAL_GATES)
    if gates:
        unknown = set(gates) - set(resolved)
        if unknown:
            raise ValueError(
                f'unknown directional CMDP gates: {sorted(unknown)}'
            )
        resolved.update({key: float(value) for key, value in gates.items()})
    if any(not math.isfinite(value) or value < 0.0
           for value in resolved.values()):
        raise ValueError('directional CMDP gates must be finite and nonnegative')

    metric_gate_pairs = [
        ('cmdp_discounted_cost',
         'minimum_cmdp_discounted_cost_reduction'),
        ('cmdp_budget_exceedance_rate',
         'minimum_budget_exceedance_reduction_percentage_points'),
        ('safety_override_rate',
         'minimum_safety_override_reduction_percentage_points'),
        ('physical_constraint_violation_rate',
         'minimum_physical_violation_reduction_percentage_points'),
    ]
    raw_action_gate_enabled = bool(
        gates is not None
        and 'minimum_raw_action_energy_infeasible_reduction_percentage_points'
        in gates
    )
    if raw_action_gate_enabled:
        metric_gate_pairs.append((
            'raw_action_energy_infeasible_rate',
            'minimum_raw_action_energy_infeasible_reduction_percentage_points',
        ))
    reductions = {}
    improvement_checks = {}
    for metric, gate_name in metric_gate_pairs:
        full_value = _finite_metric(full_metrics, metric, math.nan)
        reference_value = _finite_metric(no_cmdp_metrics, metric, math.nan)
        reduction = (
            reference_value - full_value
            if math.isfinite(full_value) and math.isfinite(reference_value)
            else math.nan
        )
        reductions[metric] = reduction
        improvement_checks[metric] = bool(
            math.isfinite(reduction) and reduction >= resolved[gate_name]
        )

    full_physical = _finite_metric(
        full_metrics, 'physical_constraint_violation_rate', math.nan
    )
    reference_physical = _finite_metric(
        no_cmdp_metrics, 'physical_constraint_violation_rate', math.nan
    )
    physical_noninferior = bool(
        math.isfinite(full_physical)
        and math.isfinite(reference_physical)
        and full_physical <= reference_physical + resolved[
            'maximum_physical_violation_increase_percentage_points'
        ]
    )
    full_completion = _finite_metric(
        full_metrics, 'task_completion_rate', math.nan
    )
    reference_completion = _finite_metric(
        no_cmdp_metrics, 'task_completion_rate', math.nan
    )
    completion_retained = bool(
        math.isfinite(full_completion)
        and math.isfinite(reference_completion)
        and (
            reference_completion <= 1e-12
            or full_completion >= reference_completion * resolved[
                'minimum_task_completion_retention_ratio'
            ]
        )
    )
    primary_names = [
        'cmdp_discounted_cost',
        'cmdp_budget_exceedance_rate',
        'safety_override_rate',
    ]
    if raw_action_gate_enabled:
        primary_names.append('raw_action_energy_infeasible_rate')
    primary_improvement = any(
        improvement_checks[name] for name in primary_names
    )
    return {
        'passed': bool(
            primary_improvement
            and physical_noninferior
            and completion_retained
        ),
        'primary_improvement_detected': primary_improvement,
        'improvement_checks': improvement_checks,
        'signed_reductions_no_cmdp_minus_full': reductions,
        'physical_constraint_noninferior': physical_noninferior,
        'task_completion_retained': completion_retained,
        'gates': resolved,
        'interpretation': (
            'Development-only directional gate: positive reductions favour '
            'Full over the matched No-CMDP controller.'
        ),
    }


def recommend_matched_nocmdp_budget(costs, target_fraction=0.90, grid=0.5,
                                    rule='fraction_of_mean', quantile=0.75):
    """Set a positive budget from matched, same-safety No-CMDP costs.

    fraction_of_mean preserves the historical v3d rule.  The v3e development
    protocol uses empirical_quantile so the unconstrained reference has a
    declared, non-degenerate exceedance rate before Full is trained.  Both
    rules use only No-CMDP development trajectories.
    """
    values = [float(value) for value in costs]
    if not values:
        raise ValueError('at least one matched No-CMDP cost is required')
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError('matched No-CMDP costs must be finite and nonnegative')
    grid = float(grid)
    if grid <= 0.0 or not math.isfinite(grid):
        raise ValueError('budget grid must be finite and positive')
    mean = sum(values) / len(values)
    if mean <= 0.0:
        raise ValueError(
            'matched No-CMDP costs are all zero; CMDP budget is not identifiable'
        )
    rule = str(rule).strip().lower()
    if rule == 'fraction_of_mean':
        target_fraction = float(target_fraction)
        if not 0.0 < target_fraction < 1.0:
            raise ValueError('budget target_fraction must lie in (0, 1)')
        unrounded = target_fraction * mean
        budget_rule = 'fraction_of_same_safety_nocmdp_reference_mean'
        reference_quantile = None
    elif rule == 'empirical_quantile':
        quantile = float(quantile)
        if not 0.0 < quantile < 1.0:
            raise ValueError('budget quantile must lie in (0, 1)')
        ordered = sorted(values)
        position = (len(ordered) - 1) * quantile
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        weight = position - lower
        unrounded = (
            ordered[lower] * (1.0 - weight) + ordered[upper] * weight
        )
        budget_rule = 'empirical_quantile_of_same_safety_nocmdp_cost'
        reference_quantile = quantile
        target_fraction = None
    else:
        raise ValueError(
            'matched No-CMDP budget rule must be fraction_of_mean or '
            'empirical_quantile'
        )
    recommended = max(grid, math.ceil((unrounded - 1e-12) / grid) * grid)
    exceedance_rate = sum(value > recommended for value in values) / len(values)
    return {
        'reference_algorithm': 'SAC-LSTM-EP-NoCMDP',
        'reference_scope': 'same_safety_layer',
        'sample_count': len(values),
        'reference_mean': float(mean),
        'reference_min': float(min(values)),
        'reference_max': float(max(values)),
        'target_fraction': target_fraction,
        'reference_quantile': reference_quantile,
        'unrounded_target': float(unrounded),
        'rounding_grid': grid,
        'recommended_cmdp_cost_limit': float(recommended),
        'reference_budget_exceedance_rate': float(exceedance_rate * 100.0),
        'budget_rule': budget_rule,
    }


def recommend_cmdp_budget(costs, confidence=0.95, grid=0.5,
                          absolute_buffer=0.0):
    """Recommend an expectation budget from the mean-CI upper bound.

    Costs must come from an independently seeded feasible reference
    controller. The upper bound is rounded upward to a declared grid; this is
    an empirical expectation-budget calibration, not a per-trajectory proof.
    """
    values = [float(value) for value in costs]
    if not values:
        raise ValueError('at least one CMDP cost is required')
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError('CMDP costs must be finite and nonnegative')
    confidence = float(confidence)
    grid = float(grid)
    absolute_buffer = float(absolute_buffer)
    if not 0.0 < confidence < 1.0:
        raise ValueError('confidence must lie in (0, 1)')
    if grid <= 0.0:
        raise ValueError('budget grid must be positive')
    if absolute_buffer < 0.0:
        raise ValueError('absolute budget buffer must be nonnegative')

    count = len(values)
    mean = sum(values) / count
    if count > 1:
        variance = sum((value - mean) ** 2 for value in values) / (count - 1)
        sample_std = math.sqrt(variance)
        z_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
        ci_half_width = z_value * sample_std / math.sqrt(count)
    else:
        sample_std = 0.0
        z_value = 0.0
        ci_half_width = 0.0
    upper_bound = mean + ci_half_width
    unrounded = upper_bound + absolute_buffer
    recommended = math.ceil((unrounded - 1e-12) / grid) * grid
    return {
        'sample_count': count,
        'mean': float(mean),
        'sample_std': float(sample_std),
        'confidence': confidence,
        'normal_quantile': float(z_value),
        'mean_ci_half_width': float(ci_half_width),
        'mean_ci_upper_bound': float(upper_bound),
        'absolute_buffer': absolute_buffer,
        'rounding_grid': grid,
        'recommended_cmdp_cost_limit': float(recommended),
    }


def analyse_cmdp_margin_signal(episodes, margin_targets, cost_scale,
                               selected_margin_target, gamma=0.99,
                               confidence=0.95, budget_grid=0.5,
                               absolute_buffer=0.0,
                               budget_target_fraction=0.90,
                               min_activation_rate=0.01,
                               max_activation_rate=0.99,
                               max_saturation_rate=0.95):
    """Measure CMDP cost density on preregistered development trajectories."""
    records = list(episodes)
    targets = sorted({float(value) for value in margin_targets})
    selected = float(selected_margin_target)
    scale = float(cost_scale)
    gamma = float(gamma)
    budget_fraction = float(budget_target_fraction)
    if not records:
        raise ValueError('at least one operational trajectory is required')
    if not targets or selected not in targets:
        raise ValueError('selected_margin_target must appear in margin_targets')
    if scale <= 0.0 or not 0.0 < gamma <= 1.0:
        raise ValueError('cost_scale must be positive and gamma in (0, 1]')
    if not 0.0 < budget_fraction <= 1.0:
        raise ValueError('budget_target_fraction must lie in (0, 1]')
    if not 0.0 <= min_activation_rate < max_activation_rate <= 1.0:
        raise ValueError('invalid CMDP activation-rate gates')
    if not 0.0 <= max_saturation_rate <= 1.0:
        raise ValueError('invalid CMDP saturation-rate gate')

    candidates = []
    selected_costs = None
    for target in targets:
        discounted_costs = []
        undiscounted_costs = []
        active_steps = saturated_steps = total_steps = 0
        for episode in records:
            trajectory = list(episode.get('trajectory', ()))
            if not trajectory:
                raise ValueError('every episode must contain a trajectory')
            discounted = undiscounted = 0.0
            for index, sample in enumerate(trajectory):
                margin = float(sample['margin'])
                if not math.isfinite(margin):
                    raise ValueError('trajectory margins must be finite')
                cost = min(max((target - margin) / scale, 0.0), 1.0)
                discounted += (gamma ** index) * cost
                undiscounted += cost
                active_steps += int(cost > 0.0)
                saturated_steps += int(cost >= 1.0 - 1e-12)
                total_steps += 1
            discounted_costs.append(discounted)
            undiscounted_costs.append(undiscounted)
        candidate = {
            'cmdp_margin_target': target,
            'cmdp_cost_scale': scale,
            'episode_count': len(discounted_costs),
            'step_count': total_steps,
            'cost_activation_rate': active_steps / max(total_steps, 1),
            'cost_saturation_rate': saturated_steps / max(total_steps, 1),
            'mean_discounted_cost': sum(discounted_costs) / len(discounted_costs),
            'minimum_discounted_cost': min(discounted_costs),
            'maximum_discounted_cost': max(discounted_costs),
            'mean_undiscounted_cost': (
                sum(undiscounted_costs) / len(undiscounted_costs)
            ),
        }
        candidates.append(candidate)
        if target == selected:
            selected_costs = discounted_costs

    budget = recommend_cmdp_budget(
        selected_costs, confidence=confidence, grid=budget_grid,
        absolute_buffer=absolute_buffer,
    )
    ci_upper_budget = budget['recommended_cmdp_cost_limit']
    unrounded_budget = (
        budget_fraction * budget['mean'] + float(absolute_buffer)
    )
    selected_budget = (
        math.ceil((unrounded_budget - 1e-12) / float(budget_grid))
        * float(budget_grid)
    )
    budget.update({
        'budget_rule': 'fraction_of_unconstrained_reference_mean',
        'target_fraction': budget_fraction,
        'unrounded_target': float(unrounded_budget),
        'mean_ci_upper_budget_diagnostic': float(ci_upper_budget),
        'recommended_cmdp_cost_limit': float(selected_budget),
    })
    selected_record = next(
        item for item in candidates
        if item['cmdp_margin_target'] == selected
    )
    checks = {
        'selected_cost_signal_is_nonzero': (
            selected_record['cost_activation_rate'] > 0.0
            and selected_record['mean_discounted_cost'] > 0.0
        ),
        'selected_activation_rate_above_floor': (
            selected_record['cost_activation_rate'] >= min_activation_rate
        ),
        'selected_activation_rate_below_ceiling': (
            selected_record['cost_activation_rate'] <= max_activation_rate
        ),
        'selected_saturation_rate_below_ceiling': (
            selected_record['cost_saturation_rate'] <= max_saturation_rate
        ),
        'recommended_budget_is_positive': (
            budget['recommended_cmdp_cost_limit'] > 0.0
        ),
    }
    return {
        'passed': all(checks.values()),
        'checks': checks,
        'selected_margin_target': selected,
        'selected_cost_scale': scale,
        'selected_candidate': selected_record,
        'candidate_diagnostics': candidates,
        'budget_calibration': budget,
        'recommended_cmdp_cost_limit': budget['recommended_cmdp_cost_limit'],
        'gates': {
            'min_activation_rate': float(min_activation_rate),
            'max_activation_rate': float(max_activation_rate),
            'max_saturation_rate': float(max_saturation_rate),
        },
        'selection_rule': (
            f'cmdp_margin_target={selected:.3f} was preregistered from the '
            'safety-zone geometry; neighbouring targets are diagnostics only'
        ),
        'interpretation': (
            'Development-only operational-trajectory signal calibration. '
            'The expectation budget is a preregistered fraction of the '
            'unconstrained reference mean; the CI upper bound is diagnostic '
            'only. This is not a confirmatory outcome.'
        ),
    }


def generate_threshold_candidates(yellow_values, orange_values, red_values):
    """Generate unique threshold triples satisfying red < orange < yellow."""
    candidates = []
    seen = set()
    for yellow in yellow_values:
        for orange in orange_values:
            for red in red_values:
                try:
                    values = validate_safety_thresholds({
                        'threshold_yellow': yellow,
                        'threshold_orange': orange,
                        'threshold_red': red,
                    })
                except ValueError:
                    continue
                key = (values['yellow'], values['orange'], values['red'])
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({
                    'threshold_yellow': values['yellow'],
                    'threshold_orange': values['orange'],
                    'threshold_red': values['red'],
                })
    if not candidates:
        raise ValueError('threshold grid contains no ordered candidate')
    return sorted(candidates, key=lambda item: (
        item['threshold_yellow'],
        item['threshold_orange'],
        item['threshold_red'],
    ))


def _finite_metric(metrics, name, default):
    value = metrics.get(name, default)
    if value is None:
        return float(default)
    value = float(value)
    return value if math.isfinite(value) else float(default)


def select_safety_candidate(candidates, gates=None):
    """Select a threshold candidate using preregistered safety-first rules.

    Evaluation metrics are percentages where applicable.  Safety gates are
    applied before any performance ranking.  When every candidate fails, the
    least-unsafe candidate is returned for diagnosis but the status remains
    ``no_feasible_candidate``.
    """
    candidate_rows = list(candidates)
    if not candidate_rows:
        raise ValueError('at least one safety-threshold candidate is required')
    resolved_gates = dict(DEFAULT_SAFETY_GATES)
    if gates:
        resolved_gates.update({key: float(value) for key, value in gates.items()})
    required_gates = set(DEFAULT_SAFETY_GATES)
    unknown = set(resolved_gates) - required_gates
    if unknown:
        raise ValueError(f'unknown safety gates: {sorted(unknown)}')
    if any(not math.isfinite(value) or value < 0.0
           for value in resolved_gates.values()):
        raise ValueError('safety gates must be finite and nonnegative')

    annotated = []
    for index, candidate in enumerate(candidate_rows):
        metrics = dict(candidate.get('metrics', {}))
        candidate_id = str(candidate.get('candidate_id', f'candidate_{index:03d}'))
        parameters = dict(candidate.get('parameters', {}))
        physical = _finite_metric(
            metrics, 'physical_constraint_violation_rate', math.inf
        )
        depletion = _finite_metric(
            metrics, 'battery_depletion_rate', math.inf
        )
        deadline = _finite_metric(
            metrics, 'mission_deadline_miss_rate', math.inf
        )
        safe_return = _finite_metric(metrics, 'safe_return_rate', -math.inf)
        completion = _finite_metric(metrics, 'task_completion_rate', 0.0)
        override = _finite_metric(metrics, 'safety_override_rate', math.inf)
        gate_checks = {
            'physical_constraint_violation_rate': (
                physical
                <= resolved_gates['max_physical_constraint_violation_rate']
            ),
            'battery_depletion_rate': (
                depletion <= resolved_gates['max_battery_depletion_rate']
            ),
            'mission_deadline_miss_rate': (
                deadline <= resolved_gates['max_mission_deadline_miss_rate']
            ),
            'safe_return_rate': (
                safe_return >= resolved_gates['min_safe_return_rate']
            ),
            'task_completion_rate': (
                completion >= resolved_gates['min_task_completion_rate']
            ),
            'safety_override_rate': (
                override <= resolved_gates['max_safety_override_rate']
            ),
        }
        annotated.append({
            **candidate,
            'candidate_id': candidate_id,
            'parameters': parameters,
            'metrics': metrics,
            'safety_gate_checks': gate_checks,
            'passes_safety_gates': all(gate_checks.values()),
        })

    def performance_key(item):
        metrics = item['metrics']
        return (
            -_finite_metric(metrics, 'mission_success_rate', -math.inf),
            -_finite_metric(metrics, 'task_completion_rate', -math.inf),
            -_finite_metric(metrics, 'core_task_completion_rate', -math.inf),
            _finite_metric(metrics, 'mean_task_delay_s', math.inf),
            _finite_metric(metrics, 'system_energy', math.inf),
            _finite_metric(metrics, 'safety_override_rate', math.inf),
            item['candidate_id'],
        )

    def unsafe_key(item):
        metrics = item['metrics']
        return (
            _finite_metric(
                metrics, 'physical_constraint_violation_rate', math.inf
            ),
            _finite_metric(metrics, 'battery_depletion_rate', math.inf),
            _finite_metric(metrics, 'mission_deadline_miss_rate', math.inf),
            -_finite_metric(metrics, 'safe_return_rate', -math.inf),
            *performance_key(item),
        )

    feasible = [item for item in annotated if item['passes_safety_gates']]
    if feasible:
        selected = min(feasible, key=performance_key)
        status = 'feasible_candidate_selected'
    else:
        selected = min(annotated, key=unsafe_key)
        status = 'no_feasible_candidate'
    return {
        'selection_status': status,
        'selected_candidate_id': selected['candidate_id'],
        'selected_parameters': selected['parameters'],
        'selected_metrics': selected['metrics'],
        'gates': resolved_gates,
        'candidates': annotated,
    }


def assess_calibration_identifiability(graded_selection, soc_selection,
                                       gates=None):
    """Evaluate preregistered smoke-test identifiability criteria.

    This gate checks whether a sweep can distinguish controllers.  It never
    requires the proposed method to win and therefore must not be interpreted
    as evidence for a paper claim.
    """
    resolved = dict(DEFAULT_IDENTIFIABILITY_GATES)
    if gates:
        resolved.update({key: float(value) for key, value in gates.items()})
    unknown = set(resolved) - set(DEFAULT_IDENTIFIABILITY_GATES)
    if unknown:
        raise ValueError(
            f'unknown identifiability gates: {sorted(unknown)}'
        )
    fraction = resolved['minimum_distinct_graded_outcome_fraction']
    outcome_tolerance = resolved['outcome_tolerance_percentage_points']
    boundary_tolerance = resolved['mission_success_boundary_tolerance']
    monotonic_tolerance = (
        resolved['soc_monotonic_tolerance_percentage_points']
    )
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            'minimum_distinct_graded_outcome_fraction must lie in (0, 1]'
        )
    if outcome_tolerance <= 0.0 or boundary_tolerance < 0.0:
        raise ValueError('identifiability tolerances must be nonnegative')
    if monotonic_tolerance < 0.0:
        raise ValueError('SOC monotonic tolerance must be nonnegative')

    graded = list((graded_selection or {}).get('candidates', []))
    signatures = set()
    mission_values = []
    for candidate in graded:
        metrics = candidate.get('metrics', {})
        signature = []
        for metric in IDENTIFIABILITY_OUTCOME_METRICS:
            value = _finite_metric(metrics, metric, math.nan)
            signature.append(
                None if not math.isfinite(value)
                else int(round(value / outcome_tolerance))
            )
        signatures.add(tuple(signature))
        mission = _finite_metric(
            metrics, 'mission_success_rate', math.nan
        )
        if math.isfinite(mission):
            mission_values.append(mission)

    required_distinct = max(2, math.ceil(len(graded) * fraction))
    graded_distinguishable = bool(
        graded and len(signatures) >= required_distinct
    )
    mission_success_discriminative = any(
        boundary_tolerance < value < 100.0 - boundary_tolerance
        for value in mission_values
    )

    soc = list((soc_selection or {}).get('candidates', []))
    soc_rows = []
    for candidate in soc:
        threshold = candidate.get('parameters', {}).get(
            'soc_return_threshold'
        )
        if threshold is None:
            continue
        metrics = candidate.get('metrics', {})
        soc_rows.append((
            float(threshold),
            _finite_metric(metrics, 'safe_return_rate', math.nan),
            _finite_metric(
                metrics, 'physical_constraint_violation_rate', math.nan
            ),
            _finite_metric(metrics, 'safety_override_rate', math.nan),
        ))
    soc_rows.sort(key=lambda row: row[0])
    soc_signatures = {
        (
            int(round(row[1] / outcome_tolerance)),
            int(round(row[2] / outcome_tolerance)),
            int(round(row[3] / outcome_tolerance)),
        )
        for row in soc_rows
        if all(math.isfinite(value) for value in row[1:])
    }
    soc_frontier_non_degenerate = len(soc_signatures) >= 2
    soc_frontier_monotone = (
        soc_frontier_non_degenerate and len(soc_rows) >= 2 and all(
            math.isfinite(previous[1])
            and math.isfinite(previous[2])
            and math.isfinite(current[1])
            and math.isfinite(current[2])
            and current[1] + monotonic_tolerance >= previous[1]
            and current[2] <= previous[2] + monotonic_tolerance
            for previous, current in zip(soc_rows, soc_rows[1:])
        )
    )
    checks = {
        'graded_thresholds_are_distinguishable': graded_distinguishable,
        'mission_success_is_discriminative': mission_success_discriminative,
        'soc_frontier_is_non_degenerate': soc_frontier_non_degenerate,
        'soc_frontier_is_monotone': soc_frontier_monotone,
    }
    return {
        'passed': all(checks.values()),
        'checks': checks,
        'gates': resolved,
        'graded_candidate_count': len(graded),
        'graded_distinct_outcome_count': len(signatures),
        'graded_required_distinct_outcome_count': required_distinct,
        'mission_success_values': mission_values,
        'soc_frontier': [
            {
                'soc_return_threshold': row[0],
                'safe_return_rate': row[1],
                'physical_constraint_violation_rate': row[2],
                'safety_override_rate': row[3],
            }
            for row in soc_rows
        ],
        'interpretation': (
            'Identifiability only; passing does not establish superiority.'
        ),
    }


def calibration_completion_status(graded_selection, identifiability=None):
    """Return primary calibration status without gating on the SOC baseline."""
    if (
        (graded_selection or {}).get('selection_status')
        != 'feasible_candidate_selected'
    ):
        return 'no_feasible_primary_candidate'
    if identifiability is not None and not identifiability.get('passed', False):
        return 'failed_identifiability'
    return 'complete'


def validate_disjoint_seed_ranges(train_seed_bases, train_episodes,
                                  calibration_seed, calibration_episodes,
                                  final_evaluation_seed,
                                  final_evaluation_episodes):
    """Reject overlapping episode-seed ranges across calibration stages."""
    counts = {
        'training': int(train_episodes),
        'calibration': int(calibration_episodes),
        'final_evaluation': int(final_evaluation_episodes),
    }
    if any(value <= 0 for value in counts.values()):
        raise ValueError('all seed-range episode counts must be positive')
    train_bases = [int(value) for value in train_seed_bases]
    if len(train_bases) < 3 or len(set(train_bases)) != len(train_bases):
        raise ValueError('at least three unique training seed bases are required')

    ranges = []
    for index, base in enumerate(train_bases):
        ranges.append({
            'label': f'training_seed_{index}',
            'start': base,
            'end': base + counts['training'] - 1,
        })
    for label, base, count in (
        ('calibration', int(calibration_seed), counts['calibration']),
        ('final_evaluation', int(final_evaluation_seed),
         counts['final_evaluation']),
    ):
        ranges.append({
            'label': label,
            'start': base,
            'end': base + count - 1,
        })

    ordered = sorted(ranges, key=lambda item: (item['start'], item['end']))
    for previous, current in zip(ordered, ordered[1:]):
        if current['start'] <= previous['end']:
            raise ValueError(
                'episode seed ranges overlap: '
                f"{previous['label']} [{previous['start']}, {previous['end']}] "
                f"and {current['label']} [{current['start']}, {current['end']}]"
            )
    return ranges
