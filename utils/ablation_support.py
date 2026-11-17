"""Shared helpers for safety-controller ablation variants.
"""

from utils.safety_constraints import safety_constraints


def configure_profile_environment(env, profile):
    """Apply the safety choice declared by an AlgorithmProfile."""
    env.set_energy_safety_supervisor(
        profile.use_energy_safety_supervisor,
        controller=profile.safety_controller,
    )
    return env


def profile_safety_decision(profile, env, emergency, prediction):
    """Resolve the action constraints used by one declared controller."""
    if profile.safety_controller == 'none':
        return 'normal', safety_constraints(env.config, 'normal')
    if profile.safety_controller in (
        'soc_threshold', 'margin_threshold', 'margin_single_level'
    ):
        mode = env.get_autonomous_safety_mode()
        return mode, safety_constraints(env.config, mode)
    energy_context = env.get_energy_context()
    return emergency.check(
        env.E_rem,
        prediction,
        return_energy=energy_context['return_energy'],
        safety_reserve=energy_context['safety_reserve'],
    )


def profile_observation_controller(profile):
    """Return the controller whose mode should be encoded in the policy state.

    The no-safety full-method ablation still observes predictive margin modes;
    only its action projection is disabled. This keeps its policy input
    matched to the full method.
    """
    if profile.safety_controller == 'margin' or profile.observe_margin_state:
        return 'margin'
    return profile.safety_controller
