"""评估阶段的统一指标口径。"""
import math


def annotate_cmdp_projection_risk(agent, state, info, raw_action=None):
    """Attach exact same-state raw/applied cost-critic risk diagnostics.

    The function deliberately mutates the per-step ``info`` record so the
    existing action logger can persist the values next to both action vectors.
    Algorithms without a learned CMDP critic receive NaN rather than a false
    zero-risk value.
    """
    estimator = getattr(agent, 'estimate_cmdp_risk', None)
    pre_risk = float('nan')
    post_risk = float('nan')
    if callable(estimator):
        requested = info.get('raw_action', raw_action)
        applied = info.get('applied_action')
        if requested is not None and applied is not None:
            pre_risk = float(estimator(state, requested))
            post_risk = float(estimator(state, applied))
    info['cmdp_pre_projection_critic_risk'] = pre_risk
    info['cmdp_post_projection_critic_risk'] = post_risk
    info['cmdp_projection_critic_risk_delta'] = (
        post_risk - pre_risk
        if math.isfinite(pre_risk) and math.isfinite(post_risk)
        else float('nan')
    )
    return info


def update_safety_return_flags(triggered, returned, info):
    """根据环境实际执行结果累计单回合的返航触发与返回状态。"""
    applied_red = info.get('safety_mode_applied') == 'red'
    return (
        bool(triggered or applied_red),
        bool(returned or info.get('returned_safe', False)),
    )


def successful_safety_return(triggered, returned):
    """只有触发过红色安全返航且最终返回才计为安全返航成功。"""
    return bool(triggered and returned)


def conditional_safe_return_rate(successful_returns, triggered_returns):
    """返回以返航触发回合为分母的安全返回率。"""
    triggered = int(triggered_returns)
    if triggered <= 0:
        return 0.0
    return float(successful_returns) / triggered


def update_mission_outcome(success, deadline_missed, info):
    """Accumulate authoritative episode-level mission outcomes from the env."""
    return (
        bool(success or info.get('mission_success', False)),
        bool(deadline_missed or info.get('mission_deadline_missed', False)),
    )
