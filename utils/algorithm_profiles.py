"""Explicit component profiles for native-scene baselines and ablations."""
from dataclasses import dataclass


@dataclass(frozen=True)
class AlgorithmProfile:
    use_lstm: bool = False
    use_cmdp: bool = False
    safety_controller: str = "none"
    prediction_source: str = "none"
    offline_only: bool = False
    observe_margin_state: bool = False
    cmdp_cost_mode: str = "energy_margin"

    @property
    def use_energy_safety_supervisor(self):
        return self.safety_controller != "none"

    @property
    def use_oracle_prediction(self):
        return self.prediction_source == "oracle"

    @property
    def use_prediction(self):
        """Whether the policy receives a nontrivial future-energy estimate."""
        return self.prediction_source in (
            "persistence", "lstm", "shuffled_lstm", "oracle",
        )


_PROFILES = {
    # Paper-facing profiles introduced after the CMDP contribution study.
    # These are new experiment identifiers, not renames of historical runs.
    # Keeping the legacy identifiers below unchanged preserves every v3.x
    # checkpoint, result path and source-fingerprint audit trail.
    "SAC-LSTM-EP-PaperMain": AlgorithmProfile(
        use_lstm=True, safety_controller="margin",
        prediction_source="lstm",
    ),
    "SAC-LSTM-EP-CMDP": AlgorithmProfile(
        use_lstm=True, use_cmdp=True,
        safety_controller="margin", prediction_source="lstm",
    ),
    "SAC-LSTM-EP-NoPrediction": AlgorithmProfile(
        safety_controller="margin", prediction_source="current",
    ),
    "SAC-LSTM-EP-FixedSOC": AlgorithmProfile(
        safety_controller="soc_threshold", prediction_source="current",
    ),

    # Main method and primary incremental ablations from Table 5-7.
    "SAC-LSTM-EP": AlgorithmProfile(
        use_lstm=True, use_cmdp=True,
        safety_controller="margin", prediction_source="lstm",
    ),
    "SAC": AlgorithmProfile(),
    "SAC+LSTM": AlgorithmProfile(
        use_lstm=True, prediction_source="lstm",
    ),
    "SAC+CMDP": AlgorithmProfile(
        use_cmdp=True,
    ),
    # Adapted from Alvi et al.: same SAC/Lagrangian machinery in the common
    # task-offloading environment, but the constraint signal is a normalized
    # deadline-miss event plus post-transition latency overshoot rather than
    # predicted energy margin.
    "C-SAC-Alvi-Adapted": AlgorithmProfile(
        use_cmdp=True, cmdp_cost_mode="latency_violation",
    ),
    "SAC+SafetyProjection": AlgorithmProfile(
        safety_controller="margin",
    ),
    "SAC-LSTM-EP-NoSafety": AlgorithmProfile(
        use_lstm=True, use_cmdp=True, prediction_source="lstm",
        offline_only=True, observe_margin_state=True,
    ),
    "SAC-Oracle": AlgorithmProfile(
        use_cmdp=True, safety_controller="margin",
        prediction_source="oracle", offline_only=True,
    ),
    "SAC-SOCThresholdReturn": AlgorithmProfile(
        safety_controller="soc_threshold",
    ),
    # Prediction-only matched controls.  All four variants keep the same SAC,
    # CMDP and graded safety controller; only the source of the future-energy
    # feature changes.  This isolates the contribution of prediction quality.
    "SAC-Current": AlgorithmProfile(
        use_cmdp=True, safety_controller="margin",
        prediction_source="current",
    ),
    "SAC-Persistence-EP": AlgorithmProfile(
        use_cmdp=True, safety_controller="margin",
        prediction_source="persistence",
    ),
    "SAC-Shuffled-EP": AlgorithmProfile(
        use_lstm=True, use_cmdp=True, safety_controller="margin",
        prediction_source="shuffled_lstm",
    ),
    "SAC-FixedSOC": AlgorithmProfile(
        use_cmdp=True, safety_controller="soc_threshold",
        prediction_source="current",
    ),
    # Matched controls: each retains the full method except for the named
    # trigger/controller/CMDP change.
    "SAC-LSTM-EP-SOC": AlgorithmProfile(
        use_lstm=True, use_cmdp=True,
        safety_controller="soc_threshold", prediction_source="lstm",
    ),
    "SAC-LSTM-EP-PredictiveReturn": AlgorithmProfile(
        use_lstm=True, use_cmdp=True,
        safety_controller="margin_threshold", prediction_source="lstm",
    ),
    "SAC-LSTM-EP-NoCMDP": AlgorithmProfile(
        use_lstm=True, safety_controller="margin",
        prediction_source="lstm",
    ),
    "SAC-LSTM-EP-SingleLevel": AlgorithmProfile(
        use_lstm=True, use_cmdp=True,
        safety_controller="margin_single_level", prediction_source="lstm",
    ),

    # Legacy leave-one-out names retained for reproducibility.
    "SAC-NoLSTM": AlgorithmProfile(
        use_cmdp=True, safety_controller="margin",
    ),
    "SAC-NoEmergency": AlgorithmProfile(
        use_lstm=True, use_cmdp=True, prediction_source="lstm",
        offline_only=True,
    ),
    "SAC-NoConstraint": AlgorithmProfile(
        use_lstm=True, safety_controller="margin",
        prediction_source="lstm",
    ),

    # Traditional baselines do not inherit proposed modules.
    "PPO": AlgorithmProfile(),
    "DDPG": AlgorithmProfile(),
    "DDPG-MatchedSafety": AlgorithmProfile(
        use_lstm=True, safety_controller="margin",
        prediction_source="lstm",
    ),
    "Greedy": AlgorithmProfile(),
    "Random": AlgorithmProfile(),
}


PRIMARY_ABLATION_ALGORITHMS = (
    "SAC",
    "SAC+LSTM",
    "SAC+CMDP",
    "SAC+SafetyProjection",
    "SAC-LSTM-EP",
    "SAC-LSTM-EP-SOC",
    "SAC-LSTM-EP-PredictiveReturn",
    "SAC-LSTM-EP-NoCMDP",
    "SAC-LSTM-EP-SingleLevel",
    "SAC-LSTM-EP-NoSafety",
)

MATCHED_CONTROL_ALGORITHMS = (
    "SAC-LSTM-EP-SOC",
    "SAC-LSTM-EP-PredictiveReturn",
    "SAC-LSTM-EP-NoCMDP",
    "SAC-LSTM-EP-SingleLevel",
)

OPTIONAL_ORACLE_ALGORITHM = "SAC-Oracle"

PREDICTION_ABLATION_ALGORITHMS = (
    "SAC-Current",
    "SAC-Persistence-EP",
    "SAC-Shuffled-EP",
    "SAC-LSTM-EP",
)

BASELINE_ALGORITHMS = (
    "SAC", "C-SAC-Alvi-Adapted", "PPO", "DDPG",
    "DDPG-MatchedSafety", "Greedy", "Random",
)


# New paper protocol.  Internal experiment identifiers remain explicit while
# manuscript tables use concise labels through ``paper_algorithm_label``.
PAPER_MAIN_ALGORITHM = "SAC-LSTM-EP-PaperMain"
PAPER_CMDP_VARIANT = "SAC-LSTM-EP-CMDP"
PAPER_PREDICTION_CONTROLS = (
    "SAC-LSTM-EP-NoPrediction",
    "SAC-LSTM-EP-FixedSOC",
)
PAPER_CORE_ALGORITHMS = (
    PAPER_MAIN_ALGORITHM,
    *PAPER_PREDICTION_CONTROLS,
    PAPER_CMDP_VARIANT,
)

_PAPER_LABELS = {
    "SAC-LSTM-EP-PaperMain": "SAC-LSTM-EP",
    "SAC-LSTM-EP-CMDP": "SAC-LSTM-EP-CMDP",
    "SAC-LSTM-EP-NoPrediction": "NoPrediction",
    "SAC-LSTM-EP-FixedSOC": "Fixed SOC",
}


# Experiment specifications may calibrate a constraint budget for one named
# algorithm while leaving every shared environment and component choice
# unchanged.  Keep this list deliberately narrow: component switches and cost
# definitions belong to ``AlgorithmProfile`` and cannot be changed by YAML.
ALGORITHM_OVERRIDE_KEYS = frozenset({
    "cmdp_cost_limit",
})


def algorithm_profile(name):
    """Return the declared experimental component profile for ``name``."""
    try:
        return _PROFILES[str(name)]
    except KeyError as exc:
        raise ValueError(f"No experimental profile declared for {name!r}") from exc


def paper_algorithm_label(name):
    """Return a manuscript label without changing the experiment identifier."""
    value = str(name)
    return _PAPER_LABELS.get(value, value)


def algorithm_config(config, name):
    """Return algorithm-specific settings without mutating the base config."""
    profile = algorithm_profile(name)
    resolved = dict(config)
    resolved["cmdp_enabled"] = bool(profile.use_cmdp)
    resolved["prediction_source"] = profile.prediction_source
    resolved["safety_controller"] = profile.safety_controller
    resolved["offline_only"] = bool(profile.offline_only)
    resolved["cmdp_cost_mode"] = profile.cmdp_cost_mode
    overrides = config.get("algorithm_overrides", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError("algorithm_overrides must be a mapping")
    selected = overrides.get(str(name), {})
    if selected is None:
        selected = {}
    if not isinstance(selected, dict):
        raise ValueError(
            f"algorithm_overrides[{name!r}] must be a mapping"
        )
    unsupported = sorted(set(selected) - ALGORITHM_OVERRIDE_KEYS)
    if unsupported:
        raise ValueError(
            f"unsupported algorithm override keys for {name!r}: "
            f"{unsupported}"
        )
    resolved.update(selected)
    return resolved
