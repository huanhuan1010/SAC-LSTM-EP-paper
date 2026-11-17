"""Independent reproduction of Li et al.'s UPTSM + VC-WOA framework."""

from .config import PaperConfig
from .scenario import Scenario, build_scenario
from .uptsm import UPTSMResult, solve_uptsm
from .vcwoa import OptimizationResult, optimize
from .sac_adapter import AdapterEvaluation, SACPaperProblemAdapter

__all__ = [
    "PaperConfig",
    "Scenario",
    "build_scenario",
    "UPTSMResult",
    "solve_uptsm",
    "OptimizationResult",
    "optimize",
    "AdapterEvaluation",
    "SACPaperProblemAdapter",
]
