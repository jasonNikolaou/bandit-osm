"""The five algorithms used by the experiment suite."""

from .sub_ew import SubEW
from .related_work import BanditIG, MSE3, OpaqueGreedy, POLICIES, RandomPolicy
from .wan_baselines import BanditMLSM, PoissonBE

Random = RandomPolicy

__all__ = ["SubEW", "Random", "RandomPolicy", "MSE3",
           "OpaqueGreedy", "BanditIG", "BanditMLSM", "PoissonBE", "POLICIES"]
