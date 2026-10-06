"""The five algorithms used by the experiment suite."""

from .sub_ew import SubEW
from .related_work import BanditIG, MSE3, OpaqueGreedy, POLICIES, RandomPolicy

Random = RandomPolicy

__all__ = ["SubEW", "Random", "RandomPolicy", "MSE3",
           "OpaqueGreedy", "BanditIG", "POLICIES"]
