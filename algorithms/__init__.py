"""The five algorithms used by the experiment suite."""

from .bandit_osm import BanditOSM
from .related_work import BanditIG, MSE3, OpaqueGreedy, POLICIES, RandomPolicy

EW = BanditOSM
Random = RandomPolicy

__all__ = ["EW", "BanditOSM", "Random", "RandomPolicy", "MSE3",
           "OpaqueGreedy", "BanditIG", "POLICIES"]
