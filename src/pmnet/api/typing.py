from torch import Tensor
from typing import Any


MultiScaleFeature = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
HotspotInfo = dict[str, Any]
