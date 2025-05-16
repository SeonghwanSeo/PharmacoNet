from typing import Any

from torch import Tensor

MultiScaleFeature = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
HotspotInfo = dict[str, Any]
