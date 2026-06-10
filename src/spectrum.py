"""
光谱数据模型
定义通用的光谱数据结构，支持XRD、XRF、拉曼、红外四种类型
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from enum import Enum


class SpectrumType(str, Enum):
    XRD = "XRD"
    XRF = "XRF"
    RAMAN = "RAMAN"
    IR = "IR"
    UNKNOWN = "UNKNOWN"


@dataclass
class Spectrum:
    name: str
    spectrum_type: SpectrumType
    x: np.ndarray
    y: np.ndarray
    x_unit: str = ""
    y_unit: str = ""
    metadata: Dict = field(default_factory=dict)
    preprocessing_history: List[Dict] = field(default_factory=list)
    peaks: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)

    def copy(self):
        return Spectrum(
            name=self.name,
            spectrum_type=self.spectrum_type,
            x=self.x.copy(),
            y=self.y.copy(),
            x_unit=self.x_unit,
            y_unit=self.y_unit,
            metadata=dict(self.metadata),
            preprocessing_history=list(self.preprocessing_history),
            peaks=list(self.peaks),
        )

    @property
    def x_range(self) -> Tuple[float, float]:
        return float(self.x.min()), float(self.x.max())

    @property
    def num_points(self) -> int:
        return len(self.x)

    def interpolate_to(self, new_x: np.ndarray) -> 'Spectrum':
        new_y = np.interp(new_x, self.x, self.y)
        result = self.copy()
        result.x = new_x
        result.y = new_y
        return result
