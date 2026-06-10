"""
光谱预处理模块
包含基线校正、平滑去噪、归一化等功能，支持步骤组合和顺序调整
"""
import numpy as np
from scipy import signal
from typing import List, Dict, Callable, Tuple, Optional
from .spectrum import Spectrum


class PreprocessingStep:
    """预处理步骤基类"""
    name: str = "base"
    params: Dict = {}
    
    def apply(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
    
    def to_dict(self) -> Dict:
        return {"name": self.name, "params": self.params.copy()}


class BaselineCorrection(PreprocessingStep):
    """基线校正"""
    name = "baseline_correction"
    
    def __init__(self, method: str = "als", **kwargs):
        self.method = method
        self.params = {"method": method, **kwargs}
    
    def apply(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        method = self.params.get("method", "als")
        
        if method == "polynomial":
            degree = self.params.get("degree", 5)
            baseline = self._polynomial_baseline(x, y, degree)
        elif method == "als":
            lam = self.params.get("lambda", 1e5)
            p = self.params.get("p", 0.01)
            niter = self.params.get("niter", 10)
            baseline = self._als_baseline(y, lam, p, niter)
        elif method == "snip":
            niter = self.params.get("niter", 20)
            baseline = self._snip_baseline(y, niter)
        else:
            return x, y
        
        return x, y - baseline
    
    def _polynomial_baseline(self, x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
        coeffs = np.polyfit(x, y, degree)
        return np.polyval(coeffs, x)
    
    def _als_baseline(self, y: np.ndarray, lam: float, p: float, niter: int) -> np.ndarray:
        L = len(y)
        D = np.diff(np.eye(L), 2)
        D = lam * D.dot(D.T)
        w = np.ones(L)
        W = np.diag(w)
        
        for _ in range(niter):
            W = np.diag(w)
            Z = W + D
            z = np.linalg.solve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        
        return z
    
    def _snip_baseline(self, y: np.ndarray, niter: int) -> np.ndarray:
        baseline = y.copy()
        for m in range(1, niter + 1):
            for i in range(m, len(y) - m):
                baseline[i] = min(baseline[i], (baseline[i - m] + baseline[i + m]) / 2)
        return baseline


class Smoothing(PreprocessingStep):
    """平滑去噪"""
    name = "smoothing"
    
    def __init__(self, method: str = "savgol", **kwargs):
        self.method = method
        self.params = {"method": method, **kwargs}
    
    def apply(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        method = self.params.get("method", "savgol")
        
        if method == "savgol":
            window = self.params.get("window", 11)
            polyorder = self.params.get("polyorder", 2)
            if window % 2 == 0:
                window += 1
            y_smoothed = signal.savgol_filter(y, window, polyorder)
        elif method == "wavelet":
            wavelet = self.params.get("wavelet", "db4")
            level = self.params.get("level", 3)
            y_smoothed = self._wavelet_denoise(y, wavelet, level)
        else:
            y_smoothed = y
        
        return x, y_smoothed
    
    def _wavelet_denoise(self, y: np.ndarray, wavelet: str, level: int) -> np.ndarray:
        try:
            import pywt
            coeffs = pywt.wavedec(y, wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(y)))
            
            new_coeffs = [coeffs[0]]
            for c in coeffs[1:]:
                new_coeffs.append(pywt.threshold(c, threshold, mode='soft'))
            
            return pywt.waverec(new_coeffs, wavelet)[:len(y)]
        except ImportError:
            return y


class Normalization(PreprocessingStep):
    """归一化"""
    name = "normalization"
    
    def __init__(self, method: str = "area", **kwargs):
        self.method = method
        self.params = {"method": method, **kwargs}
    
    def apply(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        method = self.params.get("method", "area")
        
        if method == "area":
            area = np.trapz(y, x)
            if abs(area) > 1e-10:
                y_norm = y / area
            else:
                y_norm = y
        elif method == "max":
            y_max = np.max(y)
            if y_max > 1e-10:
                y_norm = y / y_max
            else:
                y_norm = y
        elif method == "peak":
            peak_x = self.params.get("peak_x", None)
            if peak_x is not None:
                idx = np.argmin(np.abs(x - peak_x))
                peak_val = y[idx]
                if peak_val > 1e-10:
                    y_norm = y / peak_val
                else:
                    y_norm = y
            else:
                y_norm = y
        else:
            y_norm = y
        
        return x, y_norm


class PreprocessingPipeline:
    """预处理流水线，支持步骤组合和顺序调整"""
    
    def __init__(self, steps: Optional[List[PreprocessingStep]] = None):
        self.steps: List[PreprocessingStep] = steps or []
        self.history: List[List[Dict]] = []
    
    def add_step(self, step: PreprocessingStep, position: Optional[int] = None):
        if position is None:
            self.steps.append(step)
        else:
            self.steps.insert(position, step)
    
    def remove_step(self, index: int):
        if 0 <= index < len(self.steps):
            self.steps.pop(index)
    
    def move_step(self, from_idx: int, to_idx: int):
        if 0 <= from_idx < len(self.steps) and 0 <= to_idx < len(self.steps):
            step = self.steps.pop(from_idx)
            self.steps.insert(to_idx, step)
    
    def apply(self, spectrum: Spectrum) -> Spectrum:
        result = spectrum.copy()
        x, y = result.x, result.y
        
        step_dicts = []
        for step in self.steps:
            x, y = step.apply(x, y)
            step_dicts.append(step.to_dict())
        
        result.x = x
        result.y = y
        result.preprocessing_history.extend(step_dicts)
        
        return result
    
    def apply_step(self, spectrum: Spectrum, step_index: int) -> Spectrum:
        """应用到指定步骤（含该步骤之前的所有步骤）"""
        result = spectrum.copy()
        x, y = result.x, result.y
        
        for i in range(step_index + 1):
            if i < len(self.steps):
                x, y = self.steps[i].apply(x, y)
        
        result.x = x
        result.y = y
        
        return result
    
    def to_dict_list(self) -> List[Dict]:
        return [step.to_dict() for step in self.steps]
    
    @classmethod
    def from_dict_list(cls, dict_list: List[Dict]) -> 'PreprocessingPipeline':
        steps = []
        for d in dict_list:
            name = d.get("name", "")
            params = d.get("params", {})
            if name == "baseline_correction":
                steps.append(BaselineCorrection(**params))
            elif name == "smoothing":
                steps.append(Smoothing(**params))
            elif name == "normalization":
                steps.append(Normalization(**params))
        return cls(steps)
