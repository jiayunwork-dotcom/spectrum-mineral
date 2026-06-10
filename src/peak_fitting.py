"""
峰检测与拟合模块
支持自动寻峰、高斯/洛伦兹/Voigt峰型拟合、分段拟合
"""
import numpy as np
from scipy import signal, optimize
from typing import List, Dict, Tuple, Optional
from .spectrum import Spectrum


def detect_peaks(x: np.ndarray, y: np.ndarray, 
                 sensitivity: float = 3.0, 
                 min_peak_distance: float = 0.0) -> List[Dict]:
    """
    基于二阶导数过零点检测的自动寻峰算法
    
    参数:
        x: x轴数据
        y: y轴数据
        sensitivity: 灵敏度阈值（信噪比倍数）
        min_peak_distance: 最小峰间距（x轴单位）
    
    返回:
        峰列表，每个峰包含 position, intensity, fwhm_estimate
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    
    noise = np.std(y[:min(50, len(y))])
    if noise < 1e-10:
        noise = 1e-10
    
    threshold = sensitivity * noise
    
    second_deriv = np.gradient(np.gradient(y, x), x)
    
    sign_changes = np.diff(np.sign(second_deriv))
    zero_crossings = np.where(sign_changes != 0)[0]
    
    peak_indices = []
    for zc in zero_crossings:
        if zc + 1 < len(second_deriv):
            if second_deriv[zc] > 0 and second_deriv[zc + 1] < 0:
                peak_indices.append(zc + 1)
    
    peaks = []
    for idx in peak_indices:
        if 0 < idx < len(y) - 1:
            if y[idx] > y[idx - 1] and y[idx] > y[idx + 1]:
                if y[idx] > threshold:
                    peaks.append(idx)
    
    if min_peak_distance > 0:
        peaks = _filter_by_distance(x, y, peaks, min_peak_distance)
    
    result = []
    for idx in peaks:
        fwhm = _estimate_fwhm(x, y, idx)
        result.append({
            'position': float(x[idx]),
            'intensity': float(y[idx]),
            'fwhm_estimate': float(fwhm),
            'index': int(idx),
        })
    
    return result


def _filter_by_distance(x: np.ndarray, y: np.ndarray, 
                        peak_indices: List[int], 
                        min_distance: float) -> List[int]:
    """根据最小峰间距过滤峰"""
    if len(peak_indices) == 0:
        return []
    
    sorted_peaks = sorted(peak_indices, key=lambda i: y[i], reverse=True)
    filtered = []
    
    for idx in sorted_peaks:
        too_close = False
        for f_idx in filtered:
            if abs(x[idx] - x[f_idx]) < min_distance:
                too_close = True
                break
        if not too_close:
            filtered.append(idx)
    
    return sorted(filtered)


def _estimate_fwhm(x: np.ndarray, y: np.ndarray, peak_idx: int) -> float:
    """估算半高宽"""
    half_max = y[peak_idx] / 2.0
    
    left_idx = peak_idx
    while left_idx > 0 and y[left_idx] > half_max:
        left_idx -= 1
    
    right_idx = peak_idx
    while right_idx < len(y) - 1 and y[right_idx] > half_max:
        right_idx += 1
    
    if y[left_idx] < half_max and left_idx < peak_idx:
        frac = (half_max - y[left_idx]) / (y[left_idx + 1] - y[left_idx])
        left_x = x[left_idx] + frac * (x[left_idx + 1] - x[left_idx])
    else:
        left_x = x[max(0, left_idx)]
    
    if y[right_idx] < half_max and right_idx > peak_idx:
        frac = (half_max - y[right_idx - 1]) / (y[right_idx] - y[right_idx - 1])
        right_x = x[right_idx - 1] + frac * (x[right_idx] - x[right_idx - 1])
    else:
        right_x = x[min(len(y) - 1, right_idx)]
    
    return right_x - left_x


def gaussian(x, amplitude, center, fwhm):
    """高斯峰型"""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))


def lorentzian(x, amplitude, center, fwhm):
    """洛伦兹峰型"""
    gamma = fwhm / 2.0
    return amplitude * gamma**2 / ((x - center)**2 + gamma**2)


def voigt(x, amplitude, center, gaussian_fwhm, lorentzian_fwhm):
    """Voigt峰型（高斯与洛伦兹的卷积近似）"""
    from scipy.special import voigt_profile
    sigma = gaussian_fwhm / (2 * np.sqrt(2 * np.log(2)))
    gamma = lorentzian_fwhm / 2.0
    return amplitude * voigt_profile(x - center, sigma, gamma)


def fit_peaks(x: np.ndarray, y: np.ndarray, 
              peaks: List[Dict], 
              peak_type: str = 'gaussian',
              background: bool = True) -> Tuple[List[Dict], np.ndarray, float]:
    """
    峰拟合
    
    参数:
        x: x轴数据
        y: y轴数据
        peaks: 初始峰参数列表
        peak_type: 峰类型 'gaussian', 'lorentzian', 'voigt'
        background: 是否拟合线性背景
    
    返回:
        拟合后的峰参数, 拟合曲线, R平方值
    """
    if len(peaks) == 0:
        return peaks, y, 0.0
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    num_peaks = len(peaks)
    p0 = []
    bounds_lower = []
    bounds_upper = []
    
    x_range = x.max() - x.min()
    
    for peak in peaks:
        p0.append(peak['intensity'])
        bounds_lower.append(0)
        bounds_upper.append(np.inf)
        
        p0.append(peak['position'])
        bounds_lower.append(peak['position'] - x_range * 0.1)
        bounds_upper.append(peak['position'] + x_range * 0.1)
        
        if peak_type == 'voigt':
            fwhm_est = peak.get('fwhm_estimate', x_range * 0.01)
            p0.append(fwhm_est * 0.5)
            p0.append(fwhm_est * 0.5)
            bounds_lower.extend([x_range * 0.001, x_range * 0.001])
            bounds_upper.extend([x_range * 0.1, x_range * 0.1])
        else:
            fwhm_est = peak.get('fwhm_estimate', x_range * 0.01)
            p0.append(fwhm_est)
            bounds_lower.append(x_range * 0.001)
            bounds_upper.append(x_range * 0.1)
    
    if background:
        p0.extend([0, 0])
        bounds_lower.extend([-np.inf, -np.inf])
        bounds_upper.extend([np.inf, np.inf])
    
    def residual(params):
        y_fit = _combined_peaks(x, params, num_peaks, peak_type, background)
        return y - y_fit
    
    try:
        result = optimize.least_squares(
            residual, p0, 
            bounds=(bounds_lower, bounds_upper),
            method='trf',
            max_nfev=1000
        )
        
        fitted_params = result.x
        y_fitted = _combined_peaks(x, fitted_params, num_peaks, peak_type, background)
        
        ss_res = np.sum(result.fun**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        fitted_peaks = []
        param_idx = 0
        for i in range(num_peaks):
            intensity = fitted_params[param_idx]
            center = fitted_params[param_idx + 1]
            
            if peak_type == 'voigt':
                g_fwhm = fitted_params[param_idx + 2]
                l_fwhm = fitted_params[param_idx + 3]
                fwhm = 0.5346 * l_fwhm + np.sqrt(0.2166 * l_fwhm**2 + g_fwhm**2)
                fitted_peaks.append({
                    'position': float(center),
                    'intensity': float(intensity),
                    'fwhm': float(fwhm),
                    'gaussian_fwhm': float(g_fwhm),
                    'lorentzian_fwhm': float(l_fwhm),
                })
                param_idx += 4
            else:
                fwhm = fitted_params[param_idx + 2]
                fitted_peaks.append({
                    'position': float(center),
                    'intensity': float(intensity),
                    'fwhm': float(fwhm),
                })
                param_idx += 3
        
        return fitted_peaks, y_fitted, float(r_squared)
        
    except Exception as e:
        print(f"拟合失败: {e}")
        return peaks, y, 0.0


def _combined_peaks(x: np.ndarray, params: np.ndarray, 
                    num_peaks: int, peak_type: str, 
                    background: bool) -> np.ndarray:
    """生成组合峰曲线"""
    y = np.zeros_like(x)
    param_idx = 0
    
    for i in range(num_peaks):
        amplitude = params[param_idx]
        center = params[param_idx + 1]
        
        if peak_type == 'voigt':
            g_fwhm = params[param_idx + 2]
            l_fwhm = params[param_idx + 3]
            y += voigt(x, amplitude, center, g_fwhm, l_fwhm)
            param_idx += 4
        elif peak_type == 'lorentzian':
            fwhm = params[param_idx + 2]
            y += lorentzian(x, amplitude, center, fwhm)
            param_idx += 3
        else:
            fwhm = params[param_idx + 2]
            y += gaussian(x, amplitude, center, fwhm)
            param_idx += 3
    
    if background:
        y += params[-2] + params[-1] * x
    
    return y


def segment_fit(x: np.ndarray, y: np.ndarray, 
                peaks: List[Dict], 
                peak_type: str = 'gaussian',
                max_peaks_per_segment: int = 15) -> Tuple[List[Dict], np.ndarray, float]:
    """
    分段拟合：当峰数量较多时，将光谱分割为多个区间分别拟合
    
    参数:
        x: x轴数据
        y: y轴数据
        peaks: 峰列表
        peak_type: 峰类型
        max_peaks_per_segment: 每段最大峰数
    
    返回:
        拟合后的峰参数, 拟合曲线, R平方值
    """
    if len(peaks) <= max_peaks_per_segment:
        return fit_peaks(x, y, peaks, peak_type)
    
    sorted_peaks = sorted(peaks, key=lambda p: p['position'])
    
    segments = []
    current_segment = [sorted_peaks[0]]
    
    for peak in sorted_peaks[1:]:
        if len(current_segment) >= max_peaks_per_segment:
            segments.append(current_segment)
            current_segment = [peak]
        else:
            current_segment.append(peak)
    
    if current_segment:
        segments.append(current_segment)
    
    all_fitted_peaks = []
    y_fitted = np.zeros_like(y)
    
    for i, seg_peaks in enumerate(segments):
        if i == 0:
            left_bound = x[0]
        else:
            prev_seg_last = segments[i - 1][-1]['position']
            left_bound = (prev_seg_last + seg_peaks[0]['position']) / 2
        
        if i == len(segments) - 1:
            right_bound = x[-1]
        else:
            next_seg_first = segments[i + 1][0]['position']
            right_bound = (seg_peaks[-1]['position'] + next_seg_first) / 2
        
        mask = (x >= left_bound) & (x <= right_bound)
        x_seg = x[mask]
        y_seg = y[mask]
        
        if len(x_seg) > 0 and len(seg_peaks) > 0:
            fitted_seg, y_seg_fit, r2 = fit_peaks(x_seg, y_seg, seg_peaks, peak_type)
            all_fitted_peaks.extend(fitted_seg)
            y_fitted[mask] = y_seg_fit
    
    ss_res = np.sum((y - y_fitted)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return all_fitted_peaks, y_fitted, float(r_squared)


def calculate_peak_areas(x: np.ndarray, fitted_peaks: List[Dict], 
                         peak_type: str = 'gaussian') -> List[float]:
    """计算各峰面积"""
    areas = []
    for peak in fitted_peaks:
        if peak_type == 'gaussian':
            sigma = peak['fwhm'] / (2 * np.sqrt(2 * np.log(2)))
            area = peak['intensity'] * sigma * np.sqrt(2 * np.pi)
        elif peak_type == 'lorentzian':
            gamma = peak['fwhm'] / 2.0
            area = peak['intensity'] * np.pi * gamma
        elif peak_type == 'voigt':
            g_sigma = peak.get('gaussian_fwhm', peak['fwhm'] * 0.5) / (2 * np.sqrt(2 * np.log(2)))
            l_gamma = peak.get('lorentzian_fwhm', peak['fwhm'] * 0.5) / 2.0
            area = peak['intensity'] * (g_sigma * np.sqrt(2 * np.pi) + l_gamma * np.pi) * 0.5
        else:
            area = peak['intensity'] * peak['fwhm']
        
        areas.append(float(area))
    
    return areas
