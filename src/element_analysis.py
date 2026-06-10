"""
元素分析模块（XRF）
元素识别、定量分析、基体校正
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import optimize
from .spectrum import Spectrum, SpectrumType
from .peak_fitting import detect_peaks, fit_peaks, calculate_peak_areas
from .xray_database import XRayLine, find_elements_by_energy, get_element_lines, get_xray_database


def identify_elements(spectrum: Spectrum,
                      sensitivity: float = 3.0,
                      min_peak_distance: float = 0.1,
                      energy_tolerance: float = 0.1) -> List[Dict]:
    """
    识别XRF光谱中的元素
    
    参数:
        spectrum: XRF光谱数据
        sensitivity: 峰检测灵敏度
        min_peak_distance: 最小峰间距（keV）
        energy_tolerance: 能量匹配容差（keV）
    
    返回:
        识别到的元素列表
    """
    if spectrum.spectrum_type not in [SpectrumType.XRF, SpectrumType.UNKNOWN]:
        raise ValueError("元素分析仅适用于XRF数据")
    
    peaks = detect_peaks(spectrum.x, spectrum.y, sensitivity, min_peak_distance)
    
    identified = []
    
    for peak in peaks:
        energy = peak['position']
        lines = find_elements_by_energy(energy, energy_tolerance)
        
        if lines:
            best_line = lines[0]
            identified.append({
                'peak_energy': float(energy),
                'peak_intensity': float(peak['intensity']),
                'element': best_line.element,
                'line': best_line.line,
                'expected_energy': best_line.energy,
                'delta_energy': float(energy - best_line.energy),
                'relative_intensity': best_line.relative_intensity,
                'all_candidates': [
                    {'element': l.element, 'line': l.line, 'energy': l.energy}
                    for l in lines[:5]
                ],
            })
    
    elements_dict = {}
    for item in identified:
        elem = item['element']
        if elem not in elements_dict:
            elements_dict[elem] = {
                'element': elem,
                'lines': [],
                'total_intensity': 0.0,
            }
        elements_dict[elem]['lines'].append(item)
        elements_dict[elem]['total_intensity'] += item['peak_intensity']
    
    result = list(elements_dict.values())
    result.sort(key=lambda x: x['total_intensity'], reverse=True)
    
    return result


def get_element_peak_labels(spectrum: Spectrum,
                            identified_elements: List[Dict]) -> List[Dict]:
    """
    获取用于在光谱图上标注的元素标签
    
    参数:
        spectrum: 光谱数据
        identified_elements: 识别到的元素列表
    
    返回:
        标签列表
    """
    labels = []
    for elem_data in identified_elements:
        for line_info in elem_data['lines']:
            labels.append({
                'x': line_info['peak_energy'],
                'y': line_info['peak_intensity'],
                'text': f"{line_info['element']} {line_info['line']}",
            })
    return labels


class CalibrationCurve:
    """校准曲线"""
    
    def __init__(self, method: str = 'linear'):
        self.method = method  # linear, quadratic
        self.coefficients = None
        self.concentrations = []
        self.peak_areas = []
    
    def fit(self, concentrations: List[float], peak_areas: List[float]):
        """拟合校准曲线"""
        self.concentrations = np.array(concentrations)
        self.peak_areas = np.array(peak_areas)
        
        if self.method == 'linear':
            coeffs = np.polyfit(self.peak_areas, self.concentrations, 1)
        elif self.method == 'quadratic':
            coeffs = np.polyfit(self.peak_areas, self.concentrations, 2)
        else:
            raise ValueError(f"不支持的拟合方法: {self.method}")
        
        self.coefficients = coeffs
    
    def predict(self, peak_area: float) -> float:
        """根据峰面积预测浓度"""
        if self.coefficients is None:
            raise ValueError("请先拟合校准曲线")
        
        return float(np.polyval(self.coefficients, peak_area))
    
    def predict_confidence_interval(self, peak_area: float, confidence: float = 0.95) -> Tuple[float, float]:
        """计算预测的置信区间"""
        if self.coefficients is None:
            raise ValueError("请先拟合校准曲线")
        
        n = len(self.peak_areas)
        if n < 3:
            pred = self.predict(peak_area)
            return pred * 0.9, pred * 1.1
        
        predicted_conc = np.polyval(self.coefficients, self.peak_areas)
        residuals = self.concentrations - predicted_conc
        std_error = np.sqrt(np.sum(residuals**2) / (n - 2))
        
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, n - 2)
        
        mean_area = np.mean(self.peak_areas)
        ss_xx = np.sum((self.peak_areas - mean_area)**2)
        
        if ss_xx > 0:
            interval = t_value * std_error * np.sqrt(
                1 + 1/n + (peak_area - mean_area)**2 / ss_xx
            )
        else:
            interval = t_value * std_error
        
        pred = self.predict(peak_area)
        return float(pred - interval), float(pred + interval)


def quantitative_analysis(spectrum: Spectrum,
                          element: str,
                          calibration_curve: CalibrationCurve,
                          sensitivity: float = 3.0,
                          peak_type: str = 'gaussian',
                          matrix_correction: Optional[Dict] = None) -> Dict:
    """
    定量分析（简化FP法，基于校准曲线）
    
    参数:
        spectrum: XRF光谱
        element: 目标元素
        calibration_curve: 校准曲线
        sensitivity: 峰检测灵敏度
        peak_type: 峰类型
        matrix_correction: 基体校正参数
    
    返回:
        定量分析结果
    """
    element_lines = get_element_lines(element)
    
    if not element_lines:
        raise ValueError(f"未找到元素 {element} 的特征线数据")
    
    k_lines = [l for l in element_lines if l.line.startswith('K')]
    if not k_lines:
        k_lines = element_lines
    
    main_line = k_lines[0]
    target_energy = main_line.energy
    
    peaks = detect_peaks(spectrum.x, spectrum.y, sensitivity, 0.05)
    
    target_peak = None
    min_delta = float('inf')
    for peak in peaks:
        delta = abs(peak['position'] - target_energy)
        if delta < min_delta:
            min_delta = delta
            target_peak = peak
    
    if target_peak is None:
        return {
            'element': element,
            'line': main_line.line,
            'peak_found': False,
            'concentration': 0.0,
            'confidence_interval': (0.0, 0.0),
        }
    
    fitted_peaks, y_fitted, r2 = fit_peaks(
        spectrum.x, spectrum.y, [target_peak], peak_type
    )
    
    areas = calculate_peak_areas(spectrum.x, fitted_peaks, peak_type)
    peak_area = areas[0] if areas else 0.0
    
    if matrix_correction:
        correction_factor = matrix_correction.get('factor', 1.0)
        peak_area_corrected = peak_area * correction_factor
    else:
        peak_area_corrected = peak_area
    
    concentration = calibration_curve.predict(peak_area_corrected)
    ci_lower, ci_upper = calibration_curve.predict_confidence_interval(peak_area_corrected)
    
    return {
        'element': element,
        'line': main_line.line,
        'peak_energy': target_peak['position'],
        'peak_area': peak_area,
        'peak_area_corrected': peak_area_corrected,
        'concentration': concentration,
        'confidence_interval': (ci_lower, ci_upper),
        'r_squared': r2,
        'peak_found': True,
    }


def matrix_correction_empirical(peak_area: float,
                                matrix_type: str = 'general') -> float:
    """
    经验系数法基体校正
    
    参数:
        peak_area: 原始峰面积
        matrix_type: 基体类型
    
    返回:
        校正后的峰面积
    """
    correction_factors = {
        'general': 1.0,
        'heavy_matrix': 0.85,
        'light_matrix': 1.15,
        'soil': 0.92,
        'rock': 0.95,
        'water': 1.05,
        'biological': 1.1,
    }
    
    factor = correction_factors.get(matrix_type, 1.0)
    return peak_area * factor


def multi_element_analysis(spectrum: Spectrum,
                           elements: List[str],
                           calibration_curves: Dict[str, CalibrationCurve],
                           matrix_type: str = 'general') -> List[Dict]:
    """多元素同时定量分析"""
    results = []
    
    for element in elements:
        if element in calibration_curves:
            curve = calibration_curves[element]
        else:
            curve = CalibrationCurve('linear')
            curve.fit([0, 100], [0, 1000])
        
        result = quantitative_analysis(
            spectrum, element, curve,
            matrix_correction={'factor': matrix_correction_empirical(1.0, matrix_type)}
        )
        results.append(result)
    
    results.sort(key=lambda x: x.get('concentration', 0), reverse=True)
    
    return results
