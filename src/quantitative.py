"""
定量分析模块
标准曲线法、内标法
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats
from dataclasses import dataclass, field


@dataclass
class StandardCurveResult:
    """标准曲线法结果"""
    method: str  # linear, quadratic, cubic
    coefficients: np.ndarray
    r_squared: float
    concentrations: np.ndarray
    responses: np.ndarray
    predicted_concentration: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    confidence_level: float = 0.95


def standard_curve_method(concentrations: List[float],
                          responses: List[float],
                          unknown_response: Optional[float] = None,
                          method: str = 'linear',
                          confidence_level: float = 0.95) -> StandardCurveResult:
    """
    标准曲线法
    
    参数:
        concentrations: 标准样品浓度
        responses: 对应响应值（峰面积/峰高）
        unknown_response: 未知样品响应值
        method: 拟合方法 (linear, quadratic, cubic)
        confidence_level: 置信水平
    
    返回:
        标准曲线结果
    """
    conc = np.array(concentrations, dtype=float)
    resp = np.array(responses, dtype=float)
    
    if method == 'linear':
        degree = 1
    elif method == 'quadratic':
        degree = 2
    elif method == 'cubic':
        degree = 3
    else:
        degree = 1
    
    coeffs = np.polyfit(resp, conc, degree)
    
    predicted_conc = np.polyval(coeffs, resp)
    ss_res = np.sum((conc - predicted_conc)**2)
    ss_tot = np.sum((conc - np.mean(conc))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    predicted_concentration = 0.0
    ci_lower = 0.0
    ci_upper = 0.0
    
    if unknown_response is not None:
        predicted_concentration = float(np.polyval(coeffs, unknown_response))
        
        n = len(conc)
        if n > degree + 1:
            mse = ss_res / (n - degree - 1)
            rmse = np.sqrt(mse)
            
            t_value = stats.t.ppf((1 + confidence_level) / 2, n - degree - 1)
            
            mean_resp = np.mean(resp)
            ss_xx = np.sum((resp - mean_resp)**2)
            
            if ss_xx > 0:
                se_pred = rmse * np.sqrt(
                    1 + 1/n + (unknown_response - mean_resp)**2 / ss_xx
                )
            else:
                se_pred = rmse
            
            ci_lower = predicted_concentration - t_value * se_pred
            ci_upper = predicted_concentration + t_value * se_pred
        else:
            ci_lower = predicted_concentration * 0.9
            ci_upper = predicted_concentration * 1.1
    
    return StandardCurveResult(
        method=method,
        coefficients=coeffs,
        r_squared=float(r_squared),
        concentrations=conc,
        responses=resp,
        predicted_concentration=float(predicted_concentration),
        confidence_interval=(float(ci_lower), float(ci_upper)),
        confidence_level=confidence_level,
    )


def internal_standard_method(analyte_responses: List[float],
                             internal_standard_responses: List[float],
                             concentrations: List[float],
                             unknown_analyte_response: float,
                             unknown_is_response: float,
                             method: str = 'linear',
                             confidence_level: float = 0.95) -> StandardCurveResult:
    """
    内标法
    
    参数:
        analyte_responses: 分析物响应值列表
        internal_standard_responses: 内标物响应值列表
        concentrations: 对应浓度列表
        unknown_analyte_response: 未知样品分析物响应
        unknown_is_response: 未知样品内标物响应
        method: 拟合方法
        confidence_level: 置信水平
    
    返回:
        定量结果
    """
    analyte = np.array(analyte_responses, dtype=float)
    is_resp = np.array(internal_standard_responses, dtype=float)
    
    ratios = analyte / is_resp
    unknown_ratio = unknown_analyte_response / unknown_is_response
    
    return standard_curve_method(
        concentrations=concentrations,
        responses=list(ratios),
        unknown_response=unknown_ratio,
        method=method,
        confidence_level=confidence_level,
    )


def calculate_peak_area_from_spectrum(x: np.ndarray, y: np.ndarray,
                                      peak_position: float,
                                      window: float = 0.1) -> float:
    """
    从光谱中计算指定峰的面积（积分法）
    
    参数:
        x: x轴数据
        y: y轴数据
        peak_position: 峰位
        window: 积分窗口宽度
    
    返回:
        峰面积
    """
    mask = (x >= peak_position - window/2) & (x <= peak_position + window/2)
    x_window = x[mask]
    y_window = y[mask]
    
    if len(x_window) < 2:
        return 0.0
    
    baseline = np.interp(peak_position, [x_window[0], x_window[-1]], [y_window[0], y_window[-1]])
    y_corrected = y_window - baseline
    y_corrected = np.maximum(y_corrected, 0)
    
    area = np.trapz(y_corrected, x_window)
    
    return float(area)


def limit_of_detection(concentrations: np.ndarray, 
                       responses: np.ndarray,
                       method: str = 'linear',
                       k: float = 3.0) -> float:
    """
    计算检出限 (LOD)
    
    参数:
        concentrations: 浓度
        responses: 响应
        method: 拟合方法
        k: 信噪比倍数（通常为3）
    
    返回:
        检出限
    """
    if method == 'linear':
        slope, intercept, r_value, p_value, std_err = stats.linregress(responses, concentrations)
        residual_std = np.std(concentrations - (slope * responses + intercept))
        lod = k * residual_std / slope
        return float(abs(lod))
    else:
        coeffs = np.polyfit(responses, concentrations, 2 if method == 'quadratic' else 3)
        predicted = np.polyval(coeffs, responses)
        residual_std = np.std(concentrations - predicted)
        slope_at_low = coeffs[-2] if len(coeffs) > 2 else coeffs[-1]
        if abs(slope_at_low) > 1e-10:
            lod = k * residual_std / abs(slope_at_low)
        else:
            lod = k * residual_std
        return float(lod)


def limit_of_quantitation(concentrations: np.ndarray,
                          responses: np.ndarray,
                          method: str = 'linear',
                          k: float = 10.0) -> float:
    """
    计算定量限 (LOQ)
    
    参数:
        concentrations: 浓度
        responses: 响应
        method: 拟合方法
        k: 信噪比倍数（通常为10）
    
    返回:
        定量限
    """
    return limit_of_detection(concentrations, responses, method, k)
