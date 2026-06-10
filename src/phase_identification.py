"""
物相鉴定模块
XRD数据的物相检索、匹配和半定量分析
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from .spectrum import Spectrum, SpectrumType
from .pdf_database import PDFCard, get_pdf_database
from .peak_fitting import detect_peaks


def match_phases(sample_peaks: List[Dict], 
                 tolerance: float = 0.05,
                 top_n: int = 10,
                 intensity_weight: bool = True) -> List[Dict]:
    """
    将样品峰位与标准PDF卡片匹配
    
    参数:
        sample_peaks: 样品峰列表，每个峰包含 position 和 intensity
        tolerance: 2θ偏移容差（度）
        top_n: 返回前N个匹配结果
        intensity_weight: 是否考虑强度权重
    
    返回:
        匹配结果列表，按匹配度排序
    """
    db = get_pdf_database()
    
    sample_positions = np.array([p['position'] for p in sample_peaks])
    sample_intensities = np.array([p.get('intensity', 1.0) for p in sample_peaks])
    sample_intensities_norm = sample_intensities / sample_intensities.max() if len(sample_intensities) > 0 else np.array([])
    
    results = []
    
    for card in db:
        card_positions = np.array(card.two_theta_list)
        card_intensities = np.array(card.intensity_list)
        card_intensities_norm = card_intensities / card_intensities.max()
        
        matched_peaks = []
        matched_card_indices = []
        
        for i, sample_pos in enumerate(sample_positions):
            for j, card_pos in enumerate(card_positions):
                if abs(sample_pos - card_pos) <= tolerance:
                    if j not in matched_card_indices:
                        matched_peaks.append({
                            'sample_position': float(sample_pos),
                            'card_position': float(card_pos),
                            'sample_intensity': float(sample_intensities[i]),
                            'card_intensity': float(card_intensities[j]),
                            'delta': float(sample_pos - card_pos),
                        })
                        matched_card_indices.append(j)
                    break
        
        num_matched = len(matched_peaks)
        num_card_peaks = card.num_peaks
        
        if num_card_peaks > 0:
            match_ratio = num_matched / num_card_peaks
        else:
            match_ratio = 0.0
        
        if intensity_weight and len(matched_peaks) > 0:
            weighted_score = 0.0
            total_weight = 0.0
            
            for peak in matched_peaks:
                weight = peak['card_intensity'] / 100.0
                sample_int_norm = peak['sample_intensity'] / sample_intensities.max() if sample_intensities.max() > 0 else 0
                card_int_norm = peak['card_intensity'] / 100.0
                intensity_similarity = 1.0 - abs(sample_int_norm - card_int_norm)
                weighted_score += weight * intensity_similarity
                total_weight += weight
            
            if total_weight > 0:
                intensity_match_score = weighted_score / total_weight
            else:
                intensity_match_score = 0.0
            
            final_score = 0.6 * match_ratio + 0.4 * intensity_match_score
        else:
            final_score = match_ratio
        
        results.append({
            'card': card,
            'match_score': float(final_score),
            'match_ratio': float(match_ratio),
            'num_matched': num_matched,
            'num_card_peaks': num_card_peaks,
            'matched_peaks': matched_peaks,
        })
    
    results.sort(key=lambda x: x['match_score'], reverse=True)
    
    return results[:top_n]


def identify_phases(spectrum: Spectrum, 
                    sensitivity: float = 3.0,
                    min_peak_distance: float = 0.1,
                    tolerance: float = 0.05,
                    top_n: int = 10) -> Tuple[List[Dict], List[Dict]]:
    """
    物相鉴定完整流程
    
    参数:
        spectrum: XRD光谱数据
        sensitivity: 峰检测灵敏度
        min_peak_distance: 最小峰间距
        tolerance: 匹配容差
        top_n: 返回前N个结果
    
    返回:
        (峰列表, 匹配结果列表)
    """
    if spectrum.spectrum_type not in [SpectrumType.XRD, SpectrumType.UNKNOWN]:
        raise ValueError("物相鉴定仅适用于XRD数据")
    
    peaks = detect_peaks(spectrum.x, spectrum.y, sensitivity, min_peak_distance)
    
    if len(peaks) == 0:
        return [], []
    
    match_results = match_phases(peaks, tolerance, top_n)
    
    return peaks, match_results


def rir_quantitative(selected_phases: List[PDFCard],
                     sample_peaks: List[Dict],
                     tolerance: float = 0.05) -> List[Dict]:
    """
    RIR（Reference Intensity Ratio）半定量分析
    
    参数:
        selected_phases: 选定的矿物相列表
        sample_peaks: 样品峰列表
        tolerance: 峰位匹配容差
    
    返回:
        各相含量百分比
    """
    if len(selected_phases) == 0:
        return []
    
    phase_intensities = {}
    
    for phase in selected_phases:
        card_positions = phase.two_theta_list
        card_intensities = phase.intensity_list
        
        max_peak_intensity = 0.0
        
        for sample_peak in sample_peaks:
            sample_pos = sample_peak['position']
            sample_int = sample_peak['intensity']
            
            for i, card_pos in enumerate(card_positions):
                if abs(sample_pos - card_pos) <= tolerance:
                    rel_int = card_intensities[i] / 100.0
                    if rel_int > 0:
                        estimated_int = sample_int / rel_int
                        if estimated_int > max_peak_intensity:
                            max_peak_int = estimated_int
                    break
        
        phase_intensities[phase.name] = max_peak_int
    
    rir_values = {phase.name: phase.rir for phase in selected_phases}
    
    contents = {}
    total = 0.0
    
    for phase_name, intensity in phase_intensities.items():
        rir = rir_values.get(phase_name, 1.0)
        if rir > 0:
            content = intensity / rir
            contents[phase_name] = content
            total += content
    
    if total > 0:
        for phase_name in contents:
            contents[phase_name] = (contents[phase_name] / total) * 100.0
    
    result = []
    for phase in selected_phases:
        result.append({
            'phase': phase.name,
            'formula': phase.formula,
            'rir': phase.rir,
            'weight_percent': float(contents.get(phase.name, 0.0)),
        })
    
    result.sort(key=lambda x: x['weight_percent'], reverse=True)
    
    return result


def estimate_phase_abundance(spectrum: Spectrum,
                             selected_phase_names: List[str],
                             sensitivity: float = 3.0,
                             min_peak_distance: float = 0.1,
                             tolerance: float = 0.05) -> List[Dict]:
    """
    估算指定物相的含量
    
    参数:
        spectrum: XRD光谱
        selected_phase_names: 选定的物相名称列表
        sensitivity: 峰检测灵敏度
        min_peak_distance: 最小峰间距
        tolerance: 匹配容差
    
    返回:
        各相含量估算结果
    """
    from .pdf_database import get_pdf_card_by_name
    
    selected_phases = []
    for name in selected_phase_names:
        card = get_pdf_card_by_name(name)
        if card:
            selected_phases.append(card)
    
    if len(selected_phases) < 2:
        raise ValueError("请至少选择2个物相进行半定量分析")
    
    if len(selected_phases) > 5:
        raise ValueError("最多只能选择5个物相进行半定量分析")
    
    peaks = detect_peaks(spectrum.x, spectrum.y, sensitivity, min_peak_distance)
    
    return rir_quantitative(selected_phases, peaks, tolerance)
