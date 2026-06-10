"""
光谱检索与多样品对比模块
相似度匹配、差谱、比值谱、图片导出
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import signal
from .spectrum import Spectrum


def cosine_similarity(spec1: Spectrum, spec2: Spectrum, 
                      num_points: int = 1000) -> float:
    """
    余弦相似度
    
    将光谱离散化为等间距向量后计算余弦相似度
    """
    x_min = max(spec1.x.min(), spec2.x.min())
    x_max = min(spec1.x.max(), spec2.x.max())
    
    if x_max <= x_min:
        return 0.0
    
    x_new = np.linspace(x_min, x_max, num_points)
    
    y1 = np.interp(x_new, spec1.x, spec1.y)
    y2 = np.interp(x_new, spec2.x, spec2.y)
    
    y1 = y1 - y1.mean()
    y2 = y2 - y2.mean()
    
    norm1 = np.linalg.norm(y1)
    norm2 = np.linalg.norm(y2)
    
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    
    similarity = np.dot(y1, y2) / (norm1 * norm2)
    
    return float(max(0, min(1, similarity)))


def pearson_correlation(spec1: Spectrum, spec2: Spectrum,
                        num_points: int = 1000) -> float:
    """
    皮尔逊相关系数
    """
    x_min = max(spec1.x.min(), spec2.x.min())
    x_max = min(spec1.x.max(), spec2.x.max())
    
    if x_max <= x_min:
        return 0.0
    
    x_new = np.linspace(x_min, x_max, num_points)
    
    y1 = np.interp(x_new, spec1.x, spec1.y)
    y2 = np.interp(x_new, spec2.x, spec2.y)
    
    y1_mean = y1.mean()
    y2_mean = y2.mean()
    
    numerator = np.sum((y1 - y1_mean) * (y2 - y2_mean))
    denominator = np.sqrt(np.sum((y1 - y1_mean)**2) * np.sum((y2 - y2_mean)**2))
    
    if denominator < 1e-10:
        return 0.0
    
    correlation = numerator / denominator
    
    return float(max(-1, min(1, correlation)))


def spectral_band_matching(spec1: Spectrum, spec2: Spectrum,
                           tolerance: float = 0.02,
                           sensitivity: float = 3.0) -> float:
    """
    谱带匹配法
    
    只比较峰位不比较峰型
    """
    from .peak_fitting import detect_peaks
    
    peaks1 = detect_peaks(spec1.x, spec1.y, sensitivity)
    peaks2 = detect_peaks(spec2.x, spec2.y, sensitivity)
    
    if len(peaks1) == 0 or len(peaks2) == 0:
        return 0.0
    
    positions1 = np.array([p['position'] for p in peaks1])
    positions2 = np.array([p['position'] for p in peaks2])
    
    matched = 0
    for pos1 in positions1:
        for pos2 in positions2:
            if abs(pos1 - pos2) <= tolerance:
                matched += 1
                break
    
    total = max(len(peaks1), len(peaks2))
    
    return float(matched / total) if total > 0 else 0.0


def search_spectrum(query_spec: Spectrum,
                    library: List[Spectrum],
                    method: str = 'cosine',
                    top_n: int = 5) -> List[Dict]:
    """
    光谱检索
    
    参数:
        query_spec: 查询光谱
        library: 光谱库
        method: 相似度方法 (cosine, pearson, band)
        top_n: 返回前N个结果
    
    返回:
        检索结果列表
    """
    results = []
    
    for spec in library:
        if method == 'cosine':
            similarity = cosine_similarity(query_spec, spec)
        elif method == 'pearson':
            similarity = pearson_correlation(query_spec, spec)
        elif method == 'band':
            similarity = spectral_band_matching(query_spec, spec)
        else:
            similarity = cosine_similarity(query_spec, spec)
        
        results.append({
            'spectrum': spec,
            'similarity': similarity,
            'method': method,
        })
    
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return results[:top_n]


def compute_difference_spectrum(spec_a: Spectrum, 
                                spec_b: Spectrum,
                                normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算差谱 (A - B)
    
    返回: (x, y_diff)
    """
    x_min = max(spec_a.x.min(), spec_b.x.min())
    x_max = min(spec_a.x.max(), spec_b.x.max())
    
    num_points = max(spec_a.num_points, spec_b.num_points)
    x_new = np.linspace(x_min, x_max, num_points)
    
    y_a = np.interp(x_new, spec_a.x, spec_a.y)
    y_b = np.interp(x_new, spec_b.x, spec_b.y)
    
    if normalize:
        if y_a.max() > 0:
            y_a = y_a / y_a.max()
        if y_b.max() > 0:
            y_b = y_b / y_b.max()
    
    y_diff = y_a - y_b
    
    return x_new, y_diff


def compute_ratio_spectrum(spec_a: Spectrum,
                           spec_b: Spectrum,
                           normalize: bool = True,
                           epsilon: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算比值谱 (A / B)
    
    返回: (x, y_ratio)
    """
    x_min = max(spec_a.x.min(), spec_b.x.min())
    x_max = min(spec_a.x.max(), spec_b.x.max())
    
    num_points = max(spec_a.num_points, spec_b.num_points)
    x_new = np.linspace(x_min, x_max, num_points)
    
    y_a = np.interp(x_new, spec_a.x, spec_a.y)
    y_b = np.interp(x_new, spec_b.x, spec_b.y)
    
    if normalize:
        if y_a.max() > 0:
            y_a = y_a / y_a.max()
        if y_b.max() > 0:
            y_b = y_b / y_b.max()
    
    y_ratio = np.where(np.abs(y_b) > epsilon, y_a / (y_b + epsilon), 0.0)
    
    return x_new, y_ratio


def align_spectra(spectra: List[Spectrum]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    将多条光谱对齐到相同的x轴
    
    返回: (x_common, [y1, y2, ...])
    """
    if len(spectra) == 0:
        return np.array([]), []
    
    x_min = max(spec.x.min() for spec in spectra)
    x_max = min(spec.x.max() for spec in spectra)
    
    max_points = max(spec.num_points for spec in spectra)
    x_common = np.linspace(x_min, x_max, max_points)
    
    y_list = []
    for spec in spectra:
        y = np.interp(x_common, spec.x, spec.y)
        y_list.append(y)
    
    return x_common, y_list


def normalize_spectra_for_comparison(spectra: List[Spectrum],
                                     method: str = 'max') -> List[Spectrum]:
    """
    归一化多条光谱用于对比
    
    参数:
        spectra: 光谱列表
        method: 归一化方法 (max, area, None)
    
    返回:
        归一化后的光谱列表
    """
    from .preprocessing import Normalization
    
    result = []
    for spec in spectra:
        if method == 'none' or method is None:
            result.append(spec.copy())
        else:
            norm = Normalization(method=method)
            x, y = norm.apply(spec.x, spec.y)
            new_spec = spec.copy()
            new_spec.x = x
            new_spec.y = y
            result.append(new_spec)
    
    return result


def export_spectrum_plot(spectra: List[Spectrum],
                         output_path: str,
                         title: str = '',
                         x_label: str = '',
                         y_label: str = '',
                         dpi: int = 300,
                         format: str = 'png') -> bool:
    """
    导出光谱图为高分辨率图片
    
    参数:
        spectra: 光谱列表
        output_path: 输出路径
        title: 图表标题
        x_label: x轴标签
        y_label: y轴标签
        dpi: 分辨率
        format: 格式 (png, svg, pdf)
    
    返回:
        是否成功
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(spectra)))
        
        for i, spec in enumerate(spectra):
            ax.plot(spec.x, spec.y, label=spec.name, color=colors[i % len(colors)], linewidth=1.5)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        if x_label:
            ax.set_xlabel(x_label, fontsize=12)
        if y_label:
            ax.set_ylabel(y_label, fontsize=12)
        
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, format=format)
        plt.close()
        
        return True
    except Exception as e:
        print(f"导出图片失败: {e}")
        return False


def build_sample_library() -> List[Spectrum]:
    """
    构建示例光谱库（用于演示）
    """
    from .pdf_database import get_pdf_database, PDFCard
    from .spectrum import Spectrum, SpectrumType
    
    library = []
    db = get_pdf_database()
    
    for card in db[:20]:
        x = np.array(card.two_theta_list)
        y = np.array(card.intensity_list)
        
        x_smooth = np.linspace(x.min() - 2, x.max() + 2, 500)
        y_smooth = np.zeros_like(x_smooth)
        
        for xi, yi in zip(x, y):
            y_smooth += yi * np.exp(-(x_smooth - xi)**2 / (2 * 0.1**2))
        
        spec = Spectrum(
            name=card.name,
            spectrum_type=SpectrumType.XRD,
            x=x_smooth,
            y=y_smooth,
            x_unit='2θ (°)',
            y_unit='强度',
            metadata={'formula': card.formula},
        )
        library.append(spec)
    
    return library
