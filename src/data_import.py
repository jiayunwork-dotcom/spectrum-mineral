"""
数据导入模块
支持CSV和JCAMP-DX格式，自动识别光谱类型
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from .spectrum import Spectrum, SpectrumType


def detect_spectrum_type(x: np.ndarray, header_info: dict = None) -> SpectrumType:
    """
    根据横轴范围和文件头信息自动识别光谱类型
    
    XRD: 2θ角度，通常5-90度
    XRF: 能量keV，通常0-40keV
    Raman: 拉曼位移cm⁻¹，通常100-4000cm⁻¹
    IR: 波数cm⁻¹，通常400-4000cm⁻¹
    """
    x_min, x_max = float(x.min()), float(x.max())
    
    if header_info:
        title = str(header_info.get('title', '')).lower()
        if any(kw in title for kw in ['xrd', 'diffraction', '衍射']):
            return SpectrumType.XRD
        if any(kw in title for kw in ['xrf', 'fluorescence', '荧光']):
            return SpectrumType.XRF
        if any(kw in title for kw in ['raman', '拉曼']):
            return SpectrumType.RAMAN
        if any(kw in title for kw in ['ir', 'infrared', '红外', 'ftir']):
            return SpectrumType.IR
    
    x_range = x_max - x_min
    
    if 2 <= x_min and x_max <= 180 and x_range > 10:
        return SpectrumType.XRD
    
    if 0 <= x_min and x_max <= 150 and x_range > 5:
        return SpectrumType.XRF
    
    if 50 <= x_min and x_max <= 5000 and x_range > 100:
        if x_min < 200:
            return SpectrumType.RAMAN
        else:
            if x_max > 2500:
                return SpectrumType.IR
            return SpectrumType.RAMAN
    
    return SpectrumType.UNKNOWN


def read_csv(file_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """读取CSV格式光谱数据"""
    try:
        df = pd.read_csv(file_path, header=None, comment='#')
    except:
        df = pd.read_csv(file_path, comment='#')
    
    if df.shape[1] < 2:
        raise ValueError("CSV文件至少需要两列数据")
    
    x = df.iloc[:, 0].values.astype(float)
    y = df.iloc[:, 1].values.astype(float)
    
    header_info = {}
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or line.startswith('//'):
                    if ':' in line:
                        key, val = line.lstrip('#/').split(':', 1)
                        header_info[key.strip()] = val.strip()
                else:
                    break
    except:
        pass
    
    return x, y, header_info


def read_jcamp(file_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """读取JCAMP-DX格式光谱数据"""
    try:
        import jcamp
        data = jcamp.JCAMP_reader(file_path)
        x = data['x']
        y = data['y']
        header_info = {k: v for k, v in data.items() if k not in ['x', 'y']}
        return x, y, header_info
    except ImportError:
        return _read_jcamp_manual(file_path)


def _read_jcamp_manual(file_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """手动解析JCAMP-DX格式（后备方案）"""
    x = []
    y = []
    header_info = {}
    in_data = False
    data_format = 'XY'
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('##'):
                if '=' in line:
                    key, val = line.lstrip('#').split('=', 1)
                    header_info[key.strip()] = val.strip()
                if line.startswith('##XYDATA') or line.startswith('##PEAK TABLE'):
                    in_data = True
                continue
            
            if in_data:
                if line.startswith('##END'):
                    break
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x_val = float(parts[0])
                        for y_val_str in parts[1:]:
                            y_val = float(y_val_str)
                            x.append(x_val)
                            y.append(y_val)
                    except ValueError:
                        continue
    
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    if len(x) == 0:
        raise ValueError("未能从JCAMP-DX文件中解析出数据")
    
    return x, y, header_info


def load_spectrum(file_path: str, name: Optional[str] = None, 
                  force_type: Optional[SpectrumType] = None) -> Spectrum:
    """加载单个光谱文件"""
    import os
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.csv', '.txt']:
        x, y, header_info = read_csv(file_path)
    elif ext in ['.dx', '.jdx', '.jcm']:
        x, y, header_info = read_jcamp(file_path)
    else:
        try:
            x, y, header_info = read_csv(file_path)
        except:
            try:
                x, y, header_info = read_jcamp(file_path)
            except:
                raise ValueError(f"不支持的文件格式: {ext}")
    
    if force_type:
        spectrum_type = force_type
    else:
        spectrum_type = detect_spectrum_type(x, header_info)
    
    spectrum_name = name if name else os.path.splitext(os.path.basename(file_path))[0]
    
    x_unit = _get_x_unit(spectrum_type, header_info)
    y_unit = _get_y_unit(header_info)
    
    return Spectrum(
        name=spectrum_name,
        spectrum_type=spectrum_type,
        x=x,
        y=y,
        x_unit=x_unit,
        y_unit=y_unit,
        metadata=header_info,
    )


def load_spectra(file_paths: List[str], force_type: Optional[SpectrumType] = None) -> List[Spectrum]:
    """批量加载多个光谱文件"""
    spectra = []
    for fp in file_paths:
        try:
            spec = load_spectrum(fp, force_type=force_type)
            spectra.append(spec)
        except Exception as e:
            print(f"加载文件 {fp} 失败: {e}")
    return spectra


def _get_x_unit(spectrum_type: SpectrumType, header_info: dict) -> str:
    x_units = {
        SpectrumType.XRD: '2θ (°)',
        SpectrumType.XRF: '能量 (keV)',
        SpectrumType.RAMAN: '拉曼位移 (cm⁻¹)',
        SpectrumType.IR: '波数 (cm⁻¹)',
        SpectrumType.UNKNOWN: 'x',
    }
    return x_units.get(spectrum_type, 'x')


def _get_y_unit(header_info: dict) -> str:
    return '强度'
