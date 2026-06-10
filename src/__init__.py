"""
光谱数据分析与矿物成分鉴定工具
"""
from .spectrum import Spectrum, SpectrumType
from .data_import import load_spectrum, load_spectra, detect_spectrum_type
from .preprocessing import (
    PreprocessingPipeline, BaselineCorrection, Smoothing, Normalization
)
from .peak_fitting import (
    detect_peaks, fit_peaks, segment_fit, calculate_peak_areas,
    gaussian, lorentzian, voigt,
)
from .phase_identification import (
    identify_phases, match_phases, rir_quantitative, estimate_phase_abundance,
)
from .element_analysis import (
    identify_elements, get_element_peak_labels,
    CalibrationCurve, quantitative_analysis, multi_element_analysis,
    matrix_correction_empirical,
)
from .spectral_comparison import (
    cosine_similarity, pearson_correlation, spectral_band_matching,
    search_spectrum, compute_difference_spectrum, compute_ratio_spectrum,
    align_spectra, normalize_spectra_for_comparison,
    export_spectrum_plot, build_sample_library,
)
from .quantitative import (
    standard_curve_method, internal_standard_method,
    calculate_peak_area_from_spectrum,
    limit_of_detection, limit_of_quantitation,
    StandardCurveResult,
)
from .report_generator import generate_pdf_report, generate_svg_spectrum

__version__ = "1.0.0"
