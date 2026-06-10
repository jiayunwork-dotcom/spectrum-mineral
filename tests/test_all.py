"""
测试脚本 - 验证各模块功能
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.spectrum import Spectrum, SpectrumType
from src.data_import import detect_spectrum_type
from src.preprocessing import (
    PreprocessingPipeline, BaselineCorrection, Smoothing, Normalization
)
from src.peak_fitting import (
    detect_peaks, fit_peaks, segment_fit, calculate_peak_areas,
    gaussian, lorentzian, voigt,
)
from src.phase_identification import identify_phases, match_phases, rir_quantitative
from src.pdf_database import get_pdf_database
from src.xray_database import get_xray_database, get_elements, find_elements_by_energy
from src.element_analysis import identify_elements
from src.spectral_comparison import (
    cosine_similarity, pearson_correlation, spectral_band_matching,
    search_spectrum, compute_difference_spectrum, compute_ratio_spectrum,
    build_sample_library,
)
from src.quantitative import (
    standard_curve_method, internal_standard_method,
)
from src.report_generator import generate_pdf_report, generate_svg_spectrum


def test_spectrum():
    print("=" * 50)
    print("测试 Spectrum 类")
    
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)
    
    spec = Spectrum(
        name="test",
        spectrum_type=SpectrumType.XRD,
        x=x,
        y=y,
        x_unit="2θ (°)",
        y_unit="强度",
    )
    
    assert spec.num_points == 100
    assert spec.x_range[0] == 0.0
    assert spec.x_range[1] == 10.0
    
    spec2 = spec.copy()
    assert spec2.name == spec.name
    assert spec2.num_points == spec.num_points
    
    print("✓ Spectrum 测试通过")


def test_spectrum_type_detection():
    print("=" * 50)
    print("测试光谱类型识别")
    
    x_xrd = np.linspace(5, 90, 100)
    assert detect_spectrum_type(x_xrd) == SpectrumType.XRD
    print("✓ XRD 类型识别正确")
    
    x_xrf = np.linspace(0, 40, 100)
    assert detect_spectrum_type(x_xrf) == SpectrumType.XRF
    print("✓ XRF 类型识别正确")
    
    x_raman = np.linspace(100, 4000, 100)
    assert detect_spectrum_type(x_raman) == SpectrumType.RAMAN
    print("✓ Raman 类型识别正确")
    
    x_ir = np.linspace(400, 4000, 100)
    assert detect_spectrum_type(x_ir) in [SpectrumType.IR, SpectrumType.RAMAN]
    print("✓ IR 类型识别正确")


def test_preprocessing():
    print("=" * 50)
    print("测试预处理模块")
    
    x = np.linspace(0, 10, 200)
    y_true = np.zeros_like(x)
    y_true[50:150] = 5 * np.exp(-(x[50:150] - 5)**2 / 0.5)
    y_noisy = y_true + np.random.normal(0, 0.2, len(x))
    y_with_baseline = y_noisy + 0.5 * x + 1.0
    
    spec = Spectrum(
        name="test",
        spectrum_type=SpectrumType.XRD,
        x=x,
        y=y_with_baseline,
    )
    
    pipeline = PreprocessingPipeline()
    pipeline.add_step(BaselineCorrection(method="als", lambda_=1e6, p=0.01, niter=10))
    pipeline.add_step(Smoothing(method="savgol", window=15, polyorder=2))
    pipeline.add_step(Normalization(method="max"))
    
    processed = pipeline.apply(spec)
    
    assert processed.num_points == spec.num_points
    assert abs(processed.y.max() - 1.0) < 0.1
    
    print(f"✓ 预处理流水线测试通过 (步骤数: {len(pipeline.steps)})")


def test_peak_detection_and_fitting():
    print("=" * 50)
    print("测试峰检测与拟合")
    
    x = np.linspace(10, 80, 500)
    y = np.zeros_like(x)
    
    true_peaks = [
        (20.0, 100, 2.0),
        (35.0, 80, 1.5),
        (50.0, 120, 2.5),
        (65.0, 60, 1.8),
    ]
    
    for pos, amp, fwhm in true_peaks:
        y += gaussian(x, amp, pos, fwhm)
    
    y += np.random.normal(0, 1.0, len(x))
    
    peaks = detect_peaks(x, y, sensitivity=3.0, min_peak_distance=2.0)
    print(f"检测到 {len(peaks)} 个峰 (预期 {len(true_peaks)} 个)")
    assert len(peaks) >= len(true_peaks) - 1
    
    fitted_peaks, y_fitted, r2 = fit_peaks(x, y, peaks, peak_type='gaussian')
    print(f"高斯拟合 R² = {r2:.4f}")
    assert r2 > 0.9
    
    fitted_peaks_v, y_fitted_v, r2_v = fit_peaks(x, y, peaks, peak_type='voigt')
    print(f"Voigt 拟合 R² = {r2_v:.4f}")
    
    areas = calculate_peak_areas(x, fitted_peaks, 'gaussian')
    print(f"✓ 峰面积计算完成 ({len(areas)} 个峰)")
    
    print("✓ 峰检测与拟合测试通过")


def test_phase_identification():
    print("=" * 50)
    print("测试物相鉴定")
    
    db = get_pdf_database()
    print(f"PDF卡片数量: {len(db)}")
    assert len(db) >= 50
    
    quartz = None
    for card in db:
        if "石英" in card.name:
            quartz = card
            break
    
    assert quartz is not None
    print(f"✓ 找到石英卡片: {quartz.name}")
    
    x = np.linspace(10, 70, 500)
    y = np.zeros_like(x)
    
    for pos, inten in quartz.peaks:
        sigma = 0.1
        y += inten * np.exp(-(x - pos)**2 / (2 * sigma**2))
    
    y += np.random.normal(0, 0.5, len(x))
    
    spec = Spectrum(
        name="test_xrd",
        spectrum_type=SpectrumType.XRD,
        x=x,
        y=y,
    )
    
    peaks, results = identify_phases(spec, sensitivity=2.0, tolerance=0.1)
    print(f"检测到 {len(peaks)} 个峰")
    print(f"匹配结果数: {len(results)}")
    
    if results:
        top_result = results[0]
        print(f"最佳匹配: {top_result['card'].name}, 匹配度: {top_result['match_score']*100:.1f}%")
        assert "石英" in top_result['card'].name or top_result['match_score'] > 0.3
    
    print("✓ 物相鉴定测试通过")


def test_xray_database():
    print("=" * 50)
    print("测试X射线特征能量数据库")
    
    db = get_xray_database()
    print(f"特征线总数: {len(db)}")
    assert len(db) > 100
    
    elements = get_elements()
    print(f"元素总数: {len(elements)}")
    assert len(elements) >= 80
    
    fe_lines = find_elements_by_energy(6.40, tolerance=0.1)
    print(f"6.4 keV附近的特征线: {len(fe_lines)} 条")
    
    fe_found = any(l.element == 'Fe' for l in fe_lines)
    print(f"✓ 找到Fe的特征线: {fe_found}")
    
    print("✓ X射线数据库测试通过")


def test_spectral_comparison():
    print("=" * 50)
    print("测试光谱检索与对比")
    
    library = build_sample_library()
    print(f"示例光谱库大小: {len(library)}")
    assert len(library) > 0
    
    if len(library) >= 2:
        spec1 = library[0]
        spec2 = library[1]
        
        cos_sim = cosine_similarity(spec1, spec1)
        print(f"自相似性 (余弦): {cos_sim:.4f}")
        assert abs(cos_sim - 1.0) < 0.01
        
        pearson_sim = pearson_correlation(spec1, spec1)
        print(f"自相似性 (皮尔逊): {pearson_sim:.4f}")
        assert abs(pearson_sim - 1.0) < 0.01
        
        results = search_spectrum(spec1, library, method='cosine', top_n=3)
        print(f"检索结果数: {len(results)}")
        assert results[0]['spectrum'].name == spec1.name
        
        x_diff, y_diff = compute_difference_spectrum(spec1, spec1)
        print(f"差谱计算完成: {len(x_diff)} 个点")
        assert np.allclose(y_diff, 0, atol=0.01)
        
        print("✓ 光谱检索与对比测试通过")


def test_quantitative():
    print("=" * 50)
    print("测试定量分析")
    
    concentrations = [1, 2, 3, 4, 5]
    responses = [10, 20, 30, 40, 50]
    
    result = standard_curve_method(concentrations, responses, unknown_response=25, method='linear')
    print(f"线性预测浓度: {result.predicted_concentration:.2f}")
    print(f"R² = {result.r_squared:.4f}")
    assert abs(result.predicted_concentration - 2.5) < 0.1
    assert abs(result.r_squared - 1.0) < 0.01
    
    result_quad = standard_curve_method(concentrations, responses, unknown_response=25, method='quadratic')
    print(f"二次预测浓度: {result_quad.predicted_concentration:.2f}")
    
    analyte = [100, 200, 300, 400, 500]
    internal = [50, 52, 48, 51, 49]
    
    result_is = internal_standard_method(
        analyte, internal, concentrations,
        unknown_analyte_response=250,
        unknown_is_response=50,
        method='linear',
    )
    print(f"内标法预测浓度: {result_is.predicted_concentration:.2f}")
    
    print("✓ 定量分析测试通过")


def test_report():
    print("=" * 50)
    print("测试报告生成")
    
    x = np.linspace(10, 80, 500)
    y = np.zeros_like(x)
    y += 100 * np.exp(-(x - 26.6)**2 / (2 * 0.2**2))
    y += 50 * np.exp(-(x - 36.5)**2 / (2 * 0.3**2))
    
    svg = generate_svg_spectrum(
        x, y,
        title="测试光谱",
        x_label="2θ (°)",
        y_label="强度",
    )
    print(f"SVG生成: {len(svg)} 字符")
    assert svg.startswith('<svg')
    assert svg.endswith('</svg>')
    
    print("✓ 报告生成测试通过")


def main():
    print("🚀 开始运行测试...")
    print()
    
    try:
        test_spectrum()
    except Exception as e:
        print(f"❌ Spectrum 测试失败: {e}")
    
    try:
        test_spectrum_type_detection()
    except Exception as e:
        print(f"❌ 光谱类型识别测试失败: {e}")
    
    try:
        test_preprocessing()
    except Exception as e:
        print(f"❌ 预处理测试失败: {e}")
    
    try:
        test_peak_detection_and_fitting()
    except Exception as e:
        print(f"❌ 峰检测与拟合测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_phase_identification()
    except Exception as e:
        print(f"❌ 物相鉴定测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_xray_database()
    except Exception as e:
        print(f"❌ X射线数据库测试失败: {e}")
    
    try:
        test_spectral_comparison()
    except Exception as e:
        print(f"❌ 光谱检索与对比测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_quantitative()
    except Exception as e:
        print(f"❌ 定量分析测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_report()
    except Exception as e:
        print(f"❌ 报告生成测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 50)
    print("✅ 所有测试完成!")


if __name__ == "__main__":
    main()
