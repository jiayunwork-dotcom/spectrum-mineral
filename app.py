"""
光谱数据分析与矿物成分鉴定工具
主应用入口 - Streamlit 界面
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
from typing import List, Dict, Optional

from src.spectrum import Spectrum, SpectrumType
from src.data_import import load_spectrum, detect_spectrum_type
from src.preprocessing import (
    PreprocessingPipeline, BaselineCorrection, Smoothing, Normalization
)
from src.peak_fitting import (
    detect_peaks, fit_peaks, segment_fit, calculate_peak_areas,
    gaussian, lorentzian, voigt, deconvolve_peaks,
)
from src.phase_identification import (
    identify_phases, match_phases, estimate_phase_abundance,
)
from src.pdf_database import get_pdf_database, get_pdf_card_by_name
from src.element_analysis import (
    identify_elements, get_element_peak_labels,
    CalibrationCurve, quantitative_analysis, matrix_correction_empirical,
)
from src.spectral_comparison import (
    cosine_similarity, pearson_correlation, spectral_band_matching,
    search_spectrum, compute_difference_spectrum, compute_ratio_spectrum,
    align_spectra, normalize_spectra_for_comparison,
    export_spectrum_plot, build_sample_library,
)
from src.quantitative import (
    standard_curve_method, internal_standard_method,
    StandardCurveResult,
)
from src.report_generator import generate_pdf_report
from src.pca_classification import (
    PCAResult, ClusteringResult, AnomalyResult,
    interpolate_to_common_grid, standardize_matrix, perform_pca,
    find_loading_peaks, perform_kmeans, elbow_method,
    reconstruct_centroids, calculate_cosine_similarity_centroid,
    compute_confusion_matrix, compute_mahalanobis_distances,
    compute_hotelling_t2, compute_q_residuals,
    compute_t2_contributions, compute_q_contributions,
    classify_quadrant, compute_moving_average, compute_trend_slope,
    get_top_t2_contributions, get_top_q_contributions,
    save_pca_model, load_pca_model, predict_new_sample,
    export_results_to_csv, export_loadings_to_csv, export_variance_to_csv,
)
from src.data_import import load_spectrum

st.set_page_config(
    page_title="光谱数据分析与矿物成分鉴定工具",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

if 'spectra' not in st.session_state:
    st.session_state.spectra: List[Spectrum] = []

if 'current_spectrum_idx' not in st.session_state:
    st.session_state.current_spectrum_idx = 0

if 'preprocessing_pipeline' not in st.session_state:
    st.session_state.preprocessing_pipeline = PreprocessingPipeline()

if 'preprocessed_spectrum' not in st.session_state:
    st.session_state.preprocessed_spectrum = None

if 'peak_results' not in st.session_state:
    st.session_state.peak_results = None

if 'fitted_peaks' not in st.session_state:
    st.session_state.fitted_peaks = None

if 'fitted_curve' not in st.session_state:
    st.session_state.fitted_curve = None

if 'fit_r_squared' not in st.session_state:
    st.session_state.fit_r_squared = None

if 'phase_results' not in st.session_state:
    st.session_state.phase_results = None

if 'element_results' not in st.session_state:
    st.session_state.element_results = None

if 'sample_library' not in st.session_state:
    st.session_state.sample_library = build_sample_library()


def plot_spectrum(x, y, title="光谱图", x_label="", y_label="强度", peaks=None, 
                  fitted_curve=None, show_legend=True):
    """绘制光谱图"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        name='光谱',
        line=dict(color='#1f77b4', width=1.5),
        hovertemplate='x: %{x:.4f}<br>y: %{y:.2f}<extra></extra>',
    ))
    
    if peaks is not None and len(peaks) > 0:
        peak_x = [p.get('position', p.get('peak_energy', 0)) for p in peaks]
        peak_y = [p.get('intensity', p.get('peak_intensity', 0)) for p in peaks]
        fig.add_trace(go.Scatter(
            x=peak_x, y=peak_y,
            mode='markers',
            name='检测峰',
            marker=dict(color='red', size=6, symbol='circle'),
            hovertemplate='峰位: %{x:.4f}<br>强度: %{y:.2f}<extra></extra>',
        ))
    
    if fitted_curve is not None:
        fig.add_trace(go.Scatter(
            x=x, y=fitted_curve,
            mode='lines',
            name='拟合曲线',
            line=dict(color='green', width=1.5, dash='dash'),
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode='closest',
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400,
        margin=dict(l=10, r=10, t=40, b=40),
    )
    
    return fig


def plot_multi_spectra(spectra, title="多样品对比", x_label="", y_label="强度", normalize=False):
    """绘制多条光谱"""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, spec in enumerate(spectra):
        y_data = spec.y
        if normalize and y_data.max() > 0:
            y_data = y_data / y_data.max()
        
        fig.add_trace(go.Scatter(
            x=spec.x, y=y_data,
            mode='lines',
            name=spec.name,
            line=dict(color=colors[i % len(colors)], width=1.5),
            hovertemplate=f'{spec.name}<br>x: %{{x:.4f}}<br>y: %{{y:.2f}}<extra></extra>',
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
    )
    
    return fig


st.sidebar.title("🔬 光谱分析工具")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "功能菜单",
    [
        "📂 数据导入",
        "🧹 光谱预处理",
        "📊 峰检测与拟合",
        "💎 物相鉴定 (XRD)",
        "⚛️ 元素分析 (XRF)",
        "🔍 光谱检索",
        "📈 多样品对比",
        "📊 PCA与分类",
        "📐 定量分析",
        "📄 报告导出",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 当前样品")
if st.session_state.spectra:
    spectrum_names = [s.name for s in st.session_state.spectra]
    current_idx = st.sidebar.selectbox(
        "选择样品",
        range(len(spectrum_names)),
        format_func=lambda i: spectrum_names[i],
        index=st.session_state.current_spectrum_idx if st.session_state.current_spectrum_idx < len(spectrum_names) else 0,
        key="sidebar_spectrum_select",
    )
    st.session_state.current_spectrum_idx = current_idx
    
    current_spec = st.session_state.spectra[st.session_state.current_spectrum_idx]
    st.sidebar.info(f"类型: {current_spec.spectrum_type.value}\n数据点: {current_spec.num_points}")
else:
    st.sidebar.warning("暂无数据，请先导入")


if page == "📂 数据导入":
    st.title("📂 数据导入")
    st.markdown("支持CSV格式和JCAMP-DX格式(.dx/.jdx)，可批量导入多个样品")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "选择光谱文件",
            type=['csv', 'txt', 'dx', 'jdx', 'jcm'],
            accept_multiple_files=True,
            help="支持CSV（两列：角度/能量/波数 vs 强度）和JCAMP-DX格式",
        )
    
    with col2:
        force_type = st.selectbox(
            "强制指定光谱类型",
            ["自动识别", "XRD (X射线衍射)", "XRF (X射线荧光)", "RAMAN (拉曼)", "IR (红外)"],
            help="如果自动识别不准确，可以手动指定",
        )
    
    type_map = {
        "自动识别": None,
        "XRD (X射线衍射)": SpectrumType.XRD,
        "XRF (X射线荧光)": SpectrumType.XRF,
        "RAMAN (拉曼)": SpectrumType.RAMAN,
        "IR (红外)": SpectrumType.IR,
    }
    
    if uploaded_files:
        if st.button("导入数据", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            new_spectra = []
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"正在导入: {file.name}...")
                progress_bar.progress((i) / len(uploaded_files))
                
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = tmp.name
                    
                    spec = load_spectrum(
                        tmp_path,
                        name=os.path.splitext(file.name)[0],
                        force_type=type_map[force_type],
                    )
                    
                    new_spectra.append(spec)
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    st.error(f"导入 {file.name} 失败: {e}")
            
            progress_bar.progress(1.0)
            status_text.text(f"成功导入 {len(new_spectra)} 个样品")
            
            st.session_state.spectra.extend(new_spectra)
            st.session_state.current_spectrum_idx = len(st.session_state.spectra) - len(new_spectra)
            
            st.success(f"✅ 成功导入 {len(new_spectra)} 个光谱文件")
    
    st.markdown("---")
    
    if st.session_state.spectra:
        st.subheader("已导入样品列表")
        
        data = []
        for i, spec in enumerate(st.session_state.spectra):
            data.append({
                "序号": i + 1,
                "样品名称": spec.name,
                "光谱类型": spec.spectrum_type.value,
                "数据点数": spec.num_points,
                "x轴范围": f"{spec.x_range[0]:.2f} - {spec.x_range[1]:.2f}",
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("清空所有数据"):
                st.session_state.spectra = []
                st.session_state.current_spectrum_idx = 0
                st.rerun()
        with col2:
            if st.button("删除当前选中样品"):
                if st.session_state.spectra:
                    idx = st.session_state.current_spectrum_idx
                    st.session_state.spectra.pop(idx)
                    if st.session_state.current_spectrum_idx >= len(st.session_state.spectra):
                        st.session_state.current_spectrum_idx = max(0, len(st.session_state.spectra) - 1)
                    st.rerun()
    else:
        st.info("👆 请上传光谱文件开始分析")
        
        with st.expander("查看示例数据格式"):
            st.markdown("""
            **CSV格式示例：**
            ```
            # title: 石英样品XRD
            10.0, 12.5
            10.1, 13.2
            10.2, 14.1
            ...
            ```
            
            第一列为角度/能量/波数，第二列为强度。
            """)

elif page == "🧹 光谱预处理":
    st.title("🧹 光谱预处理")
    st.markdown("基线校正、平滑去噪、归一化，支持步骤组合和顺序调整")
    
    if not st.session_state.spectra:
        st.warning("请先导入数据")
    else:
        current_spec = st.session_state.spectra[st.session_state.current_spectrum_idx]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("预处理步骤")
            
            pipeline = st.session_state.preprocessing_pipeline
            
            step_types = st.multiselect(
                "添加预处理步骤",
                ["基线校正", "平滑去噪", "归一化"],
                key="add_steps_select",
            )
            
            for step_type in step_types:
                if step_type == "基线校正":
                    pass
                elif step_type == "平滑去噪":
                    pass
                elif step_type == "归一化":
                    pass
            
            if st.button("添加基线校正步骤"):
                method = "als"
                pipeline.add_step(BaselineCorrection(method="als", lambda_=1e5, p=0.01, niter=10))
                st.rerun()
            
            if st.button("添加平滑去噪步骤"):
                pipeline.add_step(Smoothing(method="savgol", window=11, polyorder=2))
                st.rerun()
            
            if st.button("添加归一化步骤"):
                pipeline.add_step(Normalization(method="max"))
                st.rerun()
            
            st.markdown("---")
            
            col_undo, col_redo, col_clear = st.columns(3)
            with col_undo:
                if st.button("↩️ 撤销", disabled=not pipeline.can_undo, help="撤销上一步操作"):
                    pipeline.undo()
                    st.rerun()
            with col_redo:
                if st.button("↪️ 重做", disabled=not pipeline.can_redo, help="重做已撤销的操作"):
                    pipeline.redo()
                    st.rerun()
            with col_clear:
                if st.button("🗑️ 清空", disabled=not pipeline.steps, help="清空所有步骤"):
                    pipeline.clear_steps()
                    st.rerun()
            
            st.markdown("---")
            
            if pipeline.steps:
                st.write(f"当前步骤数: {len(pipeline.steps)}")
                
                for i, step in enumerate(pipeline.steps):
                    with st.expander(f"步骤 {i+1}: {step.name}", expanded=True):
                        st.json(step.params)
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            if i > 0 and st.button("↑ 上移", key=f"up_{i}"):
                                pipeline.move_step(i, i - 1)
                                st.rerun()
                        with col_b:
                            if i < len(pipeline.steps) - 1 and st.button("↓ 下移", key=f"down_{i}"):
                                pipeline.move_step(i, i + 1)
                                st.rerun()
                        with col_c:
                            if st.button("🗑️ 删除", key=f"del_{i}"):
                                pipeline.remove_step(i)
                                st.rerun()
            else:
                st.info("暂无预处理步骤，请添加")
        
        with col2:
            st.subheader("实时预览")
            
            if pipeline.steps:
                preview_step = st.slider(
                    "预览到第几步",
                    0, len(pipeline.steps),
                    value=len(pipeline.steps),
                    help="拖动滑块查看每一步的效果",
                )
            else:
                preview_step = 0
                st.info("暂无预处理步骤，请在左侧添加")
            
            try:
                if pipeline.steps and preview_step > 0:
                    preview_spec = pipeline.apply_step(current_spec, preview_step - 1)
                else:
                    preview_spec = current_spec
                
                fig = plot_spectrum(
                    preview_spec.x, preview_spec.y,
                    title=f"{current_spec.name} - 预处理预览 (步骤 {preview_step})",
                    x_label=preview_spec.x_unit,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"预览渲染出错: {str(e)}")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ 应用预处理", type="primary"):
                    processed = pipeline.apply(current_spec)
                    processed.name = current_spec.name + " (预处理)"
                    st.session_state.spectra.append(processed)
                    st.session_state.current_spectrum_idx = len(st.session_state.spectra) - 1
                    st.session_state.preprocessed_spectrum = processed
                    st.success("预处理已应用并保存为新样品")
                    st.rerun()
            
            with col_b:
                if st.button("🔄 重置预览"):
                    st.rerun()
            
            if pipeline.steps:
                with st.expander("步骤参数详细设置"):
                    for i, step in enumerate(pipeline.steps):
                        st.markdown(f"**步骤 {i+1}: {step.name}**")
                        
                        if step.name == "baseline_correction":
                            method = st.selectbox(
                                "基线校正方法",
                                ["als", "polynomial", "snip"],
                                index=["als", "polynomial", "snip"].index(step.params.get("method", "als")),
                                key=f"baseline_method_{i}",
                            )
                            step.params["method"] = method
                            
                            if method == "polynomial":
                                degree = st.slider("多项式阶数", 2, 10, step.params.get("degree", 5), key=f"poly_degree_{i}")
                                step.params["degree"] = degree
                            elif method == "als":
                                lam_val = st.slider("惩罚因子 lambda (log10)", 2, 8, 
                                                    int(np.log10(step.params.get("lambda", 1e5))),
                                                    key=f"als_lam_{i}")
                                step.params["lambda"] = 10 ** lam_val
                                p_val = st.slider("非对称权重 p", 0.001, 0.1, step.params.get("p", 0.01),
                                                  key=f"als_p_{i}", format="%.3f")
                                step.params["p"] = p_val
                                niter = st.slider("迭代次数", 5, 50, step.params.get("niter", 10),
                                                  key=f"als_niter_{i}")
                                step.params["niter"] = niter
                            elif method == "snip":
                                niter = st.slider("迭代次数", 5, 50, step.params.get("niter", 20),
                                                  key=f"snip_niter_{i}")
                                step.params["niter"] = niter
                        
                        elif step.name == "smoothing":
                            method = st.selectbox(
                                "平滑方法",
                                ["savgol", "wavelet"],
                                index=["savgol", "wavelet"].index(step.params.get("method", "savgol")),
                                key=f"smooth_method_{i}",
                            )
                            step.params["method"] = method
                            
                            if method == "savgol":
                                window = st.slider("窗口大小", 3, 31, step.params.get("window", 11),
                                                   step=2, key=f"sg_window_{i}")
                                step.params["window"] = window
                                polyorder = st.slider("多项式阶数", 1, 5, step.params.get("polyorder", 2),
                                                      key=f"sg_poly_{i}")
                                step.params["polyorder"] = polyorder
                            elif method == "wavelet":
                                wavelet = st.selectbox("小波基函数", ["db4", "db8", "sym4", "coif4"],
                                                       key=f"wt_wavelet_{i}")
                                step.params["wavelet"] = wavelet
                                level = st.slider("分解层数", 1, 8, step.params.get("level", 3),
                                                  key=f"wt_level_{i}")
                                step.params["level"] = level
                        
                        elif step.name == "normalization":
                            method = st.selectbox(
                                "归一化方法",
                                ["max", "area", "peak"],
                                index=["max", "area", "peak"].index(step.params.get("method", "max")),
                                key=f"norm_method_{i}",
                            )
                            step.params["method"] = method
                            
                            if method == "peak":
                                peak_x = st.number_input("指定峰位", value=step.params.get("peak_x", 26.6),
                                                         key=f"norm_peak_{i}")
                                step.params["peak_x"] = peak_x
                        
                        st.markdown("---")

elif page == "📊 峰检测与拟合":
    st.title("📊 峰检测与拟合")
    st.markdown("自动寻峰、峰型拟合（高斯/洛伦兹/Voigt）、分段拟合、峰解卷积")
    
    if 'deconvolution_selected' not in st.session_state:
        st.session_state.deconvolution_selected = set()
    if 'deconvolution_result' not in st.session_state:
        st.session_state.deconvolution_result = None
    if 'pre_deconvolution_peaks' not in st.session_state:
        st.session_state.pre_deconvolution_peaks = None
    if 'pre_deconvolution_fitted' not in st.session_state:
        st.session_state.pre_deconvolution_fitted = None
    if 'pre_deconvolution_r2' not in st.session_state:
        st.session_state.pre_deconvolution_r2 = None
    
    if not st.session_state.spectra:
        st.warning("请先导入数据")
    else:
        current_spec = st.session_state.spectra[st.session_state.current_spectrum_idx]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("峰检测参数")
            
            sensitivity = st.slider(
                "灵敏度 (信噪比倍数)",
                1.0, 10.0, 3.0, 0.1,
                help="值越小检测到的峰越多",
            )
            
            x_range = current_spec.x_range[1] - current_spec.x_range[0]
            default_dist = x_range * 0.01
            min_peak_distance = st.number_input(
                "最小峰间距",
                value=float(default_dist),
                step=float(default_dist / 2),
                format="%.4f",
            )
            
            if st.button("🔍 自动寻峰", type="primary"):
                peaks = detect_peaks(
                    current_spec.x, current_spec.y,
                    sensitivity=sensitivity,
                    min_peak_distance=min_peak_distance,
                )
                st.session_state.peak_results = peaks
                st.session_state.deconvolution_selected = set()
                st.session_state.deconvolution_result = None
                st.session_state.pre_deconvolution_peaks = None
                st.success(f"检测到 {len(peaks)} 个峰")
            
            st.markdown("---")
            
            st.subheader("峰拟合参数")
            
            peak_type = st.selectbox(
                "峰型",
                ["gaussian", "lorentzian", "voigt"],
                format_func=lambda x: {"gaussian": "高斯", "lorentzian": "洛伦兹", "voigt": "Voigt"}[x],
            )
            
            use_segment_fit = st.checkbox("使用分段拟合（峰数较多时）", value=True)
            max_peaks_per_segment = st.slider("每段最大峰数", 5, 30, 15)
            
            if st.button("📈 拟合峰", disabled=st.session_state.peak_results is None):
                peaks = st.session_state.peak_results
                
                if use_segment_fit and len(peaks) > max_peaks_per_segment:
                    fitted_peaks, y_fitted, r2 = segment_fit(
                        current_spec.x, current_spec.y,
                        peaks, peak_type, max_peaks_per_segment,
                    )
                else:
                    fitted_peaks, y_fitted, r2 = fit_peaks(
                        current_spec.x, current_spec.y,
                        peaks, peak_type,
                    )
                
                st.session_state.fitted_peaks = fitted_peaks
                st.session_state.fitted_curve = y_fitted
                st.session_state.fit_r_squared = r2
                st.session_state.deconvolution_selected = set()
                st.session_state.deconvolution_result = None
                st.session_state.pre_deconvolution_peaks = None
                
                st.success(f"拟合完成，R² = {r2:.4f}")
            
            st.markdown("---")
            
            st.subheader("峰解卷积")
            
            selected_set = st.session_state.deconvolution_selected
            display_peaks_all = st.session_state.fitted_peaks or st.session_state.peak_results
            
            if display_peaks_all:
                selected_count = len(selected_set)
                st.info(f"已选择 {selected_count} 个峰（需要2-5个相邻重叠峰）")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    disable_deconv = not (2 <= selected_count <= 5)
                    if st.button("🔬 解卷积", disabled=disable_deconv, type="primary",
                                 help="对选中的2-5个相邻重叠峰进行约束优化分解"):
                        selected_indices = sorted(selected_set)
                        selected_peaks_list = [display_peaks_all[i] for i in selected_indices]
                        
                        st.session_state.pre_deconvolution_peaks = [p.copy() for p in display_peaks_all]
                        if st.session_state.fitted_curve is not None:
                            st.session_state.pre_deconvolution_fitted = st.session_state.fitted_curve.copy()
                            st.session_state.pre_deconvolution_r2 = st.session_state.fit_r_squared
                        
                        deconv_result = deconvolve_peaks(
                            current_spec.x, current_spec.y,
                            selected_peaks_list, peak_type,
                        )
                        
                        if deconv_result['success']:
                            deconv_peaks = deconv_result['deconvolved_peaks']
                            for idx_in_selected, orig_idx in enumerate(selected_indices):
                                if orig_idx < len(display_peaks_all):
                                    orig_peak = display_peaks_all[orig_idx]
                                    new_peak_data = deconv_peaks[idx_in_selected]
                                    for k, v in new_peak_data.items():
                                        orig_peak[k] = v
                            
                            if st.session_state.fitted_peaks:
                                st.session_state.fitted_peaks = display_peaks_all
                            
                            st.session_state.deconvolution_result = deconv_result
                            st.success(deconv_result['message'])
                            st.rerun()
                        else:
                            st.error(deconv_result['message'])
                
                with col_b:
                    disable_reset = st.session_state.pre_deconvolution_peaks is None
                    if st.button("↺ 重置解卷积", disabled=disable_reset,
                                 help="恢复到解卷积前的峰参数状态"):
                        if st.session_state.pre_deconvolution_peaks is not None:
                            if st.session_state.fitted_peaks:
                                st.session_state.fitted_peaks = st.session_state.pre_deconvolution_peaks
                            else:
                                st.session_state.peak_results = st.session_state.pre_deconvolution_peaks
                            
                            if st.session_state.pre_deconvolution_fitted is not None:
                                st.session_state.fitted_curve = st.session_state.pre_deconvolution_fitted
                                st.session_state.fit_r_squared = st.session_state.pre_deconvolution_r2
                            
                            st.session_state.deconvolution_result = None
                            st.session_state.pre_deconvolution_peaks = None
                            st.success("已恢复到解卷积前的状态")
                            st.rerun()
            else:
                st.info("请先运行自动寻峰或拟合峰")
            
            st.markdown("---")
            
            if st.session_state.fitted_peaks:
                st.subheader("手动编辑峰")
                st.info("可在右侧表格中编辑峰参数")
        
        with col2:
            st.subheader("光谱图与峰")
            
            peaks_to_show = st.session_state.fitted_peaks or st.session_state.peak_results
            fitted_curve = st.session_state.fitted_curve
            
            fig = plot_spectrum(
                current_spec.x, current_spec.y,
                title=current_spec.name,
                x_label=current_spec.x_unit,
                peaks=peaks_to_show,
                fitted_curve=fitted_curve,
            )
            
            if peaks_to_show and st.session_state.deconvolution_selected:
                selected_indices = sorted(st.session_state.deconvolution_selected)
                selected_x = [peaks_to_show[i]['position'] for i in selected_indices if i < len(peaks_to_show)]
                selected_y = [peaks_to_show[i]['intensity'] for i in selected_indices if i < len(peaks_to_show)]
                fig.add_trace(go.Scatter(
                    x=selected_x, y=selected_y,
                    mode='markers',
                    name='选中待解卷积',
                    marker=dict(color='orange', size=12, symbol='star',
                                line=dict(color='black', width=1)),
                ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            if st.session_state.fit_r_squared is not None:
                st.metric("拟合优度 R²", f"{st.session_state.fit_r_squared:.4f}")
            
            if st.session_state.deconvolution_result and st.session_state.deconvolution_result['success']:
                deconv = st.session_state.deconvolution_result
                
                st.markdown("#### 解卷积结果")
                
                x_seg = deconv['sum_curve']['x']
                y_seg = current_spec.y[(current_spec.x >= deconv['x_range'][0]) & (current_spec.x <= deconv['x_range'][1])]
                
                fig_deconv = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    row_heights=[0.7, 0.3],
                    vertical_spacing=0.05,
                    subplot_titles=("解卷积分量分解", "残差"),
                )
                
                fig_deconv.add_trace(go.Scatter(
                    x=x_seg, y=y_seg,
                    mode='lines',
                    name='原始数据',
                    line=dict(color='#1f77b4', width=2.5),
                ), row=1, col=1)
                
                colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                for i, comp in enumerate(deconv['component_curves']):
                    fig_deconv.add_trace(go.Scatter(
                        x=comp['x'], y=comp['y'],
                        mode='lines',
                        name=f'分量 {i+1}',
                        line=dict(color=colors[i % len(colors)], width=1.5, dash='dash'),
                    ), row=1, col=1)
                
                fig_deconv.add_trace(go.Scatter(
                    x=deconv['sum_curve']['x'], y=deconv['sum_curve']['y'],
                    mode='lines',
                    name='分量之和',
                    line=dict(color='green', width=2.0),
                ), row=1, col=1)
                
                fig_deconv.add_trace(go.Scatter(
                    x=deconv['residual']['x'], y=deconv['residual']['y'],
                    mode='lines',
                    name='残差',
                    line=dict(color='red', width=1.2),
                    fill='tozeroy',
                    fillcolor='rgba(255,0,0,0.1)',
                ), row=2, col=1)
                
                fig_deconv.update_layout(
                    height=500,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=10, r=10, t=40, b=40),
                )
                fig_deconv.update_xaxes(title_text=current_spec.x_unit, row=2, col=1)
                fig_deconv.update_yaxes(title_text="强度", row=1, col=1)
                fig_deconv.update_yaxes(title_text="残差", row=2, col=1)
                
                st.plotly_chart(fig_deconv, use_container_width=True)
                
                st.metric("解卷积拟合优度 R²", f"{deconv['r_squared']:.4f}")
            
            if st.session_state.fitted_peaks and fitted_curve is not None and st.session_state.deconvolution_result is None:
                residual = current_spec.y - fitted_curve
                fig_res = plot_spectrum(
                    current_spec.x, residual,
                    title="残差曲线",
                    x_label=current_spec.x_unit,
                    y_label="残差",
                )
                st.plotly_chart(fig_res, use_container_width=True)
        
        if st.session_state.peak_results or st.session_state.fitted_peaks:
            st.markdown("---")
            st.subheader("峰列表")
            
            display_peaks = st.session_state.fitted_peaks or st.session_state.peak_results
            
            selected_set = st.session_state.deconvolution_selected
            if len(selected_set) >= len(display_peaks):
                selected_set = set()
                st.session_state.deconvolution_selected = selected_set
            
            peak_data = []
            for i, peak in enumerate(display_peaks):
                pos = peak.get('position', peak.get('peak_energy', 0))
                inten = peak.get('intensity', peak.get('peak_intensity', 0))
                fwhm = peak.get('fwhm', peak.get('fwhm_estimate', None))
                is_deconvolved = peak.get('deconvolved', False)
                is_selected = i in selected_set
                
                peak_data.append({
                    "选择": is_selected,
                    "序号": i + 1,
                    "峰位": round(pos, 4),
                    "强度": round(inten, 2),
                    "半高宽(FWHM)": round(fwhm, 4) if fwhm else "-",
                    "状态": "✅ 已解卷积" if is_deconvolved else "",
                })
            
            df = pd.DataFrame(peak_data)
            
            column_config = {
                "选择": st.column_config.CheckboxColumn(
                    "选择",
                    help="勾选2-5个相邻重叠峰进行解卷积",
                    default=False,
                )
            }
            
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                hide_index=True,
                column_config=column_config,
                disabled=["序号", "峰位", "强度", "半高宽(FWHM)", "状态"],
            )
            
            new_selected = set()
            for i, row in edited_df.iterrows():
                if row.get("选择", False):
                    if 2 <= len(display_peaks):
                        new_selected.add(i)
            
            if new_selected != selected_set:
                st.session_state.deconvolution_selected = new_selected
                st.rerun()
            
            if st.session_state.fitted_peaks:
                areas = calculate_peak_areas(
                    current_spec.x, st.session_state.fitted_peaks, peak_type
                )
                st.download_button(
                    "⬇️ 下载峰数据 (CSV)",
                    df.to_csv(index=False).encode('utf-8'),
                    "peak_results.csv",
                    "text/csv",
                )

elif page == "💎 物相鉴定 (XRD)":
    st.title("💎 物相鉴定 (XRD)")
    st.markdown("基于标准PDF卡片数据库的物相检索与半定量分析")
    
    if not st.session_state.spectra:
        st.warning("请先导入数据")
    else:
        current_spec = st.session_state.spectra[st.session_state.current_spectrum_idx]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("鉴定参数")
            
            sensitivity = st.slider("峰检测灵敏度", 1.0, 10.0, 3.0, 0.1)
            min_peak_dist = st.number_input("最小峰间距 (2θ°)", value=0.1, step=0.01, format="%.3f")
            tolerance = st.slider("匹配容差 (2θ°)", 0.01, 0.2, 0.05, 0.01)
            top_n = st.slider("返回前N个结果", 3, 20, 10)
            
            if st.button("🔬 开始物相鉴定", type="primary"):
                with st.spinner("正在进行物相鉴定..."):
                    peaks, results = identify_phases(
                        current_spec,
                        sensitivity=sensitivity,
                        min_peak_distance=min_peak_dist,
                        tolerance=tolerance,
                        top_n=top_n,
                    )
                    st.session_state.peak_results = peaks
                    st.session_state.phase_results = results
                
                st.success(f"鉴定完成，检测到 {len(peaks)} 个峰，匹配到 {len(results)} 个候选物相")
            
            st.markdown("---")
            
            st.subheader("RIR半定量分析")
            
            if st.session_state.phase_results:
                phase_names = [r['card'].name for r in st.session_state.phase_results]
                selected_phases = st.multiselect(
                    "选择2-5个物相进行定量分析",
                    phase_names,
                    max_selections=5,
                    help="选择2-5个已知物相，使用RIR法估算含量",
                )
                
                if len(selected_phases) >= 2:
                    if st.button("📊 进行半定量分析"):
                        try:
                            quant_results = estimate_phase_abundance(
                                current_spec,
                                selected_phases,
                                sensitivity=sensitivity,
                                min_peak_distance=min_peak_dist,
                                tolerance=tolerance,
                            )
                            st.session_state.quant_phase_results = quant_results
                            st.success("半定量分析完成")
                        except Exception as e:
                            st.error(f"分析失败: {e}")
                else:
                    st.info("请选择至少2个物相")
            else:
                st.info("请先运行物相鉴定")
        
        with col2:
            st.subheader("鉴定结果")
            
            if st.session_state.phase_results:
                fig = plot_spectrum(
                    current_spec.x, current_spec.y,
                    title=current_spec.name,
                    x_label=current_spec.x_unit,
                    peaks=st.session_state.peak_results,
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 候选物相列表")
                
                for i, result in enumerate(st.session_state.phase_results):
                    card = result['card']
                    score = result['match_score'] * 100
                    
                    with st.expander(
                        f"{i+1}. {card.name} - 匹配度: {score:.1f}%",
                        expanded=(i < 3),
                    ):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"**化学式**: {card.formula}")
                            st.write(f"**晶系**: {card.crystal_system}")
                            st.write(f"**卡片号**: {card.card_number}")
                            st.write(f"**RIR值**: {card.rir}")
                        with col_b:
                            st.write(f"**匹配峰数**: {result['num_matched']}/{result['num_card_peaks']}")
                            st.write(f"**匹配比例**: {result['match_ratio']*100:.1f}%")
                        
                        if result['matched_peaks']:
                            st.write("**匹配的峰:**")
                            match_data = []
                            for mp in result['matched_peaks'][:10]:
                                match_data.append({
                                    "样品峰位": f"{mp['sample_position']:.3f}°",
                                    "标准峰位": f"{mp['card_position']:.3f}°",
                                    "偏差": f"{mp['delta']:+.4f}°",
                                    "样品强度": f"{mp['sample_intensity']:.1f}",
                                    "标准强度": f"{mp['card_intensity']:.1f}",
                                })
                            st.dataframe(pd.DataFrame(match_data), use_container_width=True, hide_index=True)
            else:
                st.info("👈 请设置参数并开始鉴定")
        
        if 'quant_phase_results' in st.session_state and st.session_state.quant_phase_results:
            st.markdown("---")
            st.subheader("半定量分析结果")
            
            quant_data = []
            for result in st.session_state.quant_phase_results:
                quant_data.append({
                    "物相": result['phase'],
                    "化学式": result['formula'],
                    "RIR值": result['rir'],
                    "估算含量(wt%)": f"{result['weight_percent']:.2f}",
                })
            
            st.dataframe(pd.DataFrame(quant_data), use_container_width=True, hide_index=True)
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=[r['phase'] for r in st.session_state.quant_phase_results],
                values=[r['weight_percent'] for r in st.session_state.quant_phase_results],
                hole=0.4,
            )])
            fig_pie.update_layout(title="物相含量分布", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

elif page == "⚛️ 元素分析 (XRF)":
    st.title("⚛️ 元素分析 (XRF)")
    st.markdown("特征X射线能量峰识别与定量分析")
    
    if not st.session_state.spectra:
        st.warning("请先导入数据")
    else:
        current_spec = st.session_state.spectra[st.session_state.current_spectrum_idx]
        
        tab1, tab2 = st.tabs(["定性分析", "定量分析"])
        
        with tab1:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("识别参数")
                
                sensitivity = st.slider("峰检测灵敏度", 1.0, 10.0, 3.0, 0.1)
                min_peak_dist = st.number_input("最小峰间距 (keV)", value=0.1, step=0.01, format="%.3f")
                energy_tolerance = st.slider("能量匹配容差 (keV)", 0.02, 0.3, 0.1, 0.01)
                
                if st.button("🔍 识别元素", type="primary"):
                    with st.spinner("正在识别元素..."):
                        elements = identify_elements(
                            current_spec,
                            sensitivity=sensitivity,
                            min_peak_distance=min_peak_dist,
                            energy_tolerance=energy_tolerance,
                        )
                        st.session_state.element_results = elements
                    
                    st.success(f"识别到 {len(elements)} 种元素")
            
            with col2:
                st.subheader("结果")
                
                if st.session_state.element_results:
                    labels = get_element_peak_labels(
                        current_spec, st.session_state.element_results
                    )
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=current_spec.x, y=current_spec.y,
                        mode='lines',
                        name='光谱',
                        line=dict(color='#1f77b4', width=1.5),
                    ))
                    
                    for label in labels:
                        fig.add_annotation(
                            x=label['x'],
                            y=label['y'],
                            text=label['text'],
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1,
                            arrowcolor='red',
                            font=dict(size=10, color='red'),
                            yshift=10,
                        )
                    
                    fig.update_layout(
                        title=current_spec.name,
                        xaxis_title=current_spec.x_unit,
                        yaxis_title="强度",
                        height=500,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### 识别到的元素")
                    
                    elem_data = []
                    for elem in st.session_state.element_results:
                        elem_data.append({
                            "元素": elem['element'],
                            "特征线数": len(elem['lines']),
                            "总强度": f"{elem['total_intensity']:.1f}",
                        })
                    
                    st.dataframe(pd.DataFrame(elem_data), use_container_width=True, hide_index=True)
                    
                    with st.expander("查看详细峰信息"):
                        for elem in st.session_state.element_results:
                            st.markdown(f"**{elem['element']}**")
                            line_data = []
                            for line in elem['lines']:
                                line_data.append({
                                    "谱线": line['line'],
                                    "实测能量(keV)": f"{line['peak_energy']:.3f}",
                                    "标准能量(keV)": f"{line['expected_energy']:.3f}",
                                    "偏差(keV)": f"{line['delta_energy']:+.4f}",
                                    "强度": f"{line['peak_intensity']:.1f}",
                                })
                            st.dataframe(pd.DataFrame(line_data), use_container_width=True, hide_index=True)
                else:
                    st.info("👈 请设置参数并识别元素")
        
        with tab2:
            st.subheader("定量分析")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 校准曲线设置")
                
                element_name = st.text_input("目标元素", "Fe")
                method = st.selectbox(
                    "拟合方法",
                    ["linear", "quadratic"],
                    format_func=lambda x: {"linear": "线性", "quadratic": "二次多项式"}[x],
                )
                matrix_type = st.selectbox(
                    "基体类型",
                    ["general", "heavy_matrix", "light_matrix", "soil", "rock", "water", "biological"],
                    format_func=lambda x: {
                        "general": "通用",
                        "heavy_matrix": "重基体",
                        "light_matrix": "轻基体",
                        "soil": "土壤",
                        "rock": "岩石",
                        "water": "水体",
                        "biological": "生物",
                    }[x],
                )
                
                st.markdown("#### 标样数据")
                
                num_standards = st.number_input("标样数量", 2, 10, 3)
                
                concs = []
                areas = []
                for i in range(int(num_standards)):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        c = st.number_input(f"标样{i+1}浓度 (%)", value=10.0 * (i+1), key=f"conc_{i}")
                        concs.append(c)
                    with col_b:
                        a = st.number_input(f"标样{i+1}峰面积", value=1000.0 * (i+1), key=f"area_{i}")
                        areas.append(a)
                
                if st.button("建立校准曲线"):
                    curve = CalibrationCurve(method=method)
                    curve.fit(concs, areas)
                    st.session_state.calibration_curve = curve
                    st.success("校准曲线已建立")
            
            with col2:
                if 'calibration_curve' in st.session_state and st.session_state.calibration_curve:
                    curve = st.session_state.calibration_curve
                    
                    fig_cal = go.Figure()
                    
                    fig_cal.add_trace(go.Scatter(
                        x=curve.peak_areas,
                        y=curve.concentrations,
                        mode='markers',
                        name='标样点',
                        marker=dict(size=10, color='red'),
                    ))
                    
                    x_curve = np.linspace(min(curve.peak_areas), max(curve.peak_areas), 100)
                    y_curve = np.polyval(curve.coefficients, x_curve)
                    fig_cal.add_trace(go.Scatter(
                        x=x_curve, y=y_curve,
                        mode='lines',
                        name='校准曲线',
                        line=dict(color='blue'),
                    ))
                    
                    fig_cal.update_layout(
                        title="校准曲线",
                        xaxis_title="峰面积",
                        yaxis_title="浓度 (%)",
                        height=400,
                    )
                    st.plotly_chart(fig_cal, use_container_width=True)
                    
                    st.info(f"曲线方程: {' + '.join([f'{c:.4f}x^{len(curve.coefficients)-1-i}' for i, c in enumerate(curve.coefficients[:-1])])} + {curve.coefficients[-1]:.4f}")
                    
                    if st.button("对当前样品定量分析"):
                        try:
                            result = quantitative_analysis(
                                current_spec,
                                element_name,
                                curve,
                                sensitivity=3.0,
                                peak_type='gaussian',
                                matrix_correction={'factor': matrix_correction_empirical(1.0, matrix_type)},
                            )
                            
                            st.success("定量分析完成")
                            st.metric(f"{element_name} 含量", f"{result['concentration']:.2f} %")
                            st.write(f"95%置信区间: {result['confidence_interval'][0]:.2f}% - {result['confidence_interval'][1]:.2f}%")
                            st.write(f"R² = {result['r_squared']:.4f}")
                            
                        except Exception as e:
                            st.error(f"分析失败: {e}")
                else:
                    st.info("👈 请先建立校准曲线")

elif page == "🔍 光谱检索":
    st.title("🔍 光谱检索")
    st.markdown("将未知光谱与数据库做相似度匹配")
    
    if not st.session_state.spectra:
        st.warning("请先导入数据")
    else:
        current_spec = st.session_state.spectra[st.session_state.current_spectrum_idx]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("检索参数")
            
            method = st.selectbox(
                "相似度算法",
                ["cosine", "pearson", "band"],
                format_func=lambda x: {
                    "cosine": "余弦相似度",
                    "pearson": "皮尔逊相关系数",
                    "band": "谱带匹配法",
                }[x],
            )
            
            top_n = st.slider("返回前N个结果", 1, 10, 5)
            
            if st.button("🔍 开始检索", type="primary"):
                with st.spinner("正在检索..."):
                    library = st.session_state.sample_library
                    results = search_spectrum(
                        current_spec, library, method, top_n
                    )
                    st.session_state.search_results = results
                
                st.success(f"检索完成，找到 {len(results)} 个匹配结果")
        
        with col2:
            st.subheader("检索结果")
            
            if 'search_results' in st.session_state and st.session_state.search_results:
                for i, result in enumerate(st.session_state.search_results):
                    spec = result['spectrum']
                    sim = result['similarity']
                    
                    with st.expander(
                        f"{i+1}. {spec.name} - 相似度: {sim*100:.1f}%",
                        expanded=(i == 0),
                    ):
                        fig = go.Figure()
                        
                        y1 = current_spec.y / current_spec.y.max() if current_spec.y.max() > 0 else current_spec.y
                        y2 = spec.y / spec.y.max() if spec.y.max() > 0 else spec.y
                        
                        x_min = max(current_spec.x.min(), spec.x.min())
                        x_max = min(current_spec.x.max(), spec.x.max())
                        
                        mask1 = (current_spec.x >= x_min) & (current_spec.x <= x_max)
                        mask2 = (spec.x >= x_min) & (spec.x <= x_max)
                        
                        fig.add_trace(go.Scatter(
                            x=current_spec.x[mask1], y=y1[mask1],
                            mode='lines',
                            name='未知光谱',
                            line=dict(color='#1f77b4', width=1.5),
                        ))
                        fig.add_trace(go.Scatter(
                            x=spec.x[mask2], y=y2[mask2],
                            mode='lines',
                            name=spec.name,
                            line=dict(color='#ff7f0e', width=1.5, dash='dash'),
                        ))
                        
                        fig.update_layout(
                            title="光谱对比（已归一化）",
                            xaxis_title=current_spec.x_unit,
                            yaxis_title="归一化强度",
                            height=300,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if 'formula' in spec.metadata:
                            st.write(f"**化学式**: {spec.metadata['formula']}")
            else:
                st.info("👈 请开始检索")
        
        st.markdown("---")
        
        with st.expander("管理样品库"):
            st.write(f"当前样品库中有 {len(st.session_state.sample_library)} 个样品")
            
            if st.button("添加当前样品到样品库"):
                st.session_state.sample_library.append(current_spec.copy())
                st.success("已添加到样品库")
                st.rerun()
            
            if st.button("重置样品库为默认"):
                st.session_state.sample_library = build_sample_library()
                st.success("样品库已重置")
                st.rerun()

elif page == "📈 多样品对比":
    st.title("📈 多样品对比")
    st.markdown("叠加显示、差谱、比值谱")
    
    if len(st.session_state.spectra) < 2:
        st.warning("请至少导入2个样品才能进行对比")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("选择样品")
            
            spectrum_names = [s.name for s in st.session_state.spectra]
            selected_indices = st.multiselect(
                "选择要对比的样品 (2-10个)",
                range(len(spectrum_names)),
                format_func=lambda i: spectrum_names[i],
                default=[0, 1] if len(spectrum_names) >= 2 else [],
                max_selections=10,
            )
            
            normalize = st.checkbox("统一归一化", value=True)
            
            comparison_mode = st.selectbox(
                "对比模式",
                ["叠加显示", "差谱 (A-B)", "比值谱 (A/B)"],
            )
            
            if comparison_mode in ["差谱 (A-B)", "比值谱 (A/B)"]:
                if len(selected_indices) >= 2:
                    spec_a_idx = st.selectbox(
                        "样品 A",
                        selected_indices,
                        format_func=lambda i: spectrum_names[i],
                        index=0,
                    )
                    spec_b_idx = st.selectbox(
                        "样品 B",
                        selected_indices,
                        format_func=lambda i: spectrum_names[i],
                        index=1 if len(selected_indices) > 1 else 0,
                    )
        
        with col2:
            st.subheader("对比图")
            
            if len(selected_indices) >= 2:
                selected_spectra = [st.session_state.spectra[i] for i in selected_indices]
                
                if comparison_mode == "叠加显示":
                    fig = plot_multi_spectra(
                        selected_spectra,
                        title="多样品叠加对比",
                        x_label=selected_spectra[0].x_unit,
                        normalize=normalize,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif comparison_mode == "差谱 (A-B)":
                    spec_a = st.session_state.spectra[spec_a_idx]
                    spec_b = st.session_state.spectra[spec_b_idx]
                    
                    x_diff, y_diff = compute_difference_spectrum(
                        spec_a, spec_b, normalize=normalize,
                    )
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=x_diff, y=y_diff,
                        mode='lines',
                        name='差谱 (A-B)',
                        line=dict(color='#2ca02c', width=1.5),
                        fill='tozeroy',
                        fillcolor='rgba(44, 160, 44, 0.2)',
                    ))
                    
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    
                    fig.update_layout(
                        title=f"差谱: {spec_a.name} - {spec_b.name}",
                        xaxis_title=spec_a.x_unit,
                        yaxis_title="强度差",
                        height=500,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif comparison_mode == "比值谱 (A/B)":
                    spec_a = st.session_state.spectra[spec_a_idx]
                    spec_b = st.session_state.spectra[spec_b_idx]
                    
                    x_ratio, y_ratio = compute_ratio_spectrum(
                        spec_a, spec_b, normalize=normalize,
                    )
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=x_ratio, y=y_ratio,
                        mode='lines',
                        name='比值谱 (A/B)',
                        line=dict(color='#d62728', width=1.5),
                    ))
                    
                    fig.add_hline(y=1, line_dash="dash", line_color="gray")
                    
                    fig.update_layout(
                        title=f"比值谱: {spec_a.name} / {spec_b.name}",
                        xaxis_title=spec_a.x_unit,
                        yaxis_title="强度比",
                        height=500,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    export_format = st.selectbox("导出格式", ["png", "svg", "pdf"])
                with col_b:
                    dpi = st.slider("分辨率 (DPI)", 72, 600, 300)
                
                if st.button("📥 导出对比图"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{export_format}') as tmp:
                        tmp_path = tmp.name
                    
                    success = export_spectrum_plot(
                        selected_spectra if comparison_mode == "叠加显示" else [spec_a, spec_b],
                        tmp_path,
                        title="光谱对比图",
                        x_label=selected_spectra[0].x_unit,
                        y_label="强度",
                        dpi=dpi,
                        format=export_format,
                    )
                    
                    if success:
                        with open(tmp_path, 'rb') as f:
                            st.download_button(
                                f"⬇️ 下载对比图 ({export_format.upper()})",
                                f,
                                f"spectrum_comparison.{export_format}",
                                f"image/{export_format}",
                            )
                        os.unlink(tmp_path)
                    else:
                        st.error("导出失败")
            else:
                st.info("👈 请至少选择2个样品")

elif page == "📊 PCA与分类":
    st.title("📊 主成分分析(PCA)与光谱分类")
    st.markdown("对多个样品光谱进行降维可视化和自动聚类分类")
    
    if 'pca_result' not in st.session_state:
        st.session_state.pca_result = None
    if 'clustering_result' not in st.session_state:
        st.session_state.clustering_result = None
    if 'anomaly_result' not in st.session_state:
        st.session_state.anomaly_result = None
    if 'pca_model_data' not in st.session_state:
        st.session_state.pca_model_data = None
    if 'selected_anomaly_idx' not in st.session_state:
        st.session_state.selected_anomaly_idx = None
    if 'trend_window' not in st.session_state:
        st.session_state.trend_window = 3
    
    if len(st.session_state.spectra) < 3:
        st.warning("请至少导入3个样品才能进行PCA分析")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 PCA分析", "🎯 聚类分析", "⚠️ 异常检测", "💾 模型与预测"])
        
        with tab1:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("数据准备")
                
                spectrum_names = [s.name for s in st.session_state.spectra]
                selected_indices = st.multiselect(
                    "选择分析样品 (≥3个)",
                    range(len(spectrum_names)),
                    format_func=lambda i: spectrum_names[i],
                    default=list(range(min(5, len(spectrum_names)))),
                )
                
                spacing = st.number_input(
                    "x轴网格间距",
                    min_value=0.01,
                    max_value=10.0,
                    value=0.5,
                    step=0.1,
                    format="%.2f",
                    help="所有光谱将插值到统一的等间距x轴网格"
                )
                
                n_components = st.slider(
                    "主成分数量",
                    min_value=2,
                    max_value=10,
                    value=3,
                    help="保留的主成分数"
                )
                
                if st.button("🚀 执行PCA分析", type="primary"):
                    if len(selected_indices) < 3:
                        st.error("请至少选择3个样品")
                    else:
                        with st.spinner("正在进行PCA分析..."):
                            selected_spectra = [st.session_state.spectra[i] for i in selected_indices]
                            
                            common_x, spectral_matrix, excluded_names, valid_spectra = interpolate_to_common_grid(
                                selected_spectra, spacing=spacing
                            )
                            
                            if len(valid_spectra) < 3:
                                st.error("有效样品不足3个，无法进行PCA分析")
                                if excluded_names:
                                    st.warning(f"以下样品因x轴范围不重叠被排除: {', '.join(excluded_names)}")
                            else:
                                X_std, mean, std = standardize_matrix(spectral_matrix)
                                
                                pca_dict = perform_pca(X_std, n_components=n_components)
                                
                                pca_result = PCAResult()
                                pca_result.selected_spectra = valid_spectra
                                pca_result.excluded_spectra = excluded_names
                                pca_result.common_x = common_x
                                pca_result.spectral_matrix = spectral_matrix
                                pca_result.mean = mean
                                pca_result.std = std
                                pca_result.eigenvalues = pca_dict['eigenvalues']
                                pca_result.eigenvectors = pca_dict['eigenvectors']
                                pca_result.variance_ratio = pca_dict['variance_ratio']
                                pca_result.cumulative_variance = pca_dict['cumulative_variance']
                                pca_result.scores = pca_dict['scores']
                                pca_result.loadings = pca_dict['loadings']
                                pca_result.n_components = pca_dict['n_components']
                                
                                st.session_state.pca_result = pca_result
                                st.session_state.clustering_result = None
                                st.session_state.anomaly_result = None
                                
                                st.success(f"PCA分析完成，保留 {pca_result.n_components} 个主成分")
                                if excluded_names:
                                    st.warning(f"以下样品因x轴范围不重叠被排除: {', '.join(excluded_names)}")
                
                if st.session_state.pca_result:
                    st.markdown("---")
                    st.subheader("得分图设置")
                    
                    plot_3d = st.checkbox("3D得分图", value=False)
                    color_by_type = st.checkbox("按光谱类型着色", value=True)
                    
                    loading_threshold = st.slider(
                        "载荷峰标注阈值",
                        min_value=0.05,
                        max_value=0.3,
                        value=0.1,
                        step=0.01,
                        help="标注载荷绝对值超过此阈值的峰位"
                    )
            
            with col2:
                if st.session_state.pca_result:
                    pca_result = st.session_state.pca_result
                    
                    st.subheader("方差解释比例")
                    
                    fig_var = go.Figure()
                    
                    pc_labels = [f'PC{i+1}' for i in range(pca_result.n_components)]
                    
                    fig_var.add_trace(go.Bar(
                        x=pc_labels,
                        y=pca_result.variance_ratio * 100,
                        name='方差解释比例',
                        marker_color='#1f77b4',
                        hovertemplate='%{x}: %{y:.2f}%<extra></extra>',
                    ))
                    
                    fig_var.add_trace(go.Scatter(
                        x=pc_labels,
                        y=pca_result.cumulative_variance * 100,
                        name='累积方差解释比例',
                        mode='lines+markers',
                        line=dict(color='#ff7f0e', width=2),
                        marker=dict(size=8),
                        yaxis='y2',
                        hovertemplate='%{x}: %{y:.2f}%<extra></extra>',
                    ))
                    
                    idx_95 = np.where(pca_result.cumulative_variance >= 0.95)[0]
                    if len(idx_95) > 0:
                        idx_95 = idx_95[0]
                        fig_var.add_hline(
                            y=95,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="95% 阈值",
                            annotation_position="top right",
                            yref='y2',
                        )
                    
                    fig_var.update_layout(
                        title="方差解释比例",
                        xaxis_title="主成分",
                        yaxis=dict(
                            title="方差解释比例 (%)",
                            side='left',
                        ),
                        yaxis2=dict(
                            title="累积方差解释比例 (%)",
                            side='right',
                            overlaying='y',
                            range=[0, 105],
                        ),
                        height=350,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=10, r=10, t=40, b=40),
                    )
                    
                    st.plotly_chart(fig_var, use_container_width=True)
                    
                    var_table = []
                    for i in range(pca_result.n_components):
                        var_table.append({
                            "主成分": f'PC{i+1}',
                            "特征值": f"{pca_result.eigenvalues[i]:.4f}",
                            "方差解释比例": f"{pca_result.variance_ratio[i]*100:.2f}%",
                            "累积方差解释比例": f"{pca_result.cumulative_variance[i]*100:.2f}%",
                        })
                    st.dataframe(pd.DataFrame(var_table), use_container_width=True, hide_index=True)
                    
                    st.markdown("---")
                    st.subheader("载荷图")
                    
                    fig_load = go.Figure()
                    
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                    for i in range(min(2, pca_result.n_components)):
                        fig_load.add_trace(go.Scatter(
                            x=pca_result.common_x,
                            y=pca_result.loadings[:, i],
                            mode='lines',
                            name=f'PC{i+1} 载荷',
                            line=dict(color=colors[i], width=1.5),
                            hovertemplate='x: %{x:.4f}<br>载荷: %{y:.4f}<extra></extra>',
                        ))
                        
                        loading_peaks = find_loading_peaks(
                            pca_result.common_x,
                            pca_result.loadings[:, i],
                            threshold=loading_threshold
                        )
                        
                        for peak in loading_peaks[:10]:
                            fig_load.add_vline(
                                x=peak['x'],
                                line_dash="dash",
                                line_color=colors[i],
                                opacity=0.5,
                            )
                            fig_load.add_annotation(
                                x=peak['x'],
                                y=peak['loading'],
                                text=f"{peak['x']:.2f}",
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowwidth=1,
                                arrowcolor=colors[i],
                                font=dict(size=9, color=colors[i]),
                                yshift=10 if peak['loading'] > 0 else -15,
                            )
                    
                    fig_load.add_hline(y=0, line_dash="solid", line_color="gray", line_width=0.5)
                    
                    fig_load.update_layout(
                        title="前两个主成分的载荷向量",
                        xaxis_title=pca_result.selected_spectra[0].x_unit,
                        yaxis_title="载荷值",
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=10, r=10, t=40, b=40),
                    )
                    
                    st.plotly_chart(fig_load, use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("得分图")
                    
                    sample_names = [s.name for s in pca_result.selected_spectra]
                    spectrum_types = [s.spectrum_type.value for s in pca_result.selected_spectra]
                    
                    unique_types = list(set(spectrum_types))
                    type_colors = {t: colors[i % len(colors)] for i, t in enumerate(unique_types)}
                    
                    if pca_result.n_components >= 2:
                        if plot_3d and pca_result.n_components >= 3:
                            fig_score = go.Figure()
                            
                            for i, name in enumerate(sample_names):
                                color = type_colors[spectrum_types[i]] if color_by_type else '#1f77b4'
                                fig_score.add_trace(go.Scatter3d(
                                    x=[pca_result.scores[i, 0]],
                                    y=[pca_result.scores[i, 1]],
                                    z=[pca_result.scores[i, 2]],
                                    mode='markers+text',
                                    name=name,
                                    text=[name],
                                    textposition="top center",
                                    marker=dict(size=8, color=color, line=dict(width=1, color='DarkSlateGrey')),
                                    hovertemplate=f'{name}<br>PC1: %{{x:.4f}}<br>PC2: %{{y:.4f}}<br>PC3: %{{z:.4f}}<extra></extra>',
                                ))
                            
                            fig_score.update_layout(
                                title=f"3D得分图 (PC{1}, PC{2}, PC{3})",
                                scene=dict(
                                    xaxis_title=f'PC1 ({pca_result.variance_ratio[0]*100:.2f}%)',
                                    yaxis_title=f'PC2 ({pca_result.variance_ratio[1]*100:.2f}%)',
                                    zaxis_title=f'PC3 ({pca_result.variance_ratio[2]*100:.2f}%)',
                                ),
                                height=500,
                                showlegend=True,
                            )
                        else:
                            fig_score = go.Figure()
                            
                            for i, name in enumerate(sample_names):
                                color = type_colors[spectrum_types[i]] if color_by_type else '#1f77b4'
                                fig_score.add_trace(go.Scatter(
                                    x=[pca_result.scores[i, 0]],
                                    y=[pca_result.scores[i, 1]],
                                    mode='markers+text',
                                    name=name,
                                    text=[name],
                                    textposition="top center",
                                    marker=dict(size=10, color=color, line=dict(width=1, color='DarkSlateGrey')),
                                    hovertemplate=f'{name}<br>PC1: %{{x:.4f}}<br>PC2: %{{y:.4f}}<extra></extra>',
                                ))
                            
                            fig_score.update_layout(
                                title=f"2D得分图 (PC{1} vs PC{2})",
                                xaxis_title=f'PC1 ({pca_result.variance_ratio[0]*100:.2f}%)',
                                yaxis_title=f'PC2 ({pca_result.variance_ratio[1]*100:.2f}%)',
                                height=500,
                                showlegend=True,
                            )
                        
                        st.plotly_chart(fig_score, use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("得分矩阵")
                    scores_df = pd.DataFrame(
                        pca_result.scores,
                        columns=[f'PC{i+1}' for i in range(pca_result.n_components)],
                        index=sample_names
                    )
                    st.dataframe(scores_df, use_container_width=True)
                else:
                    st.info("👈 请选择样品并执行PCA分析")
        
        with tab2:
            if not st.session_state.pca_result:
                st.warning("请先在「PCA分析」标签页执行PCA分析")
            else:
                pca_result = st.session_state.pca_result
                scores = pca_result.scores
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("聚类设置")
                    
                    auto_k = st.checkbox("自动确定K值 (Elbow法)", value=True)
                    
                    if auto_k:
                        if st.button("🔍 计算最佳K值", type="primary"):
                            with st.spinner("正在计算Elbow曲线..."):
                                elbow_result = elbow_method(scores, k_min=2, k_max=10)
                                
                                clustering_result = ClusteringResult()
                                clustering_result.elbow_scores = elbow_result['sse_scores']
                                clustering_result.calculated_k = elbow_result['recommended_k']
                                clustering_result.k_value = elbow_result['recommended_k']
                                
                                st.session_state.clustering_result = clustering_result
                                st.success(f"推荐K值: {clustering_result.k_value}")
                    else:
                        k_value = st.slider("聚类数 K", min_value=2, max_value=10, value=3)
                        
                        if st.button("🎯 执行K-Means聚类", type="primary"):
                            with st.spinner("正在进行K-Means聚类..."):
                                kmeans_result = perform_kmeans(scores, k=k_value)
                                
                                centroids_original = reconstruct_centroids(
                                    kmeans_result['centroids_pca'],
                                    pca_result.eigenvectors,
                                    pca_result.mean,
                                    pca_result.std
                                )
                                
                                cosine_sims = {}
                                for cluster_id in range(k_value):
                                    mask = kmeans_result['labels'] == cluster_id
                                    if mask.sum() > 0:
                                        cluster_spectra = pca_result.spectral_matrix[mask]
                                        sim = calculate_cosine_similarity_centroid(
                                            centroids_original[cluster_id],
                                            cluster_spectra
                                        )
                                        cosine_sims[cluster_id] = sim
                                
                                true_labels = [s.spectrum_type for s in pca_result.selected_spectra]
                                if len(set(true_labels)) > 1:
                                    try:
                                        cm, label_names = compute_confusion_matrix(
                                            None, kmeans_result['labels'], spectrum_types=true_labels
                                        )
                                    except:
                                        cm, label_names = None, None
                                else:
                                    cm, label_names = None, None
                                
                                clustering_result = ClusteringResult()
                                clustering_result.labels = kmeans_result['labels']
                                clustering_result.centroids_pca = kmeans_result['centroids_pca']
                                clustering_result.centroids_original = centroids_original
                                clustering_result.k_value = k_value
                                clustering_result.cosine_similarities = cosine_sims
                                clustering_result.confusion_matrix = cm
                                clustering_result.true_labels = label_names
                                
                                st.session_state.clustering_result = clustering_result
                                st.success(f"聚类完成，K={k_value}")
                
                with col2:
                    if st.session_state.clustering_result:
                        clustering_result = st.session_state.clustering_result
                        
                        if clustering_result.elbow_scores:
                            st.subheader("Elbow曲线")
                            
                            k_values = sorted(clustering_result.elbow_scores.keys())
                            sse_values = [clustering_result.elbow_scores[k] for k in k_values]
                            
                            fig_elbow = go.Figure()
                            fig_elbow.add_trace(go.Scatter(
                                x=k_values,
                                y=sse_values,
                                mode='lines+markers',
                                name='SSE',
                                line=dict(color='#1f77b4', width=2),
                                marker=dict(size=10),
                            ))
                            
                            if clustering_result.calculated_k:
                                best_idx = k_values.index(clustering_result.calculated_k)
                                fig_elbow.add_annotation(
                                    x=clustering_result.calculated_k,
                                    y=sse_values[best_idx],
                                    text=f"推荐 K={clustering_result.calculated_k}",
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowsize=1,
                                    arrowwidth=2,
                                    arrowcolor='red',
                                    font=dict(size=12, color='red'),
                                    yshift=30,
                                )
                            
                            fig_elbow.update_layout(
                                title="Elbow曲线 - SSE vs K",
                                xaxis_title="聚类数 K",
                                yaxis_title="组内平方和 (SSE)",
                                height=350,
                                margin=dict(l=10, r=10, t=40, b=40),
                            )
                            
                            st.plotly_chart(fig_elbow, use_container_width=True)
                            
                            if st.button("🎯 使用推荐K值进行聚类", type="primary"):
                                k_value = clustering_result.calculated_k
                                with st.spinner(f"正在进行K-Means聚类 (K={k_value})..."):
                                    kmeans_result = perform_kmeans(scores, k=k_value)
                                    
                                    centroids_original = reconstruct_centroids(
                                        kmeans_result['centroids_pca'],
                                        pca_result.eigenvectors,
                                        pca_result.mean,
                                        pca_result.std
                                    )
                                    
                                    cosine_sims = {}
                                    for cluster_id in range(k_value):
                                        mask = kmeans_result['labels'] == cluster_id
                                        if mask.sum() > 0:
                                            cluster_spectra = pca_result.spectral_matrix[mask]
                                            sim = calculate_cosine_similarity_centroid(
                                                centroids_original[cluster_id],
                                                cluster_spectra
                                            )
                                            cosine_sims[cluster_id] = sim
                                    
                                    true_labels = [s.spectrum_type for s in pca_result.selected_spectra]
                                    if len(set(true_labels)) > 1:
                                        try:
                                            cm, label_names = compute_confusion_matrix(
                                                None, kmeans_result['labels'], spectrum_types=true_labels
                                            )
                                        except:
                                            cm, label_names = None, None
                                    else:
                                        cm, label_names = None, None
                                    
                                    clustering_result.labels = kmeans_result['labels']
                                    clustering_result.centroids_pca = kmeans_result['centroids_pca']
                                    clustering_result.centroids_original = centroids_original
                                    clustering_result.k_value = k_value
                                    clustering_result.cosine_similarities = cosine_sims
                                    clustering_result.confusion_matrix = cm
                                    clustering_result.true_labels = label_names
                                    
                                    st.success(f"聚类完成，K={k_value}")
                                    st.rerun()
                        
                        if clustering_result.labels is not None:
                            st.markdown("---")
                            st.subheader("聚类结果得分图")
                            
                            sample_names = [s.name for s in pca_result.selected_spectra]
                            labels = clustering_result.labels
                            k = clustering_result.k_value
                            
                            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                            symbols = ['circle', 'square', 'triangle-up', 'diamond', 'cross',
                                       'x', 'pentagon', 'hexagon', 'star', 'bowtie']
                            
                            fig_clusters = go.Figure()
                            
                            for cluster_id in range(k):
                                mask = labels == cluster_id
                                cluster_indices = np.where(mask)[0]
                                
                                for idx in cluster_indices:
                                    fig_clusters.add_trace(go.Scatter(
                                        x=[scores[idx, 0]],
                                        y=[scores[idx, 1]],
                                        mode='markers+text',
                                        name=f'聚类 {cluster_id+1}',
                                        text=[sample_names[idx]],
                                        textposition="top center",
                                        marker=dict(
                                            size=12,
                                            color=colors[cluster_id % len(colors)],
                                            symbol=symbols[cluster_id % len(symbols)],
                                            line=dict(width=2, color='DarkSlateGrey')
                                        ),
                                        legendgroup=f'cluster_{cluster_id}',
                                        showlegend=(idx == cluster_indices[0]),
                                        hovertemplate=f'{sample_names[idx]}<br>聚类: {cluster_id+1}<br>PC1: %{{x:.4f}}<br>PC2: %{{y:.4f}}<extra></extra>',
                                    ))
                            
                            for cluster_id in range(k):
                                fig_clusters.add_trace(go.Scatter(
                                    x=[clustering_result.centroids_pca[cluster_id, 0]],
                                    y=[clustering_result.centroids_pca[cluster_id, 1]],
                                    mode='markers',
                                    name=f'质心 {cluster_id+1}',
                                    marker=dict(
                                        size=18,
                                        color=colors[cluster_id % len(colors)],
                                        symbol='x',
                                        line=dict(width=3, color='black')
                                    ),
                                    legendgroup=f'centroid_{cluster_id}',
                                    hovertemplate=f'质心 {cluster_id+1}<br>PC1: %{{x:.4f}}<br>PC2: %{{y:.4f}}<extra></extra>',
                                ))
                            
                            fig_clusters.update_layout(
                                title=f"聚类结果得分图 (K={k})",
                                xaxis_title=f'PC1 ({pca_result.variance_ratio[0]*100:.2f}%)',
                                yaxis_title=f'PC2 ({pca_result.variance_ratio[1]*100:.2f}%)',
                                height=500,
                            )
                            
                            st.plotly_chart(fig_clusters, use_container_width=True)
                            
                            st.markdown("---")
                            st.subheader("聚类成员")
                            
                            cluster_data = []
                            for i, (name, label) in enumerate(zip(sample_names, labels)):
                                cluster_data.append({
                                    "样品名称": name,
                                    "聚类": int(label) + 1,
                                })
                            
                            st.dataframe(pd.DataFrame(cluster_data), use_container_width=True, hide_index=True)
                            
                            if clustering_result.cosine_similarities:
                                st.markdown("---")
                                st.subheader("质心谱验证")
                                
                                sim_data = []
                                for cluster_id in range(k):
                                    sim = clustering_result.cosine_similarities.get(cluster_id, 0)
                                    sim_data.append({
                                        "聚类": cluster_id + 1,
                                        "余弦相似度": f"{sim:.4f}",
                                        "状态": "✅ 合格" if sim > 0.9 else "⚠️ 警告"
                                    })
                                
                                st.dataframe(pd.DataFrame(sim_data), use_container_width=True, hide_index=True)
                                
                                low_sims = [c for c, s in clustering_result.cosine_similarities.items() if s <= 0.9]
                                if low_sims:
                                    st.warning(f"聚类 {', '.join([str(c+1) for c in low_sims])} 的质心谱与类内平均谱的余弦相似度 ≤ 0.9，可能需要检查聚类质量")
                            
                            if clustering_result.confusion_matrix is not None and clustering_result.true_labels:
                                st.markdown("---")
                                st.subheader("混淆矩阵 (基于光谱类型)")
                                
                                cm = clustering_result.confusion_matrix
                                labels = clustering_result.true_labels
                                
                                fig_cm = go.Figure(data=go.Heatmap(
                                    z=cm,
                                    x=[f"预测: {l}" for l in labels],
                                    y=[f"真实: {l}" for l in labels],
                                    text=cm,
                                    texttemplate="%{text}",
                                    textfont={"size": 14},
                                    colorscale='Blues',
                                ))
                                
                                fig_cm.update_layout(
                                    title="分类混淆矩阵",
                                    xaxis_title="预测标签",
                                    yaxis_title="真实标签",
                                    height=400,
                                )
                                
                                st.plotly_chart(fig_cm, use_container_width=True)
                            
                            if clustering_result.centroids_original is not None:
                                st.markdown("---")
                                st.subheader("质心重建谱")
                                
                                fig_centroid = go.Figure()
                                
                                for cluster_id in range(k):
                                    fig_centroid.add_trace(go.Scatter(
                                        x=pca_result.common_x,
                                        y=clustering_result.centroids_original[cluster_id],
                                        mode='lines',
                                        name=f'聚类 {cluster_id+1} 质心',
                                        line=dict(color=colors[cluster_id % len(colors)], width=2),
                                    ))
                                
                                fig_centroid.update_layout(
                                    title="各聚类质心的重建光谱",
                                    xaxis_title=pca_result.selected_spectra[0].x_unit,
                                    yaxis_title="强度",
                                    height=400,
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                )
                                
                                st.plotly_chart(fig_centroid, use_container_width=True)
                    else:
                        st.info("👈 请设置聚类参数并执行聚类分析")
        
        with tab3:
            if not st.session_state.pca_result or not st.session_state.clustering_result:
                st.warning("请先执行PCA分析和聚类分析")
            else:
                pca_result = st.session_state.pca_result
                clustering_result = st.session_state.clustering_result
                sample_names = [s.name for s in pca_result.selected_spectra]
                x_unit = pca_result.selected_spectra[0].x_unit if pca_result.selected_spectra else "x"

                col_left, col_right = st.columns([1, 2])

                with col_left:
                    st.subheader("异常检测设置")

                    mahal_std_multiplier = st.slider(
                        "马氏距离阈值 (标准差倍数)",
                        min_value=1.0,
                        max_value=5.0,
                        value=3.0,
                        step=0.1,
                    )

                    enable_trend = st.checkbox(
                        "启用时序趋势分析",
                        value=False,
                        help="如按顺序导入的样品代表时间序列，可启用此功能"
                    )

                    trend_window = 3
                    if enable_trend:
                        trend_window = st.slider(
                            "移动平均窗口大小",
                            min_value=2,
                            max_value=min(10, max(2, len(sample_names) - 1)),
                            value=3,
                            step=1,
                        )
                        st.session_state.trend_window = trend_window

                    if st.button("⚠️ 执行异常检测", type="primary"):
                        with st.spinner("正在进行异常检测..."):
                            anomaly_result = AnomalyResult()

                            mahal_dist, mahal_thresh, anomaly_mahal = compute_mahalanobis_distances(
                                pca_result.scores,
                                clustering_result.labels,
                                clustering_result.centroids_pca
                            )

                            custom_mahal_thresh = mahal_dist.mean() + mahal_std_multiplier * mahal_dist.std()
                            anomaly_mahal_custom = np.where(mahal_dist > custom_mahal_thresh)[0].tolist()

                            anomaly_result.mahalanobis_distances = mahal_dist
                            anomaly_result.mahalanobis_threshold = custom_mahal_thresh
                            anomaly_result.anomaly_indices_mahal = anomaly_mahal_custom

                            t2, t2_thresh = compute_hotelling_t2(
                                pca_result.scores,
                                pca_result.eigenvalues,
                                n_components=pca_result.n_components
                            )
                            anomaly_t2 = np.where(t2 > t2_thresh)[0].tolist()

                            anomaly_result.hotelling_t2 = t2
                            anomaly_result.t2_threshold = t2_thresh
                            anomaly_result.anomaly_indices_t2 = anomaly_t2

                            X_std = (pca_result.spectral_matrix - pca_result.mean) / pca_result.std
                            q_res, q_thresh = compute_q_residuals(
                                X_std,
                                pca_result.scores,
                                pca_result.eigenvectors,
                                n_components=pca_result.n_components
                            )
                            anomaly_q = np.where(q_res > q_thresh)[0].tolist()

                            anomaly_result.q_residuals = q_res
                            anomaly_result.q_threshold = q_thresh
                            anomaly_result.anomaly_indices_q = anomaly_q

                            t2_norm = t2 / t2_thresh
                            q_norm = q_res / q_thresh
                            anomaly_result.quadrants = classify_quadrant(t2_norm, q_norm)

                            t2_contrib = compute_t2_contributions(
                                pca_result.scores,
                                pca_result.eigenvalues,
                                n_components=pca_result.n_components
                            )
                            anomaly_result.t2_contributions = t2_contrib

                            q_contrib, residuals_mat = compute_q_contributions(
                                X_std,
                                pca_result.scores,
                                pca_result.eigenvectors,
                                n_components=pca_result.n_components
                            )
                            anomaly_result.q_contributions = q_contrib
                            anomaly_result.residual_matrix = residuals_mat

                            anomaly_indices = list(set(anomaly_mahal_custom + anomaly_t2 + anomaly_q))
                            anomaly_result.anomaly_samples = [sample_names[i] for i in anomaly_indices]

                            st.session_state.anomaly_result = anomaly_result
                            st.session_state.selected_anomaly_idx = None

                            anomaly_count = len(anomaly_indices)
                            if anomaly_count > 0:
                                st.warning(f"检测到 {anomaly_count} 个潜在异常样品")
                            else:
                                st.success("未检测到异常样品")

                with col_right:
                    if st.session_state.anomaly_result:
                        anomaly_result = st.session_state.anomaly_result

                        t2 = anomaly_result.hotelling_t2
                        t2_thresh = anomaly_result.t2_threshold
                        q_res = anomaly_result.q_residuals
                        q_thresh = anomaly_result.q_threshold
                        t2_norm = t2 / t2_thresh
                        q_norm = q_res / q_thresh
                        quadrants = anomaly_result.quadrants

                        quadrant_colors = {
                            'both': '#d62728',
                            't2_only': '#ff7f0e',
                            'q_only': '#ffcc00',
                            'normal': '#7f7f7f',
                        }
                        quadrant_names = {
                            'both': '双指标超限',
                            't2_only': '仅T²超限',
                            'q_only': '仅Q超限',
                            'normal': '正常',
                        }
                        point_colors = [quadrant_colors[q] for q in quadrants]

                        fig_joint = go.Figure()

                        for q_name in ['normal', 't2_only', 'q_only', 'both']:
                            mask = [q == q_name for q in quadrants]
                            if not any(mask):
                                continue
                            x_vals = [t2_norm[i] for i in range(len(t2_norm)) if mask[i]]
                            y_vals = [q_norm[i] for i in range(len(q_norm)) if mask[i]]
                            names_sub = [sample_names[i] for i in range(len(sample_names)) if mask[i]]
                            indices_sub = [i for i in range(len(sample_names)) if mask[i]]

                            is_both = (q_name == 'both')
                            marker_size = 14 if is_both else 10
                            marker_width = 3 if is_both else 1

                            text_labels = names_sub if is_both else None
                            text_pos = 'top center' if is_both else None

                            customdata = np.array([[idx, n] for idx, n in zip(indices_sub, names_sub)])

                            fig_joint.add_trace(go.Scatter(
                                x=x_vals,
                                y=y_vals,
                                mode='markers' + ('+text' if is_both else ''),
                                name=quadrant_names[q_name],
                                text=text_labels,
                                textposition=text_pos,
                                textfont=dict(color=quadrant_colors[q_name], size=11, family='Arial Black'),
                                marker=dict(
                                    size=marker_size,
                                    color=quadrant_colors[q_name],
                                    line=dict(width=marker_width, color='black' if is_both else 'DarkSlateGrey'),
                                ),
                                customdata=customdata,
                                hovertemplate=(
                                    '<b>%{customdata[1]}</b><br>'
                                    '归一化T²: %{x:.4f}<br>'
                                    '归一化Q: %{y:.4f}<br>'
                                    '象限: ' + quadrant_names[q_name] + '<extra></extra>'
                                ),
                            ))

                        x_max = max(2.0, float(np.max(t2_norm) * 1.15))
                        y_max = max(2.0, float(np.max(q_norm) * 1.15))

                        fig_joint.add_vline(
                            x=1.0,
                            line_dash="dash",
                            line_color="red",
                            line_width=1.5,
                            annotation_text="T² 阈值",
                            annotation_position="top right",
                        )
                        fig_joint.add_hline(
                            y=1.0,
                            line_dash="dash",
                            line_color="red",
                            line_width=1.5,
                            annotation_text="Q 阈值",
                            annotation_position="top right",
                        )

                        fig_joint.add_shape(
                            type="rect",
                            x0=1.0, y0=1.0, x1=x_max, y1=y_max,
                            fillcolor="rgba(214,39,40,0.06)",
                            line_width=0,
                        )
                        fig_joint.add_shape(
                            type="rect",
                            x0=1.0, y0=0, x1=x_max, y1=1.0,
                            fillcolor="rgba(255,127,14,0.06)",
                            line_width=0,
                        )
                        fig_joint.add_shape(
                            type="rect",
                            x0=0, y0=1.0, x1=1.0, y1=y_max,
                            fillcolor="rgba(255,204,0,0.06)",
                            line_width=0,
                        )

                        fig_joint.add_annotation(
                            x=x_max * 0.92, y=y_max * 0.92,
                            text="🔴 双超限",
                            showarrow=False,
                            font=dict(color='#d62728', size=12),
                            xanchor="right", yanchor="top",
                        )
                        fig_joint.add_annotation(
                            x=x_max * 0.92, y=y_max * 0.08,
                            text="🟠 仅T²",
                            showarrow=False,
                            font=dict(color='#ff7f0e', size=12),
                            xanchor="right", yanchor="bottom",
                        )
                        fig_joint.add_annotation(
                            x=x_max * 0.08, y=y_max * 0.92,
                            text="🟡 仅Q",
                            showarrow=False,
                            font=dict(color='#c9a400', size=12),
                            xanchor="left", yanchor="top",
                        )
                        fig_joint.add_annotation(
                            x=x_max * 0.08, y=y_max * 0.08,
                            text="⚪ 正常",
                            showarrow=False,
                            font=dict(color='#7f7f7f', size=12),
                            xanchor="left", yanchor="bottom",
                        )

                        fig_joint.update_layout(
                            title="T²-Q 联合诊断图 (归一化)",
                            xaxis_title="归一化 T² (T² / T²_limit)",
                            yaxis_title="归一化 Q (Q / Q_limit)",
                            height=520,
                            xaxis_range=[0, x_max],
                            yaxis_range=[0, y_max],
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            margin=dict(l=10, r=10, t=60, b=40),
                        )

                        st.plotly_chart(fig_joint, use_container_width=True)

                        st.markdown(
                            f"<div style='font-size:13px;color:#666;margin-top:-10px;'>"
                            f"💡 使用下方下拉框选择样品查看贡献度分析</div>",
                            unsafe_allow_html=True
                        )

                        st.markdown("---")

                        anomaly_sample_options = []
                        for i, name in enumerate(sample_names):
                            tag = ""
                            if quadrants[i] == 'both':
                                tag = " 🔴"
                            elif quadrants[i] == 't2_only':
                                tag = " 🟠"
                            elif quadrants[i] == 'q_only':
                                tag = " 🟡"
                            anomaly_sample_options.append((i, name + tag))

                        selected_label = st.selectbox(
                            "🔍 选择样品查看贡献度分析 (或直接点击上图散点)",
                            range(len(anomaly_sample_options)),
                            format_func=lambda idx: anomaly_sample_options[idx][1],
                            index=st.session_state.selected_anomaly_idx if st.session_state.selected_anomaly_idx is not None else 0,
                            key="contrib_sample_select",
                        )
                        st.session_state.selected_anomaly_idx = selected_label

                        if st.session_state.selected_anomaly_idx is not None:
                            sel_idx = int(st.session_state.selected_anomaly_idx)
                            sel_name = sample_names[sel_idx]

                            st.markdown(f"### 📊 贡献度分析 — <span style='color:#d62728;'>{sel_name}</span>", unsafe_allow_html=True)

                            t2_contrib_sel = anomaly_result.t2_contributions[sel_idx]
                            pc_labels = [f'PC{i+1}' for i in range(len(t2_contrib_sel))]

                            fig_t2_contrib = go.Figure()
                            fig_t2_contrib.add_trace(go.Bar(
                                x=pc_labels,
                                y=t2_contrib_sel,
                                marker_color='#1f77b4',
                                hovertemplate='%{x}<br>贡献度: %{y:.4f}<extra></extra>',
                            ))
                            fig_t2_contrib.update_layout(
                                title=f"T² 贡献度 (主成分维度) — {sel_name}",
                                xaxis_title="主成分",
                                yaxis_title="贡献度 (score² / eigenvalue)",
                                height=340,
                                margin=dict(l=10, r=10, t=40, b=40),
                            )
                            st.plotly_chart(fig_t2_contrib, use_container_width=True)

                            q_contrib_sel = anomaly_result.q_contributions[sel_idx]
                            q_mean = float(np.mean(q_contrib_sel))
                            q_threshold_mark = q_mean * 3.0

                            fig_q_contrib = go.Figure()
                            fig_q_contrib.add_trace(go.Bar(
                                x=pca_result.common_x,
                                y=q_contrib_sel,
                                marker_color='#2ca02c',
                                hovertemplate=(
                                    f'{x_unit}: ' + '%{x:.4f}<br>'
                                    'Q贡献度: %{y:.6f}<extra></extra>'
                                ),
                            ))
                            fig_q_contrib.add_hline(
                                y=q_threshold_mark,
                                line_dash="dash",
                                line_color="red",
                                line_width=1.5,
                                annotation_text=f"3倍均值阈值 ({q_threshold_mark:.4f})",
                                annotation_position="top right",
                            )

                            high_mask = q_contrib_sel > q_threshold_mark
                            if np.any(high_mask):
                                high_x = pca_result.common_x[high_mask]
                                high_y = q_contrib_sel[high_mask]
                                for hx, hy in zip(high_x, high_y):
                                    fig_q_contrib.add_annotation(
                                        x=hx,
                                        y=hy,
                                        text=f"{hx:.2f}",
                                        showarrow=True,
                                        arrowhead=2,
                                        arrowsize=1,
                                        arrowwidth=1,
                                        arrowcolor='red',
                                        font=dict(size=9, color='red'),
                                        yshift=12,
                                    )

                            fig_q_contrib.update_layout(
                                title=f"Q 残差贡献度 ({x_unit}维度) — {sel_name}",
                                xaxis_title=x_unit,
                                yaxis_title="贡献度 (残差平方)",
                                height=340,
                                margin=dict(l=10, r=10, t=40, b=40),
                            )
                            st.plotly_chart(fig_q_contrib, use_container_width=True)

                            st.markdown("---")

                        st.subheader("📋 异常样品诊断")

                        sorted_indices = list(range(len(sample_names)))

                        for i in sorted_indices:
                            name = sample_names[i]
                            is_mahal = i in anomaly_result.anomaly_indices_mahal
                            is_t2 = i in anomaly_result.anomaly_indices_t2
                            is_q = i in anomaly_result.anomaly_indices_q
                            is_anomaly = is_mahal or is_t2 or is_q
                            quad = quadrants[i]

                            icon = "🔴" if quad == 'both' else ("🟠" if quad == 't2_only' else ("🟡" if quad == 'q_only' else "🟢"))
                            exp_title = f"{icon} {name} — {quadrant_names.get(quad, '正常')}"

                            with st.expander(exp_title, expanded=is_anomaly):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("归一化 T²", f"{t2_norm[i]:.4f}",
                                              delta="超限" if t2_norm[i] > 1 else "正常",
                                              delta_color="inverse")
                                    st.metric("归一化 Q", f"{q_norm[i]:.4f}",
                                              delta="超限" if q_norm[i] > 1 else "正常",
                                              delta_color="inverse")
                                with col_b:
                                    st.metric("所属象限", quadrant_names.get(quad, "正常"))
                                    st.metric("马氏距离", f"{anomaly_result.mahalanobis_distances[i]:.4f}")

                                st.markdown("**Top-3 T² 贡献主成分**:")
                                top_t2 = get_top_t2_contributions(
                                    anomaly_result.t2_contributions, i, top_k=3
                                )
                                t2_cols = st.columns(3)
                                for ci, (pc_num, val) in enumerate(top_t2):
                                    with t2_cols[ci]:
                                        st.info(f"**PC{pc_num}**\n\n贡献值: {val:.4f}")

                                st.markdown(f"**Top-5 Q 贡献{x_unit}位置**:")
                                top_q = get_top_q_contributions(
                                    anomaly_result.q_contributions, pca_result.common_x, i, top_k=5
                                )
                                q_data = []
                                for xv, val in top_q:
                                    q_data.append({
                                        f"{x_unit}位置": f"{xv:.4f}",
                                        "贡献值": f"{val:.6f}",
                                    })
                                if q_data:
                                    st.dataframe(pd.DataFrame(q_data), use_container_width=True, hide_index=True)

                        st.markdown("---")

                        if enable_trend and len(sample_names) >= 3:
                            st.subheader("📈 趋势分析面板")

                            ma_window = trend_window
                            t2_ma = compute_moving_average(t2_norm, ma_window)
                            q_ma = compute_moving_average(q_norm, ma_window)

                            t2_total_slope, t2_recent_slope = compute_trend_slope(t2_ma, window=3)
                            q_total_slope, q_recent_slope = compute_trend_slope(q_ma, window=3)

                            t2_ma_valid = t2_ma[~np.isnan(t2_ma)]
                            q_ma_valid = q_ma[~np.isnan(q_ma)]

                            t2_warn = False
                            q_warn = False

                            if len(t2_ma_valid) >= 4:
                                t2_end_gt_start = t2_ma_valid[-1] > t2_ma_valid[0] * 1.05
                                t2_mean_first3 = float(np.mean(t2_ma_valid[:3]))
                                t2_mean_last3 = float(np.mean(t2_ma_valid[-3:]))
                                t2_mean_gt = t2_mean_last3 > t2_mean_first3 * 1.1
                                t2_recent_pos = t2_recent_slope > 0
                                t2_warn = (t2_end_gt_start or t2_mean_gt) and t2_recent_pos

                            if len(q_ma_valid) >= 4:
                                q_end_gt_start = q_ma_valid[-1] > q_ma_valid[0] * 1.05
                                q_mean_first3 = float(np.mean(q_ma_valid[:3]))
                                q_mean_last3 = float(np.mean(q_ma_valid[-3:]))
                                q_mean_gt = q_mean_last3 > q_mean_first3 * 1.1
                                q_recent_pos = q_recent_slope > 0
                                q_warn = (q_end_gt_start or q_mean_gt) and q_recent_pos

                            if t2_warn or q_warn:
                                st.error("⚠️ **检测到异常趋势，建议关注**")
                                warn_details = []
                                if t2_warn:
                                    warn_details.append(f"T²统计量移动平均呈显著递增趋势 (近期斜率={t2_recent_slope:.4f})")
                                if q_warn:
                                    warn_details.append(f"Q残差移动平均呈显著递增趋势 (近期斜率={q_recent_slope:.4f})")
                                for w in warn_details:
                                    st.warning(w)

                            fig_trend = make_subplots(
                                rows=2, cols=1,
                                shared_xaxes=True,
                                row_heights=[0.5, 0.5],
                                vertical_spacing=0.08,
                                subplot_titles=("归一化 T² 时序", "归一化 Q 时序"),
                            )

                            fig_trend.add_trace(go.Scatter(
                                x=sample_names,
                                y=t2_norm,
                                mode='lines+markers',
                                name='T² 原始',
                                marker=dict(color='#1f77b4', size=6),
                                line=dict(color='#1f77b4', width=1),
                                hovertemplate='%{x}<br>T²: %{y:.4f}<extra></extra>',
                            ), row=1, col=1)
                            fig_trend.add_trace(go.Scatter(
                                x=sample_names,
                                y=t2_ma,
                                mode='lines',
                                name=f'T² MA({ma_window})',
                                line=dict(color='#d62728', width=2.5, dash='solid'),
                                hovertemplate='%{x}<br>T² MA: %{y:.4f}<extra></extra>',
                            ), row=1, col=1)
                            fig_trend.add_hline(
                                y=1.0,
                                line_dash="dash",
                                line_color="red",
                                line_width=1.2,
                                annotation_text="阈值",
                                annotation_position="top right",
                                row=1, col=1,
                            )

                            fig_trend.add_trace(go.Scatter(
                                x=sample_names,
                                y=q_norm,
                                mode='lines+markers',
                                name='Q 原始',
                                marker=dict(color='#2ca02c', size=6),
                                line=dict(color='#2ca02c', width=1),
                                hovertemplate='%{x}<br>Q: %{y:.4f}<extra></extra>',
                            ), row=2, col=1)
                            fig_trend.add_trace(go.Scatter(
                                x=sample_names,
                                y=q_ma,
                                mode='lines',
                                name=f'Q MA({ma_window})',
                                line=dict(color='#ff7f0e', width=2.5, dash='solid'),
                                hovertemplate='%{x}<br>Q MA: %{y:.4f}<extra></extra>',
                            ), row=2, col=1)
                            fig_trend.add_hline(
                                y=1.0,
                                line_dash="dash",
                                line_color="red",
                                line_width=1.2,
                                annotation_text="阈值",
                                annotation_position="top right",
                                row=2, col=1,
                            )

                            fig_trend.update_layout(
                                height=520,
                                xaxis_tickangle=-45,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                margin=dict(l=10, r=10, t=60, b=60),
                            )
                            fig_trend.update_yaxes(title_text="归一化 T²", row=1, col=1)
                            fig_trend.update_yaxes(title_text="归一化 Q", row=2, col=1)
                            fig_trend.update_xaxes(title_text="样品 (时序)", row=2, col=1)

                            st.plotly_chart(fig_trend, use_container_width=True)

                            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                            with col_s1:
                                st.metric("T² 总斜率", f"{t2_total_slope:.4f}")
                            with col_s2:
                                st.metric("T² 近期(3点)斜率", f"{t2_recent_slope:.4f}",
                                          delta="⚠ 上升" if t2_warn else "",
                                          delta_color="inverse")
                            with col_s3:
                                st.metric("Q 总斜率", f"{q_total_slope:.4f}")
                            with col_s4:
                                st.metric("Q 近期(3点)斜率", f"{q_recent_slope:.4f}",
                                          delta="⚠ 上升" if q_warn else "",
                                          delta_color="inverse")

                            st.markdown("---")

                        total_anomaly = len(anomaly_result.anomaly_samples)
                        if total_anomaly > 0:
                            st.warning(
                                f"⚠️ 检测到 {total_anomaly} 个异常样品: "
                                f"{', '.join(anomaly_result.anomaly_samples)}"
                            )
                        else:
                            st.success("✅ 所有样品均为正常")
                    else:
                        st.info("👈 请执行异常检测")
        
        with tab4:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("模型保存与导出")
                
                if st.session_state.pca_result:
                    st.success("✅ 已有PCA分析结果可保存")
                    
                    include_clustering = st.checkbox(
                        "包含聚类结果",
                        value=st.session_state.clustering_result is not None,
                        disabled=st.session_state.clustering_result is None,
                    )
                    
                    if st.button("💾 保存PCA模型", type="primary"):
                        pca_result = st.session_state.pca_result
                        clustering_result = st.session_state.clustering_result if include_clustering else None
                        
                        model_bytes = save_pca_model(
                            pca_result,
                            clustering_result=clustering_result,
                            common_x=pca_result.common_x
                        )
                        
                        st.download_button(
                            "⬇️ 下载PCA模型 (.pkl)",
                            model_bytes,
                            "pca_model.pkl",
                            "application/octet-stream",
                        )
                        st.success("模型已准备好下载")
                    
                    st.markdown("---")
                    st.subheader("导出分析结果")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("📊 导出得分矩阵"):
                            pca_result = st.session_state.pca_result
                            clustering_result = st.session_state.clustering_result
                            anomaly_result = st.session_state.anomaly_result
                            sample_names = [s.name for s in pca_result.selected_spectra]
                            
                            csv_data = export_results_to_csv(
                                pca_result,
                                clustering_result=clustering_result,
                                anomaly_result=anomaly_result,
                                spectrum_names=sample_names
                            )
                            
                            st.download_button(
                                "⬇️ 下载得分矩阵 (CSV)",
                                csv_data.encode('utf-8-sig'),
                                "pca_scores.csv",
                                "text/csv",
                            )
                    
                    with col_b:
                        if st.button("📈 导出载荷矩阵"):
                            pca_result = st.session_state.pca_result
                            csv_data = export_loadings_to_csv(pca_result, pca_result.common_x)
                            
                            st.download_button(
                                "⬇️ 下载载荷矩阵 (CSV)",
                                csv_data.encode('utf-8-sig'),
                                "pca_loadings.csv",
                                "text/csv",
                            )
                    
                    if st.button("📋 导出差值解释"):
                        pca_result = st.session_state.pca_result
                        csv_data = export_variance_to_csv(pca_result)
                        
                        st.download_button(
                            "⬇️ 下载方差解释 (CSV)",
                            csv_data.encode('utf-8-sig'),
                            "pca_variance.csv",
                            "text/csv",
                        )
                else:
                    st.info("请先执行PCA分析")
            
            with col2:
                st.subheader("加载模型与预测")
                
                uploaded_model = st.file_uploader(
                    "上传PCA模型文件 (.pkl)",
                    type=['pkl'],
                )
                
                if uploaded_model is not None:
                    try:
                        model_data = load_pca_model(uploaded_model.getvalue())
                        st.session_state.pca_model_data = model_data
                        st.success("✅ 模型加载成功")
                        
                        st.markdown("#### 模型信息")
                        st.write(f"主成分数: {model_data.get('n_components', 'N/A')}")
                        if 'k_value' in model_data:
                            st.write(f"聚类数: {model_data['k_value']}")
                        if 'common_x' in model_data:
                            st.write(f"x轴范围: {model_data['common_x'].min():.2f} - {model_data['common_x'].max():.2f}")
                    except Exception as e:
                        st.error(f"模型加载失败: {e}")
                
                if st.session_state.pca_model_data:
                    st.markdown("---")
                    st.subheader("预测新样品")
                    
                    uploaded_sample = st.file_uploader(
                        "上传新光谱文件进行预测",
                        type=['csv', 'txt', 'dx', 'jdx', 'jcm'],
                    )
                    
                    if uploaded_sample is not None:
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_sample.name)[1]) as tmp:
                                tmp.write(uploaded_sample.getvalue())
                                tmp_path = tmp.name
                            
                            new_spec = load_spectrum(
                                tmp_path,
                                name=os.path.splitext(uploaded_sample.name)[0],
                            )
                            os.unlink(tmp_path)
                            
                            model_data = st.session_state.pca_model_data
                            prediction = predict_new_sample(new_spec, model_data)
                            
                            st.markdown("#### 预测结果")
                            
                            scores_new = prediction['scores']
                            n_comp = len(scores_new)
                            
                            scores_data = {
                                "主成分": [f'PC{i+1}' for i in range(n_comp)],
                                "得分": [f"{scores_new[i]:.4f}" for i in range(n_comp)],
                            }
                            st.dataframe(pd.DataFrame(scores_data), use_container_width=True, hide_index=True)
                            
                            if 'predicted_cluster' in prediction:
                                st.markdown("#### 分类结果")
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("预测聚类", prediction['predicted_cluster'] + 1)
                                    st.metric("欧氏距离", f"{prediction['euclidean_distance']:.4f}")
                                with col_b:
                                    st.metric("马氏距离", f"{prediction['mahalanobis_distance']:.4f}")
                                    anomaly_status = "⚠️ 异常" if prediction['is_anomaly'] else "✅ 正常"
                                    st.metric("异常判断", anomaly_status)
                                
                                if prediction['is_anomaly']:
                                    st.warning("该样品可能为异常样品，马氏距离超过3倍标准差")
                            
                            fig_pred = go.Figure()
                            fig_pred.add_trace(go.Scatter(
                                x=prediction['common_x'],
                                y=prediction['spectrum_interpolated'],
                                mode='lines',
                                name='插值后光谱',
                                line=dict(color='#1f77b4', width=2),
                            ))
                            fig_pred.update_layout(
                                title=f"{new_spec.name} - 插值后光谱",
                                xaxis_title=new_spec.x_unit,
                                yaxis_title="强度",
                                height=300,
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"预测失败: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    st.info("请先上传PCA模型文件")

elif page == "📐 定量分析":
    st.title("📐 定量分析")
    st.markdown("标准曲线法、内标法")
    
    tab1, tab2 = st.tabs(["标准曲线法", "内标法"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("标准曲线法")
            
            method = st.selectbox(
                "拟合方法",
                ["linear", "quadratic", "cubic"],
                format_func=lambda x: {
                    "linear": "线性",
                    "quadratic": "二次多项式",
                    "cubic": "三次多项式",
                }[x],
                key="std_method",
            )
            
            confidence = st.slider("置信水平", 0.90, 0.99, 0.95, 0.01)
            
            st.markdown("#### 标样数据")
            
            num_standards = st.number_input("标样数量", 3, 20, 5, key="std_num")
            
            concs = []
            responses = []
            for i in range(int(num_standards)):
                col_a, col_b = st.columns(2)
                with col_a:
                    c = st.number_input(f"浓度 #{i+1}", value=10.0 * (i+1), key=f"std_conc_{i}")
                    concs.append(c)
                with col_b:
                    r = st.number_input(f"响应值 #{i+1}", value=1000.0 * (i+1), key=f"std_resp_{i}")
                    responses.append(r)
            
            unknown_response = st.number_input("未知样品响应值", value=3500.0)
            
            if st.button("计算浓度", type="primary", key="std_calc"):
                try:
                    result = standard_curve_method(
                        concs, responses, unknown_response, method, confidence
                    )
                    st.session_state.std_curve_result = result
                    st.success("计算完成")
                except Exception as e:
                    st.error(f"计算失败: {e}")
        
        with col2:
            if 'std_curve_result' in st.session_state:
                result = st.session_state.std_curve_result
                
                st.metric("预测浓度", f"{result.predicted_concentration:.2f} %")
                st.write(f"{int(confidence*100)}% 置信区间: {result.confidence_interval[0]:.2f}% - {result.confidence_interval[1]:.2f}%")
                st.write(f"R² = {result.r_squared:.4f}")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=result.responses,
                    y=result.concentrations,
                    mode='markers',
                    name='标样点',
                    marker=dict(size=10, color='red'),
                ))
                
                x_curve = np.linspace(min(result.responses), max(result.responses), 100)
                y_curve = np.polyval(result.coefficients, x_curve)
                fig.add_trace(go.Scatter(
                    x=x_curve, y=y_curve,
                    mode='lines',
                    name='校准曲线',
                    line=dict(color='blue', width=2),
                ))
                
                fig.add_vline(
                    x=unknown_response,
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"未知样品",
                    annotation_position="top",
                )
                
                fig.add_hline(
                    y=result.predicted_concentration,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"预测浓度: {result.predicted_concentration:.2f}%",
                    annotation_position="right",
                )
                
                fig.update_layout(
                    title="标准曲线",
                    xaxis_title="响应值",
                    yaxis_title="浓度 (%)",
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)
                
                eq_terms = []
                degree = len(result.coefficients) - 1
                for i, coeff in enumerate(result.coefficients[:-1]):
                    power = degree - i
                    if power > 1:
                        eq_terms.append(f"{coeff:.4f}x^{power}")
                    else:
                        eq_terms.append(f"{coeff:.4f}x")
                eq_terms.append(f"{result.coefficients[-1]:.4f}")
                st.info(f"曲线方程: y = {' + '.join(eq_terms)}")
            else:
                st.info("👈 请输入数据并计算")
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("内标法")
            
            method = st.selectbox(
                "拟合方法",
                ["linear", "quadratic", "cubic"],
                format_func=lambda x: {
                    "linear": "线性",
                    "quadratic": "二次多项式",
                    "cubic": "三次多项式",
                }[x],
                key="is_method",
            )
            
            confidence = st.slider("置信水平", 0.90, 0.99, 0.95, 0.01, key="is_conf")
            
            st.markdown("#### 标样数据")
            
            num_standards = st.number_input("标样数量", 3, 20, 5, key="is_num")
            
            concs = []
            analyte_resps = []
            is_resps = []
            for i in range(int(num_standards)):
                c = st.number_input(f"浓度 #{i+1}", value=10.0 * (i+1), key=f"is_conc_{i}")
                concs.append(c)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    a = st.number_input(f"分析物响应 #{i+1}", value=1000.0 * (i+1), key=f"is_analyte_{i}")
                    analyte_resps.append(a)
                with col_b:
                    is_r = st.number_input(f"内标物响应 #{i+1}", value=800.0 + 50*i, key=f"is_std_{i}")
                    is_resps.append(is_r)
            
            st.markdown("#### 未知样品")
            col_a, col_b = st.columns(2)
            with col_a:
                unknown_analyte = st.number_input("未知样品分析物响应", value=3500.0, key="is_unk_analyte")
            with col_b:
                unknown_is = st.number_input("未知样品内标物响应", value=900.0, key="is_unk_is")
            
            if st.button("计算浓度 (内标法)", type="primary", key="is_calc"):
                try:
                    result = internal_standard_method(
                        analyte_resps, is_resps, concs,
                        unknown_analyte, unknown_is,
                        method, confidence,
                    )
                    st.session_state.is_result = result
                    st.success("计算完成")
                except Exception as e:
                    st.error(f"计算失败: {e}")
        
        with col2:
            if 'is_result' in st.session_state:
                result = st.session_state.is_result
                
                st.metric("预测浓度", f"{result.predicted_concentration:.2f} %")
                st.write(f"{int(confidence*100)}% 置信区间: {result.confidence_interval[0]:.2f}% - {result.confidence_interval[1]:.2f}%")
                st.write(f"R² = {result.r_squared:.4f}")
                
                ratios = np.array(analyte_resps) / np.array(is_resps) if num_standards > 0 else np.array([])
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=ratios,
                    y=result.concentrations,
                    mode='markers',
                    name='标样点',
                    marker=dict(size=10, color='red'),
                ))
                
                x_curve = np.linspace(min(ratios) if len(ratios) > 0 else 0, max(ratios) if len(ratios) > 0 else 1, 100)
                y_curve = np.polyval(result.coefficients, x_curve)
                fig.add_trace(go.Scatter(
                    x=x_curve, y=y_curve,
                    mode='lines',
                    name='校准曲线',
                    line=dict(color='blue', width=2),
                ))
                
                fig.update_layout(
                    title="内标法校准曲线 (峰面积比 vs 浓度)",
                    xaxis_title="分析物/内标物 响应比",
                    yaxis_title="浓度 (%)",
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("👈 请输入数据并计算")

elif page == "📄 报告导出":
    st.title("📄 报告导出")
    st.markdown("生成PDF格式分析报告")
    
    col1, col2 = st.columns([1, 1])
    
    deconv_has_result = (
        'deconvolution_result' in st.session_state
        and st.session_state.deconvolution_result is not None
        and st.session_state.deconvolution_result.get('success', False)
        and 'pre_deconvolution_peaks' in st.session_state
        and st.session_state.pre_deconvolution_peaks is not None
    )
    
    with col1:
        st.subheader("报告设置")
        
        report_title = st.text_input("报告标题", "光谱分析报告")
        report_notes = st.text_area("备注", height=100, placeholder="输入报告备注信息...")
        
        st.markdown("#### 包含内容")
        
        include_sample_info = st.checkbox("样品信息", value=True)
        include_spectrum_plot = st.checkbox("光谱图", value=True)
        include_preprocessing = st.checkbox("预处理参数", value=True)
        include_peak_results = st.checkbox("峰检测结果", value=True)
        include_deconvolution = st.checkbox(
            "峰解卷积分析",
            value=deconv_has_result,
            disabled=not deconv_has_result,
            help="只有对样品执行解卷积分析后才能包含该部分",
        )
        include_phase_results = st.checkbox("物相鉴定结果", value=True)
        include_quantitative = st.checkbox("定量结果", value=True)
    
    with col2:
        st.subheader("预览")
        st.info("报告将以PDF格式生成，包含矢量光谱图")
        
        if deconv_has_result:
            st.success("✅ 检测到峰解卷积分析结果，可在报告中包含")
        else:
            st.info("💡 提示：在「峰检测与拟合」页面对重叠峰执行解卷积后，可在此报告中包含解卷积分析结果")
        
        if st.button("📄 生成PDF报告", type="primary"):
            if not st.session_state.spectra:
                st.warning("请先导入数据")
            else:
                current_spec = st.session_state.spectra[st.session_state.current_spectrum_idx]
                
                tmp_path = None
                tmp_deconv_path = None
                
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                        tmp_path = tmp.name
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(current_spec.x, current_spec.y)
                    ax.set_title(current_spec.name)
                    ax.set_xlabel(current_spec.x_unit)
                    ax.set_ylabel('强度')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(tmp_path, dpi=150)
                    plt.close()
                    
                    deconv_result_for_report = None
                    deconv_before_for_report = None
                    
                    if include_deconvolution and deconv_has_result:
                        deconv = st.session_state.deconvolution_result
                        deconv_result_for_report = deconv
                        deconv_before_for_report = st.session_state.pre_deconvolution_peaks
                        
                        x_seg = deconv['sum_curve']['x']
                        mask_seg = (current_spec.x >= deconv['x_range'][0]) & (current_spec.x <= deconv['x_range'][1])
                        y_seg = current_spec.y[mask_seg]
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_d:
                            tmp_deconv_path = tmp_d.name
                        
                        fig_d = plt.figure(figsize=(10, 7))
                        gs = fig_d.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
                        ax_top = fig_d.add_subplot(gs[0])
                        ax_bot = fig_d.add_subplot(gs[1], sharex=ax_top)
                        
                        ax_top.plot(x_seg, y_seg, color='#1f77b4', linewidth=2.0, label='原始数据')
                        
                        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                        for i, comp in enumerate(deconv['component_curves']):
                            ax_top.plot(comp['x'], comp['y'], color=colors[i % len(colors)],
                                        linestyle='--', linewidth=1.2, label=f'分量 {i+1}')
                        
                        ax_top.plot(deconv['sum_curve']['x'], deconv['sum_curve']['y'],
                                    color='green', linewidth=1.8, label='分量之和')
                        ax_top.set_ylabel('强度')
                        ax_top.legend(loc='upper right', fontsize=8, ncol=2)
                        ax_top.grid(True, alpha=0.3)
                        ax_top.set_title(f"峰解卷积结果 (R²={deconv['r_squared']:.4f})")
                        
                        ax_bot.plot(deconv['residual']['x'], deconv['residual']['y'],
                                    color='red', linewidth=1.0)
                        ax_bot.fill_between(deconv['residual']['x'], deconv['residual']['y'],
                                            alpha=0.15, color='red')
                        ax_bot.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
                        ax_bot.set_xlabel(current_spec.x_unit)
                        ax_bot.set_ylabel('残差')
                        ax_bot.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        plt.savefig(tmp_deconv_path, dpi=150)
                        plt.close(fig_d)
                    
                    sample_info = {
                        "样品名称": current_spec.name,
                        "光谱类型": current_spec.spectrum_type.value,
                        "数据点数": current_spec.num_points,
                        "x轴范围": f"{current_spec.x_range[0]:.2f} - {current_spec.x_range[1]:.2f}",
                        "x轴单位": current_spec.x_unit,
                    }
                    
                    preprocessing_params = st.session_state.preprocessing_pipeline.to_dict_list() if st.session_state.preprocessing_pipeline.steps else []
                    
                    peak_results = st.session_state.fitted_peaks or st.session_state.peak_results or []
                    phase_results = st.session_state.phase_results or []
                    quant_results = st.session_state.quant_phase_results if 'quant_phase_results' in st.session_state else []
                    
                    pdf_bytes = generate_pdf_report(
                        sample_info=sample_info,
                        preprocessing_params=preprocessing_params,
                        spectrum_image_path=tmp_path if include_spectrum_plot else None,
                        peak_results=peak_results if include_peak_results else None,
                        phase_results=phase_results if include_phase_results else None,
                        quantitative_results=quant_results if include_quantitative else None,
                        deconvolution_result=deconv_result_for_report if include_deconvolution else None,
                        deconvolution_before_peaks=deconv_before_for_report if include_deconvolution else None,
                        deconvolution_image_path=tmp_deconv_path if (include_deconvolution and deconv_has_result) else None,
                        title=report_title,
                        notes=report_notes,
                    )
                    
                    if pdf_bytes:
                        st.success("✅ 报告生成成功！")
                        st.download_button(
                            "⬇️ 下载PDF报告",
                            pdf_bytes,
                            "光谱分析报告.pdf",
                            "application/pdf",
                        )
                    else:
                        st.error("报告生成失败，请检查reportlab是否安装")
                    
                except Exception as e:
                    st.error(f"生成报告时出错: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    if tmp_deconv_path and os.path.exists(tmp_deconv_path):
                        os.unlink(tmp_deconv_path)

st.sidebar.markdown("---")
st.sidebar.markdown("### 关于")
st.sidebar.info(
    "光谱数据分析与矿物成分鉴定工具\n"
    "版本: 1.0.0\n"
    "支持: XRD / XRF / 拉曼 / 红外"
)
