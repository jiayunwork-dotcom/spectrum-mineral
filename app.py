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
    
    with col1:
        st.subheader("报告设置")
        
        report_title = st.text_input("报告标题", "光谱分析报告")
        report_notes = st.text_area("备注", height=100, placeholder="输入报告备注信息...")
        
        st.markdown("#### 包含内容")
        
        include_sample_info = st.checkbox("样品信息", value=True)
        include_spectrum_plot = st.checkbox("光谱图", value=True)
        include_preprocessing = st.checkbox("预处理参数", value=True)
        include_peak_results = st.checkbox("峰检测结果", value=True)
        include_phase_results = st.checkbox("物相鉴定结果", value=True)
        include_quantitative = st.checkbox("定量结果", value=True)
    
    with col2:
        st.subheader("预览")
        st.info("报告将以PDF格式生成，包含矢量光谱图")
        
        if st.button("📄 生成PDF报告", type="primary"):
            if not st.session_state.spectra:
                st.warning("请先导入数据")
            else:
                current_spec = st.session_state.spectra[st.session_state.current_spectrum_idx]
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    tmp_path = tmp.name
                
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(current_spec.x, current_spec.y)
                    ax.set_title(current_spec.name)
                    ax.set_xlabel(current_spec.x_unit)
                    ax.set_ylabel('强度')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(tmp_path, dpi=150)
                    plt.close()
                    
                    sample_info = {
                        "样品名称": current_spec.name,
                        "光谱类型": current_spec.spectrum_type.value,
                        "数据点数": current_spec.num_points,
                        "x轴范围": f"{current_spec.x_range[0]:.2f} - {current_spec.x_range[1]:.2f}",
                        "x轴单位": current_spec.x_unit,
                    }
                    
                    preprocessing_params = st.session_state.preprocessing_pipeline.to_dict_list() if st.session_state.preprocessing_pipeline.steps else []
                    
                    peak_results = st.session_state.peak_results or []
                    phase_results = st.session_state.phase_results or []
                    quant_results = st.session_state.quant_phase_results if 'quant_phase_results' in st.session_state else []
                    
                    pdf_bytes = generate_pdf_report(
                        sample_info=sample_info,
                        preprocessing_params=preprocessing_params,
                        spectrum_image_path=tmp_path if include_spectrum_plot else None,
                        peak_results=peak_results if include_peak_results else None,
                        phase_results=phase_results if include_phase_results else None,
                        quantitative_results=quant_results if include_quantitative else None,
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
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

st.sidebar.markdown("---")
st.sidebar.markdown("### 关于")
st.sidebar.info(
    "光谱数据分析与矿物成分鉴定工具\n"
    "版本: 1.0.0\n"
    "支持: XRD / XRF / 拉曼 / 红外"
)
