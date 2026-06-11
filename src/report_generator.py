"""
报告导出模块
生成PDF格式分析报告
"""
from typing import List, Dict, Optional
from datetime import datetime
from io import BytesIO
import numpy as np


def generate_pdf_report(sample_info: Dict,
                        preprocessing_params: List[Dict],
                        spectrum_image_path: Optional[str] = None,
                        peak_results: Optional[List[Dict]] = None,
                        phase_results: Optional[List[Dict]] = None,
                        quantitative_results: Optional[List[Dict]] = None,
                        deconvolution_result: Optional[Dict] = None,
                        deconvolution_before_peaks: Optional[List[Dict]] = None,
                        deconvolution_image_path: Optional[str] = None,
                        title: str = '光谱分析报告',
                        notes: str = '') -> bytes:
    """
    生成PDF分析报告
    
    参数:
        sample_info: 样品信息
        preprocessing_params: 预处理参数
        spectrum_image_path: 光谱图路径
        peak_results: 峰检测结果
        phase_results: 物相鉴定结果
        quantitative_results: 定量结果
        deconvolution_result: 峰解卷积结果（来自deconvolve_peaks）
        deconvolution_before_peaks: 解卷积前的峰参数列表
        deconvolution_image_path: 解卷积可视化图路径
        title: 报告标题
        notes: 备注
    
    返回:
        PDF文件字节数据
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
        )
        from reportlab.lib.units import cm
    except ImportError:
        return b""
    
    buffer = BytesIO()
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=20,
        alignment=1,
        textColor=colors.darkblue,
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=10,
        textColor=colors.darkblue,
        borderWidth=1,
        borderColor=colors.lightgrey,
        borderPadding=5,
    )
    normal_style = styles['Normal']
    
    story = []
    
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("一、样品信息", heading_style))
    sample_data = [['项目', '内容']]
    for key, value in sample_info.items():
        sample_data.append([str(key), str(value)])
    
    sample_table = Table(sample_data, colWidths=[4*cm, 11*cm])
    sample_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkblue),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    story.append(sample_table)
    
    if spectrum_image_path:
        story.append(Spacer(1, 15))
        story.append(Paragraph("二、光谱图", heading_style))
        try:
            img = Image(spectrum_image_path, width=15*cm, height=9*cm)
            story.append(img)
        except:
            story.append(Paragraph("（光谱图加载失败）", normal_style))
    
    if preprocessing_params:
        story.append(Spacer(1, 15))
        story.append(Paragraph("三、预处理参数", heading_style))
        prep_data = [['步骤', '参数']]
        for i, step in enumerate(preprocessing_params):
            step_name = step.get('name', f'步骤{i+1}')
            params_str = ', '.join([f"{k}={v}" for k, v in step.get('params', {}).items()])
            prep_data.append([step_name, params_str])
        
        prep_table = Table(prep_data, colWidths=[4*cm, 11*cm])
        prep_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkgreen),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        story.append(prep_table)
    
    if peak_results:
        story.append(Spacer(1, 15))
        story.append(Paragraph("四、峰检测结果", heading_style))
        peak_data = [['序号', '峰位', '强度', '半高宽(FWHM)']]
        for i, peak in enumerate(peak_results[:30]):
            pos = peak.get('position', peak.get('peak_energy', 0))
            inten = peak.get('intensity', peak.get('peak_intensity', 0))
            fwhm = peak.get('fwhm', peak.get('fwhm_estimate', '-'))
            peak_data.append([
                str(i+1),
                f"{pos:.4f}",
                f"{inten:.2f}",
                f"{fwhm:.4f}" if isinstance(fwhm, (int, float)) else str(fwhm),
            ])
        
        if len(peak_results) > 30:
            peak_data.append(['...', '...', f'共 {len(peak_results)} 个峰', '...'])
        
        peak_table = Table(peak_data, colWidths=[2*cm, 4*cm, 4*cm, 5*cm])
        peak_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightyellow),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkorange),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        story.append(peak_table)
    
    deconvolution_available = (
        deconvolution_result is not None
        and deconvolution_result.get('success', False)
        and deconvolution_before_peaks is not None
    )
    
    if deconvolution_available:
        story.append(Spacer(1, 15))
        story.append(Paragraph("五、峰解卷积分析", heading_style))
        
        deconv_peaks_after = deconvolution_result.get('deconvolved_peaks', [])
        deconv_indices = [p.get('original_index', i) for i, p in enumerate(deconv_peaks_after)]
        
        story.append(Paragraph("5.1 解卷积前后参数对比", styles['Heading3']))
        story.append(Spacer(1, 5))
        
        comp_data = [['分量', '参数', '解卷积前', '解卷积后', '变化量']]
        span_commands = []
        for idx_in_list, after_peak in enumerate(deconv_peaks_after):
            orig_idx = after_peak.get('original_index', idx_in_list)
            before_peak = None
            if orig_idx < len(deconvolution_before_peaks):
                before_peak = deconvolution_before_peaks[orig_idx]
            
            start_row = len(comp_data)
            for param_key, param_label in [
                ('position', '峰位'),
                ('intensity', '强度'),
                ('fwhm', 'FWHM'),
            ]:
                before_val = 0.0
                if before_peak:
                    before_val = before_peak.get(param_key, before_peak.get(
                        'fwhm_estimate' if param_key == 'fwhm' else param_key, 0.0))
                after_val = after_peak.get(param_key, 0.0)
                delta = after_val - before_val
                
                if param_key == 'position':
                    before_str = f"{before_val:.4f}"
                    after_str = f"{after_val:.4f}"
                    delta_str = f"{delta:+.4f}"
                elif param_key == 'intensity':
                    before_str = f"{before_val:.2f}"
                    after_str = f"{after_val:.2f}"
                    delta_str = f"{delta:+.2f}"
                else:
                    before_str = f"{before_val:.4f}" if isinstance(before_val, (int, float)) else "-"
                    after_str = f"{after_val:.4f}"
                    delta_str = f"{delta:+.4f}" if isinstance(before_val, (int, float)) else "-"
                
                comp_data.append([
                    f"分量{idx_in_list + 1}" if param_key == 'position' else "",
                    param_label,
                    before_str,
                    after_str,
                    delta_str,
                ])
            end_row = len(comp_data) - 1
            if end_row > start_row:
                span_commands.append(('SPAN', (0, start_row), (0, end_row)))
        
        table_style_cmds = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightcyan),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkcyan),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BACKGROUND', (0, 1), (0, -1), colors.whitesmoke),
        ]
        table_style_cmds.extend(span_commands)
        
        comp_table = Table(comp_data, colWidths=[2.2*cm, 2.5*cm, 3.2*cm, 3.2*cm, 2.5*cm])
        comp_table.setStyle(TableStyle(table_style_cmds))
        story.append(comp_table)
        
        story.append(Spacer(1, 8))
        r_squared_val = deconvolution_result.get('r_squared', 0.0)
        x_range = deconvolution_result.get('x_range', [0, 0])
        story.append(Paragraph(
            f"解卷积区间: [{x_range[0]:.3f}, {x_range[1]:.3f}]　　"
            f"拟合优度 R² = {r_squared_val:.4f}　　"
            f"分量数: {len(deconv_peaks_after)}",
            normal_style
        ))
        
        if deconvolution_image_path:
            story.append(Spacer(1, 10))
            story.append(Paragraph("5.2 解卷积可视化", styles['Heading3']))
            story.append(Spacer(1, 5))
            try:
                img_deconv = Image(deconvolution_image_path, width=15*cm, height=9*cm)
                story.append(img_deconv)
            except:
                story.append(Paragraph("（解卷积可视化图加载失败）", normal_style))
    
    if phase_results:
        story.append(Spacer(1, 15))
        next_section_num = "六" if deconvolution_available else "五"
        story.append(Paragraph(f"{next_section_num}、物相鉴定结果", heading_style))
        phase_data = [['排名', '矿物名称', '化学式', '匹配度', '匹配峰数']]
        for i, result in enumerate(phase_results[:10]):
            card = result.get('card')
            if card:
                phase_data.append([
                    str(i+1),
                    card.name,
                    card.formula,
                    f"{result.get('match_score', 0)*100:.1f}%",
                    f"{result.get('num_matched', 0)}/{result.get('num_card_peaks', 0)}",
                ])
        
        phase_table = Table(phase_data, colWidths=[1.5*cm, 4*cm, 4*cm, 3*cm, 2.5*cm])
        phase_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lavender),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.purple),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        story.append(phase_table)
    
    if quantitative_results:
        story.append(Spacer(1, 15))
        quant_section_num = "七" if (deconvolution_available and phase_results) else (
            "六" if deconvolution_available or phase_results else "五"
        )
        story.append(Paragraph(f"{quant_section_num}、定量分析结果", heading_style))
        quant_data = [['组分/元素', '含量', '置信区间', '方法']]
        for result in quantitative_results:
            name = result.get('phase', result.get('element', ''))
            content = result.get('weight_percent', result.get('concentration', 0))
            ci = result.get('confidence_interval', (0, 0))
            method = result.get('method', '标准曲线法')
            quant_data.append([
                name,
                f"{content:.2f}%",
                f"{ci[0]:.2f}% - {ci[1]:.2f}%",
                method,
            ])
        
        quant_table = Table(quant_data, colWidths=[4*cm, 3*cm, 4.5*cm, 3.5*cm])
        quant_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightpink),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkred),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        story.append(quant_table)
    
    if notes:
        story.append(Spacer(1, 15))
        notes_section_num = "八" if (deconvolution_available and phase_results and quantitative_results) else (
            "七" if ((deconvolution_available and (phase_results or quantitative_results)) or
                     (phase_results and quantitative_results)) else (
                "六" if (deconvolution_available or phase_results or quantitative_results) else "五"
            )
        )
        story.append(Paragraph(f"{notes_section_num}、备注", heading_style))
        story.append(Paragraph(notes, normal_style))
    
    doc.build(story)
    
    return buffer.getvalue()


def generate_svg_spectrum(x: np.ndarray, y: np.ndarray,
                          title: str = '',
                          x_label: str = '',
                          y_label: str = '',
                          peaks: Optional[List[Dict]] = None,
                          width: int = 800,
                          height: int = 500) -> str:
    """
    生成SVG格式光谱图（矢量图）
    
    参数:
        x: x轴数据
        y: y轴数据
        title: 标题
        x_label: x轴标签
        y_label: y轴标签
        peaks: 峰标注
        width: 宽度
        height: 高度
    
    返回:
        SVG字符串
    """
    margin_left = 60
    margin_right = 20
    margin_top = 50
    margin_bottom = 50
    
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    if y_max == y_min:
        y_max = y_min + 1
    
    def x_to_px(x_val):
        return margin_left + (x_val - x_min) / (x_max - x_min) * plot_width
    
    def y_to_px(y_val):
        return margin_top + plot_height - (y_val - y_min) / (y_max - y_min) * plot_height
    
    svg_parts = []
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    
    svg_parts.append('<rect width="100%" height="100%" fill="white"/>')
    
    svg_parts.append(f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="white" stroke="black" stroke-width="1"/>')
    
    num_x_ticks = 5
    num_y_ticks = 5
    for i in range(num_x_ticks + 1):
        x_val = x_min + i * (x_max - x_min) / num_x_ticks
        x_px = x_to_px(x_val)
        svg_parts.append(f'<line x1="{x_px}" y1="{margin_top}" x2="{x_px}" y2="{margin_top + plot_height}" stroke="#eeeeee" stroke-width="1"/>')
        svg_parts.append(f'<text x="{x_px}" y="{margin_top + plot_height + 15}" text-anchor="middle" font-size="10" fill="#333">{x_val:.1f}</text>')
    
    for i in range(num_y_ticks + 1):
        y_val = y_min + i * (y_max - y_min) / num_y_ticks
        y_px = y_to_px(y_val)
        svg_parts.append(f'<line x1="{margin_left}" y1="{y_px}" x2="{margin_left + plot_width}" y2="{y_px}" stroke="#eeeeee" stroke-width="1"/>')
        svg_parts.append(f'<text x="{margin_left - 5}" y="{y_px + 4}" text-anchor="end" font-size="10" fill="#333">{y_val:.1f}</text>')
    
    path_data = []
    for i in range(len(x)):
        px = x_to_px(x[i])
        py = y_to_px(y[i])
        if i == 0:
            path_data.append(f'M {px:.2f} {py:.2f}')
        else:
            path_data.append(f'L {px:.2f} {py:.2f}')
    
    svg_parts.append(f'<path d="{" ".join(path_data)}" fill="none" stroke="#1f77b4" stroke-width="1.5"/>')
    
    if peaks:
        for peak in peaks[:20]:
            px = peak.get('position', 0)
            py = peak.get('intensity', 0)
            x_px = x_to_px(px)
            y_px = y_to_px(py)
            svg_parts.append(f'<circle cx="{x_px:.2f}" cy="{y_px:.2f}" r="4" fill="red" opacity="0.7"/>')
    
    if title:
        svg_parts.append(f'<text x="{width/2}" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">{title}</text>')
    
    if x_label:
        svg_parts.append(f'<text x="{width/2}" y="{height - 10}" text-anchor="middle" font-size="12" fill="#333">{x_label}</text>')
    
    if y_label:
        svg_parts.append(f'<text x="15" y="{height/2}" text-anchor="middle" font-size="12" fill="#333" transform="rotate(-90, 15, {height/2})">{y_label}</text>')
    
    svg_parts.append('</svg>')
    
    return '\n'.join(svg_parts)
