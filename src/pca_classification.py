"""
PCA与光谱分类模块
对多个样品光谱进行降维可视化和自动聚类分类
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.interpolate import CubicSpline
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import pickle
import io

from .spectrum import Spectrum, SpectrumType


class PCAResult:
    """PCA分析结果容器"""
    def __init__(self):
        self.selected_spectra: List[Spectrum] = []
        self.excluded_spectra: List[str] = []
        self.common_x: np.ndarray = None
        self.spectral_matrix: np.ndarray = None
        self.mean: np.ndarray = None
        self.std: np.ndarray = None
        self.eigenvalues: np.ndarray = None
        self.eigenvectors: np.ndarray = None
        self.variance_ratio: np.ndarray = None
        self.cumulative_variance: np.ndarray = None
        self.scores: np.ndarray = None
        self.loadings: np.ndarray = None
        self.n_components: int = 3


class ClusteringResult:
    """聚类分析结果容器"""
    def __init__(self):
        self.labels: np.ndarray = None
        self.centroids_pca: np.ndarray = None
        self.centroids_original: np.ndarray = None
        self.k_value: int = 3
        self.elbow_scores: Dict[int, float] = None
        self.calculated_k: int = None
        self.confusion_matrix: np.ndarray = None
        self.true_labels: List[str] = None
        self.cosine_similarities: Dict[int, float] = None


class AnomalyResult:
    """异常检测结果容器"""
    def __init__(self):
        self.mahalanobis_distances: np.ndarray = None
        self.mahalanobis_threshold: float = None
        self.anomaly_indices_mahal: List[int] = []
        self.hotelling_t2: np.ndarray = None
        self.t2_threshold: float = None
        self.anomaly_indices_t2: List[int] = []
        self.q_residuals: np.ndarray = None
        self.q_threshold: float = None
        self.anomaly_indices_q: List[int] = []
        self.anomaly_samples: List[str] = []


def interpolate_to_common_grid(
    spectra: List[Spectrum],
    spacing: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Spectrum]]:
    """
    将多个光谱插值到统一的x轴网格
    
    参数:
        spectra: 光谱列表
        spacing: 网格间距
    
    返回:
        common_x: 统一的x轴网格
        spectral_matrix: 光谱矩阵 (行=样品, 列=强度值)
        excluded_names: 被排除的样品名称列表
        valid_spectra: 有效光谱列表
    """
    if len(spectra) < 3:
        raise ValueError("至少需要3个光谱用于PCA分析")
    
    x_ranges = [(spec.x.min(), spec.x.max()) for spec in spectra]
    
    x_min = max(r[0] for r in x_ranges)
    x_max = min(r[1] for r in x_ranges)
    
    if x_max <= x_min:
        valid_spectra = []
        excluded_names = [spec.name for spec in spectra]
        return np.array([]), np.array([]), excluded_names, valid_spectra
    
    valid_spectra = []
    excluded_names = []
    
    for spec in spectra:
        if spec.x.min() <= x_min and spec.x.max() >= x_max:
            valid_spectra.append(spec)
        else:
            excluded_names.append(spec.name)
    
    if len(valid_spectra) < 3:
        common_x = np.array([])
        spectral_matrix = np.array([])
        excluded_names = [spec.name for spec in spectra]
        valid_spectra = []
        return common_x, spectral_matrix, excluded_names, valid_spectra
    
    common_x = np.arange(x_min, x_max + spacing / 2, spacing)
    
    n_samples = len(valid_spectra)
    n_points = len(common_x)
    spectral_matrix = np.zeros((n_samples, n_points))
    
    for i, spec in enumerate(valid_spectra):
        cs = CubicSpline(spec.x, spec.y)
        spectral_matrix[i, :] = cs(common_x)
    
    return common_x, spectral_matrix, excluded_names, valid_spectra


def standardize_matrix(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    标准化矩阵 (减均值除标准差, 逐列操作)
    
    参数:
        X: 输入矩阵 (n_samples, n_features)
    
    返回:
        X_std: 标准化后的矩阵
        mean: 每列的均值
        std: 每列的标准差
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    
    std[std < 1e-10] = 1.0
    
    X_std = (X - mean) / std
    
    return X_std, mean, std


def perform_pca(
    X_std: np.ndarray,
    n_components: int = 3
) -> Dict:
    """
    执行PCA分析
    
    参数:
        X_std: 标准化后的光谱矩阵
        n_components: 保留的主成分数
    
    返回:
        包含 eigenvalues, eigenvectors, variance_ratio, cumulative_variance, scores, loadings 的字典
    """
    n_samples, n_features = X_std.shape
    max_components = min(n_components, n_samples - 1, n_features)
    
    cov_matrix = np.cov(X_std, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]
    
    eigenvalues = eigenvalues[:max_components]
    eigenvectors = eigenvectors[:, :max_components]
    
    total_variance = eigenvalues.sum() if eigenvalues.sum() > 0 else 1.0
    variance_ratio = eigenvalues / total_variance
    
    cumulative_variance = np.cumsum(variance_ratio)
    
    scores = X_std @ eigenvectors
    
    loadings = eigenvectors * np.sqrt(eigenvalues)
    
    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'variance_ratio': variance_ratio,
        'cumulative_variance': cumulative_variance,
        'scores': scores,
        'loadings': loadings,
        'n_components': max_components,
    }


def find_loading_peaks(
    x: np.ndarray,
    loading: np.ndarray,
    threshold: float = 0.1
) -> List[Dict]:
    """
    查找载荷绝对值超过阈值的峰位
    
    参数:
        x: x轴坐标
        loading: 载荷向量
        threshold: 阈值
    
    返回:
        峰位列表
    """
    from scipy.signal import find_peaks
    
    loading_abs = np.abs(loading)
    
    peaks_pos, _ = find_peaks(loading_abs, height=threshold)
    
    peak_info = []
    for idx in peaks_pos:
        peak_info.append({
            'x': float(x[idx]),
            'loading': float(loading[idx]),
            'abs_loading': float(loading_abs[idx])
        })
    
    return peak_info


def perform_kmeans(
    scores: np.ndarray,
    k: int = 3,
    random_state: int = 42
) -> Dict:
    """
    执行K-Means聚类
    
    参数:
        scores: PCA得分矩阵
        k: 聚类数
        random_state: 随机种子
    
    返回:
        包含 labels, centroids 的字典
    """
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(scores)
    centroids = kmeans.cluster_centers_
    
    return {
        'labels': labels,
        'centroids_pca': centroids,
        'inertia': kmeans.inertia_,
    }


def elbow_method(
    scores: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42
) -> Dict:
    """
    使用Elbow法自动确定K值
    
    参数:
        scores: PCA得分矩阵
        k_min: 最小K值
        k_max: 最大K值
        random_state: 随机种子
    
    返回:
        包含各K值的SSE和推荐K值的字典
    """
    sse_scores = {}
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(scores)
        sse_scores[k] = kmeans.inertia_
    
    best_k = _find_elbow_point(sse_scores)
    
    return {
        'sse_scores': sse_scores,
        'recommended_k': best_k,
    }


def _find_elbow_point(sse_scores: Dict[int, float]) -> int:
    """
    查找肘部拐点
    
    使用二阶差分法查找SSE曲线的最大曲率点
    """
    k_values = sorted(sse_scores.keys())
    sse_values = [sse_scores[k] for k in k_values]
    
    if len(k_values) < 3:
        return k_values[0]
    
    sse_array = np.array(sse_values)
    first_diff = np.diff(sse_array)
    second_diff = np.diff(first_diff)
    
    max_curvature_idx = np.argmax(np.abs(second_diff))
    best_k = k_values[max_curvature_idx + 1]
    
    return best_k


def reconstruct_centroids(
    centroids_pca: np.ndarray,
    eigenvectors: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray
) -> np.ndarray:
    """
    将聚类质心从PCA空间逆变换回原始光谱空间
    
    参数:
        centroids_pca: PCA空间的质心 (n_clusters, n_components)
        eigenvectors: 特征向量矩阵
        mean: 原始数据均值
        std: 原始数据标准差
    
    返回:
        原始空间的质心谱 (n_clusters, n_features)
    """
    n_clusters = centroids_pca.shape[0]
    n_components = centroids_pca.shape[1]
    
    eigenvectors_subset = eigenvectors[:, :n_components]
    
    centroids_std = centroids_pca @ eigenvectors_subset.T
    
    centroids_original = centroids_std * std + mean
    
    return centroids_original


def calculate_cosine_similarity_centroid(
    centroid_spectrum: np.ndarray,
    cluster_spectra: np.ndarray
) -> float:
    """
    计算质心重建谱与类内平均谱的余弦相似度
    
    参数:
        centroid_spectrum: 质心重建谱
        cluster_spectra: 该类内所有样品的原始光谱矩阵
    
    返回:
        余弦相似度
    """
    mean_spectrum = cluster_spectra.mean(axis=0)
    
    dot_product = np.dot(centroid_spectrum, mean_spectrum)
    norm_centroid = np.linalg.norm(centroid_spectrum)
    norm_mean = np.linalg.norm(mean_spectrum)
    
    if norm_centroid < 1e-10 or norm_mean < 1e-10:
        return 0.0
    
    similarity = dot_product / (norm_centroid * norm_mean)
    
    return float(max(0, min(1, similarity)))


def compute_confusion_matrix(
    true_labels: List[str],
    pred_labels: np.ndarray,
    spectrum_types: List[SpectrumType] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    计算混淆矩阵
    
    参数:
        true_labels: 真实标签列表 (光谱类型)
        pred_labels: 预测聚类标签
        spectrum_types: 光谱类型列表 (用于获取真实标签)
    
    返回:
        混淆矩阵和标签名称列表
    """
    if spectrum_types is not None:
        true_labels = [st.value for st in spectrum_types]
    
    unique_true = sorted(list(set(true_labels)))
    unique_pred = sorted(list(set(pred_labels)))
    
    label_map = {label: i for i, label in enumerate(unique_true)}
    true_numeric = np.array([label_map[label] for label in true_labels])
    
    pred_mapped = np.zeros_like(true_numeric)
    for pred_label in unique_pred:
        mask = pred_labels == pred_label
        if mask.sum() > 0:
            mode_true = stats.mode(true_numeric[mask], keepdims=False)[0]
            pred_mapped[mask] = mode_true
    
    cm = confusion_matrix(true_numeric, pred_mapped, labels=range(len(unique_true)))
    
    return cm, unique_true


def compute_mahalanobis_distances(
    scores: np.ndarray,
    labels: np.ndarray,
    centroids_pca: np.ndarray
) -> Tuple[np.ndarray, float, List[int]]:
    """
    计算每个样品到其所属聚类质心的马氏距离
    
    参数:
        scores: PCA得分矩阵
        labels: 聚类标签
        centroids_pca: 聚类质心
    
    返回:
        马氏距离数组, 阈值, 异常索引列表
    """
    n_samples = scores.shape[0]
    mahal_distances = np.zeros(n_samples)
    
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        cluster_scores = scores[mask]
        centroid = centroids_pca[cluster_id]
        
        if len(cluster_scores) > 2:
            cov = np.cov(cluster_scores, rowvar=False)
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(cov)
            
            for i in np.where(mask)[0]:
                diff = scores[i] - centroid
                mahal_distances[i] = np.sqrt(diff @ inv_cov @ diff)
        else:
            for i in np.where(mask)[0]:
                diff = scores[i] - centroid
                mahal_distances[i] = np.linalg.norm(diff)
    
    threshold = mahal_distances.mean() + 3 * mahal_distances.std()
    
    anomaly_indices = np.where(mahal_distances > threshold)[0].tolist()
    
    return mahal_distances, threshold, anomaly_indices


def compute_hotelling_t2(
    scores: np.ndarray,
    eigenvalues: np.ndarray,
    n_components: int = None
) -> Tuple[np.ndarray, float]:
    """
    计算Hotelling T2统计量
    
    参数:
        scores: PCA得分矩阵
        eigenvalues: 特征值
        n_components: 使用的主成分数
    
    返回:
        T2统计量数组, 95%置信限阈值
    """
    if n_components is None:
        n_components = scores.shape[1]
    
    eigenvalues_subset = eigenvalues[:n_components]
    eigenvalues_subset[eigenvalues_subset < 1e-10] = 1e-10
    
    T2 = np.sum((scores[:, :n_components] ** 2) / eigenvalues_subset, axis=1)
    
    n_samples = scores.shape[0]
    alpha = 0.05
    
    if n_samples > n_components:
        F_crit = stats.f.ppf(1 - alpha, n_components, n_samples - n_components)
        threshold = (n_components * (n_samples - 1) / (n_samples - n_components)) * F_crit
    else:
        chi2_crit = stats.chi2.ppf(1 - alpha, n_components)
        threshold = chi2_crit
    
    return T2, threshold


def compute_q_residuals(
    X_std: np.ndarray,
    scores: np.ndarray,
    eigenvectors: np.ndarray,
    n_components: int = None
) -> Tuple[np.ndarray, float]:
    """
    计算Q残差(SPE - Squared Prediction Error)
    
    参数:
        X_std: 标准化后的光谱矩阵
        scores: PCA得分矩阵
        eigenvectors: 特征向量矩阵
        n_components: 使用的主成分数
    
    返回:
        Q残差数组, 95%置信限阈值
    """
    if n_components is None:
        n_components = scores.shape[1]
    
    eigenvectors_subset = eigenvectors[:, :n_components]
    
    X_reconstructed = scores[:, :n_components] @ eigenvectors_subset.T
    
    residuals = X_std - X_reconstructed
    Q = np.sum(residuals ** 2, axis=1)
    
    eigenvalues_all = np.linalg.eigvalsh(np.cov(X_std, rowvar=False))
    eigenvalues_all = np.sort(eigenvalues_all)[::-1]
    
    theta1 = np.sum(eigenvalues_all[n_components:])
    theta2 = np.sum(eigenvalues_all[n_components:] ** 2)
    theta3 = np.sum(eigenvalues_all[n_components:] ** 3)
    
    if theta2 > 1e-10 and theta3 > 1e-10:
        h0 = 1 - (2 * theta1 * theta3) / (3 * theta2 ** 2)
        z_alpha = stats.norm.ppf(0.95)
        
        term1 = z_alpha * np.sqrt(2 * theta2 * h0 ** 2)
        term2 = theta2 * h0 * (h0 - 1) / (theta1 ** 2)
        threshold = theta1 * (1 + term1 / theta1 + term2) ** (1 / h0)
    else:
        threshold = np.percentile(Q, 95)
    
    return Q, threshold


def save_pca_model(
    pca_result: PCAResult,
    clustering_result: ClusteringResult = None,
    common_x: np.ndarray = None
) -> bytes:
    """
    保存PCA模型为pickle格式
    
    参数:
        pca_result: PCA分析结果
        clustering_result: 聚类结果 (可选)
        common_x: 统一的x轴网格
    
    返回:
        pickle序列化后的字节
    """
    model_data = {
        'mean': pca_result.mean,
        'std': pca_result.std,
        'eigenvalues': pca_result.eigenvalues,
        'eigenvectors': pca_result.eigenvectors,
        'loadings': pca_result.loadings,
        'variance_ratio': pca_result.variance_ratio,
        'n_components': pca_result.n_components,
        'common_x': common_x,
    }
    
    if clustering_result is not None:
        model_data.update({
            'clustering_labels': clustering_result.labels,
            'centroids_pca': clustering_result.centroids_pca,
            'centroids_original': clustering_result.centroids_original,
            'k_value': clustering_result.k_value,
        })
    
    buffer = io.BytesIO()
    pickle.dump(model_data, buffer)
    return buffer.getvalue()


def load_pca_model(model_bytes: bytes) -> Dict:
    """
    加载PCA模型
    
    参数:
        model_bytes: pickle序列化的模型数据
    
    返回:
        模型数据字典
    """
    buffer = io.BytesIO(model_bytes)
    model_data = pickle.load(buffer)
    return model_data


def predict_new_sample(
    new_spectrum: Spectrum,
    model_data: Dict,
    spacing: float = 0.5
) -> Dict:
    """
    使用已保存的PCA模型预测新样品
    
    参数:
        new_spectrum: 新的光谱样品
        model_data: 加载的PCA模型数据
        spacing: 网格间距 (需与训练时一致)
    
    返回:
        预测结果字典
    """
    common_x = model_data.get('common_x')
    
    if common_x is None:
        raise ValueError("模型中缺少common_x数据，无法进行预测")
    
    cs = CubicSpline(new_spectrum.x, new_spectrum.y)
    new_y = cs(common_x)
    
    mean = model_data['mean']
    std = model_data['std']
    std[std < 1e-10] = 1.0
    
    new_y_std = (new_y - mean) / std
    
    eigenvectors = model_data['eigenvectors']
    n_components = model_data['n_components']
    
    scores_new = new_y_std @ eigenvectors[:, :n_components]
    
    result = {
        'scores': scores_new,
        'spectrum_interpolated': new_y,
        'common_x': common_x,
    }
    
    if 'centroids_pca' in model_data:
        centroids_pca = model_data['centroids_pca']
        
        distances = np.linalg.norm(centroids_pca - scores_new, axis=1)
        predicted_cluster = int(np.argmin(distances))
        min_distance = float(distances[predicted_cluster])
        
        if len(centroids_pca) > 1:
            cov = np.cov(centroids_pca, rowvar=False)
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(cov)
            
            diff = scores_new - centroids_pca[predicted_cluster]
            mahal_distance = float(np.sqrt(diff @ inv_cov @ diff))
        else:
            mahal_distance = min_distance
        
        result.update({
            'predicted_cluster': predicted_cluster,
            'euclidean_distance': min_distance,
            'mahalanobis_distance': mahal_distance,
            'is_anomaly': mahal_distance > 3.0,
        })
    
    return result


def export_results_to_csv(
    pca_result: PCAResult,
    clustering_result: ClusteringResult = None,
    anomaly_result: AnomalyResult = None,
    spectrum_names: List[str] = None
) -> str:
    """
    将PCA分析结果导出为CSV格式
    
    参数:
        pca_result: PCA分析结果
        clustering_result: 聚类结果 (可选)
        anomaly_result: 异常检测结果 (可选)
        spectrum_names: 光谱名称列表
    
    返回:
        CSV格式字符串
    """
    if spectrum_names is None:
        spectrum_names = [f"Sample_{i+1}" for i in range(pca_result.scores.shape[0])]
    
    data = {}
    data['Sample'] = spectrum_names
    
    n_components = pca_result.scores.shape[1]
    for i in range(n_components):
        data[f'PC{i+1}_Score'] = pca_result.scores[:, i]
    
    if clustering_result is not None:
        data['Cluster'] = clustering_result.labels
    
    if anomaly_result is not None:
        data['Mahalanobis_Distance'] = anomaly_result.mahalanobis_distances
        data['Hotelling_T2'] = anomaly_result.hotelling_t2
        data['Q_Residual'] = anomaly_result.q_residuals
        data['Is_Anomaly'] = [i in anomaly_result.anomaly_samples for i in range(len(spectrum_names))]
    
    df = pd.DataFrame(data)
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    
    return csv_buffer.getvalue()


def export_loadings_to_csv(
    pca_result: PCAResult,
    common_x: np.ndarray
) -> str:
    """
    导出载荷矩阵为CSV
    
    参数:
        pca_result: PCA分析结果
        common_x: 统一的x轴网格
    
    返回:
        CSV格式字符串
    """
    data = {'x': common_x}
    
    n_components = pca_result.loadings.shape[1]
    for i in range(n_components):
        data[f'PC{i+1}_Loading'] = pca_result.loadings[:, i]
    
    df = pd.DataFrame(data)
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    
    return csv_buffer.getvalue()


def export_variance_to_csv(
    pca_result: PCAResult
) -> str:
    """
    导出差值解释比例为CSV
    
    参数:
        pca_result: PCA分析结果
    
    返回:
        CSV格式字符串
    """
    n_components = len(pca_result.variance_ratio)
    
    data = {
        'PC': [f'PC{i+1}' for i in range(n_components)],
        'Eigenvalue': pca_result.eigenvalues,
        'Variance_Ratio': pca_result.variance_ratio,
        'Cumulative_Variance': pca_result.cumulative_variance,
    }
    
    df = pd.DataFrame(data)
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    
    return csv_buffer.getvalue()
