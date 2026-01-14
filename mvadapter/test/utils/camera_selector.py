"""
相机筛选工具：基于空间多样性和投影质量评分的智能相机选择
采用方案1和3：
- 方案1：相机空间多样性选择（位置距离和角度差异）
- 方案3：投影质量评分（可选）
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class CameraSelectorConfig:
    """相机选择器配置参数"""
    
    # 选择策略参数
    selection_strategy: str = "greedy_max_min"  # 选择策略: "greedy_max_min", "clustering", "uniform"
    target_camera_ratio: float = 0.6  # 目标保留相机比例 (0.0-1.0)
    min_cameras: int = 3  # 最少保留相机数
    max_cameras: Optional[int] = None  # 最多保留相机数
    
    # 距离和角度权重
    position_weight: float = 1.0  # 位置距离权重
    angle_weight: float = 1.0  # 角度差异权重
    
    # 距离约束
    min_position_distance: float = 0.1  # 相机之间最小位置距离（相对于场景范围）
    min_angle_distance: float = 5.0  # 相机之间最小角度差异（度）
    
    # 质量评分参数
    use_quality_score: bool = False  # 是否使用投影质量评分
    quality_score_weight: float = 0.2  # 质量评分的权重 (0.0-1.0)
    
    # 其他参数
    random_seed: Optional[int] = None  # 随机种子
    verbose: bool = True  # 是否打印调试信息


class CameraSelector:
    """相机筛选工具类"""
    
    def __init__(self, config: Optional[CameraSelectorConfig] = None):
        """
        初始化相机选择器
        
        Args:
            config: 配置对象，若为None则使用默认配置
        """
        self.config = config or CameraSelectorConfig()
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            torch.manual_seed(self.config.random_seed)
    
    def select_cameras(
        self,
        cameras,
        quality_scores: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        筛选相机帧
        
        Args:
            cameras: Camera对象 (包含c2w矩阵) 或相机矩阵 [N, 4, 4]
                    包含位置和方向信息
            quality_scores: 可选，相机质量评分 [N]，范围[0, 1]，值越大越好
        
        Returns:
            筛选后的相机序号列表 [M]，其中 M <= N
        """
        num_cameras = self._get_num_cameras(cameras)
        
        if num_cameras <= self.config.min_cameras:
            if self.config.verbose:
                print(f"相机数量({num_cameras}) <= 最少保留数({self.config.min_cameras}), 保留所有相机")
            return np.arange(num_cameras)
        
        # 提取相机位置和朝向
        cam_positions, cam_forwards = self._extract_camera_geometry(cameras)
        
        # 计算距离矩阵
        position_distances = self._compute_position_distances(cam_positions)
        angle_distances = self._compute_angle_distances(cam_forwards)
        
        # 组合距离矩阵
        combined_distances = (
            self.config.position_weight * position_distances +
            self.config.angle_weight * angle_distances
        )
        
        # 选择相机
        if self.config.selection_strategy == "greedy_max_min":
            selected_indices = self._greedy_max_min_selection(
                combined_distances, num_cameras, quality_scores
            )
        elif self.config.selection_strategy == "clustering":
            selected_indices = self._clustering_selection(
                combined_distances, num_cameras, quality_scores
            )
        elif self.config.selection_strategy == "uniform":
            selected_indices = self._uniform_selection(num_cameras, quality_scores)
        else:
            raise ValueError(f"未知的选择策略: {self.config.selection_strategy}")
        
        selected_indices = np.sort(selected_indices)
        
        if self.config.verbose:
            print(f"相机筛选: {num_cameras} -> {len(selected_indices)} "
                  f"({len(selected_indices)/num_cameras*100:.1f}%)")
            print(f"选择的相机序号: {selected_indices.tolist()}")
        
        return selected_indices
    
    def _get_num_cameras(self, cameras) -> int:
        """获取相机数量"""
        try:
            # Camera对象有__len__方法
            return len(cameras)
        except:
            # 尝试使用shape
            return cameras.shape[0]
    
    def _extract_camera_geometry(
        self, cameras
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从Camera对象或相机矩阵中提取位置和朝向
        
        Args:
            cameras: Camera对象或相机矩阵
        
        Returns:
            positions: [N, 3] 相机位置
            forwards: [N, 3] 相机朝向（单位向量）
        """
        num_cameras = self._get_num_cameras(cameras)
        
        # 检查是否为Camera对象
        if hasattr(cameras, 'c2w') and hasattr(cameras, 'w2c'):
            # 这是一个Camera对象
            if cameras.c2w is not None:
                c2w = cameras.c2w
            else:
                # 如果c2w不存在，从w2c计算
                c2w = torch.linalg.inv(cameras.w2c)
            
            # 转换为numpy
            if isinstance(c2w, torch.Tensor):
                c2w_np = c2w.cpu().numpy()
            else:
                c2w_np = c2w
            
            positions = c2w_np[:, :3, 3]
            forwards = -c2w_np[:, :3, 2]  # 相机看向-Z方向
        
        elif isinstance(cameras, torch.Tensor) and cameras.shape[-2:] == (4, 4):
            # 相机矩阵 [N, 4, 4]
            positions = cameras[:, :3, 3].cpu().numpy()
            forwards = -cameras[:, :3, 2].cpu().numpy()
        
        elif isinstance(cameras, np.ndarray) and cameras.shape[-2:] == (4, 4):
            # NumPy相机矩阵
            positions = cameras[:, :3, 3]
            forwards = -cameras[:, :3, 2]
        
        else:
            raise RuntimeError(
                f"未知的相机输入类型: {type(cameras)}。"
                "期望Camera对象或相机矩阵 [N, 4, 4]"
            )
        
        # 归一化朝向
        forwards = forwards / (np.linalg.norm(forwards, axis=-1, keepdims=True) + 1e-8)
        
        return positions, forwards
    
    def _compute_position_distances(self, positions: np.ndarray) -> np.ndarray:
        """
        计算相机位置之间的距离矩阵（欧氏距离，归一化）
        
        Args:
            positions: [N, 3]
        
        Returns:
            distances: [N, N]，范围[0, 1]
        """
        # 计算欧氏距离
        diff = positions[:, None, :] - positions[None, :, :]  # [N, N, 3]
        distances = np.linalg.norm(diff, axis=-1)  # [N, N]
        
        # 归一化到[0, 1]
        max_dist = np.max(distances)
        if max_dist > 1e-8:
            distances = distances / max_dist
        
        return distances
    
    def _compute_angle_distances(self, forwards: np.ndarray) -> np.ndarray:
        """
        计算相机朝向之间的角度差异（度数，归一化）
        
        Args:
            forwards: [N, 3]
        
        Returns:
            angles: [N, N]，范围[0, 1]（角度/180度）
        """
        # 计算点积（夹角余弦值）
        dots = np.dot(forwards, forwards.T)  # [N, N]
        dots = np.clip(dots, -1.0, 1.0)
        
        # 转换为角度（弧度）
        angles_rad = np.arccos(dots)
        
        # 转换为度数并归一化到[0, 1]
        angles_deg = np.degrees(angles_rad) / 180.0
        
        return angles_deg
    
    def _greedy_max_min_selection(
        self,
        distances: np.ndarray,
        num_cameras: int,
        quality_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        贪心最大最小距离选择策略
        
        逐步选择距离已选择相机集合最远的相机，确保多样性
        
        Args:
            distances: [N, N]
            num_cameras: N
            quality_scores: [N] 可选的质量评分
        
        Returns:
            selected_indices: [M]
        """
        target_num = max(
            self.config.min_cameras,
            min(
                self.config.max_cameras or num_cameras,
                int(np.ceil(num_cameras * self.config.target_camera_ratio))
            )
        )
        
        selected = []
        remaining = set(range(num_cameras))
        
        # 第一步：选择质量最好的相机（如果提供了质量评分）
        if quality_scores is not None and self.config.use_quality_score:
            first_idx = int(np.argmax(quality_scores))
        else:
            # 否则选择到其他相机距离最远的相机
            first_idx = int(np.argmax(distances.sum(axis=1)))
        
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # 贪心选择
        while len(selected) < target_num and remaining:
            # 计算剩余相机到已选相机的最小距离
            min_distances_to_selected = distances[
                np.array(list(remaining)), :
            ][:, selected].min(axis=1)
            
            # 如果使用质量评分，组合距离和质量
            if quality_scores is not None and self.config.use_quality_score:
                remaining_indices = np.array(list(remaining))
                scores = (
                    (1 - self.config.quality_score_weight) * min_distances_to_selected +
                    self.config.quality_score_weight * quality_scores[remaining_indices]
                )
                next_idx_local = int(np.argmax(scores))
            else:
                next_idx_local = int(np.argmax(min_distances_to_selected))
            
            next_idx = list(remaining)[next_idx_local]
            selected.append(next_idx)
            remaining.remove(next_idx)
        
        return np.array(selected)
    
    def _clustering_selection(
        self,
        distances: np.ndarray,
        num_cameras: int,
        quality_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        基于聚类的选择策略
        
        使用层次聚类，然后从每个簇中选择最优相机
        
        Args:
            distances: [N, N]
            num_cameras: N
            quality_scores: [N] 可选的质量评分
        
        Returns:
            selected_indices: [M]
        """
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
        except ImportError:
            if self.config.verbose:
                print("scipy不可用，降级到greedy_max_min策略")
            return self._greedy_max_min_selection(distances, num_cameras, quality_scores)
        
        target_num = max(
            self.config.min_cameras,
            min(
                self.config.max_cameras or num_cameras,
                int(np.ceil(num_cameras * self.config.target_camera_ratio))
            )
        )
        
        # 将距离矩阵转换为凝聚距离矩阵
        condensed_dist = squareform(distances, checks=False)
        
        # 层次聚类
        Z = linkage(condensed_dist, method='average')
        
        # 根据目标数量切割树
        clusters = fcluster(Z, target_num, criterion='maxclust')
        
        # 从每个簇中选择最优相机
        selected = []
        for cluster_id in np.unique(clusters):
            cluster_members = np.where(clusters == cluster_id)[0]
            
            if quality_scores is not None and self.config.use_quality_score:
                # 选择质量最好的
                best_idx = cluster_members[np.argmax(quality_scores[cluster_members])]
            else:
                # 选择到其他相机距离最远的
                best_idx = cluster_members[
                    np.argmax(distances[cluster_members, :].sum(axis=1))
                ]
            
            selected.append(best_idx)
        
        return np.array(selected)
    
    def _uniform_selection(
        self,
        num_cameras: int,
        quality_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        均匀选择策略（按时间顺序均匀间隔）
        
        Args:
            num_cameras: N
            quality_scores: [N] 可选的质量评分
        
        Returns:
            selected_indices: [M]
        """
        target_num = max(
            self.config.min_cameras,
            min(
                self.config.max_cameras or num_cameras,
                int(np.ceil(num_cameras * self.config.target_camera_ratio))
            )
        )
        
        if quality_scores is not None and self.config.use_quality_score:
            # 优先选择高质量的相机
            sorted_indices = np.argsort(-quality_scores)
            selected = sorted(sorted_indices[:target_num])
        else:
            # 均匀间隔选择
            selected = np.linspace(0, num_cameras - 1, target_num, dtype=int)
        
        return np.array(selected)
    
    def compute_coverage_stats(
        self,
        cameras: torch.Tensor,
        selected_indices: np.ndarray,
    ) -> Dict[str, Any]:
        """
        计算选择后的覆盖统计信息
        
        Args:
            cameras: 相机对象或矩阵
            selected_indices: 选择的相机序号
        
        Returns:
            统计信息字典
        """
        cam_positions, cam_forwards = self._extract_camera_geometry(cameras)
        selected_positions = cam_positions[selected_indices]
        
        # 计算相机间距离统计
        distances = self._compute_position_distances(cam_positions)
        selected_distances = distances[selected_indices, :][:, selected_indices]
        
        # 移除对角线
        selected_distances_no_diag = selected_distances[
            ~np.eye(selected_distances.shape[0], dtype=bool)
        ]
        
        stats = {
            "total_cameras": len(cam_positions),
            "selected_cameras": len(selected_indices),
            "selection_ratio": len(selected_indices) / len(cam_positions),
            "avg_distance_between_selected": float(
                np.mean(selected_distances_no_diag) if len(selected_distances_no_diag) > 0 else 0
            ),
            "min_distance_between_selected": float(np.min(selected_distances_no_diag)) if len(selected_distances_no_diag) > 0 else 0,
            "max_distance_between_selected": float(np.max(selected_distances_no_diag)) if len(selected_distances_no_diag) > 0 else 0,
            "scene_extent": float(np.max(distances)),
        }
        
        return stats


# 使用示例
if __name__ == "__main__":
    # 创建模拟相机数据
    num_cameras = 30
    
    # 方式1：使用相机矩阵 [N, 4, 4]
    c2w = torch.randn(num_cameras, 4, 4)
    c2w[:, 3, :] = torch.tensor([0, 0, 0, 1])  # 设置齐次坐标
    c2w[:, :3, 3] = torch.linspace(0, 10, num_cameras).unsqueeze(1) * torch.randn(num_cameras, 3)
    
    # 方式2：质量评分
    quality_scores = torch.rand(num_cameras)
    
    # 创建选择器
    config = CameraSelectorConfig(
        selection_strategy="greedy_max_min",
        target_camera_ratio=0.5,
        position_weight=1.0,
        angle_weight=1.0,
        use_quality_score=True,
        quality_score_weight=0.2,
        verbose=True,
    )
    
    selector = CameraSelector(config)
    
    # 执行选择
    selected_indices = selector.select_cameras(c2w, quality_scores)
    
    # 计算统计信息
    stats = selector.compute_coverage_stats(c2w, selected_indices)
    print("\n覆盖统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
