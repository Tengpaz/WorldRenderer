"""
相机筛选集成测试脚本
演示从camera.json加载相机 -> 构建Camera对象 -> 筛选相机 -> 输出结果
"""

import sys
import torch
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.camera import load_camera_from_json
from utils.camera_selector import CameraSelector, CameraSelectorConfig


def test_camera_selection_from_json():
    """
    完整的集成测试：
    1. 从camera.json加载相机
    2. 使用CameraSelector筛选
    3. 输出选择的相机索引
    """
    print("=" * 70)
    print("相机筛选集成测试 - 从JSON文件加载相机")
    print("=" * 70)
    
    # 路径配置
    test_dir = Path(__file__).resolve().parent
    camera_json_path = test_dir / "camera_path.json"
    
    if not camera_json_path.exists():
        print(f"✗ 错误：相机JSON文件不存在: {camera_json_path}")
        return
    
    print(f"\n✓ 相机JSON文件: {camera_json_path}")
    print(f"✓ 文件大小: {camera_json_path.stat().st_size / 1024:.1f} KB")
    
    # 设置渲染参数
    height = 512
    width = 512
    device = "cpu"  # 使用CPU避免CUDA依赖
    max_views = 0  # 0表示加载所有相机
    axis_convert = False
    
    print(f"\n加载相机参数:")
    print(f"  - 分辨率: {width}x{height}")
    print(f"  - 设备: {device}")
    print(f"  - 最大相机数: {'无限制' if max_views == 0 else max_views}")
    
    # 第一步：从JSON加载相机
    print(f"\n[步骤1] 从JSON加载相机...")
    try:
        cameras, clip_near, clip_far = load_camera_from_json(
            camera_json_path,
            height,
            width,
            device,
            max_views,
            axis_convert
        )
        print(f"✓ 成功加载！")
        print(f"  - 相机数量: {len(cameras)}")
        print(f"  - 近裁剪面: {clip_near:.4f}")
        print(f"  - 远裁剪面: {clip_far:.4f}")
        print(f"  - Camera类型: {type(cameras).__name__}")
        
        # 打印Camera对象的属性
        if hasattr(cameras, 'c2w'):
            print(f"  - c2w矩阵形状: {cameras.c2w.shape}")
        if hasattr(cameras, 'w2c'):
            print(f"  - w2c矩阵形状: {cameras.w2c.shape}")
        
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 第二步：使用CameraSelector筛选相机
    print(f"\n[步骤2] 使用CameraSelector筛选相机...")
    
    # 创建多个配置进行测试
    configs = [
        CameraSelectorConfig(
            selection_strategy="greedy_max_min",
            target_camera_ratio=0.5,  # 保留50%
            position_weight=1.0,
            angle_weight=1.0,
            use_quality_score=False,
            verbose=False,
        ),
        CameraSelectorConfig(
            selection_strategy="greedy_max_min",
            target_camera_ratio=0.6,  # 保留60%
            position_weight=1.0,
            angle_weight=0.8,
            use_quality_score=False,
            verbose=False,
        ),
        CameraSelectorConfig(
            selection_strategy="uniform",
            target_camera_ratio=0.4,  # 保留40%
            verbose=False,
        ),
    ]
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n  [配置{i}] {config.selection_strategy}")
        print(f"    - 目标比例: {config.target_camera_ratio*100:.0f}%")
        print(f"    - 位置权重: {config.position_weight}")
        print(f"    - 角度权重: {config.angle_weight}")
        
        # 执行筛选
        selector = CameraSelector(config)
        selected_indices = selector.select_cameras(cameras)
        
        # 获取统计信息
        stats = selector.compute_coverage_stats(cameras, selected_indices)
        
        print(f"    ✓ 筛选完成!")
        print(f"      - 选择相机数: {len(selected_indices)}")
        print(f"      - 实际保留比例: {stats['selection_ratio']*100:.1f}%")
        print(f"      - 相机平均间距: {stats['avg_distance_between_selected']:.4f}")
        print(f"      - 最小相机间距: {stats['min_distance_between_selected']:.4f}")
        print(f"      - 最大相机间距: {stats['max_distance_between_selected']:.4f}")
        
        results.append({
            'config': config,
            'selected_indices': selected_indices,
            'stats': stats,
        })
    
    # 第三步：输出详细结果
    print(f"\n[步骤3] 详细的筛选结果")
    print("-" * 70)
    
    for i, result in enumerate(results, 1):
        config = result['config']
        selected_indices = result['selected_indices']
        stats = result['stats']
        
        print(f"\n配置{i}: {config.selection_strategy} (目标:{config.target_camera_ratio*100:.0f}%)")
        print(f"{'='*70}")
        
        print(f"相机统计:")
        print(f"  - 总相机数: {stats['total_cameras']}")
        print(f"  - 选择相机数: {stats['selected_cameras']}")
        print(f"  - 实际选择比例: {stats['selection_ratio']*100:.2f}%")
        print(f"  - 场景范围: {stats['scene_extent']:.4f}")
        
        print(f"\n相机间距离统计:")
        print(f"  - 平均距离: {stats['avg_distance_between_selected']:.4f}")
        print(f"  - 最小距离: {stats['min_distance_between_selected']:.4f}")
        print(f"  - 最大距离: {stats['max_distance_between_selected']:.4f}")
        
        print(f"\n选择的相机序号 (共{len(selected_indices)}个):")
        # 分组显示，每行显示10个
        indices_str = ", ".join(map(str, selected_indices.tolist()))
        for j in range(0, len(indices_str), 100):
            print(f"  {indices_str[j:j+100]}")
        
        # 输出为Python数组格式
        print(f"\nPython数组格式:")
        print(f"  selected_indices = np.array({selected_indices.tolist()})")
    
    # 第四步：交互式选择
    print(f"\n[步骤4] 交互式相机选择示例")
    print("-" * 70)
    
    config = CameraSelectorConfig(
        selection_strategy="greedy_max_min",
        target_camera_ratio=0.6,
        position_weight=1.0,
        angle_weight=1.0,
        use_quality_score=False,
        verbose=True,  # 启用详细输出
    )
    
    print(f"\n执行筛选 (比例: 60%)...")
    selector = CameraSelector(config)
    selected_indices = selector.select_cameras(cameras)
    
    print(f"\n筛选结果:")
    print(f"  - 选择的相机序号: {selected_indices.tolist()}")
    print(f"  - 选择的相机数量: {len(selected_indices)}")
    
    # 获取筛选后的相机对象
    print(f"\n获取筛选后的Camera对象...")
    try:
        filtered_cameras = cameras[selected_indices.tolist()]
        print(f"✓ 成功！")
        print(f"  - 筛选后的Camera对象: {type(filtered_cameras).__name__}")
        print(f"  - 筛选后的相机数: {len(filtered_cameras)}")
        if hasattr(filtered_cameras, 'c2w'):
            print(f"  - c2w矩阵形状: {filtered_cameras.c2w.shape}")
    except Exception as e:
        print(f"✗ 获取失败: {e}")
    
    print(f"\n[完成] 相机筛选集成测试完成！")
    print("=" * 70)
    
    return {
        'cameras': cameras,
        'results': results,
        'selected_indices': selected_indices,
    }


def test_camera_selection_custom_ratios():
    """
    测试不同的相机筛选比例
    """
    print("\n" + "=" * 70)
    print("测试不同的相机筛选比例")
    print("=" * 70)
    
    test_dir = Path(__file__).resolve().parent
    camera_json_path = test_dir / "camera_path.json"
    
    if not camera_json_path.exists():
        print(f"✗ 相机JSON文件不存在: {camera_json_path}")
        return
    
    # 加载相机
    cameras, _, _ = load_camera_from_json(
        camera_json_path, 512, 512, "cpu", 0, False
    )
    
    print(f"\n总相机数: {len(cameras)}")
    
    # 测试不同的比例
    ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    print(f"\n{'比例':<8} {'选择数':<10} {'实际比例':<12} {'平均距离':<15} {'最小距离':<12}")
    print("-" * 70)
    
    for ratio in ratios:
        config = CameraSelectorConfig(
            selection_strategy="greedy_max_min",
            target_camera_ratio=ratio,
            verbose=False,
        )
        
        selector = CameraSelector(config)
        selected_indices = selector.select_cameras(cameras)
        stats = selector.compute_coverage_stats(cameras, selected_indices)
        
        print(f"{ratio*100:.0f}%     {len(selected_indices):<10} "
              f"{stats['selection_ratio']*100:>6.1f}%{'':<5} "
              f"{stats['avg_distance_between_selected']:>7.4f}{'':<7} "
              f"{stats['min_distance_between_selected']:>7.4f}")


def test_export_selected_cameras():
    """
    导出选择的相机到新的JSON格式（可选）
    """
    print("\n" + "=" * 70)
    print("导出选择的相机配置")
    print("=" * 70)
    
    test_dir = Path(__file__).resolve().parent
    camera_json_path = test_dir / "camera_path.json"
    
    if not camera_json_path.exists():
        print(f"✗ 相机JSON文件不存在: {camera_json_path}")
        return
    
    # 加载和筛选相机
    cameras, _, _ = load_camera_from_json(
        camera_json_path, 512, 512, "cpu", 0, False
    )
    
    config = CameraSelectorConfig(
        selection_strategy="greedy_max_min",
        target_camera_ratio=0.6,
        verbose=False,
    )
    
    selector = CameraSelector(config)
    selected_indices = selector.select_cameras(cameras)
    
    print(f"\n✓ 筛选完成: 从{len(cameras)}个相机中选择{len(selected_indices)}个")
    
    # 生成导出配置
    output_file = test_dir / "selected_camera_indices.txt"
    
    print(f"\n导出选择的相机索引到: {output_file}")
    
    with open(output_file, 'w') as f:
        f.write(f"# 相机筛选结果\n")
        f.write(f"# 总相机数: {len(cameras)}\n")
        f.write(f"# 选择相机数: {len(selected_indices)}\n")
        f.write(f"# 筛选比例: {len(selected_indices)/len(cameras)*100:.1f}%\n")
        f.write(f"# 选择策略: {config.selection_strategy}\n")
        f.write(f"\n")
        f.write(f"selected_indices = {selected_indices.tolist()}\n")
    
    print(f"✓ 导出成功！")
    
    # 也导出为NumPy格式
    np_file = test_dir / "selected_camera_indices.npy"
    np.save(np_file, selected_indices)
    print(f"✓ NumPy格式已保存: {np_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="相机筛选集成测试")
    parser.add_argument(
        "--test", 
        type=str, 
        default="all",
        choices=["all", "basic", "ratios", "export"],
        help="要运行的测试类型"
    )
    
    args = parser.parse_args()
    
    if args.test in ["all", "basic"]:
        test_camera_selection_from_json()
    
    if args.test in ["all", "ratios"]:
        test_camera_selection_custom_ratios()
    
    if args.test in ["all", "export"]:
        test_export_selected_cameras()
    
    print("\n✓ 所有测试完成！")
