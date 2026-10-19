# -*- coding: utf-8 -*-
# Copyright (c) 2026 GCN-LLM-UniMol 项目作者（个人）. All Rights Reserved.
# 本项目基于 DrugCLIP 与 GCN-LLM 构建，仅限非商业用途；
# 未经作者书面许可，不得用于论文、学位论文、专利等学术发表。完整条款见根目录 LICENSE。
"""
训练结果可视化
================

可视化训练历史、性能指标和对比分析

使用方法:
python visualize_results.py --history results_complete/history.json --output results_complete/
"""

import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_history(history_file):
    """加载训练历史"""
    with open(history_file, 'r') as f:
        history = json.load(f)
    return history


def plot_training_curves(history, output_dir):
    """绘制训练曲线"""
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    val_auc = [h['val_auc'] for h in history]
    test_acc = [h['test_acc'] for h in history]
    test_auc = [h['test_auc'] for h in history]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 损失曲线
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. 准确率曲线
    ax = axes[0, 1]
    ax.plot(epochs, train_acc, 'b-', linewidth=2, label='Train Acc', alpha=0.7)
    ax.plot(epochs, val_acc, 'g-', linewidth=2, label='Val Acc')
    ax.plot(epochs, test_acc, 'r-', linewidth=2, label='Test Acc')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 3. AUC曲线
    ax = axes[1, 0]
    ax.plot(epochs, val_auc, 'g-', linewidth=2, label='Val AUC', marker='o', markersize=4)
    ax.plot(epochs, test_auc, 'r-', linewidth=2, label='Test AUC', marker='s', markersize=4)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='AUC=0.8')
    ax.axhline(y=0.85, color='gray', linestyle='--', alpha=0.5, label='AUC=0.85')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('AUC Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1])
    
    # 4. 学习率曲线
    ax = axes[1, 1]
    if 'lr' in history[0]:
        lr = [h.get('lr', 0.001) for h in history]
        ax.plot(epochs, lr, 'purple', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Learning Rate\nNot Recorded', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png", dpi=300, bbox_inches='tight')
    print(f"✓ 训练曲线保存至: {output_dir}/training_curves.png")
    plt.close()


def plot_performance_summary(history, output_dir):
    """绘制性能总结"""
    # 找到最佳epoch
    best_val_idx = np.argmax([h['val_auc'] for h in history])
    best_epoch = history[best_val_idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 最佳性能柱状图
    ax = axes[0]
    metrics = ['Train Acc', 'Val Acc', 'Test Acc']
    values = [best_epoch['train_acc'], best_epoch['val_acc'], best_epoch['test_acc']]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Best Performance (Epoch {best_epoch["epoch"]})', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. AUC对比
    ax = axes[1]
    auc_metrics = ['Val AUC', 'Test AUC']
    auc_values = [best_epoch['val_auc'], best_epoch['test_auc']]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax.bar(auc_metrics, auc_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title(f'AUC Scores (Epoch {best_epoch["epoch"]})', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=0.85, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars, auc_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_summary.png", dpi=300, bbox_inches='tight')
    print(f"✓ 性能总结保存至: {output_dir}/performance_summary.png")
    plt.close()


def plot_improvement_analysis(history, output_dir):
    """绘制改进分析"""
    epochs = [h['epoch'] for h in history]
    val_auc = [h['val_auc'] for h in history]
    test_auc = [h['test_auc'] for h in history]
    
    # 计算改进
    val_improvement = [val_auc[i] - val_auc[0] for i in range(len(val_auc))]
    test_improvement = [test_auc[i] - test_auc[0] for i in range(len(test_auc))]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. AUC改进曲线
    ax = axes[0]
    ax.plot(epochs, val_improvement, 'g-', linewidth=2, label='Val AUC Improvement', marker='o', markersize=4)
    ax.plot(epochs, test_improvement, 'r-', linewidth=2, label='Test AUC Improvement', marker='s', markersize=4)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('AUC Improvement', fontsize=12)
    ax.set_title('AUC Improvement Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. 累积改进
    ax = axes[1]
    max_val_improvement = max(val_improvement)
    max_test_improvement = max(test_improvement)
    
    categories = ['Val AUC', 'Test AUC']
    initial = [val_auc[0], test_auc[0]]
    final = [max(val_auc), max(test_auc)]
    improvement = [max_val_improvement, max_test_improvement]
    
    x = np.arange(len(categories))
    width = 0.25
    
    bars1 = ax.bar(x - width, initial, width, label='Initial', color='#95a5a6', alpha=0.7)
    bars2 = ax.bar(x, final, width, label='Best', color='#2ecc71', alpha=0.7)
    bars3 = ax.bar(x + width, improvement, width, label='Improvement', color='#3498db', alpha=0.7)
    
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('AUC Improvement Summary', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/improvement_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✓ 改进分析保存至: {output_dir}/improvement_analysis.png")
    plt.close()


def plot_threshold_analysis(history, output_dir):
    """绘制阈值分析"""
    epochs = [h['epoch'] for h in history]
    thresholds = [h.get('val_threshold', 0.5) for h in history]
    val_auc = [h['val_auc'] for h in history]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 阈值变化
    ax = axes[0]
    ax.plot(epochs, thresholds, 'purple', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Optimal Threshold', fontsize=12)
    ax.set_title('Optimal Threshold Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([min(thresholds) - 0.05, max(thresholds) + 0.05])
    
    # 2. 阈值 vs AUC
    ax = axes[1]
    scatter = ax.scatter(thresholds, val_auc, c=epochs, cmap='viridis', s=100, alpha=0.6, edgecolors='black')
    ax.set_xlabel('Optimal Threshold', fontsize=12)
    ax.set_ylabel('Validation AUC', fontsize=12)
    ax.set_title('Threshold vs AUC', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Epoch', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/threshold_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✓ 阈值分析保存至: {output_dir}/threshold_analysis.png")
    plt.close()


def generate_report(history, output_dir):
    """生成文本报告"""
    # 找到最佳epoch
    best_val_idx = np.argmax([h['val_auc'] for h in history])
    best_epoch = history[best_val_idx]
    
    # 最终epoch
    final_epoch = history[-1]
    
    report = f"""
{'='*70}
训练结果报告
{'='*70}

训练配置:
----------
- 总轮数: {len(history)}
- 最佳轮数: {best_epoch['epoch']}

最佳性能 (Epoch {best_epoch['epoch']}):
----------
训练集:
  - 准确率: {best_epoch['train_acc']:.4f}
  
验证集:
  - 准确率: {best_epoch['val_acc']:.4f}
  - AUC: {best_epoch['val_auc']:.4f}
  - 最佳阈值: {best_epoch.get('val_threshold', 0.5):.3f}
  
测试集:
  - 准确率: {best_epoch['test_acc']:.4f}
  - AUC: {best_epoch['test_auc']:.4f}

最终性能 (Epoch {final_epoch['epoch']}):
----------
训练集:
  - 准确率: {final_epoch['train_acc']:.4f}
  
验证集:
  - 准确率: {final_epoch['val_acc']:.4f}
  - AUC: {final_epoch['val_auc']:.4f}
  
测试集:
  - 准确率: {final_epoch['test_acc']:.4f}
  - AUC: {final_epoch['test_auc']:.4f}

性能改进:
----------
验证AUC: {history[0]['val_auc']:.4f} → {best_epoch['val_auc']:.4f} (+{best_epoch['val_auc'] - history[0]['val_auc']:.4f})
测试AUC: {history[0]['test_auc']:.4f} → {best_epoch['test_auc']:.4f} (+{best_epoch['test_auc'] - history[0]['test_auc']:.4f})

训练时间:
----------
总时间: {sum([h.get('time', 0) for h in history]):.1f}秒
平均每轮: {np.mean([h.get('time', 0) for h in history]):.1f}秒

{'='*70}
"""
    
    report_file = f"{output_dir}/training_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ 训练报告保存至: {report_file}")
    print(report)


def main():
    parser = argparse.ArgumentParser(description='可视化训练结果')
    parser.add_argument('--history', type=str, required=True,
                       help='训练历史JSON文件路径')
    parser.add_argument('--output', type=str, default='.',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("训练结果可视化")
    print("="*70)
    print(f"\n加载训练历史: {args.history}")
    
    # 加载历史
    history = load_history(args.history)
    print(f"✓ 加载了 {len(history)} 轮训练数据")
    
    # 生成可视化
    print("\n生成可视化...")
    plot_training_curves(history, output_dir)
    plot_performance_summary(history, output_dir)
    plot_improvement_analysis(history, output_dir)
    plot_threshold_analysis(history, output_dir)
    
    # 生成报告
    print("\n生成报告...")
    generate_report(history, output_dir)
    
    print("\n" + "="*70)
    print("可视化完成！")
    print("="*70)
    print(f"\n生成的文件:")
    print(f"  - {output_dir}/training_curves.png")
    print(f"  - {output_dir}/performance_summary.png")
    print(f"  - {output_dir}/improvement_analysis.png")
    print(f"  - {output_dir}/threshold_analysis.png")
    print(f"  - {output_dir}/training_report.txt")


if __name__ == "__main__":
    main()
