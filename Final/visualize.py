# -*- coding: utf-8 -*-
# Copyright (c) 2026 GCN-LLM-UniMol 项目作者（个人）. All Rights Reserved.
# 本项目基于 DrugCLIP 与 GCN-LLM 构建，仅限非商业用途；
# 未经作者书面许可，不得用于论文、学位论文、专利等学术发表。完整条款见根目录 LICENSE。
"""
训练结果可视化
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_history(history_file, output_dir='output'):
    """绘制训练历史"""
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 如果output_dir是文件路径（以.png结尾），提取目录和文件名
    if output_dir.endswith('.png'):
        output_file = output_dir
        output_dir = os.path.dirname(output_file)
        if not output_dir:
            output_dir = '.'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    else:
        output_file = os.path.join(output_dir, 'training_results.png')
    
    # 加载历史
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    test_acc = [h['test_acc'] for h in history]
    val_f1 = [h['val_f1'] for h in history]
    val_auc = [h['val_auc'] for h in history]
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GCN-LLM-UniMol训练结果', fontsize=16, fontweight='bold')
    
    # 1. 损失曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='训练损失')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('训练损失曲线', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 准确率曲线
    ax2 = axes[0, 1]
    ax2.plot(epochs, train_acc, 'b-', linewidth=2, label='训练准确率')
    ax2.plot(epochs, val_acc, 'g-', linewidth=2, label='验证准确率')
    ax2.plot(epochs, test_acc, 'r-', linewidth=2, label='测试准确率')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('准确率曲线', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. F1和AUC曲线
    ax3 = axes[1, 0]
    ax3.plot(epochs, val_f1, 'g-', linewidth=2, label='验证F1')
    ax3.plot(epochs, val_auc, 'orange', linewidth=2, label='验证AUC')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('F1和AUC曲线', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4. 最终指标对比
    ax4 = axes[1, 1]
    final_metrics = {
        '训练准确率': train_acc[-1],
        '验证准确率': val_acc[-1],
        '测试准确率': test_acc[-1],
        '验证F1': val_f1[-1],
        '验证AUC': val_auc[-1]
    }
    
    bars = ax4.bar(range(len(final_metrics)), list(final_metrics.values()), 
                   color=['blue', 'green', 'red', 'purple', 'orange'])
    ax4.set_xticks(range(len(final_metrics)))
    ax4.set_xticklabels(list(final_metrics.keys()), rotation=45, ha='right')
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('最终指标', fontsize=14)
    ax4.set_ylim([0, 1])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上显示数值
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = os.path.join(output_dir, 'training_results.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_file}")
    
    # 显示图表
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*70)
    print("训练统计")
    print("="*70)
    print(f"总轮数: {len(history)}")
    print(f"最佳验证准确率: {max(val_acc):.4f} (Epoch {val_acc.index(max(val_acc))+1})")
    print(f"最终训练准确率: {train_acc[-1]:.4f}")
    print(f"最终验证准确率: {val_acc[-1]:.4f}")
    print(f"最终测试准确率: {test_acc[-1]:.4f}")
    print(f"最终验证F1: {val_f1[-1]:.4f}")
    print(f"最终验证AUC: {val_auc[-1]:.4f}")
    print(f"最低训练损失: {min(train_loss):.4f} (Epoch {train_loss.index(min(train_loss))+1})")
    
    # 打印详细历史表格
    print("\n" + "="*70)
    print("详细训练历史")
    print("="*70)
    print(f"{'Epoch':<8} {'Loss':<10} {'Train':<10} {'Val':<10} {'Test':<10} {'F1':<10} {'AUC':<10}")
    print("-"*70)
    for h in history:
        print(f"{h['epoch']:<8} {h['train_loss']:<10.4f} {h['train_acc']:<10.4f} "
              f"{h['val_acc']:<10.4f} {h['test_acc']:<10.4f} "
              f"{h['val_f1']:<10.4f} {h['val_auc']:<10.4f}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='可视化训练结果')
    parser.add_argument('--history', type=str, default='output/history.json',
                       help='训练历史文件路径')
    parser.add_argument('--output', type=str, default='output',
                       help='输出目录')
    args = parser.parse_args()
    
    if not os.path.exists(args.history):
        print(f"错误: 找不到历史文件 {args.history}")
        print("请先运行训练脚本生成历史文件")
        return
    
    plot_training_history(args.history, args.output)


if __name__ == "__main__":
    main()
