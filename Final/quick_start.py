# -*- coding: utf-8 -*-
# Copyright (c) 2026 GCN-LLM-UniMol 项目作者（个人）. All Rights Reserved.
# 本项目基于 DrugCLIP 与 GCN-LLM 构建，仅限非商业用途；
# 未经作者书面许可，不得用于论文、学位论文、专利等学术发表。完整条款见根目录 LICENSE。
"""
快速开始：一条命令跑通「嵌入提取 → 训练 → 可视化」
==================================================

用法（在仓库根目录执行）::

    python Final/quick_start.py --pdb-id 3NVW

前提：
    1. checkpoints/drugclip.pt           DrugCLIP 预训练权重（见 README「模型权重」）
    2. models--seyonec--ChemBERTa-zinc-base-v1/   ChemBERTa 权重
    3. Final/Data/output/processed_data.csv       训练数据（仓库已附带 XOD 数据）

产出（默认）::

    Final/complete_embeddings/    嵌入 + training_data_complete.pkl
    results/                      best_model.pt、history.json、training_plot.png
"""

import argparse
import os
import subprocess
import sys

FINAL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(FINAL_DIR)


def run(cmd, description):
    print("\n" + "=" * 70)
    print(description)
    print("=" * 70)
    print("命令: " + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=True, cwd=ROOT)
    print(f"\n✓ {description} 完成")


def main():
    parser = argparse.ArgumentParser(description="GCN-LLM-UniMol 一键训练")
    parser.add_argument("--pdb-id", type=str, required=True,
                        help="目标蛋白 PDB ID，如 3NVW")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/drugclip.pt",
                        help="DrugCLIP 权重路径（默认 checkpoints/drugclip.pt）")
    parser.add_argument("--data", type=str, default="Final/Data/output/processed_data.csv",
                        help="训练数据 CSV（默认 Final/Data/output/processed_data.csv）")
    parser.add_argument("--emb-dir", type=str, default="Final/complete_embeddings",
                        help="嵌入输出目录（默认 Final/complete_embeddings）")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数（默认 50）")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="批次大小（默认 32）")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率（默认 1e-3）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="嵌入提取设备（cuda / cpu）")
    parser.add_argument("--output", type=str, default="results",
                        help="训练结果目录（默认 results）")
    parser.add_argument("--skip-embed", action="store_true",
                        help="跳过嵌入提取，直接使用 --emb-dir 中已有的结果")
    args = parser.parse_args()

    checkpoint = os.path.join(ROOT, args.checkpoint)
    data_file = os.path.join(ROOT, args.data)
    emb_dir = os.path.join(ROOT, args.emb_dir)

    if not os.path.exists(data_file):
        sys.exit(f"错误：训练数据不存在：{data_file}\n"
                 "请先运行 python Final/Data/run_all.py --target <靶点名称>")
    if not args.skip_embed and not os.path.exists(checkpoint):
        sys.exit(f"错误：DrugCLIP 权重不存在：{checkpoint}\n"
                 "请按 README「模型权重」一节下载后放到该路径")

    print("=" * 70)
    print("GCN-LLM-UniMol 一键训练")
    print("=" * 70)
    print(f"  蛋白 PDB ID : {args.pdb_id}")
    print(f"  数据        : {args.data}")
    print(f"  DrugCLIP    : {args.checkpoint}")
    print(f"  嵌入目录    : {args.emb_dir}")
    print(f"  训练轮数    : {args.epochs}")
    print(f"  输出目录    : {args.output}")

    training_data = os.path.join(emb_dir, "training_data_complete.pkl")

    # 步骤1: 提取 ChemBERTa(768) + DrugCLIP(512) 嵌入
    if args.skip_embed:
        print(f"\n跳过嵌入提取，使用已有文件：{training_data}")
        if not os.path.exists(training_data):
            sys.exit(f"错误：{training_data} 不存在")
    else:
        run([
            sys.executable, os.path.join(FINAL_DIR, "prepare_complete_embeddings.py"),
            "--data", data_file,
            "--drugclip-checkpoint", checkpoint,
            "--pdb-id", args.pdb_id,
            "--output", emb_dir,
            "--device", args.device,
            "--batch-size", str(args.batch_size),
        ], "步骤1: 提取 ChemBERTa + DrugCLIP 嵌入")

    # 步骤2: 训练
    run([
        sys.executable, os.path.join(FINAL_DIR, "train.py"),
        "--data", training_data,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--output", args.output,
    ], "步骤2: 训练 GCN-LLM-UniMol")

    # 步骤3: 可视化
    history = os.path.join(ROOT, args.output, "history.json")
    if os.path.exists(history):
        run([
            sys.executable, os.path.join(FINAL_DIR, "visualize.py"),
            "--history", history,
            "--output", os.path.join(ROOT, args.output, "training_plot.png"),
        ], "步骤3: 绘制训练曲线")

    print("\n" + "=" * 70)
    print("✓ 全部完成")
    print("=" * 70)
    print(f"\n模型: {os.path.join(args.output, 'best_model.pt')}")
    print(f"历史: {os.path.join(args.output, 'history.json')}")
    print(f"曲线: {os.path.join(args.output, 'training_plot.png')}")


if __name__ == "__main__":
    main()
