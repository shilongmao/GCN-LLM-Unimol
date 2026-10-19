# -*- coding: utf-8 -*-
# Copyright (c) 2026 GCN-LLM-UniMol 项目作者（个人）. All Rights Reserved.
# 本项目基于 DrugCLIP 与 GCN-LLM 构建，仅限非商业用途；
# 未经作者书面许可，不得用于论文、学位论文、专利等学术发表。完整条款见根目录 LICENSE。
"""
数据准备一键脚本
================

从 ChEMBL 下载目标蛋白（及其同源蛋白）的活性数据，并清洗成
`processed_data.csv`，供后续的嵌入提取与训练使用。

用法（在任意目录均可执行）：

    python run_all.py --target HDAC1

输出：

    output/raw_data.csv         ChEMBL 原始活性记录
    output/processed_data.csv   清洗后数据（smiles + label + 理化描述符）

下一步（在 Final/ 目录下）：

    python prepare_complete_embeddings.py \
        --data Data/output/processed_data.csv \
        --drugclip-checkpoint ../checkpoints/drugclip.pt \
        --pdb-id 3NVW --output complete_embeddings

    python train.py --data complete_embeddings/training_data_complete.pkl --output results
"""

import argparse
import os
import subprocess
import sys

# 让脚本无论从哪个目录调用，都以其所在目录为工作目录
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def run_command(command, description):
    """执行一条子命令，失败则抛 CalledProcessError。"""
    print("\n" + "=" * 70)
    print(description)
    print("=" * 70)
    print(f"命令: {' '.join(command)}\n")

    result = subprocess.run(command, check=True, cwd=DATA_DIR)
    print(f"\n✓ {description} 完成")
    return result


def main():
    parser = argparse.ArgumentParser(description="一键下载 + 清洗 ChEMBL 数据")
    parser.add_argument("--target", type=str, required=True,
                        help="目标蛋白名称（如 HDAC1、Xanthine dehydrogenase/oxidase）")
    parser.add_argument("--similarity", type=float, default=0.85,
                        help="同源蛋白序列相似度阈值（默认 0.85）")
    parser.add_argument("--max_compounds", type=int, default=5000,
                        help="最大化合物数量（默认 5000）")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="输出目录（默认 output）")
    parser.add_argument("--skip_download", action="store_true",
                        help="跳过下载，直接使用已有的 raw_data.csv")
    parser.add_argument("--skip_process", action="store_true",
                        help="跳过清洗，直接使用已有的 processed_data.csv")
    args = parser.parse_args()

    output_dir = os.path.join(DATA_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    raw_file = os.path.join(output_dir, "raw_data.csv")
    processed_file = os.path.join(output_dir, "processed_data.csv")

    print("=" * 70)
    print("数据准备：ChEMBL 下载 → 分子清洗")
    print("=" * 70)
    print("\n配置:")
    print(f"  目标蛋白: {args.target}")
    print(f"  相似度阈值: {args.similarity}")
    print(f"  最大化合物数: {args.max_compounds}")
    print(f"  输出目录: {output_dir}")

    try:
        if args.skip_download:
            print(f"\n跳过步骤1（使用已有文件: {raw_file}）")
        else:
            run_command([
                sys.executable, "1_download_chembl_data.py",
                "--target", args.target,
                "--similarity", str(args.similarity),
                "--max_compounds", str(args.max_compounds),
                "--output", raw_file,
            ], "步骤1: 下载 ChEMBL 数据")

        if args.skip_process:
            print(f"\n跳过步骤2（使用已有文件: {processed_file}）")
        else:
            run_command([
                sys.executable, "2_process_molecules.py",
                "--input", raw_file,
                "--output", processed_file,
            ], "步骤2: 清洗分子数据")

    except subprocess.CalledProcessError as e:
        print(f"\n✗ 流程中断: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("✓ 数据准备完成")
    print("=" * 70)
    print("\n输出文件:")
    print(f"  原始数据:   {raw_file}")
    print(f"  清洗后数据: {processed_file}")
    print("\n下一步（在 Final/ 目录下执行）:")
    print("  python prepare_complete_embeddings.py \\")
    print("      --data Data/output/processed_data.csv \\")
    print("      --drugclip-checkpoint ../checkpoints/drugclip.pt \\")
    print("      --pdb-id 3NVW --output complete_embeddings")
    print("  python train.py --data complete_embeddings/training_data_complete.pkl")


if __name__ == "__main__":
    main()
