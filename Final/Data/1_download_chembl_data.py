# -*- coding: utf-8 -*-
# Copyright (c) 2026 GCN-LLM-UniMol 项目作者（个人）. All Rights Reserved.
# 本项目基于 DrugCLIP 与 GCN-LLM 构建，仅限非商业用途；
# 未经作者书面许可，不得用于论文、学位论文、专利等学术发表。完整条款见根目录 LICENSE。
"""
步骤1: 从ChEMBL下载数据
========================

从ChEMBL数据库下载目标蛋白及其同源蛋白的活性数据
"""

import argparse
import os
import sys
from utils.chembl_downloader import ChEMBLDownloader


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='从ChEMBL下载目标蛋白数据')
    
    # 参数
    parser.add_argument('--target', type=str, required=True,
                       help='目标蛋白名称（如HDAC1）')
    parser.add_argument('--similarity', type=float, default=0.85,
                       help='序列相似度阈值（默认0.85）')
    parser.add_argument('--max_compounds', type=int, default=5000,
                       help='最大化合物数量（默认5000）')
    parser.add_argument('--output', type=str, default='raw_data.csv',
                       help='输出文件路径（默认raw_data.csv）')
    parser.add_argument('--activity_types', type=str, nargs='+', 
                       default=['IC50', 'Ki', 'Kd'],
                       help='活性类型列表（默认IC50 Ki Kd）')
    
    args = parser.parse_args()
    
    print("="*70)
    print("步骤1: 从ChEMBL下载数据")
    print("="*70)
    print(f"\n参数设置:")
    print(f"  目标蛋白: {args.target}")
    print(f"  序列相似度阈值: {args.similarity}")
    print(f"  最大化合物数: {args.max_compounds}")
    print(f"  活性类型: {', '.join(args.activity_types)}")
    print(f"  输出文件: {args.output}")
    
    # 创建下载器
    downloader = ChEMBLDownloader(similarity_threshold=args.similarity)
    
    # 下载数据
    try:
        df = downloader.download_target_data(
            target_name=args.target,
            similarity_threshold=args.similarity,
            max_compounds=args.max_compounds,
            activity_types=args.activity_types
        )
        
        if df.empty:
            print("\n错误: 未下载到任何数据！")
            sys.exit(1)
        
        # 保存数据
        df.to_csv(args.output, index=False)
        
        print("\n" + "="*70)
        print("下载完成！")
        print("="*70)
        print(f"\n数据已保存至: {args.output}")
        print(f"总化合物数: {len(df)}")
        print(f"活性化合物数 (pchembl>=6.0): {(df['pchembl'] >= 6.0).sum()}")
        print(f"\n数据预览:")
        print(df.head())
        
        print("\n下一步: 运行 python 2_process_molecules.py --input {} --output processed_data.csv".format(args.output))
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
