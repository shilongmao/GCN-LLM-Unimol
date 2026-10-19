# -*- coding: utf-8 -*-
# Copyright (c) 2026 GCN-LLM-UniMol 项目作者（个人）. All Rights Reserved.
# 本项目基于 DrugCLIP 与 GCN-LLM 构建，仅限非商业用途；
# 未经作者书面许可，不得用于论文、学位论文、专利等学术发表。完整条款见根目录 LICENSE。
"""
步骤2: 处理分子数据
==================

清洗和标准化分子数据，计算分子性质
"""

import argparse
import sys
from utils.molecule_processor import MoleculeProcessor


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='处理分子数据')
    
    # 参数
    parser.add_argument('--input', type=str, required=True,
                       help='输入CSV文件路径')
    parser.add_argument('--output', type=str, default='processed_data.csv',
                       help='输出CSV文件路径（默认processed_data.csv）')
    parser.add_argument('--activity_threshold', type=float, default=1000,
                       help='活性阈值（nM，默认1000）')
    parser.add_argument('--min_mw', type=float, default=150,
                       help='最小分子量（默认150）')
    parser.add_argument('--max_mw', type=float, default=800,
                       help='最大分子量（默认800）')
    parser.add_argument('--min_logp', type=float, default=-2,
                       help='最小LogP（默认-2）')
    parser.add_argument('--max_logp', type=float, default=7,
                       help='最大LogP（默认7）')
    parser.add_argument('--max_atoms', type=int, default=100,
                       help='最大原子数（默认100）')
    
    args = parser.parse_args()
    
    print("="*70)
    print("步骤2: 处理分子数据")
    print("="*70)
    print(f"\n参数设置:")
    print(f"  输入文件: {args.input}")
    print(f"  输出文件: {args.output}")
    print(f"  活性阈值: {args.activity_threshold} nM")
    print(f"  分子量范围: {args.min_mw}-{args.max_mw}")
    print(f"  LogP范围: {args.min_logp}-{args.max_logp}")
    print(f"  最大原子数: {args.max_atoms}")
    
    # 创建处理器
    processor = MoleculeProcessor(activity_threshold=args.activity_threshold)
    
    # 处理数据
    try:
        df = processor.process_csv(args.input, args.output)
        
        if df.empty:
            print("\n错误: 处理后没有有效数据！")
            sys.exit(1)
        
        print("\n下一步（在 Final/ 目录下）:")
        print("  python prepare_complete_embeddings.py --data {} "
              "--drugclip-checkpoint ../checkpoints/drugclip.pt "
              "--pdb-id 3NVW --output complete_embeddings".format(args.output))
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
