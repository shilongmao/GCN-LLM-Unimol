# -*- coding: utf-8 -*-
# Copyright (c) 2026 GCN-LLM-UniMol 项目作者（个人）. All Rights Reserved.
# 本项目基于 DrugCLIP 与 GCN-LLM 构建，仅限非商业用途；
# 未经作者书面许可，不得用于论文、学位论文、专利等学术发表。完整条款见根目录 LICENSE。
"""
完整嵌入生成流程
================

整合ChemBERTa（768维LLM嵌入）和DrugCLIP（512维UniMol嵌入）

工作流程:
1. 读取处理后的数据
2. 提取ChemBERTa嵌入（768维）
3. 提取DrugCLIP嵌入（512维）
4. 剔除失败样本
5. 创建训练数据集
"""

import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

# 导入特征提取器
from chemberta_feature_extractor import ChemBERTaFeatureExtractor, CHEMBERTA_LOCAL_PATH
from unimol_feature_extractor import UniMolFeatureExtractor, download_pdb


def prepare_complete_embeddings(
    data_csv,
    chemberta_model_path,
    drugclip_checkpoint,
    target_pdb_id,
    output_dir='complete_embeddings',
    device='cuda',
    batch_size=32,
    max_pocket_atoms=512
):
    """
    准备完整的嵌入数据
    
    参数:
    ----
    data_csv : str
        处理后的数据CSV文件
    chemberta_model_path : str
        ChemBERTa模型路径
    drugclip_checkpoint : str
        DrugCLIP模型路径
    target_pdb_id : str
        目标蛋白PDB ID
    output_dir : str
        输出目录
    device : str
        计算设备
    batch_size : int
        批次大小
    max_pocket_atoms : int
        蛋白口袋最大原子数
    """
    
    print("="*70)
    print("完整嵌入生成流程")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ============================================
    # 步骤1: 读取数据
    # ============================================
    print(f"\n步骤1: 读取数据")
    print(f"数据文件: {data_csv}")
    
    df = pd.read_csv(data_csv)
    print(f"总样本数: {len(df)}")
    
    smiles_list = df['smiles'].tolist()
    labels = df['label'].values
    
    print(f"活性分子: {labels.sum()}")
    print(f"非活性分子: {len(labels) - labels.sum()}")
    
    # ============================================
    # 步骤2: 下载蛋白PDB
    # ============================================
    print(f"\n步骤2: 下载目标蛋白PDB")
    print(f"PDB ID: {target_pdb_id}")
    
    pdb_dir = os.path.join(output_dir, 'pdb_files')
    pdb_file = download_pdb(target_pdb_id, pdb_dir)
    
    # ============================================
    # 步骤3: 提取ChemBERTa嵌入（768维）
    # ============================================
    print(f"\n步骤3: 提取ChemBERTa嵌入（768维）")
    
    chemberta_emb_file = os.path.join(output_dir, 'chemberta_embeddings.pkl')
    
    if os.path.exists(chemberta_emb_file):
        print(f"加载已存在的ChemBERTa嵌入: {chemberta_emb_file}")
        with open(chemberta_emb_file, 'rb') as f:
            chemberta_embeddings = pickle.load(f)
        chemberta_failed = []
    else:
        print("初始化ChemBERTa提取器...")
        chemberta_extractor = ChemBERTaFeatureExtractor(
            model_path=chemberta_model_path,
            device=device
        )
        
        # 正确接收返回值：(embeddings, failed)
        chemberta_embeddings, chemberta_failed = chemberta_extractor.batch_extract_molecules(
            smiles_list,
            chemberta_emb_file,
            batch_size=batch_size
        )
    
    print(f"ChemBERTa成功: {len(chemberta_embeddings)}/{len(smiles_list)}")
    print(f"ChemBERTa失败: {len(chemberta_failed)}")
    
    # ============================================
    # 步骤4: 提取DrugCLIP嵌入（512维）
    # ============================================
    print(f"\n步骤4: 提取DrugCLIP嵌入（512维）")
    
    unimol_emb_file = os.path.join(output_dir, 'unimol_embeddings.pkl')
    
    if os.path.exists(unimol_emb_file):
        print(f"加载已存在的UniMol嵌入: {unimol_emb_file}")
        with open(unimol_emb_file, 'rb') as f:
            unimol_embeddings = pickle.load(f)
        unimol_failed = []
    else:
        print("初始化DrugCLIP提取器...")
        unimol_extractor = UniMolFeatureExtractor(
            drugclip_checkpoint_path=drugclip_checkpoint,
            device=device
        )
        
        # 正确接收返回值：(embeddings, failed)
        unimol_embeddings, unimol_failed = unimol_extractor.batch_extract_molecules(
            smiles_list,
            unimol_emb_file,
            batch_size=batch_size
        )
    
    print(f"DrugCLIP成功: {len(unimol_embeddings)}/{len(smiles_list)}")
    print(f"DrugCLIP失败: {len(unimol_failed)}")
    
    # ============================================
    # 步骤5: 提取蛋白嵌入
    # ============================================
    print(f"\n步骤5: 提取蛋白嵌入（512维）")
    
    protein_emb_file = os.path.join(output_dir, 'protein_embedding.npy')
    
    if os.path.exists(protein_emb_file):
        print(f"加载已存在的蛋白嵌入: {protein_emb_file}")
        protein_emb = np.load(protein_emb_file)
    else:
        if 'unimol_extractor' not in locals():
            print("初始化DrugCLIP提取器...")
            unimol_extractor = UniMolFeatureExtractor(
                drugclip_checkpoint_path=drugclip_checkpoint,
                device=device
            )
        
        protein_emb = unimol_extractor.extract_protein_embedding(
            pdb_file,
            pocket_residues=None,
            max_atoms=max_pocket_atoms
        )
        
        np.save(protein_emb_file, protein_emb)
    
    print(f"蛋白嵌入维度: {protein_emb.shape}")
    
    # ============================================
    # 步骤6: 剔除失败样本，创建训练数据集
    # ============================================
    print(f"\n步骤6: 创建训练数据集（剔除失败样本）")
    
    # 找出所有失败的SMILES
    all_failed = set(chemberta_failed) | set(unimol_failed)
    print(f"总失败样本数: {len(all_failed)}")
    
    # 过滤有效样本
    valid_indices = []
    valid_smiles = []
    valid_labels = []
    valid_chemberta_embs = []
    valid_unimol_embs = []
    
    for i, smiles in enumerate(smiles_list):
        if smiles in all_failed:
            continue
        
        if smiles not in chemberta_embeddings or smiles not in unimol_embeddings:
            continue
        
        valid_indices.append(i)
        valid_smiles.append(smiles)
        valid_labels.append(labels[i])
        valid_chemberta_embs.append(chemberta_embeddings[smiles])
        valid_unimol_embs.append(unimol_embeddings[smiles])
    
    print(f"有效样本数: {len(valid_indices)}/{len(smiles_list)}")
    print(f"剔除样本数: {len(smiles_list) - len(valid_indices)}")
    
    # 统计有效样本的标签分布
    valid_labels_array = np.array(valid_labels)
    print(f"有效样本中活性分子: {valid_labels_array.sum()}")
    print(f"有效样本中非活性分子: {len(valid_labels_array) - valid_labels_array.sum()}")
    
    # 保存为训练就绪格式
    training_data = {
        'smiles': valid_smiles,
        'labels': valid_labels_array,
        'chemberta_embeddings': np.array(valid_chemberta_embs),  # [N, 768]
        'unimol_embeddings': np.array(valid_unimol_embs),  # [N, 512]
        'protein_embedding': protein_emb,  # [512]
        'protein_pdb_id': target_pdb_id,
        'protein_pdb_file': pdb_file,
        'failed_smiles': list(all_failed)
    }
    
    output_file = os.path.join(output_dir, 'training_data_complete.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(training_data, f)
    
    print(f"\n训练数据保存至: {output_file}")
    print(f"数据统计:")
    print(f"  样本数: {len(valid_smiles)}")
    print(f"  活性分子: {valid_labels_array.sum()}")
    print(f"  非活性分子: {len(valid_labels_array) - valid_labels_array.sum()}")
    print(f"  ChemBERTa嵌入维度: {valid_chemberta_embs[0].shape}")
    print(f"  UniMol嵌入维度: {valid_unimol_embs[0].shape}")
    print(f"  蛋白嵌入维度: {protein_emb.shape}")
    
    # 保存元数据
    metadata = {
        'data_csv': data_csv,
        'chemberta_model': chemberta_model_path,
        'drugclip_checkpoint': drugclip_checkpoint,
        'target_pdb_id': target_pdb_id,
        'num_total_samples': len(smiles_list),
        'num_valid_samples': len(valid_smiles),
        'num_failed_samples': len(all_failed),
        'num_active': int(valid_labels_array.sum()),
        'num_inactive': int(len(valid_labels_array) - valid_labels_array.sum()),
        'chemberta_dim': 768,
        'unimol_dim': 512
    }
    
    metadata_file = os.path.join(output_dir, 'metadata_complete.pkl')
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"元数据保存至: {metadata_file}")
    
    print("\n" + "="*70)
    print("数据准备完成！")
    print("="*70)
    print(f"\n下一步: 运行训练脚本")
    print(f"python train.py --data {output_file}")
    
    return output_file


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='准备完整嵌入数据')
    
    # 输入数据
    parser.add_argument('--data', type=str, required=True,
                       help='处理后的数据CSV文件')
    parser.add_argument('--chemberta-model', type=str, default=None,
                       help='ChemBERTa模型路径（默认使用 CHEMBERTA_MODEL_PATH 环境变量，'
                            '否则用包内 models--seyonec--ChemBERTa-zinc-base-v1）')
    parser.add_argument('--drugclip-checkpoint', type=str, required=True,
                       help='DrugCLIP模型路径')
    parser.add_argument('--pdb-id', type=str, required=True,
                       help='目标蛋白PDB ID')
    
    # 输出
    parser.add_argument('--output', type=str, default='complete_embeddings',
                       help='输出目录')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--max-pocket-atoms', type=int, default=512,
                       help='蛋白口袋最大原子数')
    
    args = parser.parse_args()
    
    try:
        prepare_complete_embeddings(
            data_csv=args.data,
            chemberta_model_path=args.chemberta_model or CHEMBERTA_LOCAL_PATH,
            drugclip_checkpoint=args.drugclip_checkpoint,
            target_pdb_id=args.pdb_id,
            output_dir=args.output,
            device=args.device,
            batch_size=args.batch_size,
            max_pocket_atoms=args.max_pocket_atoms
        )
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
