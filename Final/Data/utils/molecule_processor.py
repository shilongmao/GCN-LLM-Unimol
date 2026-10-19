# -*- coding: utf-8 -*-
# Copyright (c) 2026 GCN-LLM-UniMol 项目作者（个人）. All Rights Reserved.
# 本项目基于 DrugCLIP 与 GCN-LLM 构建，仅限非商业用途；
# 未经作者书面许可，不得用于论文、学位论文、专利等学术发表。完整条款见根目录 LICENSE。
"""
分子处理工具
============

提供分子清洗、标准化和性质计算功能
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from tqdm import tqdm


class MoleculeProcessor:
    """分子处理器"""
    
    def __init__(self, activity_threshold=1000):
        """
        初始化处理器
        
        参数:
            activity_threshold: 活性阈值（nM），默认1000nM = 1μM
        """
        self.activity_threshold = activity_threshold
    
    def validate_smiles(self, smiles):
        """验证SMILES有效性"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # 尝试标准化
                Chem.SanitizeMol(mol)
                return True, Chem.MolToSmiles(mol)  # 返回标准化的SMILES
        except:
            pass
        return False, None
    
    def calculate_properties(self, smiles):
        """计算分子性质"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            props = {
                'mol_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'num_rings': Descriptors.RingCount(mol),
                'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
                'tpsa': Descriptors.TPSA(mol),
                'num_hbd': Descriptors.NumHDonors(mol),
                'num_hba': Descriptors.NumHAcceptors(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol)
            }
            return props
        except:
            return None
    
    def generate_activity_label(self, pchembl_value):
        """
        生成活性标签
        
        pchembl >= 6.0 表示 IC50 <= 1μM，标记为活性（1）
        pchembl < 6.0 表示 IC50 > 1μM，标记为非活性（0）
        """
        if pd.isna(pchembl_value):
            return None
        return 1 if pchembl_value >= 6.0 else 0
    
    def filter_by_properties(self, df, 
                            min_mw=150, max_mw=800,
                            min_logp=-2, max_logp=7,
                            max_atoms=100):
        """根据分子性质过滤"""
        print("\n应用性质过滤...")
        print(f"  分子量范围: {min_mw}-{max_mw}")
        print(f"  LogP范围: {min_logp}-{max_logp}")
        print(f"  最大原子数: {max_atoms}")
        
        initial_count = len(df)
        
        df = df[
            (df['mol_weight'] >= min_mw) & 
            (df['mol_weight'] <= max_mw) &
            (df['logp'] >= min_logp) & 
            (df['logp'] <= max_logp) &
            (df['num_atoms'] <= max_atoms)
        ].copy()
        
        filtered_count = initial_count - len(df)
        print(f"  过滤掉 {filtered_count} 个分子")
        print(f"  剩余 {len(df)} 个分子")
        
        return df
    
    def process_dataframe(self, df):
        """处理DataFrame"""
        print("\n开始处理分子数据...")
        print(f"初始分子数: {len(df)}")
        
        # 1. 验证和标准化SMILES
        print("\n步骤1: 验证SMILES...")
        valid_smiles = []
        standardized_smiles = []
        
        for smiles in tqdm(df['smiles'], desc="验证SMILES"):
            is_valid, std_smiles = self.validate_smiles(smiles)
            valid_smiles.append(is_valid)
            standardized_smiles.append(std_smiles if is_valid else None)
        
        df['valid'] = valid_smiles
        df['smiles_std'] = standardized_smiles
        
        # 只保留有效的
        df = df[df['valid']].copy()
        df['smiles'] = df['smiles_std']
        df = df.drop(['valid', 'smiles_std'], axis=1)
        
        print(f"  有效SMILES: {len(df)}")
        
        # 2. 去重
        print("\n步骤2: 去重...")
        df = df.drop_duplicates(subset=['smiles']).reset_index(drop=True)
        print(f"  去重后: {len(df)}")
        
        # 3. 计算分子性质
        print("\n步骤3: 计算分子性质...")
        properties_list = []
        
        for smiles in tqdm(df['smiles'], desc="计算性质"):
            props = self.calculate_properties(smiles)
            properties_list.append(props)
        
        # 添加性质列
        props_df = pd.DataFrame(properties_list)
        df = pd.concat([df, props_df], axis=1)
        
        # 移除性质计算失败的分子
        df = df.dropna(subset=['mol_weight']).reset_index(drop=True)
        print(f"  性质计算成功: {len(df)}")
        
        # 4. 生成活性标签
        print("\n步骤4: 生成活性标签...")
        df['label'] = df['pchembl'].apply(self.generate_activity_label)
        df = df.dropna(subset=['label']).reset_index(drop=True)
        df['label'] = df['label'].astype(int)
        
        print(f"  活性分子 (label=1): {(df['label']==1).sum()}")
        print(f"  非活性分子 (label=0): {(df['label']==0).sum()}")
        
        # 5. 过滤异常分子
        df = self.filter_by_properties(df)
        
        return df
    
    def process_csv(self, input_file, output_file=None):
        """
        处理CSV文件
        
        参数:
            input_file: 输入CSV文件路径
            output_file: 输出CSV文件路径（可选）
        
        返回:
            处理后的DataFrame
        """
        print("="*70)
        print("分子数据处理")
        print("="*70)
        
        # 读取数据
        print(f"\n读取文件: {input_file}")
        df = pd.read_csv(input_file)
        
        # 处理
        df_processed = self.process_dataframe(df)
        
        # 保存
        if output_file:
            df_processed.to_csv(output_file, index=False)
            print(f"\n处理后的数据已保存至: {output_file}")
        
        # 打印统计信息
        print("\n" + "="*70)
        print("处理完成！统计信息:")
        print("="*70)
        print(f"总分子数: {len(df_processed)}")
        print(f"活性分子: {(df_processed['label']==1).sum()}")
        print(f"非活性分子: {(df_processed['label']==0).sum()}")
        print(f"\n分子性质统计:")
        print(df_processed[['mol_weight', 'logp', 'num_atoms', 'tpsa']].describe())
        
        return df_processed


# 使用示例
if __name__ == "__main__":
    processor = MoleculeProcessor()
    
    # 处理数据
    df = processor.process_csv(
        input_file='hdac1_raw_data.csv',
        output_file='hdac1_processed.csv'
    )
    
    print(f"\n前5行数据:")
    print(df.head())
