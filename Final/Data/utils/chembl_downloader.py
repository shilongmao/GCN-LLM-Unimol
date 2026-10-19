# -*- coding: utf-8 -*-
# Copyright (c) 2026 GCN-LLM-UniMol 项目作者（个人）. All Rights Reserved.
# 本项目基于 DrugCLIP 与 GCN-LLM 构建，仅限非商业用途；
# 未经作者书面许可，不得用于论文、学位论文、专利等学术发表。完整条款见根目录 LICENSE。
"""
ChEMBL数据下载工具
==================

基于DOL的高级下载器，提供同源蛋白数据下载功能
"""

from chembl_webresource_client.new_client import new_client
import pandas as pd
import numpy as np
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import requests
import time


class ChEMBLDownloader:
    """ChEMBL数据下载器"""
    
    def __init__(self, similarity_threshold=0.85):
        """
        初始化下载器
        
        参数:
            similarity_threshold: 序列相似度阈值（0-1）
        """
        self.similarity_threshold = similarity_threshold
        self.target_api = new_client.target
        self.activity_api = new_client.activity
        self.molecule_api = new_client.molecule
    
    def get_uniprot_id(self, target_chembl_id):
        """获取UniProt ID"""
        try:
            target = self.target_api.get(target_chembl_id)
            components = target.get('target_components', [])
            if components:
                accessions = components[0].get('target_component_xrefs', [])
                for acc in accessions:
                    if acc.get('xref_src_db') == 'UniProt':
                        return acc.get('xref_id')
        except Exception as e:
            print(f"获取UniProt ID失败: {e}")
        return None
    
    def get_protein_sequence(self, uniprot_id):
        """从UniProt获取蛋白序列"""
        try:
            url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
            response = requests.get(url)
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                sequence = ''.join(lines[1:])
                return sequence
        except Exception as e:
            print(f"获取序列失败: {e}")
        return None
    
    def calculate_sequence_identity(self, seq1, seq2):
        """计算序列相似度"""
        if not seq1 or not seq2:
            return 0.0
        
        try:
            alignments = pairwise2.align.globalxx(seq1, seq2)
            if alignments:
                alignment = alignments[0]
                matches = sum(a == b for a, b in zip(alignment[0], alignment[1]) if a != '-' and b != '-')
                length = max(len(seq1), len(seq2))
                return matches / length if length > 0 else 0.0
        except:
            return 0.0
        
        return 0.0
    
    def search_protein_family(self, target_name, family_pattern=None):
        """搜索蛋白家族"""
        print(f"\n搜索目标蛋白: {target_name}")
        
        # 搜索目标
        targets = self.target_api.filter(
            target_synonym__icontains=target_name,
            organism='Homo sapiens'
        ).only(['target_chembl_id', 'pref_name', 'target_type'])
        
        family_members = []
        for target in targets:
            if target['target_type'] == 'SINGLE PROTEIN':
                family_members.append({
                    'target_chembl_id': target['target_chembl_id'],
                    'target_name': target['pref_name']
                })
        
        print(f"找到 {len(family_members)} 个蛋白")
        return family_members
    
    def download_activities_for_target(self, target_id, target_name, 
                                      activity_types=['IC50', 'Ki', 'Kd'],
                                      max_compounds=1000):
        """下载单个靶点的活性数据"""
        print(f"\n下载 {target_name} 的活性数据...")
        
        activities = self.activity_api.filter(
            target_chembl_id=target_id,
            standard_type__in=activity_types,
            standard_relation='=',
            assay_type='B'
        ).only([
            'molecule_chembl_id',
            'canonical_smiles',
            'standard_type',
            'standard_value',
            'standard_units',
            'pchembl_value'
        ])
        
        data_list = []
        count = 0
        
        for act in activities:
            if count >= max_compounds:
                break
            
            if act['canonical_smiles'] and act['pchembl_value']:
                data_list.append({
                    'molecule_id': act['molecule_chembl_id'],
                    'smiles': act['canonical_smiles'],
                    'activity_type': act['standard_type'],
                    'activity_value': act['standard_value'],
                    'pchembl': float(act['pchembl_value']),
                    'target_name': target_name,
                    'target_id': target_id
                })
                count += 1
        
        print(f"  下载了 {len(data_list)} 条活性数据")
        return pd.DataFrame(data_list)
    
    def download_target_data(self, target_name, 
                            similarity_threshold=None,
                            max_compounds=5000,
                            activity_types=['IC50', 'Ki', 'Kd']):
        """
        下载目标蛋白及同源蛋白的数据
        
        参数:
            target_name: 目标蛋白名称
            similarity_threshold: 序列相似度阈值
            max_compounds: 最大化合物数量
            activity_types: 活性类型列表
        
        返回:
            DataFrame包含所有活性数据
        """
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
        
        print("="*70)
        print(f"开始下载 {target_name} 的数据")
        print(f"序列相似度阈值: {similarity_threshold}")
        print(f"最大化合物数: {max_compounds}")
        print("="*70)
        
        # 搜索蛋白家族
        family_members = self.search_protein_family(target_name)
        
        if not family_members:
            print("未找到目标蛋白！")
            return pd.DataFrame()
        
        # 下载所有家族成员的数据
        all_data = []
        
        for member in family_members[:10]:  # 限制最多10个蛋白
            try:
                df = self.download_activities_for_target(
                    member['target_chembl_id'],
                    member['target_name'],
                    activity_types=activity_types,
                    max_compounds=max_compounds // len(family_members)
                )
                
                if not df.empty:
                    all_data.append(df)
                
                time.sleep(0.5)  # 避免请求过快
                
            except Exception as e:
                print(f"下载 {member['target_name']} 失败: {e}")
                continue
        
        if not all_data:
            print("未下载到任何数据！")
            return pd.DataFrame()
        
        # 合并数据
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # 去重
        combined_df = combined_df.drop_duplicates(subset=['molecule_id'])
        
        print(f"\n下载完成！")
        print(f"总化合物数: {len(combined_df)}")
        print(f"活性化合物数: {(combined_df['pchembl'] >= 6.0).sum()}")
        
        return combined_df


# 使用示例
if __name__ == "__main__":
    downloader = ChEMBLDownloader(similarity_threshold=0.85)
    
    # 下载HDAC1数据
    df = downloader.download_target_data(
        target_name='HDAC1',
        max_compounds=1000
    )
    
    # 保存
    df.to_csv('hdac1_raw_data.csv', index=False)
    print(f"\n数据已保存至: hdac1_raw_data.csv")
