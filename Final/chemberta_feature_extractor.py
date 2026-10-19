# -*- coding: utf-8 -*-
# Copyright (c) 2026 GCN-LLM-UniMol 项目作者（个人）. All Rights Reserved.
# 本项目基于 DrugCLIP 与 GCN-LLM 构建，仅限非商业用途；
# 未经作者书面许可，不得用于论文、学位论文、专利等学术发表。完整条款见根目录 LICENSE。
"""
ChemBERTa特征提取器
==================

使用ChemBERTa模型提取分子的768维LLM嵌入

依赖:
- transformers
- torch
- rdkit

使用流程:
1. 加载本地ChemBERTa模型
2. 批量提取分子嵌入
3. 保存为numpy数组
"""

import torch
import torch.nn as nn
import numpy as np
import warnings
import os
from tqdm import tqdm
import pickle

# 抑制PyTorch版本警告
warnings.filterwarnings("ignore", message=".*PyTorch.*")
warnings.filterwarnings("ignore", message=".*torch.*")

# 强制设置环境变量，让transformers识别PyTorch
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# 在导入transformers之前，确保torch已加载
import torch

# 没有 GPU 时降级到 CPU（原代码此处是 assert，会让纯 CPU 机器直接导入失败）
if not torch.cuda.is_available():
    warnings.warn("CUDA 不可用，ChemBERTa 将在 CPU 上运行（较慢）")

from transformers import AutoTokenizer, AutoModel

# 设置本地模型路径
#   - 优先读取环境变量 CHEMBERTA_MODEL_PATH（便于迁移到其他机器）
#   - 否则默认使用项目根目录下的 models--seyonec--ChemBERTa-zinc-base-v1
#     （项目根目录 = 本文件所在目录 Final/ 的上一级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEMBERTA_LOCAL_PATH = os.environ.get(
    "CHEMBERTA_MODEL_PATH",
    os.path.join(_PROJECT_ROOT, "models--seyonec--ChemBERTa-zinc-base-v1"),
)


class ChemBERTaFeatureExtractor(nn.Module):
    """
    ChemBERTa特征提取器
    
    使用预训练的ChemBERTa模型提取分子的768维嵌入
    """
    
    def __init__(self, model_path=CHEMBERTA_LOCAL_PATH, device='cuda'):
        """
        初始化特征提取器
        
        参数:
        ----
        model_path : str
            ChemBERTa模型路径
        device : str
            计算设备 ('cuda' 或 'cpu')
        """
        super(ChemBERTaFeatureExtractor, self).__init__()
        
        self.device = device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu'
        
        print("="*70)
        print("初始化ChemBERTa特征提取器")
        print("="*70)
        print(f"模型路径: {model_path}")
        print(f"设备: {self.device}")
        
        # 检查模型路径
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ChemBERTa模型不存在: {model_path}\n"
                f"请确保模型已下载到该路径"
            )
        
        # 如果是Hugging Face缓存目录结构，找到snapshots中的实际模型
        if os.path.exists(os.path.join(model_path, 'snapshots')):
            print(f"检测到Hugging Face缓存目录结构")
            snapshots_dir = os.path.join(model_path, 'snapshots')
            # 获取最新的snapshot
            snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
            if snapshots:
                # 使用第一个snapshot（通常只有一个）
                snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                print(f"使用snapshot: {snapshots[0]}")
                model_path = snapshot_path
            else:
                raise FileNotFoundError(f"在 {snapshots_dir} 中未找到snapshot")
        
        print(f"最终模型路径: {model_path}")
        
        print("\n加载ChemBERTa模型...")
        
        # ChemBERTa基于RoBERTa，直接使用RobertaTokenizer
        try:
            from transformers import RobertaTokenizer
            print("尝试使用RobertaTokenizer...")
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            print("✓ RobertaTokenizer加载成功")
        except Exception as e1:
            print(f"RobertaTokenizer失败: {e1}")
            # 回退到AutoTokenizer
            try:
                print("尝试使用AutoTokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            except Exception as e2:
                print(f"AutoTokenizer失败: {e2}")
                # 最后尝试：直接从roberta-base加载
                print("尝试从roberta-base加载tokenizer...")
                from transformers import RobertaTokenizer
                self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        # 加载模型（抑制警告）
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*were not used when initializing.*")
            warnings.filterwarnings("ignore", message=".*Some weights of the model checkpoint.*")
            warnings.filterwarnings("ignore", message=".*PyTorch.*")
            
            # 使用RobertaModel而不是AutoModel（避免版本检查）
            try:
                from transformers import RobertaModel
                print("加载RobertaModel...")
                self.model = RobertaModel.from_pretrained(
                    model_path,
                    ignore_mismatched_sizes=True
                )
                print("✓ RobertaModel加载成功")
            except Exception as e:
                print(f"RobertaModel加载失败: {e}")
                # 回退到AutoModel
                print("尝试AutoModel...")
                self.model = AutoModel.from_pretrained(
                    model_path,
                    ignore_mismatched_sizes=True
                )
        
        # 移动到设备
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("✓ ChemBERTa模型加载成功！")
        print(f"嵌入维度: 768")
    
    def extract_embedding(self, smiles):
        """
        提取单个分子的768维嵌入
        
        参数:
        ----
        smiles : str
            SMILES字符串
        
        返回:
        ----
        np.ndarray : 形状 [768]
        """
        return self.extract_batch_embeddings([smiles])[0]
    
    def extract_batch_embeddings(self, smiles_list):
        """
        批量提取分子嵌入
        
        参数:
        ----
        smiles_list : list of str
            SMILES列表
        
        返回:
        ----
        list of np.ndarray : 每个形状 [768]
        """
        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(
                smiles_list,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 前向传播
            outputs = self.model(**inputs)
            
            # 使用[CLS] token的表示（第一个token）
            # 或者使用所有token的平均
            embeddings = outputs.last_hidden_state.mean(dim=1)  # [batch_size, 768]
            
            # 转换为numpy
            embeddings = embeddings.cpu().numpy()
        
        return [emb for emb in embeddings]
    
    def batch_extract_molecules(self, smiles_list, output_file, batch_size=32):
        """
        批量提取分子嵌入并保存
        
        参数:
        ----
        smiles_list : list of str
            SMILES列表
        output_file : str
            输出文件路径（.pkl格式）
        batch_size : int
            批次大小
        """
        print(f"\n批量提取ChemBERTa嵌入...")
        print(f"总分子数: {len(smiles_list)}")
        print(f"批次大小: {batch_size}")
        
        embeddings = {}
        failed = []
        
        # 分批处理
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="提取嵌入"):
            batch_smiles = smiles_list[i:i+batch_size]
            
            try:
                batch_embs = self.extract_batch_embeddings(batch_smiles)
                
                for smiles, emb in zip(batch_smiles, batch_embs):
                    embeddings[smiles] = emb
                    
            except Exception as e:
                print(f"\n警告: 批次 {i//batch_size + 1} 失败: {e}")
                # 逐个尝试
                for smiles in batch_smiles:
                    try:
                        emb = self.extract_embedding(smiles)
                        embeddings[smiles] = emb
                    except:
                        failed.append(smiles)
        
        # 保存嵌入
        with open(output_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        print(f"\n提取完成!")
        print(f"成功: {len(embeddings)}/{len(smiles_list)}")
        print(f"失败: {len(failed)}")
        print(f"保存至: {output_file}")
        
        if failed:
            failed_file = output_file.replace('.pkl', '_failed.txt')
            with open(failed_file, 'w') as f:
                f.write('\n'.join(failed))
            print(f"失败列表: {failed_file}")
        
        return embeddings, failed


def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='提取ChemBERTa嵌入')
    
    parser.add_argument('--model-path', type=str, default=CHEMBERTA_LOCAL_PATH,
                       help='ChemBERTa模型路径')
    parser.add_argument('--smiles-file', type=str, required=True,
                       help='SMILES文件路径（每行一个SMILES）')
    parser.add_argument('--output', type=str, default='chemberta_embeddings.pkl',
                       help='输出文件路径')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    
    args = parser.parse_args()
    
    # 读取SMILES
    print(f"读取SMILES文件: {args.smiles_file}")
    with open(args.smiles_file, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    
    print(f"总SMILES数: {len(smiles_list)}")
    
    # 初始化提取器
    extractor = ChemBERTaFeatureExtractor(
        model_path=args.model_path,
        device=args.device
    )
    
    # 批量提取
    extractor.batch_extract_molecules(
        smiles_list,
        args.output,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
