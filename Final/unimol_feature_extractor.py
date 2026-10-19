# -*- coding: utf-8 -*-
# Copyright (c) 2026 GCN-LLM-UniMol 项目作者（个人）. All Rights Reserved.
# 本项目基于 DrugCLIP 与 GCN-LLM 构建，仅限非商业用途；
# 未经作者书面许可，不得用于论文、学位论文、专利等学术发表。完整条款见根目录 LICENSE。
"""
UniMol特征提取器
================

使用DrugCLIP模型提取分子和蛋白的512维UniMol嵌入

依赖:
- DrugCLIP预训练模型
- RDKit (生成3D构象)
- BioPython (处理PDB文件)
- unicore框架

使用流程:
1. 下载DrugCLIP预训练权重
2. 准备目标蛋白PDB文件
3. 运行此脚本提取嵌入
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# RDKit用于分子处理
from rdkit import Chem
from rdkit.Chem import AllChem

# BioPython用于蛋白处理
try:
    from Bio.PDB import PDBParser, Selection
except ImportError:
    print("警告: BioPython未安装，无法处理PDB文件")
    print("安装命令: pip install biopython")


class UniMolFeatureExtractor:
    """
    UniMol特征提取器
    
    使用DrugCLIP的UniMol编码器提取分子和蛋白的512维嵌入
    """
    
    def __init__(self, 
                 drugclip_checkpoint_path,
                 mol_dict_path='DrugCLIP-main/data/dict_mol.txt',
                 pocket_dict_path='DrugCLIP-main/data/dict_pkt.txt',
                 device='cuda'):
        """
        初始化特征提取器
        
        参数:
        ----
        drugclip_checkpoint_path : str
            DrugCLIP预训练模型路径
        mol_dict_path : str
            分子字典文件路径
        pocket_dict_path : str
            蛋白字典文件路径
        device : str
            计算设备 ('cuda' 或 'cpu')
        """
        self.device = device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu'
        
        print("="*70)
        print("初始化UniMol特征提取器")
        print("="*70)
        print(f"模型路径: {drugclip_checkpoint_path}")
        print(f"分子字典: {mol_dict_path}")
        print(f"蛋白字典: {pocket_dict_path}")
        print(f"设备: {self.device}")
        
        # 加载DrugCLIP模型
        try:
            # 导入必要的模块
            import sys
            
            # 获取项目根目录（处理相对路径）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)  # Final的父目录
            drugclip_path = os.path.join(project_root, 'DrugCLIP-main')
            
            # 添加DrugCLIP到Python路径
            if drugclip_path not in sys.path:
                sys.path.insert(0, drugclip_path)
            
            print(f"DrugCLIP路径: {drugclip_path}")
            print(f"项目根目录: {project_root}")
            
            # 转换为绝对路径
            if not os.path.isabs(drugclip_checkpoint_path):
                drugclip_checkpoint_path = os.path.join(project_root, drugclip_checkpoint_path)
            if not os.path.isabs(mol_dict_path):
                mol_dict_path = os.path.join(project_root, mol_dict_path)
            if not os.path.isabs(pocket_dict_path):
                pocket_dict_path = os.path.join(project_root, pocket_dict_path)
            
            # 检查文件是否存在
            if not os.path.exists(drugclip_checkpoint_path):
                raise FileNotFoundError(
                    f"DrugCLIP模型文件不存在: {drugclip_checkpoint_path}\n"
                    f"请检查路径是否正确"
                )
            if not os.path.exists(mol_dict_path):
                raise FileNotFoundError(f"分子字典文件不存在: {mol_dict_path}")
            if not os.path.exists(pocket_dict_path):
                raise FileNotFoundError(f"蛋白字典文件不存在: {pocket_dict_path}")
            
            print(f"✓ 所有文件存在")
            
            from unicore import checkpoint_utils
            from unicore.data import Dictionary
            from unicore import utils
            import argparse
            
            print("\n加载DrugCLIP模型...")
            
            # 加载checkpoint（已经是绝对路径）
            state = checkpoint_utils.load_checkpoint_to_cpu(drugclip_checkpoint_path)
            
            # 加载字典（已经是绝对路径）
            self.mol_dictionary = Dictionary.load(mol_dict_path)
            self.pocket_dictionary = Dictionary.load(pocket_dict_path)
            
            print(f"分子字典大小: {len(self.mol_dictionary)}")
            print(f"蛋白字典大小: {len(self.pocket_dictionary)}")
            
            # 创建模型参数
            from unimol.models.drugclip import BindingAffinityModel, drugclip_architecture
            
            # 构建args
            parser = argparse.ArgumentParser()
            args = parser.parse_args([])
            
            # 设置必要的参数
            args.data = 'DrugCLIP-main/data'
            drugclip_architecture(args)
            
            # 创建模型
            print("\n构建DrugCLIP模型...")
            self.model = BindingAffinityModel(args, self.mol_dictionary, self.pocket_dictionary)
            
            # 加载权重（使用strict=False忽略大小不匹配）
            missing_keys, unexpected_keys = self.model.load_state_dict(state["model"], strict=False)
            
            if missing_keys:
                print(f"警告: 缺少的键 ({len(missing_keys)}个)")
            if unexpected_keys:
                print(f"警告: 意外的键 ({len(unexpected_keys)}个)")
            
            print("✓ 权重加载完成（部分层可能未加载）")
            
            # 移动到设备
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print("✓ DrugCLIP模型加载成功！")
            
        except Exception as e:
            print(f"\n错误: 无法加载DrugCLIP模型")
            print(f"详细信息: {e}")
            print("\n回退到模拟嵌入方案")
            self.model = None
            self.mol_dictionary = None
            self.pocket_dictionary = None
    
    def smiles_to_3d(self, smiles, num_confs=10):
        """
        将SMILES转换为3D构象
        
        参数:
        ----
        smiles : str
            SMILES字符串
        num_confs : int
            生成的构象数量
        
        返回:
        ----
        dict : {
            'atoms': list of int,  # 原子类型
            'coords': np.ndarray,  # 3D坐标 [num_atoms, 3]
            'mol': RDKit Mol对象
        }
        """
        # 解析SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无效的SMILES: {smiles}")
        
        # 添加氢原子
        mol = Chem.AddHs(mol)
        
        # 生成3D构象
        try:
            # 使用ETKDG方法生成构象
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
            
            # 优化构象
            for conf_id in range(mol.GetNumConformers()):
                AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
            
        except Exception as e:
            print(f"警告: 3D构象生成失败 ({smiles}): {e}")
            # 使用简单的距离几何方法
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        
        # 提取第一个构象的坐标
        if mol.GetNumConformers() == 0:
            raise ValueError(f"无法生成3D构象: {smiles}")
        
        conf = mol.GetConformer(0)
        coords = conf.GetPositions()  # [num_atoms, 3]
        
        # 提取原子类型
        atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        
        return {
            'atoms': atoms,
            'coords': coords,
            'mol': mol,
            'smiles': smiles
        }
    
    def pdb_to_pocket(self, pdb_file, pocket_residues=None, distance_cutoff=10.0, max_atoms=None):
        """
        从PDB文件提取蛋白口袋
        
        参数:
        ----
        pdb_file : str
            PDB文件路径
        pocket_residues : list of int, optional
            口袋残基ID列表。如果为None，则使用整个蛋白
        distance_cutoff : float
            如果指定了配体，则提取距离配体该距离内的残基
        max_atoms : int, optional
            最大原子数限制（避免内存问题）
        
        返回:
        ----
        dict : {
            'atoms': list of int,  # 原子类型
            'coords': np.ndarray,  # 3D坐标 [num_atoms, 3]
        }
        """
        print(f"  解析PDB文件: {pdb_file}")
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        
        # 提取所有原子
        atoms_list = []
        coords_list = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    # 跳过水分子和杂原子
                    if residue.id[0] != ' ':
                        continue
                    
                    # 如果指定了口袋残基，则过滤
                    if pocket_residues is not None:
                        if residue.id[1] not in pocket_residues:
                            continue
                    
                    for atom in residue:
                        # 获取原子类型（简化映射）
                        element = atom.element.upper()
                        atomic_num = self._element_to_atomic_num(element)
                        
                        atoms_list.append(atomic_num)
                        coords_list.append(atom.coord)
                        
                        # 如果指定了最大原子数，检查是否超过
                        if max_atoms and len(atoms_list) >= max_atoms:
                            print(f"  警告: 原子数达到限制({max_atoms})，截断处理")
                            break
                    
                    if max_atoms and len(atoms_list) >= max_atoms:
                        break
                if max_atoms and len(atoms_list) >= max_atoms:
                    break
            if max_atoms and len(atoms_list) >= max_atoms:
                break
        
        if len(atoms_list) == 0:
            raise ValueError(f"未能从PDB文件提取任何原子: {pdb_file}")
        
        print(f"  提取了{len(atoms_list)}个原子")
        
        return {
            'atoms': atoms_list,
            'coords': np.array(coords_list),
            'pdb_file': pdb_file
        }
    
    def _element_to_atomic_num(self, element):
        """元素符号转原子序数"""
        element_map = {
            'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
            'P': 15, 'S': 16, 'CL': 17, 'BR': 35, 'I': 53
        }
        return element_map.get(element, 6)  # 默认为碳
    
    def extract_molecule_embedding(self, smiles):
        """
        提取分子的512维UniMol嵌入
        
        参数:
        ----
        smiles : str
            SMILES字符串
        
        返回:
        ----
        np.ndarray : 形状 [512]
        
        异常:
        ----
        如果提取失败，抛出异常（不再回退到随机嵌入）
        """
        # 如果模型未加载，抛出异常
        if self.model is None:
            raise RuntimeError("DrugCLIP模型未加载，无法提取嵌入")
        
        # 使用DrugCLIP模型提取真实嵌入
        # 生成3D构象
        mol_data = self.smiles_to_3d(smiles)
        atoms = mol_data['atoms']
        coords = mol_data['coords']
        
        # 准备输入数据（参考encode_mols_once方法）
        # 1. Tokenize原子类型
        atoms_list = [self.mol_dictionary.index(str(a)) if str(a) in self.mol_dictionary.indices else self.mol_dictionary.unk() for a in atoms]
        
        # 2. 添加BOS和EOS token
        bos_token = self.mol_dictionary.bos()
        eos_token = self.mol_dictionary.eos()
        atoms_tensor = torch.tensor([[bos_token] + atoms_list + [eos_token]], dtype=torch.long)
        
        # 3. 归一化坐标（中心化）
        coords = coords - coords.mean(axis=0)
        
        # 4. 计算距离矩阵
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        dist_matrix = torch.cdist(coords_tensor, coords_tensor)
        
        # 5. 添加BOS/EOS的padding（距离为0）
        n_atoms = dist_matrix.size(0)
        dist_padded = torch.zeros(n_atoms + 2, n_atoms + 2)
        dist_padded[1:-1, 1:-1] = dist_matrix
        dist_matrix = dist_padded.unsqueeze(0)  # [1, n+2, n+2]
        
        # 6. 计算edge type（基于原子类型）
        # 简化版本：edge_type[i,j] = atoms[i] * len(dict) + atoms[j]
        edge_type = torch.zeros_like(dist_matrix, dtype=torch.long)
        for i in range(len(atoms_tensor[0])):
            for j in range(len(atoms_tensor[0])):
                edge_type[0, i, j] = atoms_tensor[0, i] * len(self.mol_dictionary) + atoms_tensor[0, j]
        
        # 移动到设备
        atoms_tensor = atoms_tensor.to(self.device)
        dist_matrix = dist_matrix.to(self.device)
        edge_type = edge_type.to(self.device)
        
        # 前向传播（参考encode_mols_once）
        with torch.no_grad():
            # Padding mask
            mol_padding_mask = atoms_tensor.eq(self.model.mol_model.padding_idx)
            
            # Embed tokens
            mol_x = self.model.mol_model.embed_tokens(atoms_tensor)
            
            # GBF特征（Gaussian Basis Function）
            n_node = dist_matrix.size(-1)
            gbf_feature = self.model.mol_model.gbf(dist_matrix, edge_type)
            gbf_result = self.model.mol_model.gbf_proj(gbf_feature)
            
            # Graph attention bias
            graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            
            # Encoder forward
            mol_outputs = self.model.mol_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
            )
            
            # 提取CLS token的表示（第一个token）
            mol_encoder_rep = mol_outputs[0][:, 0, :]  # [1, hidden_dim]
            
            # 使用512维的encoder输出（不经过projection）
            embedding = mol_encoder_rep.squeeze(0).cpu().numpy()
        
        return embedding
    
    def extract_protein_embedding(self, pdb_file, pocket_residues=None, max_atoms=None):
        """
        提取蛋白的512维UniMol嵌入
        
        参数:
        ----
        pdb_file : str
            PDB文件路径
        pocket_residues : list of int, optional
            口袋残基ID列表
        max_atoms : int, optional
            最大原子数限制
        
        返回:
        ----
        np.ndarray : 形状 [512]
        
        异常:
        ----
        如果提取失败，抛出异常（不再回退到随机嵌入）
        """
        # 如果模型未加载，抛出异常
        if self.model is None:
            raise RuntimeError("DrugCLIP模型未加载，无法提取嵌入")
        
        # 使用DrugCLIP模型提取真实嵌入
        # 提取口袋
        pocket_data = self.pdb_to_pocket(pdb_file, pocket_residues, max_atoms=max_atoms)
        atoms = pocket_data['atoms']
        coords = pocket_data['coords']
        
        print(f"  准备输入数据...")
        # 准备输入数据（参考encode_mols_once方法）
        # 1. Tokenize原子类型
        atoms_list = [self.pocket_dictionary.index(str(a)) if str(a) in self.pocket_dictionary.indices else self.pocket_dictionary.unk() for a in atoms]
        
        # 2. 添加BOS和EOS token
        bos_token = self.pocket_dictionary.bos()
        eos_token = self.pocket_dictionary.eos()
        atoms_tensor = torch.tensor([[bos_token] + atoms_list + [eos_token]], dtype=torch.long)
        
        # 3. 归一化坐标（中心化）
        coords = coords - coords.mean(axis=0)
        
        print(f"  计算距离矩阵 ({len(atoms)}个原子)...")
        # 4. 计算距离矩阵
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        dist_matrix = torch.cdist(coords_tensor, coords_tensor)
        
        # 5. 添加BOS/EOS的padding（距离为0）
        n_atoms = dist_matrix.size(0)
        dist_padded = torch.zeros(n_atoms + 2, n_atoms + 2)
        dist_padded[1:-1, 1:-1] = dist_matrix
        dist_matrix = dist_padded.unsqueeze(0)  # [1, n+2, n+2]
        
        print(f"  计算edge type...")
        # 6. 计算edge type（基于原子类型）
        edge_type = torch.zeros_like(dist_matrix, dtype=torch.long)
        for i in range(len(atoms_tensor[0])):
            for j in range(len(atoms_tensor[0])):
                edge_type[0, i, j] = atoms_tensor[0, i] * len(self.pocket_dictionary) + atoms_tensor[0, j]
        
        print(f"  移动到GPU...")
        # 移动到设备
        atoms_tensor = atoms_tensor.to(self.device)
        dist_matrix = dist_matrix.to(self.device)
        edge_type = edge_type.to(self.device)
        
        print(f"  模型前向传播...")
        # 前向传播（参考encode_mols_once）
        with torch.no_grad():
            # Padding mask
            pocket_padding_mask = atoms_tensor.eq(self.model.pocket_model.padding_idx)
            
            # Embed tokens
            pocket_x = self.model.pocket_model.embed_tokens(atoms_tensor)
            
            # GBF特征
            n_node = dist_matrix.size(-1)
            gbf_feature = self.model.pocket_model.gbf(dist_matrix, edge_type)
            gbf_result = self.model.pocket_model.gbf_proj(gbf_feature)
            
            # Graph attention bias
            graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            
            # Encoder forward
            pocket_outputs = self.model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
            )
            
            # 提取CLS token的表示
            pocket_encoder_rep = pocket_outputs[0][:, 0, :]  # [1, hidden_dim]
            
            # 使用512维的encoder输出
            embedding = pocket_encoder_rep.squeeze(0).cpu().numpy()
        
        print(f"  ✓ 提取完成")
        return embedding
    
    def batch_extract_molecules(self, smiles_list, output_file, batch_size=32):
        """
        批量提取分子嵌入
        
        参数:
        ----
        smiles_list : list of str
            SMILES列表
        output_file : str
            输出文件路径（.pkl格式）
        batch_size : int
            批次大小
        
        返回:
        ----
        tuple: (embeddings_dict, failed_list)
        """
        print(f"\n批量提取分子嵌入...")
        print(f"总分子数: {len(smiles_list)}")
        
        embeddings = {}
        failed = []
        
        for i, smiles in enumerate(tqdm(smiles_list, desc="提取嵌入")):
            try:
                emb = self.extract_molecule_embedding(smiles)
                embeddings[smiles] = emb
            except Exception as e:
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


def download_pdb(pdb_id, output_dir='pdb_files'):
    """
    从RCSB PDB下载蛋白结构
    
    参数:
    ----
    pdb_id : str
        PDB ID (如 '1T64')
    output_dir : str
        输出目录
    
    返回:
    ----
    str : PDB文件路径
    """
    import urllib.request
    
    os.makedirs(output_dir, exist_ok=True)
    
    pdb_file = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    if os.path.exists(pdb_file):
        print(f"PDB文件已存在: {pdb_file}")
        return pdb_file
    
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    print(f"下载PDB文件: {pdb_id}")
    print(f"URL: {url}")
    
    try:
        urllib.request.urlretrieve(url, pdb_file)
        print(f"下载成功: {pdb_file}")
        return pdb_file
    except Exception as e:
        print(f"下载失败: {e}")
        raise


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='提取UniMol嵌入')
    
    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='DrugCLIP模型路径')
    parser.add_argument('--mol-dict', type=str, 
                       default='DrugCLIP-main/data/dict_mol.txt',
                       help='分子字典路径')
    parser.add_argument('--pocket-dict', type=str,
                       default='DrugCLIP-main/data/dict_pkt.txt',
                       help='蛋白字典路径')
    
    # 输入数据
    parser.add_argument('--smiles-file', type=str,
                       help='SMILES文件路径（每行一个SMILES）')
    parser.add_argument('--pdb-file', type=str,
                       help='蛋白PDB文件路径')
    parser.add_argument('--pdb-id', type=str,
                       help='PDB ID（自动下载）')
    
    # 输出
    parser.add_argument('--output-mol', type=str, default='molecule_embeddings.pkl',
                       help='分子嵌入输出文件')
    parser.add_argument('--output-protein', type=str, default='protein_embedding.pkl',
                       help='蛋白嵌入输出文件')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    
    args = parser.parse_args()
    
    print("="*70)
    print("UniMol特征提取")
    print("="*70)
    
    # 初始化提取器
    try:
        extractor = UniMolFeatureExtractor(
            drugclip_checkpoint_path=args.checkpoint,
            mol_dict_path=args.mol_dict,
            pocket_dict_path=args.pocket_dict,
            device=args.device
        )
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n提示: 如果没有DrugCLIP模型，可以使用简化方案（预计算嵌入）")
        return
    
    # 提取分子嵌入
    if args.smiles_file:
        print(f"\n读取SMILES文件: {args.smiles_file}")
        with open(args.smiles_file, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        
        extractor.batch_extract_molecules(
            smiles_list, 
            args.output_mol,
            batch_size=args.batch_size
        )
    
    # 提取蛋白嵌入
    if args.pdb_file or args.pdb_id:
        if args.pdb_id:
            pdb_file = download_pdb(args.pdb_id)
        else:
            pdb_file = args.pdb_file
        
        print(f"\n提取蛋白嵌入: {pdb_file}")
        protein_emb = extractor.extract_protein_embedding(pdb_file)
        
        with open(args.output_protein, 'wb') as f:
            pickle.dump(protein_emb, f)
        
        print(f"蛋白嵌入保存至: {args.output_protein}")
        print(f"嵌入维度: {protein_emb.shape}")


if __name__ == "__main__":
    main()
