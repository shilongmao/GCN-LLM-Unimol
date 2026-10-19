# -*- coding: utf-8 -*-
# Copyright (c) 2026 GCN-LLM-UniMol 项目作者（个人）. All Rights Reserved.
# 本项目基于 DrugCLIP 与 GCN-LLM 构建，仅限非商业用途；
# 未经作者书面许可，不得用于论文、学位论文、专利等学术发表。完整条款见根目录 LICENSE。
"""
GCN-LLM-UniMol模型训练脚本
使用InfoNCE对比损失进行单蛋白专属虚拟筛选
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import time
import json

from gcn_llm_unimol_encoder import GCN_LLM_UniMol_Classifier


# ============================================
# InfoNCE对比损失
# ============================================
class InfoNCELoss(nn.Module):
    """单向InfoNCE对比损失"""
    
    def __init__(self, temperature=0.07, learnable=True):
        super().__init__()
        if learnable:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/temperature))
        else:
            self.register_buffer('logit_scale', torch.tensor(np.log(1/temperature)))
    
    def forward(self, protein_emb, molecule_embs, pos_mask):
        if protein_emb.dim() == 1:
            protein_emb = protein_emb.unsqueeze(0)
        
        protein_emb = F.normalize(protein_emb, p=2, dim=1)
        molecule_embs = F.normalize(molecule_embs, p=2, dim=1)
        
        logits = protein_emb @ molecule_embs.t() * self.logit_scale.exp()
        probs = F.softmax(logits, dim=1)
        pos_probs = probs[0, pos_mask]
        
        if pos_probs.numel() == 0:
            return torch.tensor(0.0, device=protein_emb.device)
        
        return -torch.log(pos_probs + 1e-8).mean()


# ============================================
# 数据处理
# ============================================
def get_atom_features(atom):
    """提取原子特征（139维）"""
    features = []
    atom_types = list(range(1, 119))
    features.extend([1 if atom.GetAtomicNum() == t else 0 for t in atom_types])
    features.extend([1 if atom.GetDegree() == d else 0 for d in range(7)])
    features.extend([1 if atom.GetFormalCharge() == c else 0 for c in [-2, -1, 0, 1, 2]])
    
    hybridizations = [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
                     Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
                     Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]
    features.extend([1 if atom.GetHybridization() == h else 0 for h in hybridizations])
    features.append(1 if atom.GetIsAromatic() else 0)
    features.extend([1 if atom.GetTotalNumHs() == h else 0 for h in range(5)])
    
    while len(features) < 139:
        features.append(0)
    return features[:139]


def smiles_to_graph(smiles):
    """SMILES转图"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)
    
    edge_indices = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])
    
    if not edge_indices:
        edge_indices = [[0, 0]]
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
    return Data(x=x, edge_index=edge_index)


def generate_mock_embedding(smiles, dim):
    """生成模拟嵌入（仅用于LLM，UniMol使用预计算嵌入）"""
    seed = hash(smiles) % (2**32)
    np.random.seed(seed)
    return np.random.randn(dim).astype(np.float32)


def load_unimol_embeddings(embedding_file):
    """
    加载预计算的UniMol嵌入
    
    参数:
    ----
    embedding_file : str
        嵌入文件路径（.pkl格式）
    
    返回:
    ----
    dict : {smiles: embedding}
    """
    import pickle
    
    if not os.path.exists(embedding_file):
        raise FileNotFoundError(f"嵌入文件不存在: {embedding_file}")
    
    with open(embedding_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    print(f"加载UniMol嵌入: {len(embeddings)} 个分子")
    
    # 检查嵌入维度
    first_emb = next(iter(embeddings.values()))
    print(f"嵌入维度: {first_emb.shape}")
    
    return embeddings


class MoleculeDataset:
    """分子数据集"""
    
    def __init__(self, smiles_list, labels, unimol_embeddings=None, llm_embeddings=None):
        """
        参数:
        ----
        smiles_list : list of str
            SMILES列表
        labels : array-like
            标签
        unimol_embeddings : dict, optional
            预计算的UniMol嵌入 {smiles: embedding}
            如果为None，则使用模拟嵌入
        llm_embeddings : dict, optional
            预计算的LLM嵌入 {smiles: embedding}
            如果为None，则使用模拟嵌入
        """
        self.smiles_list = smiles_list
        self.labels = labels
        self.unimol_embeddings = unimol_embeddings
        self.llm_embeddings = llm_embeddings
        self.valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            if smiles_to_graph(smiles) is not None:
                # 如果使用预计算嵌入，检查是否存在
                if unimol_embeddings is not None:
                    if smiles not in unimol_embeddings:
                        continue
                if llm_embeddings is not None:
                    if smiles not in llm_embeddings:
                        continue
                self.valid_indices.append(i)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        smiles = self.smiles_list[real_idx]
        
        # LLM嵌入（真实或模拟）
        if self.llm_embeddings is not None and smiles in self.llm_embeddings:
            llm_emb = torch.tensor(self.llm_embeddings[smiles], dtype=torch.float)
        else:
            # 回退到模拟嵌入（768维）
            llm_emb = torch.tensor(generate_mock_embedding(smiles, 768), dtype=torch.float)
        
        # UniMol嵌入（真实或模拟）
        if self.unimol_embeddings is not None and smiles in self.unimol_embeddings:
            unimol_emb = torch.tensor(self.unimol_embeddings[smiles], dtype=torch.float)
        else:
            # 回退到模拟嵌入（512维）
            unimol_emb = torch.tensor(generate_mock_embedding(smiles, 512), dtype=torch.float)
        
        return {
            'graph': smiles_to_graph(smiles),
            'llm_emb': llm_emb,
            'unimol_emb': unimol_emb,
            'label': torch.tensor(self.labels[real_idx], dtype=torch.long),
            'smiles': smiles
        }


def collate_fn(batch):
    """批处理函数"""
    from torch_geometric.data import Batch
    
    # 过滤掉无效的图
    valid_batch = [item for item in batch if item['graph'] is not None]
    
    if len(valid_batch) == 0:
        return None
    
    graphs = [item['graph'] for item in valid_batch]
    llm_embs = torch.stack([item['llm_emb'] for item in valid_batch])
    unimol_embs = torch.stack([item['unimol_emb'] for item in valid_batch])
    labels = torch.stack([item['label'] for item in valid_batch])
    
    batched_graph = Batch.from_data_list(graphs)
    
    # 返回字典，避免解包错误
    return {
        'graph': batched_graph,
        'llm_emb': llm_embs,
        'unimol_emb': unimol_embs,
        'label': labels
    }


# ============================================
# 训练和评估
# ============================================
def train_epoch(model, loader, optimizer, criterion, get_protein_vector, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_similarities, all_labels = [], []
    
    for batch_data in loader:
        if batch_data is None:
            continue
        
        batch_graph = batch_data['graph'].to(device)
        llm_embs = batch_data['llm_emb'].to(device)
        unimol_embs = batch_data['unimol_emb'].to(device)
        labels = batch_data['label'].to(device)
        
        mol_vectors = model.get_molecule_embeddings(
            batch_graph.x, batch_graph.edge_index, batch_graph.batch,
            llm_embs, unimol_embs
        )
        
        # 每次获取最新的蛋白向量
        protein_vector = get_protein_vector()
        
        pos_mask = (labels == 1)
        loss = criterion(protein_vector, mol_vectors, pos_mask)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        
        total_loss += loss.item()
        
        # 收集相似度用于动态阈值评估
        with torch.no_grad():
            similarities = F.cosine_similarity(
                protein_vector.unsqueeze(0).expand(mol_vectors.size(0), -1),
                mol_vectors, dim=1
            )
            all_similarities.extend(similarities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 使用动态阈值评估训练集
    all_similarities = np.array(all_similarities)
    all_labels = np.array(all_labels)
    
    # 找到最佳阈值
    best_acc = 0
    best_thr = 0
    for thr in np.linspace(all_similarities.min(), all_similarities.max(), 100):
        preds = (all_similarities > thr).astype(int)
        acc = accuracy_score(all_labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    
    return total_loss / len(loader), best_acc


def evaluate(model, loader, get_protein_vector, device, find_best_threshold=False):
    """评估模型"""
    model.eval()
    all_preds, all_labels, all_similarities = [], [], []
    
    # 获取当前的蛋白向量
    protein_vector = get_protein_vector()
    
    with torch.no_grad():
        for batch_data in loader:
            if batch_data is None:
                continue
            
            batch_graph = batch_data['graph'].to(device)
            llm_embs = batch_data['llm_emb'].to(device)
            unimol_embs = batch_data['unimol_emb'].to(device)
            labels = batch_data['label']
            
            mol_vectors = model.get_molecule_embeddings(
                batch_graph.x, batch_graph.edge_index, batch_graph.batch,
                llm_embs, unimol_embs
            )
            
            similarities = F.cosine_similarity(
                protein_vector.unsqueeze(0).expand(mol_vectors.size(0), -1),
                mol_vectors, dim=1
            )
            
            all_similarities.extend(similarities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy() if labels.is_cuda else labels.numpy())
    
    all_similarities = np.array(all_similarities)
    all_labels = np.array(all_labels)
    
    # 寻找最佳阈值（基于F1分数）
    if find_best_threshold:
        best_threshold = 0.5
        best_f1 = 0.0
        for thresh in np.arange(-0.5, 1.0, 0.05):
            preds = (all_similarities > thresh).astype(int)
            f1 = f1_score(all_labels, preds, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        threshold = best_threshold
    else:
        # 使用中位数作为阈值
        threshold = np.median(all_similarities)
    
    # 基于阈值预测
    preds = (all_similarities > threshold).astype(int)
    
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, average='binary', zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_similarities)
    except:
        auc = 0.5
    
    return acc, f1, auc, threshold


# ============================================
# 主函数
# ============================================
def main():
    parser = argparse.ArgumentParser(description='GCN-LLM-UniMol训练')
    parser.add_argument('--data', type=str, required=True, help='数据文件路径（CSV、TXT或PKL）')
    parser.add_argument('--unimol-emb', type=str, help='预计算的UniMol嵌入文件（.pkl）')
    parser.add_argument('--protein-emb', type=str, help='蛋白嵌入文件（.npy）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数（默认100，早停会提前结束）')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率（会自适应调整）')
    parser.add_argument('--temperature', type=float, default=0.07, help='InfoNCE温度参数')
    parser.add_argument('--learnable_temp', action='store_true', help='是否使用可学习温度')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("="*70)
    print("GCN-LLM-UniMol模型训练")
    print("="*70)
    
    # 加载数据
    print("\n加载数据...")
    
    # 支持三种数据格式
    if args.data.endswith('.pkl'):
        # 从 prepare_complete_embeddings.py 生成的训练数据
        import pickle
        with open(args.data, 'rb') as f:
            data = pickle.load(f)
        
        smiles_list = data['smiles']
        labels = data['labels']
        
        # 使用预计算的UniMol嵌入
        unimol_embeddings = {smiles: emb for smiles, emb in zip(smiles_list, data['unimol_embeddings'])}
        
        # 使用预计算的LLM嵌入（如果存在）
        if 'chemberta_embeddings' in data:
            llm_embeddings = {smiles: emb for smiles, emb in zip(smiles_list, data['chemberta_embeddings'])}
            print(f"从PKL文件加载数据（包含真实ChemBERTa嵌入）")
        else:
            llm_embeddings = None
            print(f"从PKL文件加载数据（使用模拟LLM嵌入）")
        
        protein_vector_data = data['protein_embedding']
        
        print(f"目标蛋白: {data.get('protein_pdb_id', 'Unknown')}")
        
    elif args.data.endswith('.txt'):
        with open(args.data, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        np.random.seed(42)
        labels = np.random.randint(0, 2, len(smiles_list))
        unimol_embeddings = None
        llm_embeddings = None
        protein_vector_data = None
        
    else:
        df = pd.read_csv(args.data)
        smiles_list = df['smiles'].tolist()
        labels = df['label'].values
        unimol_embeddings = None
        llm_embeddings = None
        protein_vector_data = None
    
    # 如果命令行指定了嵌入文件，则覆盖
    if args.unimol_emb:
        print(f"\n加载UniMol嵌入: {args.unimol_emb}")
        unimol_embeddings = load_unimol_embeddings(args.unimol_emb)
    
    if args.protein_emb:
        print(f"\n加载蛋白嵌入: {args.protein_emb}")
        protein_vector_data = np.load(args.protein_emb)
    
    print(f"总样本数: {len(smiles_list)}")
    print(f"活性分子: {labels.sum()}, 非活性分子: {len(labels) - labels.sum()}")
    
    if unimol_embeddings:
        print(f"使用预计算的UniMol嵌入: {len(unimol_embeddings)} 个分子")
    else:
        print(f"使用模拟UniMol嵌入（512维）")
    
    if llm_embeddings:
        print(f"使用预计算的ChemBERTa嵌入: {len(llm_embeddings)} 个分子")
    else:
        print(f"使用模拟LLM嵌入（768维）")
    
    # 创建数据集（传入llm_embeddings）
    dataset = MoleculeDataset(smiles_list, labels, unimol_embeddings, llm_embeddings)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size*2, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size*2, shuffle=False, collate_fn=collate_fn)
    
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    model = GCN_LLM_UniMol_Classifier(
        num_node_features=139, num_classes=2, hidden_channels=64,
        llm_dim=768, unimol_dim=512, proj_dim=64,  # UniMol为512维，投影到64维
        num_gcn_layers=6, dropout=0.2
    ).to(device)
    
    # 蛋白向量（真实或模拟）
    # 蛋白向量也是512维，需要投影到192维（与分子嵌入对齐）
    if protein_vector_data is not None:
        print(f"\n使用真实蛋白嵌入: {protein_vector_data.shape}")
        protein_vector_512 = torch.tensor(protein_vector_data, device=device, dtype=torch.float32)
    else:
        print(f"\n使用模拟蛋白嵌入（512维）")
        np.random.seed(123)
        protein_vector_512 = torch.tensor(np.random.randn(512).astype(np.float32), device=device)
    
    # 归一化512维蛋白向量
    protein_vector_512 = F.normalize(protein_vector_512, p=2, dim=0)
    
    # 使用与模型相同的投影方式：512→64，然后复制3次得到192维
    # 创建蛋白投影层（与UniMol投影层结构相同）
    protein_proj = torch.nn.Sequential(
        torch.nn.Linear(512, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2)
    ).to(device)
    
    # 损失和优化器
    criterion = InfoNCELoss(temperature=args.temperature, learnable=args.learnable_temp).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()) + list(protein_proj.parameters()),
        lr=args.lr, weight_decay=1e-5
    )
    
    # 学习率调度器：验证AUC不提升时降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True, min_lr=1e-6
    )
    
    # 定义函数来获取蛋白向量（每次调用都重新计算）
    def get_protein_vector():
        """获取当前的蛋白向量（192维）"""
        protein_proj_64 = protein_proj(protein_vector_512.unsqueeze(0)).squeeze(0)  # [64]
        pv = torch.cat([protein_proj_64, protein_proj_64, protein_proj_64], dim=0)  # [192]
        return F.normalize(pv, p=2, dim=0)
    
    print(f"\n训练配置:")
    print(f"  初始学习率: {args.lr} (自适应调整)")
    print(f"  温度参数: {args.temperature} ({'可学习' if args.learnable_temp else '固定'})")
    print(f"  批次大小: {args.batch_size}")
    print(f"  早停patience: 20轮")
    
    print(f"\n开始训练 ({args.epochs}轮)...")
    
    # 训练
    history = []
    best_val_auc = 0
    best_threshold = 0.5
    patience_counter = 0
    patience = 20  # 早停patience
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, get_protein_vector, device)
        
        # 验证集：寻找最佳阈值
        val_acc, val_f1, val_auc, val_threshold = evaluate(model, val_loader, get_protein_vector, device, find_best_threshold=True)
        
        # 测试集：使用验证集的最佳阈值
        test_acc, test_f1, test_auc, _ = evaluate(model, test_loader, get_protein_vector, device, find_best_threshold=False)
        
        epoch_time = time.time() - start_time
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_acc': float(val_acc),
            'val_f1': float(val_f1),
            'val_auc': float(val_auc),
            'val_threshold': float(val_threshold),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_auc': float(test_auc),
            'time': float(epoch_time)
        })
        
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) - "
              f"Loss: {train_loss:.4f}, Train: {train_acc:.4f}, "
              f"Val: {val_acc:.4f} (AUC: {val_auc:.4f}, Thr: {val_threshold:.3f}), "
              f"Test: {test_acc:.4f} (AUC: {test_auc:.4f}), "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 学习率调度
        scheduler.step(val_auc)
        
        # 基于AUC保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_threshold = val_threshold
            patience_counter = 0  # 重置patience
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'protein_proj_state_dict': protein_proj.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_threshold': val_threshold,
                'test_auc': test_auc,
            }, os.path.join(args.output, 'best_model.pt'))
            print(f"  → 新的最佳模型！(AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停触发！验证AUC已{patience}轮未提升")
                break
    
    # 保存历史
    with open(os.path.join(args.output, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n训练完成！")
    print(f"最佳验证AUC: {best_val_auc:.4f} (阈值: {best_threshold:.3f})")
    print(f"模型保存至: {args.output}/best_model.pt")
    print(f"历史保存至: {args.output}/history.json")


if __name__ == "__main__":
    main()
