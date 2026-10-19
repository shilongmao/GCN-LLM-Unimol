# -*- coding: utf-8 -*-
# Copyright (c) 2026 GCN-LLM-UniMol 项目作者（个人）. All Rights Reserved.
# 本项目基于 DrugCLIP 与 GCN-LLM 构建，仅限非商业用途；
# 未经作者书面许可，不得用于论文、学位论文、专利等学术发表。完整条款见根目录 LICENSE。
"""
GCN-LLM-UniMol 多模态分子编码器
==================================

架构说明：
---------
本模型融合三种分子表示方式：
1. 2D图结构 (GCN) - 捕捉分子拓扑信息
2. SMILES语义 (LLM) - 捕捉化学语义信息  
3. 3D构象 (UniMol) - 捕捉空间几何信息

维度变化流程：
-------------
输入:
  - 2D图: 节点特征 [num_nodes, node_feat_dim]
  - LLM嵌入: [batch_size, 768] → 投影 → [batch_size, 64]
  - UniMol嵌入: [batch_size, 512] → 投影 → [batch_size, 64] (DrugCLIP输出512维)

每层GCN处理:
  第1层: node_feat → GCN → [num_nodes, 64]
         拼接: [64(GCN) + 64(LLM) + 64(UniMol)] = [num_nodes, 192]
  
  第2-6层: [num_nodes, 192] → GCN → [num_nodes, 64]
           拼接: [64(GCN) + 64(LLM) + 64(UniMol)] = [num_nodes, 192]

输出:
  全局平均池化: [num_nodes, 192] → [batch_size, 192]
  最终分子表示: 192维向量 (64+64+64)

关键设计：
---------
- 每层GCN输出64维，拼接后变192维
- LLM和UniMol特征在每层都参与，确保多模态信息充分融合
- 不压缩维度，保持192维到最后，最大化信息保留
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN_LLM_UniMol_Encoder(nn.Module):
    """
    GCN-LLM-UniMol 多模态分子编码器
    
    参数说明:
    ---------
    num_node_features : int
        2D图的节点特征维度（原子特征数量）
    hidden_channels : int, default=64
        GCN每层的输出维度
    llm_dim : int, default=768
        LLM嵌入的原始维度（如ChemBERTa的768维）
    unimol_dim : int, default=512
        UniMol嵌入的原始维度（DrugCLIP输出512维）
    proj_dim : int, default=64
        LLM和UniMol投影后的维度
    num_gcn_layers : int, default=6
        GCN的层数
    dropout : float, default=0.2
        Dropout比率
    
    输入:
    ----
    x : torch.Tensor
        节点特征矩阵，形状 [num_nodes, num_node_features]
    edge_index : torch.Tensor
        边索引矩阵，形状 [2, num_edges]
    batch : torch.Tensor
        批次索引，形状 [num_nodes]，指示每个节点属于哪个分子
    llm_emb : torch.Tensor
        LLM嵌入，形状 [batch_size, llm_dim]
    unimol_emb : torch.Tensor
        UniMol嵌入，形状 [batch_size, unimol_dim] (512维，来自DrugCLIP)
    
    输出:
    ----
    torch.Tensor
        分子表示向量，形状 [batch_size, 192]
    """
    
    def __init__(self, 
                 num_node_features,
                 hidden_channels=64,
                 llm_dim=768,
                 unimol_dim=512,
                 proj_dim=64,
                 num_gcn_layers=6,
                 dropout=0.2):
        super(GCN_LLM_UniMol_Encoder, self).__init__()
        
        self.num_gcn_layers = num_gcn_layers
        self.hidden_channels = hidden_channels
        self.proj_dim = proj_dim
        
        # ============================================
        # 1. 投影层：将LLM和UniMol嵌入投影到统一维度
        # ============================================
        
        # LLM投影层: 768维 → 64维
        self.llm_proj = nn.Sequential(
            nn.Linear(llm_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # UniMol投影层: 512维 → 64维 (DrugCLIP输出)
        self.unimol_proj = nn.Sequential(
            nn.Linear(unimol_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ============================================
        # 2. GCN卷积层：6层图卷积网络
        # ============================================
        
        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 第1层GCN: 输入为原始节点特征
        # 输入维度: num_node_features
        # 输出维度: hidden_channels (64)
        self.gcn_layers.append(
            GCNConv(num_node_features, hidden_channels)
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # 第2-6层GCN: 输入为拼接后的特征
        # 输入维度: hidden_channels * 3 (192) = 64(GCN) + 64(LLM) + 64(UniMol)
        # 输出维度: hidden_channels (64)
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(
                GCNConv(hidden_channels * 3, hidden_channels)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        self.dropout = nn.Dropout(dropout)
        
        # 最终输出维度: hidden_channels * 3 = 192
        self.output_dim = hidden_channels * 3
    
    def forward(self, x, edge_index, batch, llm_emb, unimol_emb):
        """
        前向传播
        
        详细流程:
        --------
        1. 投影LLM和UniMol嵌入到64维
        2. 将投影后的全局向量复制到每个节点
        3. 第1层GCN处理原始节点特征
        4. 拼接GCN输出与全局向量 (64+64+64=192)
        5. 第2-6层GCN处理拼接后的192维特征
        6. 每层后都重新拼接，保持192维
        7. 全局平均池化得到分子级表示
        """
        
        # ============================================
        # 步骤1: 投影全局向量
        # ============================================
        # LLM: [batch_size, 768] → [batch_size, 64]
        llm_proj = self.llm_proj(llm_emb)
        
        # UniMol: [batch_size, 512] → [batch_size, 64] (DrugCLIP输出)
        unimol_proj = self.unimol_proj(unimol_emb)
        
        # ============================================
        # 步骤2: 将全局向量复制到每个节点
        # ============================================
        # batch是一个索引向量，指示每个节点属于哪个分子
        # 例如: batch = [0,0,0,1,1,2,2,2,2] 表示前3个节点属于分子0，接下来2个属于分子1...
        
        # 使用batch索引将分子级特征广播到节点级
        # llm_proj[batch]: [num_nodes, 64]
        # unimol_proj[batch]: [num_nodes, 64]
        llm_expanded = llm_proj[batch]
        unimol_expanded = unimol_proj[batch]
        
        # ============================================
        # 步骤3: 第1层GCN
        # ============================================
        # 输入: 原始节点特征 x [num_nodes, num_node_features]
        # 输出: h [num_nodes, 64]
        h = self.gcn_layers[0](x, edge_index)
        h = self.batch_norms[0](h)
        h = F.relu(h)
        h = self.dropout(h)
        
        # 拼接: [64(GCN) + 64(LLM) + 64(UniMol)] = [num_nodes, 192]
        h = torch.cat([h, llm_expanded, unimol_expanded], dim=1)
        
        # ============================================
        # 步骤4: 第2-6层GCN（每层都拼接）
        # ============================================
        for i in range(1, self.num_gcn_layers):
            # 输入: h [num_nodes, 192]
            # GCN处理: [num_nodes, 192] → [num_nodes, 64]
            h = self.gcn_layers[i](h, edge_index)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            h = self.dropout(h)
            
            # 再次拼接: [64(GCN) + 64(LLM) + 64(UniMol)] = [num_nodes, 192]
            h = torch.cat([h, llm_expanded, unimol_expanded], dim=1)
        
        # ============================================
        # 步骤5: 全局平均池化
        # ============================================
        # 将每个分子的所有节点特征取平均
        # 输入: h [num_nodes, 192]
        # 输出: h_mol [batch_size, 192]
        h_mol = global_mean_pool(h, batch)
        
        return h_mol
    
    def get_output_dim(self):
        """返回输出维度"""
        return self.output_dim


class GCN_LLM_UniMol_Classifier(nn.Module):
    """
    基于GCN-LLM-UniMol编码器的分类器
    
    在编码器基础上添加分类头，用于分子活性预测任务
    """
    
    def __init__(self,
                 num_node_features,
                 num_classes=2,
                 hidden_channels=64,
                 llm_dim=768,
                 unimol_dim=128,
                 proj_dim=64,
                 num_gcn_layers=6,
                 dropout=0.2):
        super(GCN_LLM_UniMol_Classifier, self).__init__()
        
        # 编码器
        self.encoder = GCN_LLM_UniMol_Encoder(
            num_node_features=num_node_features,
            hidden_channels=hidden_channels,
            llm_dim=llm_dim,
            unimol_dim=unimol_dim,
            proj_dim=proj_dim,
            num_gcn_layers=num_gcn_layers,
            dropout=dropout
        )
        
        # 分类头
        encoder_output_dim = self.encoder.get_output_dim()  # 192
        
        self.classifier = nn.Sequential(
            nn.Linear(encoder_output_dim, encoder_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_output_dim, num_classes)
        )
    
    def forward(self, x, edge_index, batch, llm_emb, unimol_emb):
        """
        前向传播
        
        返回:
        ----
        logits : torch.Tensor
            分类logits，形状 [batch_size, num_classes]
        """
        # 获取分子表示
        h_mol = self.encoder(x, edge_index, batch, llm_emb, unimol_emb)
        
        # 分类
        logits = self.classifier(h_mol)
        
        return logits
    
    def get_embedding(self, x, edge_index, batch, llm_emb, unimol_emb):
        """
        获取分子嵌入（用于下游任务或可视化）
        
        返回:
        ----
        torch.Tensor
            分子嵌入，形状 [batch_size, 192]
        """
        with torch.no_grad():
            h_mol = self.encoder(x, edge_index, batch, llm_emb, unimol_emb)
        return h_mol
    
    def get_molecule_embeddings(self, x, edge_index, batch, llm_emb, unimol_emb):
        """
        获取分子嵌入（用于对比学习训练）
        
        与get_embedding的区别：不使用no_grad，支持梯度回传
        
        返回:
        ----
        torch.Tensor
            分子嵌入，形状 [batch_size, 192]
        """
        h_mol = self.encoder(x, edge_index, batch, llm_emb, unimol_emb)
        return h_mol


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    """
    测试代码：验证模型的维度变化
    """
    
    print("="*60)
    print("GCN-LLM-UniMol 编码器测试")
    print("="*60)
    
    # 模拟数据
    batch_size = 4
    num_nodes = 20  # 总节点数（4个分子的原子总数）
    num_node_features = 139  # 节点特征维度
    num_edges = 40
    
    # 创建模拟输入
    x = torch.randn(num_nodes, num_node_features)  # 节点特征
    edge_index = torch.randint(0, num_nodes, (2, num_edges))  # 边索引
    batch = torch.tensor([0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3])  # 批次索引
    llm_emb = torch.randn(batch_size, 768)  # LLM嵌入
    unimol_emb = torch.randn(batch_size, 512)  # UniMol嵌入 (DrugCLIP输出512维)
    
    print(f"\n输入维度:")
    print(f"  节点特征 x: {x.shape}")
    print(f"  边索引 edge_index: {edge_index.shape}")
    print(f"  批次索引 batch: {batch.shape}")
    print(f"  LLM嵌入: {llm_emb.shape}")
    print(f"  UniMol嵌入: {unimol_emb.shape}")
    
    # 创建编码器
    encoder = GCN_LLM_UniMol_Encoder(
        num_node_features=num_node_features,
        hidden_channels=64,
        llm_dim=768,
        unimol_dim=128,
        proj_dim=64,
        num_gcn_layers=6,
        dropout=0.2
    )
    
    # 前向传播
    h_mol = encoder(x, edge_index, batch, llm_emb, unimol_emb)
    
    print(f"\n输出维度:")
    print(f"  分子表示 h_mol: {h_mol.shape}")
    print(f"  预期维度: [batch_size={batch_size}, 192]")
    
    # 验证维度
    assert h_mol.shape == (batch_size, 192), "输出维度不正确！"
    print(f"\n✓ 维度验证通过！")
    
    # 测试分类器
    print(f"\n" + "="*60)
    print("测试分类器")
    print("="*60)
    
    classifier = GCN_LLM_UniMol_Classifier(
        num_node_features=num_node_features,
        num_classes=2
    )
    
    logits = classifier(x, edge_index, batch, llm_emb, unimol_emb)
    print(f"\n分类logits: {logits.shape}")
    print(f"预期维度: [batch_size={batch_size}, num_classes=2]")
    
    assert logits.shape == (batch_size, 2), "分类器输出维度不正确！"
    print(f"\n✓ 分类器测试通过！")
    
    # 模型参数统计
    print(f"\n" + "="*60)
    print("模型参数统计")
    print("="*60)
    
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"\n编码器总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    classifier_params = sum(p.numel() for p in classifier.parameters())
    print(f"\n分类器总参数量: {classifier_params:,}")
    
    print(f"\n" + "="*60)
    print("测试完成！")
    print("="*60)
