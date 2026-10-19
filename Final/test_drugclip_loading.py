# -*- coding: utf-8 -*-
# Copyright (c) 2026 GCN-LLM-UniMol 项目作者（个人）. All Rights Reserved.
# 本项目基于 DrugCLIP 与 GCN-LLM 构建，仅限非商业用途；
# 未经作者书面许可，不得用于论文、学位论文、专利等学术发表。完整条款见根目录 LICENSE。
"""
DrugCLIP 权重加载自检
====================

在任意目录执行均可：

    python Final/test_drugclip_loading.py

成功标志：打印字典大小、模型参数规模，并输出 512 维分子嵌入。
"""

import os
import sys

FINAL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(FINAL_DIR)
sys.path.insert(0, FINAL_DIR)

print("=" * 70)
print("DrugCLIP 权重加载自检")
print("=" * 70)

CHECKPOINT = os.path.join(ROOT, "checkpoints", "drugclip.pt")
MOL_DICT = os.path.join(ROOT, "DrugCLIP-main", "data", "dict_mol.txt")
PKT_DICT = os.path.join(ROOT, "DrugCLIP-main", "data", "dict_pkt.txt")

for path, label in [(CHECKPOINT, "DrugCLIP 权重"),
                    (MOL_DICT, "分子词表"),
                    (PKT_DICT, "蛋白词表")]:
    if not os.path.exists(path):
        sys.exit(f"✗ 缺少{label}: {path}\n  请先按 README「模型权重」准备文件")

try:
    import torch
    from unimol_feature_extractor import UniMolFeatureExtractor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    extractor = UniMolFeatureExtractor(
        drugclip_checkpoint_path=CHECKPOINT,
        mol_dict_path=MOL_DICT,
        pocket_dict_path=PKT_DICT,
        device=device,
    )

    if extractor.model is None:
        sys.exit("✗ DrugCLIP 加载失败（通常是没有安装 unicore，见 README「环境」）")

    print(f"\n✓ 模型加载成功: {type(extractor.model).__name__}")

    print("\n测试分子嵌入提取（CCO）...")
    emb = extractor.extract_molecule_embedding("CCO")
    print(f"✓ 嵌入维度: {emb.shape}（应为 (512,)）")

    print("\n✓ 自检通过")

except Exception as exc:  # noqa: BLE001
    print(f"\n✗ 错误: {exc}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
