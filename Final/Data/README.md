# Data — 数据获取与清洗

本模块只做两件事：**从 ChEMBL 下载活性数据** → **清洗成统一的 CSV**。
得到的 `output/processed_data.csv` 是后续所有嵌入提取与训练的输入。

> 嵌入（768 维 ChemBERTa + 512 维 UniMol）由 `Final/prepare_complete_embeddings.py`
> 生成，不在本模块内完成。

## 目录结构

```
Data/
├── 1_download_chembl_data.py    # 步骤1：按靶点 + 同源蛋白下载活性数据
├── 2_process_molecules.py       # 步骤2：SMILES 校验、去重、理化性质、打标签
├── run_all.py                   # 一键执行步骤1 + 步骤2
├── utils/
│   ├── chembl_downloader.py     # ChEMBL API 封装（靶点搜索 / 同源比对 / 活性拉取）
│   └── molecule_processor.py    # RDKit 清洗与描述符计算
└── output/
    ├── raw_data.csv             # 原始下载结果
    ├── processed_data.csv       # 清洗后数据（训练输入）
    └── metadata.pkl             # 靶点、来源、样本数等元信息
```

## 快速开始

```bash
cd Final/Data

# 一键（推荐）
python run_all.py --target HDAC1

# 或分步
python 1_download_chembl_data.py --target HDAC1 --output output/raw_data.csv
python 2_process_molecules.py --input output/raw_data.csv --output output/processed_data.csv
```

常用参数：

| 脚本 | 参数 | 默认 | 说明 |
|---|---|---|---|
| `run_all.py` | `--target` | 必填 | 靶点名称，如 `HDAC1`、`Xanthine dehydrogenase/oxidase` |
| | `--similarity` | 0.85 | 同源蛋白序列相似度阈值 |
| | `--max_compounds` | 5000 | 最大化合物数 |
| `1_download...` | `--activity_types` | IC50 Ki Kd | 拉取的活性类型 |
| `2_process...` | `--activity_threshold` | 1000 | 活性阈值（nM），≤ 阈值记为 1 |
| | `--min_mw / --max_mw` | 150 / 800 | 分子量过滤 |
| | `--min_logp / --max_logp` | -2 / 7 | LogP 过滤 |
| | `--max_atoms` | 100 | 原子数上限 |

## 输出格式

`processed_data.csv`（每行一个分子）共 18 列：

| 列 | 说明 |
|---|---|
| `molecule_id`, `smiles` | ChEMBL ID 与标准化 SMILES |
| `activity_type`, `activity_value`, `pchembl` | 原始活性类型 / 数值 / pChEMBL |
| `target_name`, `target_id` | 靶点名称与 ChEMBL 靶点 ID |
| `mol_weight`, `logp`, `num_atoms`, `num_bonds`, `num_rings`, `num_aromatic_rings`, `tpsa`, `num_hbd`, `num_hba`, `num_rotatable_bonds` | RDKit 理化描述符 |
| `label` | 二分类活性标签（1 = 活性） |

仓库中已附带一份 XOD（PDB 3NVW，靶点 `CHEMBL1929`）的数据：
`output/processed_data.csv`，共 **630 个分子**。

## 下一步

```bash
cd ..   # 回到 Final/

# 提取 ChemBERTa(768) + DrugCLIP(512) 嵌入，并打包成训练集
python prepare_complete_embeddings.py \
    --data Data/output/processed_data.csv \
    --drugclip-checkpoint ../checkpoints/drugclip.pt \
    --pdb-id 3NVW \
    --output complete_embeddings

# 训练
python train.py --data complete_embeddings/training_data_complete.pkl --output results
```

## 注意

- 需要联网访问 ChEMBL API；`chembl_webresource_client` 偶发超时，重跑即可（已下载部分会落盘）。
- 同源蛋白比对用 BioPython 的 pairwise 对齐，靶点较多时较慢，可调低 `--max_compounds`。
- ChEMBL 返回的活性值单位不统一，`2_process_molecules.py` 统一换算为 nM 后再按阈值打标签。
