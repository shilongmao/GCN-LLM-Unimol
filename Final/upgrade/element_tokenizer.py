# -*- coding: utf-8 -*-
# Copyright (c) 2026 GCN-LLM-UniMol 项目作者（个人）. All Rights Reserved.
# 本项目基于 DrugCLIP 与 GCN-LLM 构建，仅限非商业用途；
# 未经作者书面许可，不得用于论文、学位论文、专利等学术发表。完整条款见根目录 LICENSE。
"""
element_tokenizer.py — 修复 GCN-LLM-UniMol 最致命的缺陷
=========================================================

问题
----
`Final/unimol_feature_extractor.py` 里：

    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]   # 6, 7, 8, ...
    atoms_list = [self.mol_dictionary.index(str(a))
                  if str(a) in self.mol_dictionary.indices
                  else self.mol_dictionary.unk() for a in atoms]

而 `DrugCLIP-main/data/dict_mol.txt` 的 token 是**元素符号**：

    [PAD] [CLS] [SEP] [UNK] C N O S H Cl F Br I Si P B Na K
    Al Ca Sn As Hg Fe Zn Cr Se Gd Au Li [MASK]

`str(6) == "6"` 永远不在字典里 → **每一个原子都退化为 [UNK]**。
连带 `edge_type = a_i * V + a_j` 变成常数，GBF 的 per-edge-type 仿射失效。

净效果：Uni-Mol 塔看到的是「一堆匿名点 + 距离矩阵」，碳和氯完全无法区分。

修复
----
用**元素符号**而不是原子序数去查字典。本模块提供：

    atomic_num_to_symbol(z)   原子序数 → 标准元素符号
    normalize_element(sym)    PDB/RDKit 各种写法的符号归一化（"CL"→"Cl", "C"→"C"）
    AtomTokenizer             holds a Dictionary-like token→idx map, with fallback to unk

用法（替换原文件里的两处）
--------------------------
分子侧（unimol_feature_extractor.py:224 + :343）::

    # 原来
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    ...
    atoms_list = [self.mol_dictionary.index(str(a)) if str(a) in ... else ...]

    # 改为
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    ...
    atoms_list = [tok(s) for s in symbols]

蛋白侧（:278 + :439）::

    # 原来
    element = atom.element.upper()
    atomic_num = self._element_to_atomic_num(element)

    # 改为
    element = normalize_element(atom.element)

自测
----
    python element_tokenizer.py --dict ../DrugCLIP-main/data/dict_mol.txt
    python element_tokenizer.py --selftest        # 不需要字典文件
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence

# 仓库根目录 = 本文件（Final/upgrade/）上两级
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_DICT = os.path.join(_REPO_ROOT, "DrugCLIP-main", "data", "dict_mol.txt")


# ---------------------------------------------------------------------------
# 1. 原子序数 -> 标准元素符号（1..118，IUPAC 常规拼写）
# ---------------------------------------------------------------------------
_SYMBOLS: List[str] = [
    "",   # 0  占位
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

_SYMBOL_TO_Z: Dict[str, int] = {
    s: i for i, s in enumerate(_SYMBOLS) if s
}


def atomic_num_to_symbol(atomic_num: int) -> str:
    """6 -> 'C'，17 -> 'Cl'，35 -> 'Br'。越界返回 'X'。"""
    try:
        z = int(atomic_num)
    except (TypeError, ValueError):
        return "X"
    if 1 <= z < len(_SYMBOLS):
        return _SYMBOLS[z]
    return "X"


def normalize_element(raw: Optional[str], name_fallback: Optional[str] = None) -> str:
    """把 PDB / Biopython / RDKit 各种元素写法统一成字典里的形式。

    'CL' -> 'Cl', 'cl' -> 'Cl', ' C' -> 'C', 'BR' -> 'Br', 'NA' -> 'Na'

    PDB 的 element 列有时为空，Biopython 会从原子名猜。若 ``atom.element``
    为空/异常，可用 ``name_fallback``（如原子名 "CA"）兜底：
    这里对两字母名称做歧义处理 —— 若首字母是大写、次字母也大写，
    优先当作两字母元素（CL/BR/NA），否则取首字母。
    """
    s = (raw or "").strip()
    if not s and name_fallback:
        s = (name_fallback or "").strip()
    if not s:
        return "X"
    s = s.strip()
    # 已经是标准形式
    if s in _SYMBOL_TO_Z:
        return s
    # 全大写 / 全小写 / 混合 -> 首字母大写、其余小写
    cand = s[0].upper() + s[1:].lower()
    if cand in _SYMBOL_TO_Z:
        return cand
    # 单字母兜底
    if s[0].upper() in _SYMBOL_TO_Z:
        return s[0].upper()
    return "X"


# ---------------------------------------------------------------------------
# 2. 一个极简的 Dictionary 兼容层（不依赖 unicore）
# ---------------------------------------------------------------------------
class AtomTokenizer:
    """把元素符号映射成 DrugCLIP 字典里的 token id。

    与 unicore.data.Dictionary 的行为对齐：
        - 未命中的符号 -> ``unk()``
        - 可在末尾追加 [CLS]/[SEP]（DrugCLIP 的输入约定）

    参数
    ----
    tokens : 字典文件里按行读出的 token 列表
    """

    def __init__(self, tokens: Sequence[str]) -> None:
        self.tokens: List[str] = [t for t in tokens]
        self.indices: Dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        if "[UNK]" not in self.indices:
            raise ValueError("字典缺少 [UNK]")
        self._unk = self.indices["[UNK]"]
        self._cls = self.indices.get("[CLS]")
        self._sep = self.indices.get("[SEP]")

    # -- 构造 ---------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str) -> "AtomTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls([line.rstrip("\n") for line in f if line.strip()])

    # -- 查询 ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.tokens)

    def unk(self) -> int:
        return self._unk

    def index(self, token: str) -> int:
        return self.indices[token]

    def token_to_id(self, symbol: str) -> int:
        return self.indices.get(symbol, self._unk)

    def element_to_id(self, raw: str, name_fallback: Optional[str] = None) -> int:
        return self.token_to_id(normalize_element(raw, name_fallback))

    # -- 序列化 -------------------------------------------------------------
    def encode(
        self,
        symbols: Iterable[str],
        add_cls_sep: bool = True,
    ) -> List[int]:
        """符号序列 -> id 序列（默认首尾加 [CLS]/[SEP]）。"""
        ids = [self.token_to_id(s) for s in symbols]
        if add_cls_sep and self._cls is not None and self._sep is not None:
            ids = [self._cls] + ids + [self._sep]
        return ids

    def coverage(self, symbols: Iterable[str]) -> Dict[str, int]:
        """诊断：统计命中/未命中的符号（修复前应看到 unk 占满）。"""
        hit, miss = {}, {}
        for s in symbols:
            if s in self.indices:
                hit[s] = hit.get(s, 0) + 1
            else:
                miss[s] = miss.get(s, 0) + 1
        return {"n_hit_types": len(hit), "n_miss_types": len(miss),
                "n_hit": sum(hit.values()), "n_miss": sum(miss.values()),
                "missing": miss}


# ---------------------------------------------------------------------------
# 3. 自测
# ---------------------------------------------------------------------------
def _selftest() -> int:
    fails = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal fails
        if not cond:
            fails += 1
            print(f"  [FAIL] {msg}")
        else:
            print(f"  [ok]   {msg}")

    print("== atomic_num_to_symbol ==")
    check(atomic_num_to_symbol(1) == "H", "1 -> H")
    check(atomic_num_to_symbol(6) == "C", "6 -> C")
    check(atomic_num_to_symbol(7) == "N", "7 -> N")
    check(atomic_num_to_symbol(17) == "Cl", "17 -> Cl")
    check(atomic_num_to_symbol(35) == "Br", "35 -> Br")
    check(atomic_num_to_symbol(53) == "I", "53 -> I")
    check(atomic_num_to_symbol(999) == "X", "999 -> X")

    print("== normalize_element ==")
    check(normalize_element("CL") == "Cl", "'CL' -> 'Cl'")
    check(normalize_element("cl") == "Cl", "'cl' -> 'Cl'")
    check(normalize_element(" C") == "C", "' C' -> 'C'")
    check(normalize_element("BR") == "Br", "'BR' -> 'Br'")
    check(normalize_element("NA") == "Na", "'NA' -> 'Na'")
    check(normalize_element("C") == "C", "'C' -> 'C'")
    check(normalize_element("") == "X", "'' -> 'X'")
    check(normalize_element(None, "CA") == "Ca", "None + 'CA' -> 'Ca'")

    print("== AtomTokenizer（模拟 dict_mol.txt）==")
    fake = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "C", "N", "O", "S", "H",
            "Cl", "F", "Br", "I", "[MASK]"]
    tok = AtomTokenizer(fake)
    check(tok.token_to_id("C") == 4, "'C' -> 4")
    check(tok.token_to_id("Cl") == 9, "'Cl' -> 9")
    check(tok.token_to_id("P") == tok.unk(), "'P' 不在仿真字典 -> unk")

    # 模拟一个乙醇 + 氯苯的原子符号序列
    symbols = ["C", "C", "O", "C", "C", "C", "C", "C", "C", "Cl"]
    ids = tok.encode(symbols)
    check(ids[0] == 1 and ids[-1] == 2, "encode 加 [CLS]/[SEP]")
    cov = tok.coverage(symbols)
    check(cov["n_miss"] == 0, f"全部命中（miss={cov['n_miss']}）")

    print("== 复现「修复前」的行为，证明 bug ==")
    # 修复前：用原子序数去查字典
    old_ids = []
    nums = [6, 6, 8, 6, 6, 6, 6, 6, 6, 17]
    for a in nums:
        old_ids.append(tok.token_to_id(str(a)))   # str(6) == "6"
    n_unk = sum(1 for i in old_ids if i == tok.unk())
    check(n_unk == len(nums), f"修复前 {n_unk}/{len(nums)} 个原子退化为 [UNK]")

    print()
    print("全部通过" if fails == 0 else f"{fails} 项失败")
    return fails


def _check_real_dict(dict_path: str) -> int:
    """用仓库里真实的字典检查覆盖率。"""
    tok = AtomTokenizer.from_file(dict_path)
    print(f"字典: {dict_path}")
    print(f"  token 数: {len(tok)}")
    print(f"  元素种类: {len([t for t in tok.tokens if t.isalpha()])}")

    # 常见药物化学元素
    common = ["C", "N", "O", "S", "H", "P", "F", "Cl", "Br", "I", "B", "Si"]
    cov = tok.coverage(common)
    print(f"  常见元素命中: {len(common) - len(cov['missing'])}/{len(common)}")
    if cov["missing"]:
        print(f"  [!] 缺失: {sorted(cov['missing'])}")
    else:
        print("  [OK] 常见元素全覆盖")

    # 演示：C6H5Cl
    sym = ["C"] * 6 + ["H"] * 5 + ["Cl"]
    ids = tok.encode(sym)
    unk_rate = sum(1 for i in ids if i == tok.unk()) / len(ids)
    print(f"  氯苯 tokenize: {len(ids)} token, UNK 率 = {unk_rate:.1%}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="修复 Uni-Mol 原子 tokenization")
    ap.add_argument("--selftest", action="store_true", help="只跑内置自测")
    ap.add_argument("--dict", type=str,
                    default=_DEFAULT_DICT,
                    help="dict_mol.txt 路径")
    args = ap.parse_args()

    code = _selftest()
    if not args.selftest:
        import os
        if os.path.isfile(args.dict):
            print()
            code += _check_real_dict(args.dict)
        else:
            print(f"\n(跳过真实字典检查：{args.dict} 不存在)")
    sys.exit(0 if code == 0 else 1)
