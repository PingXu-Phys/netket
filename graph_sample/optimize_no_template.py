"""
optimize_no 调用模版
====================

前提
----
- rdm        : 当前基底的 1-RDM，(n, n) 实对称矩阵
- H          : 同一基底的 FermionOperator2nd（可选）
- hop        : 同一基底的单体 hopping 矩阵 tᵢⱼ，(n, n)（可选）
- J          : 每个格点的 Kondo 耦合强度，(n,) 数组或标量（可选）

所有矩阵必须在同一个基底下表达。代码内部会自动对 rdm 做对角化，
得到自然轨道 U₀ 和自然轨道占据数，再在各块内优化旋转矩阵 V，
最终变换为 U = U₀ @ V。

输出中最重要的量
----------------
site_to_optimized_orbital : (n, n)
    列 j = 优化后的第 j 个轨道（用当前基底展开）
    这就是你需要的基底变换矩阵，可直接用来旋转任意算符

optimized_hamiltonian      : FermionOperator2nd（仅当传入 H 时有效）
    H 在优化后基底下的表示
"""

import numpy as np
from optimize_no import NOOptimizationConfig, optimize_no_basis

# ──────────────────────────────────────────────
# 1. 准备输入（替换成你自己的数据）
# ──────────────────────────────────────────────

# 1-RDM：当前基底，实对称，(n, n)
rdm: np.ndarray = ...          # 你的 1-RDM

# 单体 hopping 矩阵：当前基底，实对称，(n, n)
# 如果有完整的 FermionOperator2nd，直接传 H=H 即可，hopping_matrix 可省略
hop: np.ndarray = ...          # 你的 tᵢⱼ 矩阵（或不传）

# Kondo 耦合强度：每个格点一个值，(n,)
# 不需要 Kondo 项时传 None，会自动关闭 C_K
J: np.ndarray = ...            # 你的 Jᵢ 数组（或 None）

# ──────────────────────────────────────────────
# 2. 配置（按需调整）
# ──────────────────────────────────────────────

config = NOOptimizationConfig(
    # ----- 损失权重 -----
    lambda_occupancy = 0.1,   # C_γ 正则项，保持接近 NO 基底，不宜过大
    lambda_hopping   = 1.0,   # C_t hopping 稀疏性
    lambda_kondo     = 2.0,   # C_K Kondo footprint × 相空间（主要优化目标）
    lambda_locality  = 0.1,   # C_loc 轨道局域性正则

    # ----- 块划分 -----
    filled_tol     = 0.05,    # n > 1-filled_tol → near-full 块（冻结）
    empty_tol      = 0.05,    # n < empty_tol    → near-empty 块（冻结）
    degeneracy_tol = 0.05,    # 同块内相邻占据数之差超过此值时再拆块

    # ----- 优化器 -----
    optimizer_name = "adam",
    learning_rate  = 1e-2,
    gradient_clip  = 1.0,
    n_steps        = 300,
    log_every      = 50,

    # ----- 其他 -----
    structure_metric    = "hybrid",   # "participation" | "decay" | "hybrid"
    hamiltonian_cutoff  = 1e-10,      # 旋转 H 时截断小项（传 H 时有效）
    real_orbitals       = True,       # 实数问题保持 True（默认）
)

# ──────────────────────────────────────────────
# 3. 运行优化
# ──────────────────────────────────────────────

result = optimize_no_basis(
    rdm             = rdm,
    hopping_matrix  = hop,    # 有完整 H 时改为 H=H，此行删掉
    exchange_profile= J,
    config          = config,
    # H             = H,      # 传入时会自动提取 hopping 并输出旋转后的 H
)

# ──────────────────────────────────────────────
# 4. 读取结果
# ──────────────────────────────────────────────

# 基底变换矩阵：U = U₀ @ V，列 j = 第 j 个优化轨道（当前基底下展开）
U = result["site_to_optimized_orbital"]          # (n, n)

# 自然轨道信息（优化前）
U0  = result["natural_orbitals"]                 # (n, n)，列 = 自然轨道
occ = result["natural_occupations"]              # (n,)，降序排列

# 优化后的占据数和 1-RDM
occ_opt = result["optimized_diag_occupations"]   # (n,)
rdm_opt = result["optimized_rdm"]                # (n, n)

# 优化后的 hopping 矩阵（如果传入了 hopping_matrix）
hop_opt = result["optimized_hopping_matrix"]     # (n, n) or None

# 有效散射矩阵 W_ab = Λ_ab · Ξ_ab（如果传入了 exchange_profile）
W = result["effective_scattering"]               # (n, n) or None

# 旋转后的 Hamiltonian（仅当传入 H 时）
H_opt = result["optimized_hamiltonian"]          # FermionOperator2nd or None

# loss 历史
print("initial loss:", result["initial_metrics"]["loss"])
print("final   loss:", result["final_metrics"]["loss"])
print("loss breakdown:", result["final_loss_terms"])
for entry in result["history"]:
    print(f"  step {entry['step']:4d}  loss={entry.get('optax_loss', entry['loss']):.6f}"
          f"  kondo={entry['kondo_structure_cost']:.4f}")

# ──────────────────────────────────────────────
# 5. 快速诊断（可选）
# ──────────────────────────────────────────────

from optimize_no import evaluate_basis_metrics

# 在任意旋转 U_test 下评估所有指标
diag = evaluate_basis_metrics(
    rdm,
    U,                        # 替换成任意你想评估的变换矩阵
    hopping_matrix = hop,
    exchange_profile = J,
)
print("occupation offdiag cost:", diag["occupation_offdiag_cost"])
print("hopping structure cost: ", diag["hopping_metrics"]["structure_cost"])
print("kondo structure cost:   ", diag["kondo_metrics"]["structure_cost"])
