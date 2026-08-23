#!/bin/bash
# =============================================================
# TwiBot-22 批量实验脚本 (E2E 端到端训练)
# =============================================================
# 用法:
#   ./run_experiments.sh T          # T 敏感性 (E2E)
#   ./run_experiments.sh L          # L 敏感性 (E2E, 无需矩阵)
#   ./run_experiments.sh missing    # 时间缺失敏感性 (E2E)
#   ./run_experiments.sh flat       # Flat-Static 消融 (两阶段)
#   ./run_experiments.sh phase1     # T + L 敏感性 (~5h)
#   ./run_experiments.sh phase2     # 时间缺失 + Flat-Static (~8h)
#   ./run_experiments.sh all        # 全部
#
# ⚠️ 在运行前，请先完成 Phase 0 (§1.4):
#    1. E2E 验证 (seed=42)
#    2. 调 ts-lr-scale → BEST_SCALE
#    3. 调 align-beta → BEST_BETA
#    4. 最佳配置 5 种子
# =============================================================
set -e

WORK=/root/autodl-tmp/twibot22
DATA=/root/autodl-tmp/twibot22/data
CODE=/root/autodl-tmp/twibot22/code
cd "$CODE"

SEEDS="42 123 456 789 2024"

# ============================================================
# ⬇️ Phase 0 调参结果 — 运行前务必替换为实际最佳值 ⬇️
# ============================================================
BEST_SCALE=0.1   # ← Phase 0 步骤4 的最佳 ts-lr-scale
BEST_BETA=0.0    # ← Phase 0 步骤5 的最佳 align-beta

log_result() {
    echo "[RESULT] $(date +%H:%M) $1" | tee -a "$WORK/experiment_results.log"
}

# ============================================================
# 参数敏感性: 序列长度 T ∈ {8, 32, 64} (E2E)
# T=16 结果来自 Phase 0 步骤6
# ============================================================
run_T_sensitivity() {
    echo ""
    echo "########################################"
    echo "#  E2E 参数敏感性: T ∈ {8, 32, 64}     #"
    echo "########################################"

    # T=8: 从默认 T=16 矩阵截断, 不需要生成新矩阵
    echo ""
    echo "====== T=8 (E2E, 自动截断) ======"
    for SEED in $SEEDS; do
        echo "--- T=8, seed=${SEED} ---"
        python train_twibot22.py \
            --work-dir "$WORK" --e2e \
            --seq-len 8 --ts-layers 2 --ts-nhead 2 \
            --ts-lr-scale "$BEST_SCALE" --align-beta "$BEST_BETA" --align-temp 0.5 \
            --epochs 200 --patience 30 --seed "$SEED" \
            --save-suffix "_e2e_T8_s${SEED}"
        log_result "E2E T=8 seed=${SEED} 完成"
    done

    # T=32, 64: 需要先生成矩阵
    for T in 32 64; do
        MATRIX_DIR="$WORK/tmp_twibot22_T${T}"
        if [ ! -f "$MATRIX_DIR/train_matrices.npz" ]; then
            echo ""
            echo "====== T=${T}: 生成矩阵 (feature_extraction) ======"
            python feature_extraction_twibot22.py --mode pseudo \
                --data-dir "$DATA" --work-dir "$WORK" \
                --seq-len "$T" --num-layers 2
        else
            echo "====== T=${T}: 矩阵已存在, 跳过生成 ======"
        fi

        echo "====== T=${T}: E2E 训练 (5 seeds) ======"
        for SEED in $SEEDS; do
            echo "--- T=${T}, seed=${SEED} ---"
            python train_twibot22.py \
                --work-dir "$WORK" --e2e \
                --e2e-matrix-dir "$MATRIX_DIR" \
                --seq-len "$T" --ts-layers 2 --ts-nhead 2 \
                --ts-lr-scale "$BEST_SCALE" --align-beta "$BEST_BETA" --align-temp 0.5 \
                --epochs 200 --patience 30 --seed "$SEED" \
                --save-suffix "_e2e_T${T}_s${SEED}"
            log_result "E2E T=${T} seed=${SEED} 完成"
        done
    done
    echo ">>> T 敏感性 (E2E) 完成！"
}

# ============================================================
# 参数敏感性: 编码层数 L ∈ {1, 4} (E2E)
# L=2 结果来自 Phase 0 步骤6
# 优势: 改 --ts-layers 即可, 完全不需要重新生成矩阵或嵌入!
# ============================================================
run_L_sensitivity() {
    echo ""
    echo "########################################"
    echo "#  E2E 参数敏感性: L ∈ {1, 4}          #"
    echo "#  (无需额外矩阵, 直接改 --ts-layers)   #"
    echo "########################################"

    for L in 1 4; do
        echo ""
        echo "====== L=${L}: E2E 训练 (5 seeds) ======"
        for SEED in $SEEDS; do
            echo "--- L=${L}, seed=${SEED} ---"
            python train_twibot22.py \
                --work-dir "$WORK" --e2e \
                --seq-len 16 --ts-layers "$L" --ts-nhead 2 \
                --ts-lr-scale "$BEST_SCALE" --align-beta "$BEST_BETA" --align-temp 0.5 \
                --epochs 200 --patience 30 --seed "$SEED" \
                --save-suffix "_e2e_L${L}_s${SEED}"
            log_result "E2E L=${L} seed=${SEED} 完成"
        done
    done
    echo ">>> L 敏感性 (E2E) 完成！"
}

# ============================================================
# 时间缺失敏感性: ρ ∈ {25%, 50%, 75%, 100%} (E2E)
# Pseudo-TS 行使用 Phase 0 结果 (不受 ρ 影响)
# Real-TS 行需要生成不同 ρ 的矩阵, 然后用 E2E 训练
# ============================================================
run_missing_sensitivity() {
    echo ""
    echo "########################################"
    echo "#  E2E 时间缺失: ρ ∈ {25%,50%,75%,100%} #"
    echo "########################################"

    # ρ=100% (无丢弃)
    MATRIX_DIR_100="$WORK/tmp_twibot22_real"
    if [ ! -f "$MATRIX_DIR_100/train_matrices.npz" ]; then
        echo "====== ρ=100%: 构建 Real-TS 矩阵 ======"
        python feature_extraction_twibot22_real.py --mode real \
            --data-dir "$DATA" --work-dir "$WORK" --drop-rate 0.0
    fi
    echo "====== ρ=100%: E2E 训练 (5 seeds) ======"
    for SEED in $SEEDS; do
        echo "--- ρ=100%, seed=${SEED} ---"
        python train_twibot22.py \
            --work-dir "$WORK" --e2e \
            --e2e-matrix-dir "$MATRIX_DIR_100" \
            --seq-len 16 --ts-layers 2 --ts-nhead 2 \
            --ts-lr-scale "$BEST_SCALE" --align-beta "$BEST_BETA" --align-temp 0.5 \
            --epochs 200 --patience 30 --seed "$SEED" \
            --save-suffix "_e2e_real_rho100_s${SEED}"
        log_result "E2E Real-TS ρ=100% seed=${SEED} 完成"
    done

    # ρ=75%, 50%, 25%
    for DROP in 0.25 0.50 0.75; do
        RHO_PCT=$(python3 -c "print(int((1-${DROP})*100))")
        DROP_TAG=$(python3 -c "print(f'_drop{${DROP}:.2f}'.replace('.',''))")
        MATRIX_DIR="$WORK/tmp_twibot22_real${DROP_TAG}"

        if [ ! -f "$MATRIX_DIR/train_matrices.npz" ]; then
            echo ""
            echo "====== ρ=${RHO_PCT}% (drop=${DROP}): 构建 Real-TS 矩阵 ======"
            python feature_extraction_twibot22_real.py --mode real \
                --data-dir "$DATA" --work-dir "$WORK" \
                --drop-rate "$DROP"
        fi

        echo "====== ρ=${RHO_PCT}%: E2E 训练 (5 seeds) ======"
        for SEED in $SEEDS; do
            echo "--- ρ=${RHO_PCT}%, seed=${SEED} ---"
            python train_twibot22.py \
                --work-dir "$WORK" --e2e \
                --e2e-matrix-dir "$MATRIX_DIR" \
                --seq-len 16 --ts-layers 2 --ts-nhead 2 \
                --ts-lr-scale "$BEST_SCALE" --align-beta "$BEST_BETA" --align-temp 0.5 \
                --epochs 200 --patience 30 --seed "$SEED" \
                --save-suffix "_e2e_real_rho${RHO_PCT}_s${SEED}"
            log_result "E2E Real-TS ρ=${RHO_PCT}% seed=${SEED} 完成"
        done
    done
    echo ">>> 时间缺失敏感性 (E2E) 完成！"
}

# ============================================================
# Flat-Static 消融 (两阶段 — 无 Transformer, 无法 E2E)
# ============================================================
run_flat_static() {
    echo ""
    echo "########################################"
    echo "#  Flat-Static 消融 (两阶段)            #"
    echo "########################################"

    python flat_static_embed.py "$WORK" --dataset twibot22

    NPZ="$WORK/feature_model_outputs/twibot22_transformer_vectors_flat.npz"
    if [ ! -f "$NPZ" ]; then
        echo "错误: Flat-Static 嵌入未生成"
        exit 1
    fi

    for SEED in $SEEDS; do
        echo "--- Flat-Static, seed=${SEED} ---"
        python train_twibot22.py \
            --work-dir "$WORK" --model v2 \
            --temporal-npz "$NPZ" \
            --epochs 200 --patience 30 --seed "$SEED" \
            --save-suffix "_flat_s${SEED}"
        log_result "Flat-Static seed=${SEED} 完成"
    done
    echo ">>> Flat-Static 消融完成！"
}

# ============================================================
# 主入口
# ============================================================
echo "=========================================="
echo "  TwiBot-22 批量实验 (E2E 模式)"
echo "  开始时间: $(date)"
echo "  BEST_SCALE=${BEST_SCALE}, BEST_BETA=${BEST_BETA}"
echo "=========================================="

case "${1:-help}" in
    T)       run_T_sensitivity ;;
    L)       run_L_sensitivity ;;
    missing) run_missing_sensitivity ;;
    flat)    run_flat_static ;;
    phase1)  run_T_sensitivity; run_L_sensitivity ;;
    phase2)  run_missing_sensitivity; run_flat_static ;;
    all)     run_T_sensitivity; run_L_sensitivity; run_missing_sensitivity; run_flat_static ;;
    *)
        echo "用法: $0 {T|L|missing|flat|phase1|phase2|all}"
        echo ""
        echo "  T       - E2E 序列长度敏感性 T∈{8,32,64}"
        echo "  L       - E2E 编码层数敏感性 L∈{1,4} (无需额外矩阵!)"
        echo "  missing - E2E 时间缺失敏感性 ρ∈{25%,50%,75%,100%}"
        echo "  flat    - Flat-Static 消融 (两阶段)"
        echo "  phase1  - T + L (~5h)"
        echo "  phase2  - missing + flat (~8h)"
        echo "  all     - 全部 (~13h)"
        echo ""
        echo "⚠️ 运行前请修改脚本顶部的 BEST_SCALE 和 BEST_BETA 为 Phase 0 最佳值!"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "  实验完成！结束时间: $(date)"
echo "=========================================="
echo ">>> 检查结果: grep -A 5 '测试集结果' $WORK/experiment_results.log"
