#!/bin/bash
# ============================================================
# TwiBot-22 一键运行脚本 (AutoDL)
# ============================================================
# 用法:
#   chmod +x run_twibot22.sh
#   ./run_twibot22.sh
#
# 或分步运行:
#   ./run_twibot22.sh preprocess
#   ./run_twibot22.sh feature
#   ./run_twibot22.sh train
# ============================================================

set -e  # 遇到错误立即停止

# -------- 路径配置 --------
WORK_DIR="/root/autodl-tmp/twibot22"
DATA_DIR="${WORK_DIR}/data"
CODE_DIR="${WORK_DIR}/code"

echo "============================================"
echo "TwiBot-22 实验管道"
echo "工作目录: ${WORK_DIR}"
echo "数据目录: ${DATA_DIR}"
echo "代码目录: ${CODE_DIR}"
echo "============================================"

# -------- 环境检查 --------
check_env() {
    echo ""
    echo ">>> 检查环境..."
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
    echo ">>> 环境检查通过"
}

# -------- 阶段 1: 预处理 --------
run_preprocess() {
    echo ""
    echo "============================================"
    echo ">>> 阶段 1: 预处理 (生成 .pt 张量)"
    echo "============================================"

    cd "${CODE_DIR}"

    echo "--- 1.1 加载用户特征 ---"
    python preprocess_twibot22.py --stage 1 --data-dir "${DATA_DIR}" --save-dir "${WORK_DIR}/saved_data/twibot22_data"

    echo "--- 1.2 收集推文文本 ---"
    python preprocess_twibot22.py --stage 2 --data-dir "${DATA_DIR}" --save-dir "${WORK_DIR}/saved_data/twibot22_data"

    echo "--- 1.3 RoBERTa 编码 ---"
    python preprocess_twibot22.py --stage 3 --data-dir "${DATA_DIR}" --save-dir "${WORK_DIR}/saved_data/twibot22_data"

    echo "--- 1.4 构建图 ---"
    python preprocess_twibot22.py --stage 4 --data-dir "${DATA_DIR}" --save-dir "${WORK_DIR}/saved_data/twibot22_data"

    echo "--- 1.5 构建标签和划分 ---"
    python preprocess_twibot22.py --stage 5 --data-dir "${DATA_DIR}" --save-dir "${WORK_DIR}/saved_data/twibot22_data"

    echo "--- 1.6 保存最终张量 ---"
    python preprocess_twibot22.py --stage 6 --data-dir "${DATA_DIR}" --save-dir "${WORK_DIR}/saved_data/twibot22_data"

    echo ">>> 预处理完成！"
}

# -------- 阶段 2: 时序特征提取 --------
run_feature() {
    echo ""
    echo "============================================"
    echo ">>> 阶段 2: 时序特征提取 (Transformer)"
    echo "============================================"

    cd "${CODE_DIR}"
    python feature_extraction_twibot22.py \
        --mode pseudo \
        --data-dir "${DATA_DIR}" \
        --work-dir "${WORK_DIR}"

    echo ">>> 时序特征提取完成！"
}

# -------- 阶段 3: GNN 训练 (原始两阶段) --------
run_train() {
    echo ""
    echo "============================================"
    echo ">>> 阶段 3: BotRGCN_v2 训练 (两阶段, 冻结时序)"
    echo "============================================"

    cd "${CODE_DIR}"
    python train_twibot22.py \
        --work-dir "${WORK_DIR}" \
        --model v2 \
        --epochs 200 \
        --patience 30 \
        --seed 42

    echo ">>> 训练完成！"
}

# -------- 阶段 3b: E2E 端到端训练 --------
run_e2e() {
    echo ""
    echo "============================================"
    echo ">>> 阶段 3b: E2E 端到端训练 (Transformer+GNN联合优化)"
    echo "============================================"

    cd "${CODE_DIR}"
    python train_twibot22.py \
        --work-dir "${WORK_DIR}" \
        --e2e \
        --seq-len 16 \
        --ts-nhead 2 \
        --ts-layers 2 \
        --ts-lr-scale 0.1 \
        --epochs 200 \
        --patience 30 \
        --seed 42 \
        --save-suffix "_e2e"

    echo ">>> E2E 训练完成！"
}

# -------- 阶段 3c: E2E + InfoNCE 对齐损失 --------
run_e2e_align() {
    echo ""
    echo "============================================"
    echo ">>> 阶段 3c: E2E + InfoNCE 对齐损失"
    echo "============================================"

    cd "${CODE_DIR}"
    python train_twibot22.py \
        --work-dir "${WORK_DIR}" \
        --e2e \
        --seq-len 16 \
        --ts-nhead 2 \
        --ts-layers 2 \
        --ts-lr-scale 0.1 \
        --align-beta 0.1 \
        --align-temp 0.5 \
        --epochs 200 \
        --patience 30 \
        --seed 42 \
        --save-suffix "_e2e_align"

    echo ">>> E2E+Align 训练完成！"
}

# -------- 多种子 E2E 实验 --------
run_e2e_multiseed() {
    echo ""
    echo "============================================"
    echo ">>> 多种子 E2E 实验 (seeds: 42, 123, 456, 789, 2024)"
    echo "============================================"

    cd "${CODE_DIR}"
    BETA="${1:-0.0}"  # 默认无对齐损失; 传参可覆盖
    SUFFIX_BASE="_e2e"
    if [ "$(echo "$BETA > 0" | bc -l)" = "1" ]; then
        SUFFIX_BASE="_e2e_align"
    fi

    for SEED in 42 123 456 789 2024; do
        echo "--- seed=${SEED} ---"
        python train_twibot22.py \
            --work-dir "${WORK_DIR}" \
            --e2e \
            --seq-len 16 \
            --ts-nhead 2 \
            --ts-layers 2 \
            --ts-lr-scale 0.1 \
            --align-beta "${BETA}" \
            --align-temp 0.5 \
            --epochs 200 \
            --patience 30 \
            --seed "${SEED}" \
            --save-suffix "${SUFFIX_BASE}_s${SEED}"
    done

    echo ">>> 多种子 E2E 实验完成！"
}

# -------- 主入口 --------
case "${1:-all}" in
    check)
        check_env
        ;;
    preprocess)
        run_preprocess
        ;;
    feature)
        run_feature
        ;;
    train)
        run_train
        ;;
    e2e)
        run_e2e
        ;;
    e2e_align)
        run_e2e_align
        ;;
    e2e_multi)
        run_e2e_multiseed "${2:-0.0}"
        ;;
    all)
        check_env
        run_preprocess
        run_feature
        run_train
        ;;
    *)
        echo "用法: $0 {check|preprocess|feature|train|e2e|e2e_align|e2e_multi [beta]|all}"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo ">>> 全部完成！"
echo "============================================"
