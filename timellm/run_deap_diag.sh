#!/bin/bash
# =============================================================================
# run_deap_diag.sh — 诊断 v2：最简分类通路，单看分类头能不能学
#
# 背景：v1 诊断（仅把辅助损失权重置零）准确率仍钉死在多数类，且分类 loss 全程 ≈ln2，
# 说明分类通路根本没在学。最大嫌疑是 VQ 码本坍缩 + AdaptiveLossWeighter 干扰。
# v2 把通路砍到最简：
#   --no-enable_vq          关闭 VQ（直接 patch→重编程→冻结LLM→分类头）
#   --no-use_adaptive_loss  绕过自适应加权，直接用裸 CrossEntropy
#   --domain_weight 0 --contrastive_weight 0   辅助损失置零
#   batch_size 64           A100 上 batch 8 严重欠载，加大提速且梯度更稳
#   train_epochs 5          快速看趋势；分类 loss 若要降，1~2 个 epoch 就能看出来
#
# 判读：
#   分类 loss 跌破 0.69 / 准确率开始动 → 确认是 VQ/加权机制的锅，再逐个加回来
#   仍然平 → 问题在"重编程→冻结LLM→分类头"骨架本身，下一步动学习率/架构
# =============================================================================
echo "=== EEGLLM_VQ 诊断 v2：关 VQ + 裸 CE，最简分类通路 ==="

MODEL_ID="DEAP_DIAG2"
MODEL="EEGLLM_VQ"
DATA="DEAP"
ROOT_PATH="/content/drive/MyDrive/DEAP/"   # Colab 路径
SEQ_LEN=256
N_CLASS=2
CLASSIFICATION_TYPE="valence"

python run_main_with_reconstruction.py \
    --task_name classification \
    --is_training 1 \
    --model_id $MODEL_ID \
    --model $MODEL \
    --data $DATA \
    --root_path $ROOT_PATH \
    --seq_len $SEQ_LEN \
    --n_class $N_CLASS \
    --classification_type $CLASSIFICATION_TYPE \
    --enc_in 14 \
    --d_model 32 \
    --n_heads 8 \
    --e_layers 2 \
    --d_ff 128 \
    --llm_model GPT2 \
    --llm_dim 768 \
    --llm_layers 2 \
    --patch_len 16 \
    --stride 8 \
    --no-enable_vq \
    --no-use_adaptive_loss \
    --domain_weight 0.0 \
    --contrastive_weight 0.0 \
    --alpha_schedule constant \
    --max_alpha 0.0 \
    --channel_selection comprehensive_emotion \
    --train_epochs 5 \
    --batch_size 64 \
    --patience 15 \
    --learning_rate 0.0001 \
    --loss CrossEntropyLoss \
    --itr 1 \
    --use_gpu True \
    --gpu 0

echo ""
echo "诊断 v2 完成。重点看 Classification loss 有没有跌破 0.69、Vali/Test Accuracy 有没有动。"
