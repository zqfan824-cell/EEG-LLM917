#!/bin/bash
# =============================================================================
# run_deap_diag.sh — 诊断脚本：隔离辅助损失，单看分类头能不能学
#
# 目的：917log.txt 里准确率钉死在多数类（val 0.5583 / test 0.6292）。怀疑是
# 重建(~70)/对抗/对比等辅助损失淹没了分类信号(~0.17)。这里把辅助损失全部
# 关掉/置零，只留分类(+极小的 VQ)，如果准确率开始动 → 确认是辅助损失的问题；
# 如果还是钉死 → 问题在更底层（数据/标签/分类头/VQ 本身）。
#
# 与 run_deap_vq.sh 的区别：
#   - 不传 --enable_reconstruction        （重建关闭）
#   - --domain_weight 0.0                 （域对抗置零）
#   - --contrastive_weight 0.0            （对比学习置零）
#   - --loss CrossEntropyLoss             （先用标准 CE，排除 focal 的干扰）
#   注：--enable_vq / --enable_adversarial 是 store_true 且 default=True，
#       CLI 无法关闭；这里靠把权重置零来等效隔离。VQ loss 极小，保留无妨。
# =============================================================================
echo "=== EEGLLM_VQ 诊断：纯分类（辅助损失置零）==="

MODEL_ID="DEAP_DIAG"
MODEL="EEGLLM_VQ"
DATA="DEAP"
ROOT_PATH="/content/drive/MyDrive/DEAP/"   # Colab 路径，按实际改
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
    --enable_vq \
    --vq_embed_dim 64 \
    --vq_n_embed 1024 \
    --vq_beta 0.25 \
    --domain_weight 0.0 \
    --contrastive_weight 0.0 \
    --alpha_schedule constant \
    --max_alpha 0.0 \
    --channel_selection comprehensive_emotion \
    --train_epochs 20 \
    --batch_size 8 \
    --patience 15 \
    --learning_rate 0.0001 \
    --loss CrossEntropyLoss \
    --itr 1 \
    --use_gpu True \
    --gpu 0

echo ""
echo "诊断完成。看 Vali/Test Accuracy 是否随 epoch 变化："
echo "  - 开始动  → 确认是辅助损失淹没分类，回到 run_deap_vq.sh 调权重即可"
echo "  - 仍钉死  → 问题在数据/标签/分类头/VQ，需要进一步查"
