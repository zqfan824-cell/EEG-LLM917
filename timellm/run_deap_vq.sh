#!/bin/bash

# TimeLLM with VQ and Reconstruction Loss - DEAP Dataset
# 集成NeuroLM的重建损失机制，增强域对抗学习效果

echo "=== TimeLLM with VQ and Reconstruction Loss ==="
echo "Dataset: DEAP"
echo "Task: EEG Emotion Classification"
echo "Features: VQ + Reconstruction Loss + Domain Adversarial Learning"
echo ""

# 基础配置
MODEL_ID="DEAP_TimeLLM_VQ"
MODEL="TimeLLM_VQ"
DATA="DEAP"
ROOT_PATH="/root/autodl-tmp/data_preprocessed_python/"
SEQ_LEN=256
N_CLASS=2
CLASSIFICATION_TYPE="valence"

# 训练配置
TRAIN_EPOCHS=20
BATCH_SIZE=8
PATIENCE=15
LEARNING_RATE=0.0001

# VQ配置
VQ_EMBED_DIM=64
VQ_N_EMBED=1024
VQ_BETA=0.25

# 重建损失配置
FREQ_WEIGHT=1.0
RAW_WEIGHT=1.0
RECONSTRUCTION_WEIGHT=0.5

# 模态对抗学习配置（增强版）
DOMAIN_WEIGHT=1.0
CONTRASTIVE_WEIGHT=0.1
CONTRASTIVE_TEMP=0.1
ALPHA_SCHEDULE="sigmoid"
MAX_ALPHA=1.0

# 模型配置
ENC_IN=14  # 使用综合情绪14通道组
D_MODEL=32
N_HEADS=8
E_LAYERS=2
D_FF=128
LLM_MODEL=GPT2
LLM_DIM=768
LLM_LAYERS=2

# Patch配置
PATCH_LEN=16
STRIDE=8

echo "开始训练 TimeLLM_VQ 模型..."
echo "配置参数:"
echo "  - VQ嵌入维度: $VQ_EMBED_DIM"
echo "  - VQ码本大小: $VQ_N_EMBED"
echo "  - 重建损失权重: $RECONSTRUCTION_WEIGHT"
echo "  - 域对抗权重: $DOMAIN_WEIGHT"
echo "  - 对比学习权重: $CONTRASTIVE_WEIGHT"
echo "  - 对比学习温度: $CONTRASTIVE_TEMP"
echo "  - Alpha调度策略: $ALPHA_SCHEDULE"
echo "  - 最大Alpha值: $MAX_ALPHA"
echo "  - 训练轮数: $TRAIN_EPOCHS"
echo ""

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
    --enc_in $ENC_IN \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --e_layers $E_LAYERS \
    --d_ff $D_FF \
    --llm_model $LLM_MODEL \
    --llm_dim $LLM_DIM \
    --llm_layers $LLM_LAYERS \
    --patch_len $PATCH_LEN \
    --stride $STRIDE \
    --enable_vq \
    --vq_embed_dim $VQ_EMBED_DIM \
    --vq_n_embed $VQ_N_EMBED \
    --vq_beta $VQ_BETA \
    --enable_reconstruction \
    --freq_weight $FREQ_WEIGHT \
    --raw_weight $RAW_WEIGHT \
    --reconstruction_weight $RECONSTRUCTION_WEIGHT \
    --enable_adversarial \
    --domain_weight $DOMAIN_WEIGHT \
    --contrastive_weight $CONTRASTIVE_WEIGHT \
    --contrastive_temp $CONTRASTIVE_TEMP \
    --alpha_schedule $ALPHA_SCHEDULE \
    --max_alpha $MAX_ALPHA \
    --channel_selection comprehensive_emotion \
    --train_epochs $TRAIN_EPOCHS \
    --batch_size $BATCH_SIZE \
    --patience $PATIENCE \
    --learning_rate $LEARNING_RATE \
    --loss focal \
    --itr 1 \
    --use_gpu True \
    --gpu 0

echo ""
echo "训练完成！"
echo "检查点保存在: ./checkpoints/${MODEL_ID}_${MODEL}_${DATA}_${SEQ_LEN}_0/"
echo "测试结果保存在: ./test_results/${MODEL_ID}_${MODEL}_${DATA}_${SEQ_LEN}_0/"
