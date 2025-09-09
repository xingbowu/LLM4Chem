#!/bin/bash

# 性能分析脚本 - 使用PyTorch Profiler诊断训练瓶颈
# 使用方法: bash run_profiling.sh

echo "🔍 开始性能分析训练..."

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py \
  --base_model /home/libo/.cache/modelscope/hub/models/AI-ModelScope/Mistral-7B-v0.1 \
  --data_path /home/libo/.cache/modelscope/hub/datasets/osunlp/SMolInstruct \
  --output_dir checkpoint/mistral-7b-chemistry-profiling \
  --micro_batch_size 32 \
  --batch_size 512 \
  --num_epochs 1 \
  --learning_rate 1e-4 \
  --cutoff_len 256 \
  --fsdp="full_shard auto_wrap" \
  --fsdp_config='{"fsdp_transformer_layer_cls_to_wrap":"MistralDecoderLayer","fsdp_backward_prefetch":"backward_pre","fsdp_forward_prefetch":false,"fsdp_activation_checkpointing":true,"fsdp_use_orig_params":true}' \
  --gradient_checkpointing false \
  --precision bf16 \
  --dataloader_num_workers 8 \
  --enable_profiling true \
  --profile_steps 20 \
  --profile_dir "./profiling_logs" \
  --tasks "['forward_synthesis','retrosynthesis','molecule_captioning']" \
  --swanlab_project "chemistry-llm" \
  --swanlab_run_name "mistral-7b-profiling-test"

echo "✅ 训练完成！"
echo "📊 查看性能分析结果："
echo "   tensorboard --logdir=./profiling_logs"
echo ""
echo "🔍 分析要点："
echo "   1. 查看 DataLoader 时间"
echo "   2. 查看前向传播时间"
echo "   3. 查看反向传播时间"
echo "   4. 查看梯度掩码操作时间"
echo "   5. 查看 FSDP 通信时间"
