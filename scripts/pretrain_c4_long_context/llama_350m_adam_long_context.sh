# LLaMA-350M (seq length 1024), AdamW, 4 A100, 1 Node
lr=0.005 # best lr chosen from a range of 0.001 0.0025 0.005 0.0075 0.01
torchrun --standalone --nproc_per_node 4 main_pretrain.py \
    --model_config configs/llama_350m.json \
    --lr ${lr} \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --max_length 1024 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adamw \
    --weight_decay 0 \
    --project apollo_test \
    --name apollo_test_adam_350m_long_context \
    --save_dir ./ckpts/adam_350m_s1024_lr${lr}



