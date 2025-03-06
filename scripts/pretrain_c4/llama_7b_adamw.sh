# LLaMA-7B, Adamw, 8 A100, 1 Node

torchrun --standalone --nproc_per_node 8 main_pretrain.py \
    --model_config configs/llama_7b.json \
    --eval_every 1000 \
    --dtype bfloat16 \
    --batch_size 4 \
    --total_batch_size 512 \
    --lr 0.01 \
    --warmup_steps 15000 \
    --num_training_steps 150000 \
    --optimizer adamw \
    --weight_decay 0 \
    --project apollo_test \
    --name adamw_test_apollo_7b \
    --save_dir ./ckpts/Adamw_7b_scale${apollo_scale}_rank${num_rank}_proj${proj_type}_type${scale_type}


