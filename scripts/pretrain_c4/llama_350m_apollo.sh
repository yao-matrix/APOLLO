# LLaMA-350M, APOLLO, 4 A100, 1 Node
num_rank=256
scale_type=channel
proj_type=random
apollo_scale=1

torchrun --standalone --nproc_per_node 4 main_pretrain.py \
    --model_config configs/llama_350m.json \
    --lr 0.01 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer apollo_adamw \
    --apollo_scale ${apollo_scale} \
    --rank ${num_rank} \
    --scale_type ${scale_type} \
    --proj ${proj_type} \
    --update_proj_gap 200 \
    --project apollo_test \
    --name apollo_test_apollo_350m \
    --save_dir ./ckpts/Apollo_350m_scale${apollo_scale}_rank${num_rank}_proj${proj_type}_type${scale_type}
