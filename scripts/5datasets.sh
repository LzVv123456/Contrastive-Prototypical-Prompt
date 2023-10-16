cd ..

CUDA_VISIBLE_DEVICES=0 python main.py \
--exp_name '/exp' \
--dataset '5datasets' \
--total_nc 50 \
--fg_nc 10 \
--task_num 4 \
--bs 256 \
--epochs 50 \
--seed 77 \
--pretrain_method 'mae' \
--loss 'protocon' \
--topk 3 \
--mc 5 \