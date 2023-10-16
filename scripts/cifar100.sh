cd ..

CUDA_VISIBLE_DEVICES=0 python main.py \
--exp_name '/exp' \
--dataset 'cifar100' \
--total_nc 4 \
--fg_nc 2 \
--task_num 1 \
--bs 256 \
--epochs 1 \
--seed 77 \
--pretrain_method '1k' \
--loss 'protocon' \
--topk 3 \
--mc 5 \
