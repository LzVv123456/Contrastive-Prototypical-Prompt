cd ..

CUDA_VISIBLE_DEVICES=0 python main.py \
--exp_name '/exp' \
--dataset 'cifar100' \
--total_nc 100 \
--fg_nc 10 \
--task_num 9 \
--bs 256 \
--epochs 50 \
--seed 77 \
--pretrain_method '1k' \
--loss 'protocon' \
--topk 3 \
--mc 5 \
