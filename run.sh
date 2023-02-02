set -e

# *** reproduce results ***
# see https://github.com/Verg-Avesta/CounTR/issues/23

#python FSC_test_cross\(few-shot\).py --output_dir ./data/out/aaa --resume ./weights/FSC147.pth --box_bound 3


# *** reproduce pre-training and fine-tuning described in the paper ***

CUDA_VISIBLE_DEVICES=0 python FSC_pretrain.py --epochs 300 --batch_size 16 --lr 5e-6 --output_dir ./data/out/pretrain --log_dir None --title CounTR_pretraining_paper
CUDA_VISIBLE_DEVICES=0 python FSC_finetune_cross.py --epochs 1000 --batch_size 8 --lr 1e-5 --output_dir ./data/out/finetune --log_dir None --title CounTR_finetuning_paper --resume ./data/out/pretrain/checkpoint__pretraining_299.pth
python FSC_test_cross\(few-shot\).py --output_dir ./data/out/results_base --resume ./data/out/finetune/checkpoint__finetuning_minMAE.pth --box_bound 3


# *** reproduce pre-training and fine-tuning described in the paper, using the large MAE model ***

# CUDA_VISIBLE_DEVICES=0 python FSC_pretrain.py --epochs 300 --batch_size 16 --lr 5e-6 --output_dir ./data/out/pretrain_large --log_dir None --title CounTR_pretraining_paper_large --resume ./weights/mae_pretrain_vit_large_full.pth --model mae_vit_large_patch16
# CUDA_VISIBLE_DEVICES=0 python FSC_finetune_cross.py --epochs 1000 --batch_size 8 --lr 1e-5 --output_dir ./data/out/finetune_large --log_dir None --title CounTR_finetuning_paper_large --resume ./data/out/pretrain_large/checkpoint__pretraining_299.pth --model mae_vit_large_patch16
# python FSC_test_cross\(few-shot\).py --output_dir ./data/out/results_large --resume ./data/out/finetune_large/checkpoint__finetuning_minMAE.pth --box_bound 3 --model mae_vit_large_patch16
