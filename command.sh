# nohup sh command.sh > logs/output

CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/_new_/upernet_vit-b16_mln_512x512_80k_ade20k.py --work-dir work_dirs/test_20k