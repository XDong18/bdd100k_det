export CUDA_VISIBLE_DEVICES=1,8,9
python train_tensormask.py --num-gpus 3 --dist-url auto --eval-only MODEL.WEIGHTS ./tensormask_r101_3x_multi_scale_bs16_bdd100k/model_0029999.pth