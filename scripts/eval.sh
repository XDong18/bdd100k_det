export CUDA_VISIBLE_DEVICES=1,8,9
python train_bdd.py --num-gpus 3 --dist-url auto --eval-only MODEL.WEIGHTS ./mask_rcnn_r_101_fpn_yanzaho/model_0014499.pth