export CUDA_VISIBLE_DEVICES=0,9
python train.py --num-gpus 2 --dist-url auto --eval-only MODEL.WEIGHTS ./bdd100k_faster_rcnn_R_101_FPN_1x/bdd100k_faster_rcnn_R_101_FPN_1x-d488e6cf.pth 