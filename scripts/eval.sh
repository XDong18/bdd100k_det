export CUDA_VISIBLE_DEVICES=6,7,8,9
python train_bdd.py --num-gpus 4 --dist-url auto --eval-only MODEL.WEIGHTS ./bdd100k_retinanet_R_50_FPN_1x/bdd100k_retinanet_R_50_FPN_1x-303595e5.pth 