export CUDA_VISIBLE_DEVICES=0,9
python train_bdd.py --num-gpus 2 --dist-url auto --eval-only MODEL.WEIGHTS ./bdd100k_retinanet_R_50_FPN_1x/bdd100k_retinanet_R_50_FPN_1x-303595e5.pth 