export CUDA_VISIBLE_DEVICES=0,9
python train_bdd.py --num-gpus 2 --dist-url auto --eval-only MODEL.WEIGHTS ./bdd100k_retinanet_new/Retinanet_r_101_fpn_1x-745c57cc.pth 