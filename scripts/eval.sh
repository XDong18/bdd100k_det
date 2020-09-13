export CUDA_VISIBLE_DEVICES=1,2,3,4
python train.py --num-gpus 4 --eval-only MODEL.WEIGHTS ./bdd100k_retinanet/model_final.pth