export CUDA_VISIBLE_DEVICES=5,6,7,8
python train.py --num-gpus 4 --dist-url auto --eval-only MODEL.WEIGHTS ./bdd100k_fasterrcnn/model_0004999.pth 