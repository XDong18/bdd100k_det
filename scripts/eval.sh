export CUDA_VISIBLE_DEVICES=5,6,7,8
python train.py --num-gpus 4 --eval-only MODEL.WEIGHTS ./bdd100k_fasterrcnn/model_0004999.pth --dist_url auto