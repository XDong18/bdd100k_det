export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8
python vis/demo.py --config-file configs/BDD00K-InstanceSegmentation/mask_rcnn_r_101_fpn_yanzaho.yaml \
  --input /shared/xudongliu/bdd100k/10k/val/* \
  --output show/mask_rcnn_r_101_fpn_yanzaho/ \
  --opts SOLVER.IMS_PER_BATCH 8 MODEL.WEIGHTS /shared/xudongliu/code/bdd100k_det/mask_rcnn_r_101_fpn_yanzaho/model_final.pth