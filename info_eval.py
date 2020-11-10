import json
import os


if __name__ == "__main__":
    out_dir = 'report_dir'
    temp_dir = 'temp_dir'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    eval_json_name = 'bdd100k' + '-' 'detectron2' + '-' + 'faster_rcnn_r50_fpn-1x' + '-' + 'eval' + '.json'
    eval_json_name = os.path.join(out_dir, eval_json_name)

    data = {'classwise_AP': [['person', '0.239'], ['rider', '0.187'], ['car', '0.411'], ['bus', '0.400'], ['truck', '0.382'], ['bike', '0.192'], ['motor', '0.183'], ['traffic light', '0.166'], ['traffic sign', '0.298'], ['train', '0.001']], 
    'bbox_mAP': 0.246, 'bbox_mAP_50': 0.498, 'bbox_mAP_75': 0.211, 
    'bbox_mAP_s': 0.086, 'bbox_mAP_m': 0.305, 'bbox_mAP_l': 0.473
    }

    with open(eval_json_name, 'w') as f:
        json.dump(data, f)