import json
import os


if __name__ == "__main__":
    out_dir = 'report_dir'
    temp_dir = 'temp_dir'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    eval_json_name = 'bdd100k' + '-' 'detectron2' + '-' + 'faster_rcnn_r50_fpn-1x' + '-' + 'eval' + '.json'
    eval_json_name = os.path.join(out_dir, eval_json_name)

    data = {'classwise_AP': [['person', '0.329'], ['rider', '0.243'], ['car', '0.504'], ['bus', '0.458'], ['truck', '0.443'], ['bike', '0.242'], ['motor', '0.230'], ['traffic light', '0.258'], ['traffic sign', '0.384'], ['train', '0.000']], 
    'bbox_mAP': 0.309, 'bbox_mAP_50': 0.572, 'bbox_mAP_75': 0.285, 
    'bbox_mAP_s': 0.147, 'bbox_mAP_m': 0.358, 'bbox_mAP_l': 0.518
    }

    with open(eval_json_name, 'w') as f:
        json.dump(data, f)