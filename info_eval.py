import json
import os


if __name__ == "__main__":
    out_dir = 'report_dir'
    temp_dir = 'temp_dir'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    eval_json_name = 'bdd100k' + '-' 'detectron2' + '-' + 'retinanet_r101_fpn-1x' + '-' + 'eval' + '.json'
    eval_json_name = os.path.join(out_dir, eval_json_name)

    data = {'classwise_AP': [['person', '0.306'], ['rider', '0.239'], ['car', '0.482'], ['bus', '0.464'], ['truck', '0.439'], ['bike', '0.246'], ['motor', '0.230'], ['traffic light', '0.218'], ['traffic sign', '0.354'], ['train', '0.000']], 
    'bbox_mAP': 0.298, 'bbox_mAP_50': 0.544, 'bbox_mAP_75': 0.279, 
    'bbox_mAP_s': 0.115, 'bbox_mAP_m': 0.366, 'bbox_mAP_l': 0.528
    }

    with open(eval_json_name, 'w') as f:
        json.dump(data, f)