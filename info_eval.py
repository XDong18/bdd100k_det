import json
import os


if __name__ == "__main__":
    out_dir = 'report_dir'
    temp_dir = 'temp_dir'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    eval_json_name = 'bdd100k' + '-' 'detectron2' + '-' + 'retinanet_r50_fpn-1x' + '-' + 'eval' + '.json'
    eval_json_name = os.path.join(out_dir, eval_json_name)

    data = {'classwise_AP': [['person', '0.295'], ['rider', '0.217'], ['car', '0.478'], ['bus', '0.446'], ['truck', '0.426'], ['bike', '0.226'], ['motor', '0.203'], ['traffic light', '0.213'], ['traffic sign', '0.349'], ['train', '0.000']], 
    'bbox_mAP': 0.285, 'bbox_mAP_50': 0.527, 'bbox_mAP_75': 0.264, 
    'bbox_mAP_s': 0.109, 'bbox_mAP_m': 0.350, 'bbox_mAP_l': 0.506
    }

    with open(eval_json_name, 'w') as f:
        json.dump(data, f)