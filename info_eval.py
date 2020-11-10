import json
import os


if __name__ == "__main__":
    out_dir = 'report_dir'
    temp_dir = 'temp_dir'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    eval_json_name = 'bdd100k' + '-' 'detectron2' + '-' + 'faster_rcnn_r101_fpn-1x' + '-' + 'eval' + '.json'
    eval_json_name = os.path.join(out_dir, eval_json_name)

    data = {'classwise_AP': [['person', '0.336'], ['rider', '0.255'], ['car', '0.507'], ['bus', '0.482'], ['truck', '0.455'], ['bike', '0.248'], ['motor', '0.238'], ['traffic light', '0.263'], ['traffic sign', '0.387'], ['train', '0.000']], 
    'bbox_mAP': 0.317, 'bbox_mAP_50': 0.581, 'bbox_mAP_75': 0.298, 
    'bbox_mAP_s': 0.148, 'bbox_mAP_m': 0.368, 'bbox_mAP_l': 0.535
    }

    with open(eval_json_name, 'w') as f:
        json.dump(data, f)