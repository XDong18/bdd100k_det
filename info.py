import json
import os
from pycocotools.coco import COCO

def transform(annFile):
    # transform to bdd format
    coco = COCO(annFile)
    imgIds = coco.getImgIds()
    imgIds = sorted(imgIds)
    catsIds = coco.getCatIds()
    cats = coco.loadCats(catsIds)
    nms = [cat['name'] for cat in cats]
    catsMap = dict(zip(coco.getCatIds(), nms))
    bdd_label = []
    for imgId in imgIds:
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(annIds)
        det_dict = {}
        det_dict["name"] = img["file_name"]
        det_dict["attributes"] = {"weather": "undefined",
                                  "scene": "undefined",
                                  "timeofday": "undefined"}
        det_dict["labels"] = []
        for ann in anns:
            label = {"id": ann["id"],
                     "category": catsMap[ann["category_id"]],
                     "manualShape": False,
                     "manualAttributes": False,
                     "box2d": {
                       "x1": ann["bbox"][0],
                       "y1": ann["bbox"][1],
                       "x2": ann["bbox"][0] + ann["bbox"][2] - 1,
                       "y2": ann["bbox"][1] + ann["bbox"][3] - 1,
                     },
                     "score": ann["score"]
                     }
            det_dict["labels"].append(label)
        bdd_label.append(det_dict)
    return bdd_label


if __name__ == "__main__":
    out_dir = 'report_dir'
    temp_dir = 'temp_dir'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    with open(os.path.join('bdd100k_retinanet_new', 'inference_test/coco_instances_results.json')) as f:
        temp_result_data = json.load(f)

    with open('/shared/xudongliu/bdd100k/labels/bdd100k_labels_images_det_coco_test.json') as f:
        new_result_data = json.load(f)
    
    new_result_data['annotations'] = []
    for i, instance in enumerate(temp_result_data):
        anno = {}
        anno['id'] = i + 1
        anno['image_id'] = instance['image_id']
        anno['category_id'] = instance['category_id']
        anno['bbox'] = instance['bbox']
        anno['area'] = float(instance['bbox'][2] * instance['bbox'][3])
        anno['score'] = instance['score']
        new_result_data['annotations'].append(anno)
    
    coco_temp_result_file = os.path.join(temp_dir, 'bdd100k_retinanet_new.json')
    with open(coco_temp_result_file, 'w') as f:
        json.dump(new_result_data, f)
    
    test_json_name = 'bdd100k' + '-' 'detectron2' + '-' + 'retinanet_r101_fpn-1x' + '-' + 'results' + '.json'
    test_json_name = os.path.join(out_dir, test_json_name)

    bdd_label = transform(coco_temp_result_file)
    with open(test_json_name, 'w') as outfile:
        json.dump(bdd_label, outfile)