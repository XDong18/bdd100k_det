import json

fn = "/data5/bdd100k/labels/det/coco_format/bdd100k_labels_images_val_coco_release.json"

with open(fn) as f:
    ori_coco = json.load(f)

ori_annos = ori_coco['annotations']
new_annos = []
for ori_anno in ori_annos:
    if ori_anno['category_id'] > 10:
        continue
    new_annos.append(ori_anno)

new_coco = ori_coco
new_coco['annotations'] = new_annos
new_coco['categories'] = ori_coco['categories'][:-1]

new_fn = 'val_coco.json'
with open(new_fn, 'w') as f:
    json.dump(new_coco, f)
    