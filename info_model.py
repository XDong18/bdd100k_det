
import os
import hashlib

if __name__ == "__main__":
    out_dir = 'report_dir'
    temp_dir = 'temp_dir'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    sha256 = hashlib.sha256()
    model_path = 'bdd100k_retinanet_R_50_FPN_1x/bdd100k_retinanet_R_50_FPN_1x-303595e5.pth'
    with open(model_path, 'rb') as u:
        while True:
            buffer = u.read(8192)
            if len(buffer) == 0:
                break
            sha256.update(buffer)
    
    digest = sha256.hexdigest()[:8]
    model_out_name = 'bdd100k' + '-' 'detectron2' + '-' + 'retinanet_r50_fpn-1x' + '-' + digest + '.pth'
    model_out_name = os.path.join(out_dir, model_out_name)

    with open(model_out_name, 'wb') as f:
        with open(model_path, 'rb') as u:
            model = u.read()
        f.write(model)