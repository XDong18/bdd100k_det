import logging
import os
from collections import OrderedDict
import torch

import dl_lib.utils.comm as comm
from dl_lib.checkpoint import DetectionCheckpointer
from dl_lib.config import get_cfg
from dl_lib.data import MetadataCatalog, DatasetCatalog
from dl_lib.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from dl_lib.evaluation import COCOEvaluator
from dl_lib.modeling import GeneralizedRCNNWithTTA
from dl_lib import model_zoo
from dl_lib.data.datasets import register_coco_instances


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_test")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    # set config file
    cfg.merge_from_file("bdd100k_faster_rcnn_R_101_FPN_1x/config.yaml")
    cfg.DATASETS.TRAIN = ("bdd100k_train",)
    cfg.DATASETS.TEST = ("bdd100k_test",)
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo


    cfg.OUTPUT_DIR = './bdd100k_faster_rcnn_R_101_FPN_1x'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # new added solver arguments
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    cfg.TEST.EVAL_PERIOD = 5000
    # end of new arguments

    # cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    # dataset
    register_coco_instances("bdd100k_train", {}, "train_coco.json", "/shared/xudongliu/bdd100k/100k/train")
    register_coco_instances("bdd100k_test", {}, "/shared/xudongliu/bdd100k/labels/bdd100k_labels_images_det_coco_test.json", "/shared/xudongliu/bdd100k/100k/test")

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    # dataset registration
    # global DatasetCatalog, MetadataCatalog


    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )