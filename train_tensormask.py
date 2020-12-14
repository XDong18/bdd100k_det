import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from tensormask import add_tensormask_config


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_val")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_tensormask_config(cfg)
    # set config file
    cfg.merge_from_file("configs/BDD00K-InstanceSegmentation/tensormask_r50_3x_bs8.yaml")
    cfg.DATASETS.TRAIN = ("bdd100k_train",)
    cfg.DATASETS.TEST = ("bdd100k_test",)
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo


    cfg.OUTPUT_DIR = './tensormask_r50_3x_bs8'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # new added solver arguments
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.TEST.EVAL_PERIOD = 500
    # end of new arguments

    # cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    # dataset
    register_coco_instances("bdd100k_train", {}, "/shared/xudongliu/bdd100k/labels/ins_seg/ins_seg_train.json", "/shared/xudongliu/bdd100k/10k/train")
    register_coco_instances("bdd100k_test", {}, "/shared/xudongliu/bdd100k/labels/ins_seg/ins_seg_val.json", "/shared/xudongliu/bdd100k/10k/val")

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
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
