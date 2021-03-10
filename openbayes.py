import os
import json
import argparse
import numpy as np
import pandas as pd
import time
import mmcv
from mmcv import Config

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset 
from mmseg.datasets.custom import CustomDataset

from mmseg.datasets.builder import DATASETS
from mmseg.utils.get_config_file import get_config_file_from_params
from mmseg.utils import collect_env, get_root_logger

import torch.distributed as dist
dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

@DATASETS.register_module()

class OpenbayesDataset(CustomDataset):

    def __init__(self, **params): 
        #print("*"*50,params)
        self.CLASSES = params['classes']
        self.image_width = params['image_width']
        self.image_height = params['image_height']
        self.custom_classes = True
        super(OpenbayesDataset, self).__init__(
            ann_file=params['ann_file'],
            pipeline=params['pipeline'],
            img_dir = params['args']['input'])
        # load annotations
        self.img_infos = self.load_annotation(self.img_dir, self.ann_file, self.image_width, self.image_height, self.CLASSES)
        
        
    def load_annotation(self, img_dir, ann_file, image_width, image_height, classes):
        """Load annotation from directory.
        Returns:
            list[dict]: All image info of dataset.
        """
        img_infos = []
        ann_path = os.path.join(img_dir, ann_file)
        for line in open(ann_path): 
            train_path, label_path= line.split(',')
            label_path = label_path.strip('\n')
            train_path = img_dir + train_path
            label_path = img_dir + label_path
            img_info = dict(filename=train_path)
            img_info['ann'] = dict(seg_map = label_path)
            img_infos.append(img_info)

        return img_infos

def main():
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument('--input', '-i', help='input dataset path', default="/input0/")
    parser.add_argument('--output', '-o', help='output model path', default="/output/model_output")
    parser.add_argument('--model', '-m', help='algorithm', default="faster_rcnn")
    parser.add_argument('--hparams', '-p', help='hyper params json path', default="/output/mmsegmentation/openbayes_params.json")
    args, unknown = parser.parse_known_args()

    params = json.load(open(args.hparams))
    params['args'] = vars(args)
    # default_parameters
    cfg = Config.fromfile(get_config_file_from_params(params))
    cfg.dataset_type = 'OpenbayesDataset'
    cfg.data.test.type = 'OpenbayesDataset'
    cfg.data.test.ann_file = 'test.csv'
    cfg.data.train.type = 'OpenbayesDataset'
    cfg.data.train.ann_file = 'train.csv'
    cfg.data.val.type = 'OpenbayesDataset'
    cfg.data.val.ann_file = 'val.csv'

    #cfg.model.classes = len(params['classes'])
    cfg.work_dir = params['args']['output']
    cfg.evaluation.save_best = 'mIoU'
    cfg.evaluation.rule = 'greater'

    cfg.total_epochs = params['epochs']
    cfg.data.samples_per_gpu = params['batch_size']

    cfg.optimizer.lr = 0.001
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 10

    # seed
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    meta = dict()

    datasets = [build_dataset(cfg.data.train, params)]
    model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.CLASSES = datasets[0].CLASSES
    print('model.CLASSES', model.CLASSES)
    cfg.dump('openbayes_config.py')

    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=(not False),
        timestamp=timestamp,
        meta=meta,
        params=params)

    
if __name__ == '__main__':
    main()
