import argparse
import os

import torch

from config.yolo_config import yolo_config
from data.transforms import ValTransforms
from models.yolo import build_model
from utils.misc import TestTimeAugmentation


#from evaluator.PKU_DVS_coco_evaluator import COCOAPIEvaluator
from evaluator.gen1_coco_evaluator import COCOAPIEvaluator


parser = argparse.ArgumentParser(description='YOLO Detection')
# basic
parser.add_argument('-size', '--img_size', default=640, type=int,
                    help='img_size')
                    
parser.add_argument('--cuda', action='store_true', default=True,
                    help='Use cuda')
# True
# model
parser.add_argument('-m', '--model', default='yolo_tiny_SNN',
                    help='yolo_tiny_DLN,yolo_tiny_SNN')


parser.add_argument('--weight', type=str,
                    default='./weight/yolo_tiny.pth',
                    help='Trained state_dict file path to open')

parser.add_argument('--conf_thresh', default=0.1, type=float,
                    help='NMS threshold')
# 0.001
parser.add_argument('--nms_thresh', default=0.6, type=float,
                    help='NMS threshold')
# 0.6
parser.add_argument('--center_sample', action='store_true', default=False,
                    help='center sample trick.')
# dataset
parser.add_argument('--root', default='../aaai2024/',
                    help='data root')
parser.add_argument('-d', '--dataset', default='gen1',
                    help='PKU_DVS, gen1.')
# TTA
parser.add_argument('-tta', '--test_aug', action='store_true', default=False,
                    help='use test augmentation.')

args = parser.parse_args()


    
def PKU_DVS_test(model, data_dir, device, img_size):

#    evaluator = VOCAPIEvaluator(
#            data_dir=data_dir,
#            img_size=img_size,
#            device=device,
#            display=True,
#            transform=ValTransforms(img_size)
#        )
        
    evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=img_size,
                        device=device,
                        testset=True,
                        transform=ValTransforms(img_size)
                        )

    # VOC evaluation
    evaluator.evaluate(model)   
    
def gen1_test(model, data_dir, device, img_size):

#    evaluator = VOCAPIEvaluator(
#            data_dir=data_dir,
#            img_size=img_size,
#            device=device,
#            display=True,
#            transform=ValTransforms(img_size)
#        )
        
    evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=img_size,
                        device=device,
                        testset=True,
                        transform=ValTransforms(img_size)
                        )

    # VOC evaluation
    evaluator.evaluate(model)   





if __name__ == '__main__':
    # dataset

    if args.dataset == 'PKU_DVS':
        print('eval on PKU_DVS ...')
        num_classes = 9
        data_dir = os.path.join(args.root)    
        
    elif args.dataset == 'gen1':
        print('eval on gen1 ...')
        num_classes = 2
        data_dir = os.path.join(args.root)  
        
    else:
        print('unknow dataset !!')
        exit(0)

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # YOLO Config
    cfg = yolo_config[args.model]
    # build model
    model = build_model(args=args, 
                        cfg=cfg, 
                        device=device, 
                        num_classes=num_classes, 
                        trainable=False)

    # load weight
    model.load_state_dict(torch.load(args.weight, map_location='cpu'), strict=False)
    model = model.to(device).eval()
    print('Finished loading model!')

    # TTA
    test_aug = TestTimeAugmentation(num_classes=num_classes) if args.test_aug else None
    
    # evaluation
    with torch.no_grad():
        if args.dataset == 'PKU_DVS':
            PKU_DVS_test(model, data_dir, device, args.img_size)
        elif args.dataset == 'gen1':
            gen1_test(model, data_dir, device, args.img_size)
