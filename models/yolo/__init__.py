from .yolo_tiny_SNN import yolo_tiny_SNN
from .yolo_tiny_DLN import yolo_tiny_DLN



# build YOLO detector
def build_model(args, cfg, device, num_classes=80, trainable=False):

    if args.model == 'yolo_tiny_SNN':
        print('Build YOLO-Tiny ...')
        model = yolo_tiny_SNN(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)

    elif args.model == 'yolo_tiny_DLN':
        print('Build yolo_tiny_DLN ...')
        model = yolo_tiny_DLN(cfg=cfg,
                        device=device,
                        img_size=args.img_size,
                        num_classes=num_classes,
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)


    return model
