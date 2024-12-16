# YOLO config


yolo_config = {
    'yolo_tiny_DLN': {
        # backbone
        'backbone': 'cspd_tiny',
        # neck
        'neck': 'spp-csp',
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    },
    'yolo_tiny_SNN': {
        # backbone
        'backbone': 'cspd_tiny',
        # neck
        'neck': 'spp-csp',
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    },

}