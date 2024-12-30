# CREST: An Efficient Conjointly-trained Spike-driven Framework for Event-based Object Detection Exploiting Spatiotemporal Dynamics
![image](https://github.com/madamei/CREST/blob/main/architecture.png) 
Official code implementation of the AAAI 2025 paper: [CREST: An Efficient Conjointly-trained Spike-driven Framework for Event-based Object Detection Exploiting Spatiotemporal Dynamics]

## Required Data
### Gen1
Thanks to [event_representation_study](https://github.com/uzh-rpg/event_representation_study) , the required preprocessed Gen1 datasets can be easily obtained from there.

## MESTOR
Before the Raw event stream is input into the subsequent network, it needs to be processed by MESTOR to integrate the input feature from multi-scales.   
Using MESTOR to process the GEN1 dataset:
```
python gen1data/MESTOR_gen1.py
```
## Pre-trained Checkpoints
[Gen1_yolotiny](https://drive.google.com/drive/folders/1DnfbxD-rGOvF2IIwCtaqbqxOMmUgFIBI?usp=sharing)
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.360
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.632
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.351
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.021
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.324
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.461
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.254
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.490
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.494
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.029
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.463
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.566
```
## Object Detection
### Evaluation
```
python eval_yolo.py
```
### Training
```
python train_yolo.py
```

## Object Recognition

## Code Acknowledgments
This project has used code from the following projects:  
* [FS-neurons](https://github.com/christophstoeckl/FS-neurons)
* [event_representation_study](https://github.com/uzh-rpg/event_representation_study)   
* [PyTorch_YOLO-Family](https://github.com/yjh0410/PyTorch_YOLO-Family)
* [DOTIE](https://github.com/manishnagaraj/DOTIE)

