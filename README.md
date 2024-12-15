# CREST: An Efficient Conjointly-trained Spike-driven Framework for Event-based Object Detection Exploiting Spatiotemporal Dynamics
![image](https://github.com/madamei/CREST/blob/main/architecture.png) 
Official code implementation of the AAAI 2025 paper: [CREST: An Efficient Conjointly-trained Spike-driven Framework for Event-based Object Detection Exploiting Spatiotemporal Dynamics]

## Required Data
### Gen1
Thanks to [event_representation_study](https://github.com/uzh-rpg/event_representation_study) , the required preprocessed Gen1 datasets can be easily obtained from there.
### NCAR
NCAR dataset can be obtained from [there](https://www.prophesee.ai/2018/03/13/dataset-n-cars/)
## MESTOR
Before the Raw event stream is input into the subsequent network, it needs to be processed by MESTOR to integrate the input feature from multi-scales.   
Using MESTOR to process the GEN1 dataset:
```
python gen1data/MESTOR_gen1.py
```
## Pre-trained Checkpoints

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
[event_representation_study](https://github.com/uzh-rpg/event_representation_study)
