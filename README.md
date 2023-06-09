# CFineNet

Coarse-to-Fine Network (CFineNet) is proposed for pixel-level video prediction task. 

More information and models are coming soon!

### Table of Contents
> * [Introduction](#Introduction)
> * [Requirements](#Requirements)
> * [Architecture](#Architecture)
> * [Train&Test](#Train&Test)
> * [Visualize](#Visualize)


### Introduction
Video prediction is the task of predicting future video frames conditioned on a few observed video frames.
![Image text](https://github.com/Chuanqi-Zang/CFineNet/blob/master/git_img/preview.png)
Three networks for video prediction. When we predict the next frame based on the previous frame, it is difficult to predict the accurate motion state, but no matter what the state is, we can predict its appearance according to the predicted motion state. The direct prediction network confuses the ambiguous motion prediction and the exact appearance prediction, directly generating future frames. The residual decoupling network models the motion and appearance of the frame separately. In our proposed cascaded decoupling network, we first predict the motion state and then adjust the appearance of the object according to its motion state.

### Requirements
- Ubuntu 18.04.3 LTS
- Python 3.7.4 (Anaconda)
- CUDA Version 10.0.130 && CUDNN 7.6.2
- tensorflow 1.14.0
- matplotlib(3.0.1)

### Architecture
```bash
    .
    └── data
    |   ├── human.py                  #data reader(for human3.6m)
    |   ├── ucf.py                    #data reader(for ucf101)
    |   ├── ucf_train.txt             #data split for training (ucf101)
    |   └── ucf_test.txt              #data split for testing (ucf101)
    └── main.py                       #Config and executive program
    └── layer_3d.py                   #Image encoder, decoder, coarse LSTM, refine LSTM
    └── layer_D.py                    #Temporal smoothness classification model
    └── losses.py                     #Implementation of various loss functions
    └── utils_3d.py                   #Image encoder processing
    └── data_utils.py                 #data reading and image save
    └── measure.py                    #Implementation of various measures
```

### Train&Test
1. train
```
# human3.6m
python main.py --dataset human
# ucf101
python main.py --phase ucf
```
