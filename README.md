# SelectedTopic25-HW4
#ID: 312540020

## Introduction
In this work, we focus on building a unified model capable of handling multiple degradation types within a single framework. Specifically, we aim to restore two distinct types of degraded images (snow and rain) using a shared network architecture. To address the challenge of multi-degradation restoration, we modify the components and modules of the PromptIR architecture, a state-of-the-art prompt-based image restoration model. 
My code is base on git PromptIR: https://github.com/va1shn9v/PromptIR.git

## Create environment
```
conda env create -f env.yml
```
The training data should be placed in ```data/hw4_realse_dataset```.
After placing the training data the directory structure would be as follows:
```
└───data
    ├───hw4_realse_dataset
       ├───train
       └───test
```

## Training 
```
python train.py
```
## Test
```
python demo.py
```
## Qualitative Results
![Qualitative Results](./Qual.png) 
## Performance Snapshot
 ![Performance Snapshot](./Snapshot.png)  

