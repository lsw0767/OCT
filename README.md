# K-Index Estimateion System with Deep Learning

The goal of this projet is to replace the OSA H/W device to S/W estimation system,
so that acquired raw signals can be compensated without H/W device


 ## Data structure
 > Each K-Index
 > > Each Sample * # of Sample
 > > > 2D Raw Signal * # of depth

* data would not be provided 


 ## System Overview
 ~~1. K-index curve parameter estimation system~~ 
 
 ~~2. End2End compensated signal estimation system~~
 
 upper systems (in train.py) are toy examples for future works, so there may not be further updates.  
 3. Parametric estimation system
 - system overview will ve updated
 
 

## How to run
#### Step 1. data preprocessing
    python extract_k-index.py
    python preporcess.py
#### Step 2. run train.py
    python train.py
training arguments will be updated