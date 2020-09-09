# Convolutional Character Networks


## Abstract

+ (?) involving ROI pooling

+ Propose:
    - Can process 2 tasks simultaneously (same time) in 1 pass. 

    - CharNet output: bounding boxes for words & characters in word.
    => *to optimizer text detection jointly with a RNN model recognition.*

+ Crop figures in paper: *done*.


##  1. Introduction

+ The two tasks {Text Detection + Text Recognition (RNN-based)}
    
    - Leading approaches are mainly extended from object detect/segmentation
    
    - **Problems:**

        - 2 task independently ~> a sub-optimizer problem => difficult to optimizer then requires larger amount of training samples.

        - Requires to implement multiple sequential steps

        - Involves (relative) **RoI cropping & pooling** for two-stage framework ~> difficult to crop an accurate text region where a large amount of background.

        - With latin language, many high-performance models consider words as detect units, but word-level requires to cast text recognition into a sequence labelling problem (~ RNN with CTC/attention mechanism).

+ Approacher: **Convolutional Character Network** for joint 2 tasks by leveraging character as basic level.

+ Main contributions:
    - Firstly, CharNet for joint text detection & text recognition.
    
    - Secondly, develop an *iterative (loop) character detection* method.

    - Thirdly, consistently out-performs 


##  2. Related Work

### 2.1 Text Detection

+ Recent approaches **mainly** built on general object detectors with various **text-specific modifications**.

    - Region Proposal Network

    - Connectionist Text Proposal Network (CTPN)

        - Detect a text instance in a sequence of fine-scale text proposal.

    - **Link segment** which also localizes a text instance in a sequence.
        
        - The capability for detecting multi-oriented text.

    - Single-shot text detector using *SSD* object detection to text

### 2.2 Text Recognition

+ A sequence-to-sequence recognition:
    
    - Recurrent Neural Network (RNN)

    - CNN + RNN + CTC ~> end-to-end trainable.

    - Using various attention mechanisms


## 3. Convolution Characters Networks

### 3.1 Overview

+ Consisting 2 branchs:

    - **A character branch**: 

        - Direct chracter detection & recognition

    - **Text detection branch**:

        - Predicting bouding boxes.

+ **Backbone Networks:**

    - *ResNet-50*

        - Convolution features maps with4x down-sampling ratio ~ the final convolutional maps  

    - *Hourglass*

        - Stack 2 hourglass modules.

            - Hourglass-88 <- Hourglass-104 

            - Hourglass-57

        - The final feature map be up-sampled to 1/4 resolution.

### 3.2 Charater branch

+ **Identification of characters is of greate importance to RNN-based text recognitions**

    - *Direct character recognition with an automation character localization mechanism*

    - *Character ~ basic unit*

+ Contains 3 sub-branch:

    - Text instance segmentation

        - 3 convolutional layers: filter_size : {3x3, 3x3, 1x1}

    - Character detection

        - 3 convolutional layers: filter_size : {3x3, 3x3, 1x1}

    - Character recognition

        - 4 convolutional layers: filter_size : {3x3}

### 3.3 Text Detection Branch

+ Identify text instances at a higher level concept, as **words or text-lines**

