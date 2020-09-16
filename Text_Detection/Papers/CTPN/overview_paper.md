# Detecting Text in Natural Image with Connectionist Text Proposal Network

## Abstract 

+ Detect a line text (~ a sequence of *fine-scale text*) in feature map convolution.

+ Using **vertical anchor** mechanism predict:
    - Location
    - Text/non-text score

+ The sequential proposals are naturally connected by a recurrent neural network

## 1.Introduction

+ **Architecture of the CTPN**

    - VGG16 ~> feature maps

    - Bi-directional LSTM (~ the sequential windows)

        - RNN layer is connected to a 512D fully-connected layer => predicts: text/non-text scores; y-axis coordinates; side-refinement offsets of *k* anchor.

### 1.1 Contributions
    - Using **an anchor regression mechanism** predict vertical location & text/non-text score of each text proposal. 
        -> **Detecting text in fine-scale proposals**

    - Using **an in-network recurrent mechanism** to connect sequential text proposals in the convolution feature map.
        -> **Recurrent connectionist text proposals**


    - Both methods are integrated seamlessly -> method is able to handle multi-scale and multi-lingual text.
        -> **side-refinement**

## 2. Related work

## 3. Connectionist Text Proposal Network

### 3.1 Detecting text in Fine-Scale Proposals

+ **A fully convolutional network** ~> allow input image of arbitrary size => output a sequence of fine-scale text proposals

+ **VGG16:**

    - kernel size: 3x3

    - Architecture:
        
        - the size of *conv5* feature maps is determined by the size of input img.

        - ? *the total stride & receptive field* are fixed as 16 & 228 pixels

        - *sliding-window* methods adopt multi-scale windows to detect objects with different sizes.

+ Design the *fine-scale* text proposals, that investigates each spatial location in the *conv5* densely.

    - Text proposals is defined **width of 16 pixels**

    - Design *k* vertical anchors to:

        - **? vertical & horizontal same width: 16 pixels**
        
        - predict y-coordinates for each proposals.

        - same horizontal location with a fixed width of 16 pixels but vertical locations are varied in k different heights.

        - Author used *k=10* anchors for each proposal


