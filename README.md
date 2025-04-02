# Action & Activity Recognition

**Seth Ojo**  
**Khalid Mahmoud**  
Dept. of Computer Science  
University of British Columbia  
Kelowna, Canada  

**Devstutya Pandey**  
Dept. of Computer Science  
University of British Columbia  
Kelowna, Canada  

**Ethan Grosvenor**  
Dept. of Engineering  
University of British Columbia  
Kelowna, Canada  

**Mack Bourne**  
Dept. of Computer Science  
University of British Columbia  
Kelowna, Canada  

---

## Abstract

Human action recognition is a fundamental problem in computer vision with applications in security, surveillance, human-computer interaction, and healthcare. This project presents an action recognition system that utilizes Motion History Images (MHI) and Hidden Markov Models (HMM) to classify human activities. MHI serves as an effective method for encoding motion over time by capturing temporal changes in pixel intensities. To extract meaningful features from MHI, Hu moments are computed, providing shape-based motion descriptors that remain invariant to transformations such as scaling, rotation, and reflection. These features are then used to train HMMs, which model the temporal dependencies of different actions. The system is evaluated on a benchmark dataset, demonstrating its effectiveness in recognizing a variety of human activities. Experimental results indicate that the combination of MHI and HMM provides a robust and computationally efficient approach to action recognition, achieving competitive accuracy compared to traditional methods.

**Keywords:** Action Recognition, Motion History Images (MHI), Hidden Markov Models (HMM), Hu Moments, Computer Vision, Temporal Motion Analysis

---

## Introduction

Human action recognition represents a fundamental challenge in computer vision with critical applications spanning security surveillance [1], human-computer interaction [2], and healthcare monitoring [3]. Our project addresses this challenge through an innovative integration of Motion History Images (MHI) and Hidden Markov Models (HMM), building upon established computer vision techniques while introducing novel enhancements to improve recognition accuracy.

...

## Conclusion

This project set out to explore and enhance traditional motion-based action recognition via Motion History Images (MHI) and Hidden Markov Models (HMM). While the baseline MHI-HMM system showed robust performance, the addition of **optical flow weighting** yielded the most substantial improvements, particularly for speed-dependent or subtle actions. Temporal pyramid segmentation, although conceptually appealing, did not notably boost accuracy in our uniform-segmentation approach. Directional MHI (DMHI) introduced excessive noise and degraded performance.

Notably, **combining optical flow weighting with temporal pyramids** delivered our best results, suggesting that carefully balanced motion encoding and phase-aware segmentation can synergize effectively. Overall, these findings highlight that **traditional methods**, when thoughtfully enhanced, can remain competitive with modern deep learning in settings that demand interpretability, efficiency, or a limited dataset. Future work may explore refined segmentation strategies, hybrid deep learning integrations, or more adaptive motion encoding schemes to further improve recognition accuracy and robustness.