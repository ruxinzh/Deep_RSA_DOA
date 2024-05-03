# Real World dataset for Direction of Arrival (DOA) Estimation

## Motivation
In the field of DOA estimation, the absence of publicly available real-world datasets has been a significant barrier for advancing and validating DOA estimation technologies. Historically, researchers have relied on simulated datasets to train and evaluate their models. To bridge this gap and contribute a practical resource to the community, we have developed a DOA estimation dataset gathered under real-world conditions. 

## Data acquisition vehicle platform
Data acquisition vehicle platform of Lexus RX450h with high-resolution imaging radar, LiDAR, and stereo cameras is used to carry out field experiments at the University of Alabama.
<p align="center">
  <img src="https://github.com/ruxinzh/Deep_RSA_DOA/blob/main/real_World_DOA_dataset/fig/platform.png" width="602" height="457">
</p>

## Field experiment
This dataset was generated in a parking lot scenario where a stationary vehicle, equipped with a TI Cascade Imaging Radar, collected data. The vehicle was stationed to capture signals from a corner reflector placed 15 meters away, encompassing all possible directions. 

<p align="center">
  <img src="https://github.com/ruxinzh/Deep_RSA_DOA/blob/main/real_World_DOA_dataset/fig/DOA data.png" width="500" height="400">
</p>

This comprehensive data collection resulted in 195 high-SNR signals representing unique angles of arrival from a single target. 
<p align="center">
  <img src="https://github.com/ruxinzh/Deep_RSA_DOA/blob/main/real_World_DOA_dataset/fig/example.gif" width="500" height="400">
</p>


To enhance the complexity and usability of the dataset, we superimposed these signals to simulate scenarios with multiple targets. Here are some examples of FFT spectrum on multiple targets signals. 
<p align="center">
  <img src="https://github.com/ruxinzh/Deep_RSA_DOA/blob/main/real_World_DOA_dataset/fig/multiExamples.png" width="2200" height="400">
</p>

## Dataset structure 
data.mat contains:
- ang_list (1 x 195): the ground truth DOA of each signal
- bv_list (195 x 86): 195 raw signal with 86 antennas 

## How to use 
Matlab
``` matlab
load('data.mat')
```
python 
``` python
import scipy.io
data = scipy.io.loadmat('data.mat')
```

If this dataset contributes to your research, please acknowledge its use with the following citation:
``` LATEX
@ARTICLE{10348517,
  author={Zheng, Ruxin and Sun, Shunqiao and Liu, Hongshan and Chen, Honglei and Li, Jian},
  journal={IEEE Sensors Journal}, 
  title={Interpretable and Efficient Beamforming-Based Deep Learning for Single Snapshot DOA Estimation}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  keywords={Direction-of-arrival estimation;Estimation;Deep learning;Covariance matrices;Sensors;Mathematical models;Array signal processing;Single snapshot DOA estimation;array signal processing;automotive radar;interpretability;deep learning},
  doi={10.1109/JSEN.2023.3338575}}
```
