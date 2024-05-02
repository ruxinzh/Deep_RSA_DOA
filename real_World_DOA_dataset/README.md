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


To enhance the complexity and usability of the dataset, we superimposed these signals to simulate scenarios with multiple targets. It is important to note that this dataset was specifically curated for testing and demonstrating the efficacy of our network in real-world conditions, and was not used during the training phase. This setup ensures that the dataset serves as a robust tool for accurately assessing the performance of DOA estimation models.





``` sh
```
