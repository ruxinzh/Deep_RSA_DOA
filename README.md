# Deep-Learning-Enabled-Robust-DOA-Estimation-with-Single-Snapshot-Sparse-Arrays
This is the code for paper "Antenna Failure Resilience: Deep Learning-Enabled Robust DOA Estimation with Single Snapshot Sparse Arrays" 

## Simulated dataset generation for trianing and validation 
``` sh
python scr/dataset_gen.py --output_dir './' --num_samples_val 1024 --num_samples_train 100000 --N 10 --max_targets 3 
```

## Network architectures 

<p align="center">
  <img src="https://github.com/ruxinzh/Deep_RSA_DOA/blob/main/fig/Network.png" width="700">
</p>

## Training 
Without sparse augmentation model:

```sh
python train.py --data_path './data' --checkpoint_path './checkpoint' --number_elements 10 --output_size 61 --sparsity 0.3 --use_sparse False --learning_rate 0.0001 --batch_size 1024 --epochs 300
```

With sparse augmentation model:

``` sh
python train.py --data_path './data' --checkpoint_path './checkpoint' --number_elements 10 --output_size 61 --sparsity 0.3 --use_sparse True --learning_rate 0.0001 --batch_size 1024 --epochs 300
```

## Evaluation 
The evaluation of the model can be conducted immediately using weights that we have trained and provided. These weights are available in the 'checkpoint' directory. 
Before proceeding with the following steps, ensure you are in the correct directory where the scripts or applications are located.

``` sh
cd  scr
```

### Single target accuracy

``` sh
python run_eval.py --num_simulations 1000 --num_antennas 10 --evaluation_mode 'accuracy1'
```

Expected outputs: ULA(left), SLA(right)
<p align="center">
  <img src="https://github.com/ruxinzh/Deep_RSA_DOA/blob/main/fig/Accuracy1_ULA.png" width="400">
  <img src="https://github.com/ruxinzh/Deep_RSA_DOA/blob/main/fig/Accuracy1_SLA.png" width="400">
</p>

### Two target accuracy

``` sh
python run_eval.py --num_simulations 1000 --num_antennas 10 --evaluation_mode 'accuracy2'
```

Expected outputs: ULA(left), SLA(right)
<p align="center">
  <img src="https://github.com/ruxinzh/Deep_RSA_DOA/blob/main/fig/Accuracy2_ULA.png" width="400">
  <img src="https://github.com/ruxinzh/Deep_RSA_DOA/blob/main/fig/Accuracy2_SLA.png" width="400">
</p>

### Seperatebility

``` sh
python run_eval.py --num_simulations 1000 --num_antennas 10 --evaluation_mode 'separate'
```

Expected outputs: ULA(left), SLA(right)
<p align="center">
  <img src="https://github.com/ruxinzh/Deep_RSA_DOA/blob/main/fig/Separate_ULA.png" width="400">
  <img src="https://github.com/ruxinzh/Deep_RSA_DOA/blob/main/fig/Separate_SLA.png" width="400">
</p>

### Complexity
``` sh
python run_eval.py --num_simulations 1000 --num_antennas 10 --evaluation_mode 'complexity'
```
Expected outputs:

Total trainable parameters in MLP model: 2848829 

Total trainable parameters in Ours model: 4106301

### Results examples 
#### With simulated data

``` sh
python run_eval.py --evaluation_mode 'examples'
```

Expected outputs:

<p align="center">
  <img src="https://github.com/ruxinzh/Deep_RSA_DOA/blob/main/fig/Example_ULA.png" width="800">
</p>
<p align="center">
  <img src="https://github.com/ruxinzh/Deep_RSA_DOA/blob/main/fig/Example_SLA.png" width="800">
</p>

#### With real world data 

``` sh
python run_eval.py --evaluation_mode 'examples' --real True
```

Expected outputs:
<p align="center">
  <img src="https://github.com/ruxinzh/Deep_RSA_DOA/blob/main/fig/Example_ULA_real.png" width="800">
</p>
<p align="center">
  <img src="https://github.com/ruxinzh/Deep_RSA_DOA/blob/main/fig/Example_SLA_real.png" width="800">
</p>
## Real World dataset 
please refer README in the folder 'real_World_DOA_dataset'

## Enviroment 
The Conda environment required for this project is specified in the file 'conda_env.txt'. This file contains a list of all the necessary Python packages and their versions to ensure compatibility and reproducibility of the project's code.






