# Deep-Learning-Enabled-Robust-DOA-Estimation-with-Single-Snapshot-Sparse-Arrays
This is the code for paper "Antenna Failure Resilience: Deep Learning-Enabled Robust DOA Estimation with Single Snapshot Sparse Arrays" 

## Simulated dataset generation for trianing and validation 
``` sh
python scr/dataset_gen.py --output_dir '../' --num_samples_val 1024 --num_samples_train 100000 --N 10 --max_targets 3 
```

## Network architectures 

## Training 
Without sparse augmentation model
```sh
python train.py --data_path './data' --checkpoint_path './checkpoint' --number_elements 10 --output_size 61 --sparsity 0.3 --use_sparse False --learning_rate 0.0001 --batch_size 1024 --epochs 300
```
With sparse augmentation model
``` sh
python train.py --data_path './data' --checkpoint_path './checkpoint' --number_elements 10 --output_size 61 --sparsity 0.3 --use_sparse True --learning_rate 0.0001 --batch_size 1024 --epochs 300
```
## Evaluation 
Before proceeding with the following steps, ensure you are in the correct directory where the scripts or applications are located.
``` sh
cd  scr
```
Single target accuracy
``` sh
python run_eval.py --num_simulations 1000 --num_antennas 10 --evaluation_mode 'accuracy1'
```
Two target accuracy
``` sh
python run_eval.py --num_simulations 1000 --num_antennas 10 --evaluation_mode 'accuracy2'
```
Seperatebility
``` sh
python run_eval.py --num_simulations 1000 --num_antennas 10 --evaluation_mode 'separate'
```
Complexity
``` sh
python run_eval.py --num_simulations 1000 --num_antennas 10 --evaluation_mode 'complexity'
```

### Results examples 
with simulated data 
``` sh
python run_eval.py --evaluation_mode 'examples'
```
with real world data 
``` sh
python run_eval.py --evaluation_mode 'examples' --load
```

## Real World dataset 
please refer README in the folder 'real_World_DOA_dataset'







