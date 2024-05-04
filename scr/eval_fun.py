import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import time
from helpers import *
import sys
sys.path.append('../')
from models.SADOANet import SADOANet
from tqdm import tqdm
import torch
from collections import OrderedDict

def load_sadoanet(num_elements, output_size, sparsity, is_sparse, device, model_path):
    model = SADOANet(num_elements, output_size, sparsity, is_sparse).to(device)
    state_dict = torch.load(model_path)
    # model is trained using nn.DataParallel
    # need to rename key, if the model not in DataParallel
    new_state_dict = OrderedDict()    
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.","")] = v
    model.load_state_dict(new_state_dict)
    
    return model.eval()

def randSparse(signal,sparsity):
    sparseSignal = signal.clone()
    sparseInd = torch.randperm(signal.numel())[:int(signal.numel() * sparsity)]
    sparseSignal[sparseInd] = 0
    return sparseSignal

def FFT(signal, num_antennas=10, ang_min = -30, ang_max = 31, ang_step = 1):
    ang_list = torch.arange(ang_min,ang_max,ang_step)
    a_theta = steering_vector(num_antennas,ang_list ).conj()
    AH = torch.transpose(a_theta, 0, 1)
    spec = torch.abs(torch.matmul(AH,signal)/num_antennas).squeeze().numpy()
    return ang_list, spec

def IAA(y, Niter=15):
    """
    Implementation of the IAA algorithm.
    """
    N = y.shape[0]
    ang_list = torch.arange(-90,91,1)
    A = steering_vector(N,ang_list)
    AH = torch.transpose(A.conj(), 0, 1)
    N, K = A.shape
    Ns = y.shape[1]
    Pk = np.zeros(K, dtype=np.complex128)

    y = y.squeeze().numpy()
    A = A.numpy()
    AH = AH.resolve_conj().numpy()
    # Initial computation of power
    
    Pk = (AH @ y / N) ** 2
    P = np.diag(Pk)
    R = A @ P @ AH

    # Main iteration
    for _ in range(Niter):
        R += 0e-3 * np.eye(N)
        ak_R = AH @ np.linalg.pinv(R)
        T = ak_R @ y
        B = ak_R @ A
        b = B.diagonal()
        sk = T/np.abs(b)
        Pk = np.abs(sk) ** 2
        P = np.diag(Pk)
        R = A @ P @ A.conj().T
    # spec = Pk
    spec = Pk[60:121] #-30:1:30
    ang_list = torch.arange(-30,31,1)
    return ang_list, spec

def DLapproach(signal,model,device):
    model_result = model(signal.squeeze().unsqueeze(0).to(device))
    model_result = model_result.squeeze().cpu().detach().numpy()
    ang_list = torch.arange(-30,31,1)
    spec = model_result
    return ang_list, spec

def estimate_doa(ang_list, spec, scale = 0.7):
    """
    Estimate doa from spectrum
    """
    max_height = np.max(spec)
    min_peak_height = (scale * max_height)
    peaks, properties = find_peaks(spec, height=min_peak_height)
    # Sort peaks by their magnitudes in descending order
    sorted_indices = np.argsort(properties['peak_heights'])[::-1]  # Get indices to sort in descending order
    sorted_peaks = peaks[sorted_indices]
    sorted_peak_heights = properties['peak_heights'][sorted_indices]
    doa = ang_list[sorted_peaks]

    return doa

def plot_results(snr_levels, mse_metrics):
    plt.figure(figsize=(10, 6))
    markers = {'fft': 'o', 'iaa': 's', 'mlp': '^', 'sparse': 'd', 'grid': 'x'}
    colors = {'fft': 'blue', 'iaa': 'green', 'mlp': 'red', 'sparse': 'cyan', 'grid': 'black'}
    for key, color in colors.items():
        plt.semilogy(snr_levels, mse_metrics[key], label=key.upper(), color=color, marker=markers[key])

    plt.title('MSE vs SNR for Different DOA Estimation Methods')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Mean Squared Error (deg)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_HR(snr_levels, HR_metrics):
    plt.figure(figsize=(10, 6))
    markers = {'fft': 'o', 'iaa': 's', 'mlp': '^', 'sparse': 'd'}
    colors = {'fft': 'blue', 'iaa': 'green', 'mlp': 'red', 'sparse': 'cyan'}
    for key, color in colors.items():
        plt.plot(snr_levels, HR_metrics[key], label=key.upper(), color=color, marker=markers[key])

    plt.title('Hit Rate vs SNR for Different DOA Estimation Methods')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Hit Rate (%)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def handle_estimated_angles(tmp):
    """ Handle the sorting and selection of estimated angles. """
    if tmp.nelement() == 0:
        return np.array([30, 30])  # Default error value when no angles are resolved
    elif tmp.nelement() == 1:
        return np.array([tmp[0].item(), tmp[0].item()])
    else:
        tmp = tmp[:2]
        sorted_tmp = torch.sort(tmp)[0].numpy()
        return sorted_tmp

def calculate_mse(actual_angles_1, actual_angles_2, estimates):
    """ Calculate the mean squared error for estimated angles. """
    mse_1 = np.mean((actual_angles_1 - estimates[:, 0]) ** 2)
    mse_2 = np.mean((actual_angles_2 - estimates[:, 1]) ** 2)
    return np.mean([mse_1, mse_2])
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#################################################################
##               MONTE CARLO TEST FUNCTIONS                    ##
#################################################################
def run_monte_carlo_accuracy(num_simulations, num_antennas=10, sparse_flag = True):
    snr_levels = np.arange(-5, 35, 5)  # SNR levels from -5 dB to 30 dB in 5 dB steps
    mse_metrics = {'fft': [], 'iaa': [], 'mlp': [], 'sparse': [], 'grid': []}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load models
    mlp_model = load_sadoanet(10, 61, 0.3, False, device, '../checkpoint/filled/best_model_checkpoint.pth')
    sparse_model = load_sadoanet(10, 61, 0.3, True, device, '../checkpoint/sparse/best_model_checkpoint.pth')

    methods = {
        'fft': lambda sig: FFT(sig),
        'iaa': lambda sig: IAA(sig),
        'mlp': lambda sig: DLapproach(sig, mlp_model, device),
        'sparse': lambda sig: DLapproach(sig, sparse_model, device)
    }
    
    for snr_db in tqdm(snr_levels, desc='SNR levels', unit='snr'):
        actual_angles = np.random.uniform(-30, 30, num_simulations)
        estimates = {key: np.zeros(num_simulations) for key in mse_metrics}

        for i in range(num_simulations):
            signal = generate_complex_signal(num_antennas, snr_db, torch.tensor([actual_angles[i]]))
            if sparse_flag:
                signal = randSparse(signal, 0.3)

            for method_name, method in methods.items():
                ang_list, spec = method(signal)
                tmp = estimate_doa(ang_list, spec)
                estimates[method_name][i] = actual_angles[i] if tmp.nelement() == 0 else tmp[0].item()

        for key in mse_metrics:
            if key != 'grid':
                mse_metrics[key].append(np.mean((actual_angles - estimates[key]) ** 2))
            else :
                mse_metrics[key].append(np.mean((actual_angles - np.rint(actual_angles)) ** 2))
    return snr_levels, mse_metrics


##################################################################
def run_monte_carlo_accuracy2(num_simulations, num_antennas=10, sparse_flag = True):
    """
    Run Monte Carlo simulations to evaluate various DOA estimation methods across SNR levels.
    """
    snr_levels = np.arange(-5, 35, 5)  # SNR levels from -5 dB to 30 dB in 5 dB steps
    mse_metrics = {'fft': [], 'iaa': [], 'mlp': [], 'sparse': [], 'grid': []}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load models
    mlp_model = load_sadoanet(10, 61, 0.3, False, device, '../checkpoint/filled/best_model_checkpoint.pth')
    sparse_model = load_sadoanet(10, 61, 0.3, True, device, '../checkpoint/sparse/best_model_checkpoint.pth')

    methods = {
        'fft': lambda sig: FFT(sig),
        'iaa': lambda sig: IAA(sig),
        'mlp': lambda sig: DLapproach(sig, mlp_model, device),
        'sparse': lambda sig: DLapproach(sig, sparse_model, device)
    }
    
    for snr_db in tqdm(snr_levels, desc='SNR levels', unit='snr'):
        actual_angles_1 = np.random.uniform(-0.6, 0.4, num_simulations)
        actual_angles_2 = np.random.uniform(9.6, 10.4, num_simulations)
        actual_angles = np.column_stack((actual_angles_1, actual_angles_2)).flatten()

        estimates = {key: np.zeros((num_simulations, 2)) for key in mse_metrics}
        
        for i in range(num_simulations):
            signal = generate_complex_signal(num_antennas, snr_db, torch.tensor([actual_angles_1[i], actual_angles_2[i]]))
            if sparse_flag:
                signal = randSparse(signal, 0.3)
            
            for method_name, method in methods.items():
                ang_list, spec = method(signal)
                tmp = estimate_doa(ang_list, spec,0)
                estimates[method_name][i] = handle_estimated_angles(tmp)
        
        # Calculate MSE for each method
        for key in mse_metrics:
            if key != 'grid':
                mse_metrics[key].append(calculate_mse(actual_angles_1, actual_angles_2, estimates[key]))
            else:
                tmp1 = np.mean((actual_angles_1- np.rint(actual_angles_1)) ** 2)
                tmp2 = np.mean((actual_angles_2- np.rint(actual_angles_2)) ** 2)
                mse_metrics[key].append(np.mean(tmp1+tmp2))   
    return snr_levels, mse_metrics



##################################################################
def run_monte_carlo_sep(num_simulations, num_antennas=10, sparse_flag = True):
    """
    Run Monte Carlo simulations to evaluate DOA estimation accuracy across different separation angles.
    """
    sep_angles = np.arange(2, 30, 2)  # Separation angles from 2 to 28 degrees in 2 degree steps
    HR_metrics = {'fft': [], 'iaa': [], 'mlp': [], 'sparse': []}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load models
    mlp_model = load_sadoanet(10, 61, 0.3, False, device, '../checkpoint/filled/best_model_checkpoint.pth')
    sparse_model = load_sadoanet(10, 61, 0.3, True, device, '../checkpoint/sparse/best_model_checkpoint.pth')

    methods = {
        'fft': lambda sig: FFT(sig),
        'iaa': lambda sig: IAA(sig),
        'mlp': lambda sig: DLapproach(sig, mlp_model, device),
        'sparse': lambda sig: DLapproach(sig, sparse_model, device)
    }
    

    for sep in tqdm(sep_angles, desc='Separation', unit='deg', ncols=86):
        actual_angles = np.array([-sep / 2, sep / 2])  # Centered angles
        tmp_rates = {key: np.zeros(num_simulations) for key in HR_metrics}

        for i in range(num_simulations):
            signal = generate_complex_signal(num_antennas, 40, torch.from_numpy(actual_angles))
            if sparse_flag:
                signal = randSparse(signal, 0.3)
                
            for method_name, method in methods.items():
                ang_list, spec = method(signal)
                estimated_angles = estimate_doa(ang_list, spec, 0.7 if method_name in ['fft', 'iaa'] else 0.2)
                estimated_angles, _ = torch.sort(estimated_angles)
                if estimated_angles.nelement() == 2 and np.allclose(estimated_angles.numpy(), actual_angles, atol=2):
                    tmp_rates[method_name][i] = 1

        for key in HR_metrics:
            HR_metrics[key].append(np.mean(tmp_rates[key]))

    return sep_angles, HR_metrics


##################################################################
def run_examples(signal,num_antennas=10,sparse_flag = True):
    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    mlp_model = load_sadoanet(10, 61, 0.3, False, device, '../checkpoint/filled/best_model_checkpoint.pth')
    sparse_model = load_sadoanet(10, 61, 0.3, True, device, '../checkpoint/sparse/best_model_checkpoint.pth')
    signal = torch.from_numpy(signal).to(torch.cfloat)
    sparsity= 0.3
    antennaPos = np.arange(num_antennas)
    if sparse_flag:
        signal = randSparse(signal, 0.3)
        zero_indices = torch.where(signal == 0)[0].numpy()        
        print(zero_indices)
        antennaPos = np.arange(num_antennas)
        antennaPos = np.delete(antennaPos, zero_indices)
        
    # Create a 1x4 subplot grid
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # Adjust the figsize to ensure all subplots are visible and not squished
    
    # Plot the first subplot similar to the uploaded image
    axs[0].stem(antennaPos, np.ones_like(antennaPos), linefmt='blue', markerfmt='bo', basefmt="r-")
    if sparse_flag:
        axs[0].set_title('Sparse Linear Array')
    else:            
        axs[0].set_title('Uniform Linear Array') 
    axs[0].set_xlabel('Horizontal [Half Wavelength]')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_ylim([0, 2])  # Setting the y-axis limits to match the uploaded image
    axs[0].set_xlim([0, num_antennas-1])  # Setting the x-axis limits to match the uploaded image
    axs[0].set_yticks([0, 1, 2])  
    axs[0].set_xticks(np.arange(num_antennas))  
    axs[0].grid(True)        
    
    #FFT
    ang_list, spec0 = FFT(signal,num_antennas)
    spec0 = spec0/np.max(spec0)
    #IAA
    ang_list, spec1 = IAA(signal)
    spec1 = spec1/np.max(spec1)
    axs[1].plot(ang_list, spec0, label='DBF', color='blue',linewidth=2)  
    axs[1].plot(ang_list, spec1, label='IAA', color='red',linewidth=2)         
    axs[1].set_title('DBF vs IAA') 
    axs[1].set_xlabel('Angle [degree]')
    axs[1].set_ylabel('Magnitude')
    axs[1].set_ylim([0, 1])  # Setting the y-axis limits to match the uploaded image
    axs[1].set_xlim([-30, 30])  # Setting the x-axis limits to match the uploaded image 
    axs[1].set_xticks(np.arange(-30, 31, 10))
    axs[1].set_yticks([0, 0.5, 1]) 
    axs[1].legend()
    axs[1].grid(True)  
    
    #MLP
    ang_list, spec2 = DLapproach(signal,mlp_model,device)
    spec2 = spec2/np.max(spec2)
    axs[2].plot(ang_list, spec2, color='blue',linewidth=2)         
    axs[2].set_title('MLP') 
    axs[2].set_xlabel('Angle [degree]')
    axs[2].set_ylabel('Magnitude')
    axs[2].set_ylim([0, 1])  # Setting the y-axis limits to match the uploaded image
    axs[2].set_xlim([-30, 30])  # Setting the x-axis limits to match the uploaded image 
    axs[2].set_xticks(np.arange(-30, 31, 10))  
    axs[2].set_yticks([0, 0.5, 1])
    axs[2].grid(True) 
    
    #Sparse
    ang_list, spec3 = DLapproach(signal,sparse_model,device)
    spec3 = spec3/np.max(spec3)
    axs[3].plot(ang_list, spec3, color='blue',linewidth=2)         
    axs[3].set_title('Ours') 
    axs[3].set_xlabel('Angle [degree]')
    axs[3].set_ylabel('Magnitude')
    axs[3].set_ylim([0, 1])  # Setting the y-axis limits to match the uploaded image
    axs[3].set_xlim([-30, 30])  # Setting the x-axis limits to match the uploaded image 
    axs[3].set_xticks(np.arange(-30, 31, 10))  
    axs[3].set_yticks([0, 0.5, 1])
    axs[3].grid(True) 
    return ang_list, spec0, spec1, spec2, spec3

##################################################################
def model_complexity(num_elements = 10, output_size = 61, sparsity = 0.3):
    mlp_model = SADOANet(num_elements, output_size, sparsity, False)
    num_params = count_parameters(mlp_model)
    sparse_model = SADOANet(num_elements, output_size, sparsity, True)
    num_params_sparse = count_parameters(sparse_model)
    print(f"Total trainable parameters in MLP model: {num_params}")
    print(f"Total trainable parameters in Ours model: {num_params_sparse}")
    return num_params, num_params_sparse

##################################################################
# def main():
#     parser = argparse.ArgumentParser(description="Run Monte Carlo simulations for DOA estimation accuracy")
#     parser.add_argument('--num_simulations', type=int, default=1000, help="Number of Monte Carlo simulations to run")
#     parser.add_argument('--num_antennas', type=int, default=10, help="Number of antennas in the array")
#     args = parser.parse_args()

#     # Accuracy - single target
#     snr_levels, mse_metrics = run_monte_carlo_accuracy(args.num_simulations, args.num_antennas, False)
#     plot_results(snr_levels, mse_metrics)
#     snr_levels, mse_metrics = run_monte_carlo_accuracy(args.num_simulations, args.num_antennas)
#     plot_results(snr_levels, mse_metrics)
    
#     # Accuracy - two targets
#     snr_levels, mse_metrics = run_monte_carlo_accuracy2(args.num_simulations, args.num_antennas, False)
#     plot_results(snr_levels, mse_metrics)
#     snr_levels, mse_metrics = run_monte_carlo_accuracy2(args.num_simulations, args.num_antennas)
#     plot_results(snr_levels, mse_metrics)
    
#     # Generate estimation example on real data
#     signal = generate_complex_signal(10, 40, torch.tensor([0, 7])).numpy() 
#     run_examples(signal,num_antennas=10,sparse_flag = False)
#     run_examples(signal,num_antennas=10,sparse_flag = True)
#     run_examples(signal,num_antennas=10,sparse_flag = True)
#     run_examples(signal,num_antennas=10,sparse_flag = True)
    
#     # Counts total trainable parameters 
#     model_complexity()
# if __name__ == "__main__":
#     main()
    