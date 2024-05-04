import os
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader


def steering_vector(N, deg):
    """
    Calculate the steering vector for a uniform linear array using given antenna configuration.
    
    Args:
        N (int): Number of antenna elements.
        deg (float): Angle of arrival in degrees.

    Returns:
        torch.Tensor: The steering vector as a complex-valued tensor.
    """
    d = 0.5  # Element spacing (in units of wavelength)
    wavelength = 1.0  # Wavelength of the signal (same units as d)
    k = 2 * torch.pi / wavelength  # Wavenumber
    n = torch.arange(0, N).view(N, 1)  # Antenna element indices [0, 1, ..., N-1]
    theta = deg * torch.pi / 180  # Convert degrees to radians
    phases = k * d * n * torch.sin(theta)  # Phase shift for each element

    return torch.exp(1j * phases)  # Complex exponential for each phase shift



def generate_complex_signal(N=10, snr_db=10, deg=torch.tensor([30])):
    """
    Generates a complex-valued signal for an array of N antenna elements.

    Args:
        N (int): Number of antenna elements.
        snr_db (float): Signal-to-Noise Ratio in decibels.
        deg (tensor): Angle of arrival in degrees.

    Returns:
        torch.Tensor: Complex-valued tensor of shape (N, 1) representing the received signals.
    """
    a_theta = steering_vector(N, deg)
    phase = torch.exp(2j * torch.pi * torch.randn(a_theta.size()[1])).view(-1, 1)
    signal = torch.matmul(a_theta.to(phase.dtype), phase)
    signal_power = torch.mean(torch.abs(signal)**2)
    snr_linear = 10**(snr_db / 10)

    noise_power = signal_power / snr_linear
    noise_real = torch.sqrt(noise_power / 2) * torch.randn_like(signal.real)
    noise_imag = torch.sqrt(noise_power / 2) * torch.randn_like(signal.imag)
    noise = torch.complex(noise_real, noise_imag)

    return signal + noise 


def generate_label(degrees, min_angle=-30, max_angle=30):
    """
    Generate one-hot encoded labels for the given degrees.
    
    Args:
        degrees (tensor): Target angles in degrees.

    Returns:
        torch.Tensor: One-hot encoded labels.
    """
    labels = torch.zeros(max_angle - min_angle + 1)
    indices = degrees - min_angle
    labels[indices.long()] = 1
    return labels

def generate_data(N, num_samples=1, max_targets=3, folder_path='/content/drive/MyDrive/Asilomar2024/data/'):
    """
    Generate dataset with random number of targets and varying SNR levels.
    
    Args:
        N (int): Number of antenna elements.
        num_samples (int): Number of samples to generate for each SNR level.
        max_targets (int): Maximum number of targets.
        folder_path (str): Base folder path for saving data.

    Returns:
        int: Always returns 0. Data saved in specified directory.
    """
    angles = torch.arange(-30, 31, 1)
    signal_folder = os.path.join(folder_path, 'signal')
    label_folder = os.path.join(folder_path, 'label')
    os.makedirs(signal_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    for snr_db in tqdm(range(0, 35, 5), desc='SNR levels', unit='snr', dynamic_ncols=True):
        all_signals, all_labels = [], []
        for _ in range(num_samples):
            num_targets = torch.randint(1, max_targets + 1, (1,)).item()
            deg_indices = torch.randperm(len(angles))[:num_targets]
            degs = angles[deg_indices]
            label = generate_label(degs)
            noisy_signal = generate_complex_signal(N=N, snr_db=snr_db, deg=degs)
            all_signals.append(noisy_signal)
            all_labels.append(label)
        torch.save(all_signals, os.path.join(signal_folder, f'signals_snr_{snr_db}dB.pt'))
        torch.save(all_labels, os.path.join(label_folder, f'labels_snr_{snr_db}dB.pt'))
    return None 
 

class SignalDataset(Dataset):
    def __init__(self, file_paths, label_paths):
        """
        Initializes a dataset containing signals and their corresponding labels.

        Args:
            file_paths (list): Paths to files containing signals.
            label_paths (list): Paths to files containing labels.
        """
        self.signals = [torch.stack(torch.load(file), dim=0) for file in file_paths]
        self.labels = [torch.stack(torch.load(label), dim=0) for label in label_paths]
        self.signals = torch.cat(self.signals, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

def create_dataloader(data_path, batch_size=32, shuffle=True):
    """
    Create a DataLoader for batching and shuffling the dataset.

    Args:
        data_path (str): Path to the directory containing the data files.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: Configured DataLoader for the dataset.
    """
    signal_dir_path = os.path.join(data_path, "signal")
    label_dir_path = os.path.join(data_path, "label")
    signal_files = [os.path.join(signal_dir_path, f) for f in os.listdir(signal_dir_path) if 'signals' in f]
    label_files = [os.path.join(label_dir_path, f) for f in os.listdir(label_dir_path) if 'labels' in f]
    dataset = SignalDataset(sorted(signal_files), sorted(label_files))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)  