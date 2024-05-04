import torch
import torch.nn as nn
import sys
sys.path.append('../')
from scr.helpers import steering_vector

    
class DOANet(nn.Module):
    def __init__(self, number_element=20, output_size=61):
        super(DOANet, self).__init__()
        # Layer configurations
        self.input_size = number_element
        self.output_size = output_size
        hidden_sizes = [2048, 1024, 512, 256, 128]  # Adjustable hidden layer sizes
        
        # Network layers
        layers = []
        for h1, h2 in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes):
            layers.append(nn.Linear(h1, h2))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], self.output_size))
        layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
    
class SparseLayer(nn.Module):
    def __init__(self, input_size=10, max_sparsity=0.5):
        super(SparseLayer, self).__init__()
        self.input_size = input_size
        self.max_zeros = int(input_size * max_sparsity)

    def forward(self, x):
        if self.training:
            batch_size,N = x.size()  # Get the batch size
            sparsity = torch.zeros(batch_size, dtype=torch.long).to(x.device)  # Tensor to store the number of zeros
            # Generate a random mask for each example in the batch
            masks = torch.ones((batch_size, self.input_size)).to(x.device)  # Start with all ones
            NN = torch.ones(batch_size) * N
            num_zeros = torch.randint(0, self.max_zeros + 1, (batch_size,))
            for i in range(batch_size):
              zero_indices = torch.randperm(self.input_size)[:num_zeros[i]]  # Random indices to be zeroed
              masks[i, zero_indices] = 0  # Set selected indices to zero
            sparsity = NN - num_zeros  # Store the number of zeros used for this mask
            x_sparse = x * masks

        else:
            x_sparse = x
            sparsity, masks = self.thresholding(x)

        return x_sparse, sparsity, masks.to(x_sparse.dtype).to(x_sparse.device)


    def thresholding(self, x):
        threshold = 0.001
        mask = (torch.abs(x) > threshold).float()
        return torch.sum(mask, dim=1), mask
    

class SALayer(nn.Module):
    def __init__(self, number_element=10, output_size=61, max_sparsity=0.5):
        super(SALayer, self).__init__()
        # Initialize SparseLayer
        self.sparselayer = SparseLayer(number_element, max_sparsity)
        self.output_size = output_size
        self.hidden_size = 512
        
        # Calculate steering vectors
        a_theta = steering_vector(number_element, torch.arange(-30, 31)).conj()
        self.AH = torch.transpose(a_theta, 0, 1)

        # Define the network layers
        self.linear_layer = nn.Linear(number_element * 2, self.hidden_size)
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        batch_size, N = x.size()
        # Sparse representation and FFT
        x_sparse, sparsity, masks = self.sparselayer(x)
        x_fft = self.apply_fft(x_sparse, sparsity, batch_size)
        masks_fft = self.apply_fft(masks, sparsity, batch_size)

        # Flatten the sparse input and apply linear transformation
        x_flat = torch.view_as_real(x_sparse).view(batch_size, -1)
        embedded_values = self.relu(self.linear_layer(x_flat))
        normalized_values = self.normalize(embedded_values, sparsity)

        # Concatenate features from different processing streams
        output = torch.cat((masks_fft, x_fft, normalized_values), dim=1)
        return output

    def normalize(self, x, sparsity):
        """Normalize the data by the sparsity-derived factor."""
        normalization_factor = sparsity.unsqueeze(-1).to(x.device)
        return x / normalization_factor

    def apply_fft(self, x, sparsity, batch_size):
        """Apply FFT to the input tensor and adjust based on batch size."""
        AH_batched = self.AH.unsqueeze(0).repeat(batch_size, 1, 1)
        x_expanded = x.unsqueeze(-1)
        fft_output = torch.abs(torch.matmul(AH_batched.to(x_expanded.device), x_expanded)).squeeze(-1)
        return self.normalize(fft_output,sparsity)
    

class SADOANet(nn.Module):
    def __init__(self, number_element=10, output_size=61, max_sparsity=0.5, is_sparse=True):
        super(SADOANet, self).__init__()
        self.salayer = SALayer(number_element, output_size, max_sparsity)
        input_size = 512 + output_size * 2 if is_sparse else number_element * 2
        self.doanet = DOANet(input_size, output_size)
        self.is_sparse = is_sparse

    def forward(self, x):
        if len(x.size()) == 3: # signal pass in as real value
            x = torch.view_as_complex(x)
        if self.is_sparse:
            x = self.salayer(x)
        else:
            x = torch.view_as_real(x).view(x.size(0), -1)
        return self.doanet(x)
    
    