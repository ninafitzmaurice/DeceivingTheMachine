import torch
from torch import nn
from torch.nn import functional as F
import math
import matplotlib.pyplot as plt


class InsNorm(nn.Module):
    """
    Normalize the the input by the norm of each channel. 

    = consistent feature scale across layers,
    see papers assumption of constant euclidian-norm across feedback features g(k)

    avoids divergence in MAP estimation
    """
    def forward(self, x, std=False, returnnorm=False):
        if (std==True):
            normx = torch.sqrt(torch.var(x, dim=(2,3)).reshape([x.shape[0],x.shape[1],1,1]) + 1e-8)
        else:
            normx = torch.sqrt(torch.mean(x**2, dim=(2,3)).reshape([x.shape[0],x.shape[1],1,1]) + 1e-8)
        if returnnorm==True:
            return x/normx, normx
        else:
            return x/normx

class Flatten(nn.Module):
    """
    Flattens the input into vector and reconstructs spatial feature maps from vectors.
    """
    def forward(self, x, step='forward'):
        if 'forward' in step:
            self.size = x.size()
            batch_size = x.size(0)
            return x.view(batch_size, -1)

        elif 'backward' in step:
            batch_size = x.size(0)
            # reshape tensor to original shape in backwards pass 
            # maintaining structural consistency of latent spatial information
            return x.view(batch_size, *self.size[1:])

        else:
            raise ValueError("step must be 'forward' or 'backward'")


class Conv2d(nn.Module):
    """
    Applies a 2D convolution over the input. In the feedback step,
    a transposed 2D convolution is applied to the input with the same weights
    as the 2D convolution.

    This results in Gaussian spatial latent variable priors
    ---> BECAUSE He initialization of weights in forward step
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=False, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              bias=bias, **kwargs)
        
        if self.conv.stride[0] > 1:        
            self.conv_t = nn.ConvTranspose2d(out_channels, in_channels,
                                         kernel_size, bias=bias, output_padding=1, **kwargs)
        else:
            self.conv_t = nn.ConvTranspose2d(out_channels, in_channels,
                                         kernel_size, bias=bias, **kwargs)
            
        n = self.conv.kernel_size[0] * self.conv.kernel_size[1] * self.conv.out_channels
        self.conv.weight.data.normal_(0, math.sqrt(2. / n))  # He initialization
        
        # Tie the weights between the convolution and the transposed convolution
        '''
        By sharing weights between the forward convolution (self.conv) 
        and backward transposed convolution (self.conv_t), this ensures that the feedback 
        pass aligns with the learned priors from the forward pass.
        '''
        self.conv_t.weight = self.conv.weight

    def forward(self, x, step='forward'):
        if 'forward' in step:
            return self.conv(x)

        ####### BACKWARD PASS: GENERATION
        elif 'backward' in step:

            '''This upsamples the images in backward pass, can extract
                image from here by plotting it:

                upsampled_output = self.conv_t(x)
                single_image = upsampled_output[0]

                generates higher-resolution spatial latents
                '''
            return self.conv_t(x)

        else:
            raise ValueError("step must be 'forward' or 'backward'")


class Linear(nn.Module):
    """
    Applies a linear transform over the input. In the feedback step,
    a transposed linear transform is applied to the input with the transposed
    weights of the linear transform.
    """
    def __init__(self, in_features, out_features, bias=False, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear_t = nn.Linear(out_features, in_features, bias=bias)
        
        if(bias==True):
            self.linear.bias.data.zero_()
        # Tie the weight of the transposed linear transform with the
        # transposed weights of the linear transform
        self.linear_t.weight.data = self.linear.weight.data.t()

        # Make the weight publicly accessible
        self.weight = self.linear.weight

    def forward(self, x, step='forward'):
        if 'forward' in step:
            return self.linear(x)

        elif 'backward' in step:
            # Tie the weight of the transposed linear transform with the
            # transposed weights of the linear transform
            self.linear_t.weight.data = self.linear.weight.data.t()

            return self.linear_t(x)

        else:
            raise ValueError("step must be 'forward' or 'backward'")


class ReLU(nn.Module):
    """
    AdaReLU
    """
    def __init__(self):
        super().__init__()
        self.state = None
        self.hidden = None

    def forward(self, x, null_space=None, unit_space=None, step='forward'):
        if 'forward' in step:
            # Store which weights were activated in forward pass
            if self.hidden is None:
                self.state = (x > 0) 

            else:
                self.state = (x * self.hidden) > 0
            result = x * self.state.float()
                
            return result

        elif 'backward' in step:
            # ONLY units that were activated in the forward step are passed through
            self.hidden = x
            masked_hidden = x * self.state.float()
            if unit_space is not None:
                return masked_hidden, unit_space
            else:
                return masked_hidden

        else:
            raise ValueError("step must be 'forward' or 'backward'")

    def reset(self):
        self.hidden = None
        self.state = None


class resReLU(nn.Module):
    """
    custom AdaReLU includes a "residual" effect:
    where hidden states can be partially retained across cycles of forward and 
    backward passes:

    the inverse ReLU operation enforces a form of posterior constraint (prior)
    by conditioning the reconstruction on only those features that are 
    likely under the models weights (from layer+1 to current layer). 
    
    Since ReLU in the forward pass implicitly 
    establishes this "prior" on positive activations, its inverse in the backward pass 
    is like a "posterior update", where only features that passed the ReLU 
    activation are allowed to reconstruct. aka the features are sampled. 
    """
    def __init__(self, res_param=0.1):
        super().__init__()
        # Tracks whether units were activated in the forward pass (used as a mask in backward)
        self.state = None  
        # Stores intermediate hidden states across feedback cycles
        self.hidden = None
        # Parameter that controls the strength of residual connections
        self.res_param = res_param

    def forward(self, x, null_space=None, unit_space=None, step='forward'):
        if 'forward' in step:
            '''
            after the first forward pass
            self.hidden is used in the subsequent forward pass(es) to update x with a 
            a mix of the previous hidden state from layer+1, 
            that carries over weighted (by res_param) information from the feedback cycles.
            '''
            # In the forward pass, compute ReLU activation and apply residual updates if hidden states exist
            if self.hidden is None:
                # If no hidden state is stored, simply apply ReLU by setting state to (x > 0),
                # storing which units are activated in the current forward pass.
                self.state = (x > 0)

            else:
                # If a hidden state exists (from a previous backward cycle), incorporate it using
                # a residual update: blend x with hidden using res_param, then update state.
                x = x + self.res_param * (self.hidden - x)
                # Update `state` to indicate where the units are activated after residual blending
                self.state = (x * self.hidden) > 0

            # Apply ReLU by masking `x` with `state`, keeping only positive values
            result = x * self.state.float()
            return result

        elif 'backward' in step:
            '''
            Reconstruction: In the backward pass, x is masked by self.state, 
            which retains only those units that were activated in the forward pass. 
            aka a prior over latent features, ensuring that only 
            activations likely to align with prior expectations are reconstructed.
            '''
            # In the backward pass, ReLU inversion is applied:
            # Only units activated in the forward pass are allowed to pass through

            # Save the current input as `hidden` for potential use in a future forward pass
            self.hidden = x

            # Mask `x` with `state` to reconstruct only the units that were active in the forward pass
            masked_hidden = x * self.state.float()

            if unit_space is not None:
                # If `unit_space` is provided, return it along with masked hidden states
                return masked_hidden, unit_space
            else:
                # Otherwise, return only the masked hidden states
                return masked_hidden

        else:
            # Error handling for invalid `step` values
            raise ValueError("step must be 'forward' or 'backward'")

    def reset(self):
        # Clears `hidden` and `state`, resetting the layer’s memory of past activations
        self.hidden = None
        self.state = None


class AvgPool2d(nn.Module):
    
    def __init__(self, kernel_size, scale_factor=10, **kwargs):
        super().__init__()
        
        self.avgpool = nn.AvgPool2d(kernel_size, **kwargs)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=scale_factor, **kwargs)  # feedforward, before avgpool, size is 10

    def forward(self, x, step='forward'):
        if 'forward' in step:
            return self.avgpool(x)

        elif 'backward' in step:
            return self.upsample(x)

        else:
            raise ValueError("step must be 'forward' or 'backward'")
        
        
class MaxPool2d(nn.Module):
    """
    AdaPool:
    In the feedforward pass, if the pixel in g that governs this grid is > 0, do maxpool on this grid
    In the feedforward pass, if the pixel in g that governs this grid is < 0, do min pool on this grid
    In the feedback pass, if the pixel comes from a max value, put it back to the position of the max value
    In the feedback pass, if the pixel comes from a min value, put it back to the position of the min value
    """
    def __init__(self, kernel_size, **kwargs):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, return_indices=True, **kwargs)
        self.unpool = nn.MaxUnpool2d(kernel_size, **kwargs)
        
        self.hidden = None
        self.pos_state = None
        self.neg_state = None
        self.pos_ind = None
        self.neg_ind = None
        self.unpoolsize = None

    def forward(self, x, null_space=None, unit_space=None, step='forward'):
        if 'forward' in step:
            if self.hidden is None:
                self.unpoolsize = x.shape
                max_pool, self.pos_ind = self.maxpool(x)   # self.state is the indices of maxpool
                return max_pool
                
            else:
                max_pool, self.pos_ind = self.maxpool(x)
                min_pool, self.neg_ind = self.maxpool(-x)
                min_pool = -min_pool
                self.pos_state = (self.hidden > 0).float()
                self.neg_state = (self.hidden < 0).float()
                out = self.pos_state * max_pool + self.neg_state * min_pool
                return out

        elif 'backward' in step:
            self.hidden = x
            if ((self.pos_state is None) or (self.neg_state is None)):  # reconstruction in the first iteration, using maxpool states
                max_unpool = self.unpool(x, self.pos_ind, output_size=self.unpoolsize)
                return max_unpool
            else:            
                max_unpool = self.unpool(x * self.pos_state, self.pos_ind, output_size=self.unpoolsize)
                min_unpool = self.unpool(x * self.neg_state, self.neg_ind, output_size=self.unpoolsize)
                return max_unpool + min_unpool

        else:
            raise ValueError("step must be 'forward' or 'backward'")

    def reset(self):
        self.hidden = None
        self.pos_state = None
        self.neg_state = None
        self.pos_ind = None
        self.neg_ind = None


class Bias(nn.Module):
    """
    Add a bias to the input. In the feedback step, the bias is subtracted
    from the input.
    """
    def __init__(self, size):
        super().__init__()
        self.bias = nn.Parameter(torch.Tensor(*size))
        self.bias.data.zero_()

    def forward(self, x, step='forward'):
        if 'forward' in step:
            return x + self.bias

        # no bias in backwards pass
        elif 'backward' in step:
            self.x = x
            return x

        else:
            raise ValueError("step must be 'forward' or 'backward'")

    def path_norm_loss(self, unit_space):
        return torch.mean((self.x * self.bias - unit_space * self.bias)**2)


class Dropout(nn.Module):
    """
    Performs dropout regularization to the input. In the feedback step, the 
    same dropout transformation is applied in the backwards step.
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x, step='forward'):
        if 'forward' in step:
            if self.training:
                self.dropout = (torch.rand_like(x) > self.p).float() / self.p
                return x * self.dropout
            else:
                return x

        elif 'backward' in step:
            if self.training:
                return x * self.dropout
            else:
                return x

        else:
            raise ValueError("step must be 'forward' or 'backward'")
        
