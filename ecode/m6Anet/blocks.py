r"""
This module is a collection of m6Anet building blocks
"""
import torch.nn.functional as F
import torch
from torch import nn
from typing import Dict, Optional


def get_activation(activation: str) -> torch.nn.Module:
    r'''
    Instance method to get modification probability on the site level from read features.

            Args:
                    activation (str): A string that corresponds to the desired activation function. Must be one of ('tanh', 'sigmoid', 'relu', 'softmax')
            Returns:
                    activation_func (torch.nn.Module): A PyTorch activation function
    '''
    allowed_activation = ('tanh', 'sigmoid', 'relu', 'softmax')
    activation_func = None
    if activation == 'tanh':
        activation_func = nn.Tanh()
    elif activation == 'sigmoid':
        activation_func = nn.Sigmoid()
    elif activation == 'relu':
        activation_func = nn.ReLU()
    elif activation == 'softmax':
        activation_func = nn.Softmax(dim=1)
    else:
        raise ValueError("Invalid activation, must be one of {}".format(allowed_activation))

    return activation_func


class Block(nn.Module):
    r"""
    The basic building block of m6Anet model
    """
    def __init__(self):
        super(Block, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        The forward method should be implemented in subclasses
        """
        raise NotImplementedError("Subclasses should implement this method")


class ConcatenateFeatures(Block):
    r"""
    Block object to concatenate several different features (i.e, k-mer embedding and signal features) into one high dimensional tensor
    """
    def __init__(self):
        super(ConcatenateFeatures, self).__init__()

    def forward(self, x: Dict) -> torch.Tensor:
        r'''
        Instance method to concatenate all tensor values in dictionary as one high dimensional tensor

                Args:
                        x (dict): Dictionary containing tensor values

                Returns:
                        x (torch.Tensor): PyTorch tensor from concatenating all values in the dictionary input
        '''
        x = torch.cat([x['X'],x['kmer']], axis=-1)
        return x


class ExtractSignal(Block):
    r"""
    Block object to extract only the signal features from input argument
    """
    def __init__(self):
        super(ExtractSignal, self).__init__()

    def forward(self, x):
        r'''
        Instance method to extract only the signal features from the input. The signal value must have the key 'X' in the dictionary input

                Args:
                        x (dict): Dictionary containing tensor values

                Returns:
                        x (torch.Tensor): PyTorch tensor containing the signal value corresponding to the key 'X' in the input dictionary
        '''
        return x['X']



class Flatten(Block):
    r"""
    Block object that acts as a wrapper for torch.nn.Flatten
    ...

    Attributes
    -----------
    layers (nn.Module): PyTorch nn.Flatten
    """
    def __init__(self, start_dim, end_dim):
        r'''
        Initialization function for the class

                Args:
                        start_dim (int): Starting dimension to flatten
                        end_dim (int): Ending dimension to flatten

                Returns:
                        None
        '''
        super(Flatten, self).__init__()
        self.layers = nn.Flatten(start_dim, end_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Instance method to flatten the target tensor

                Args:
                        x (torch.Tensor): Tensor input

                Returns:
                        x (torch.Tensor): Flattened tensor output according to the specified start_dim and end_dim during initialization
        '''
        return self.layers(x)


class KmerMultipleEmbedding(Block):
    r"""
    Block object that applies PyTorch embedding layer to sequence information from nanopolish input
    ...

    Attributes
    -----------
    input_channel (int): Number of unique 5-mer motifs to be embedded
    output_channel (int): Output dimension of the transformed 5-mer motif
    embedding_layer (nn.Module): PyTorch nn.Embedding layer to transform categorical variable into vectors
    n_features (int): Number of features in the signal data
    """
    def __init__(self, input_channel: int, output_channel:int, num_neighboring_features: Optional[int] = 1):
        r'''
        Initialization function for the class

                Args:
                        input_channel (int): Number of unique 5-mer motifs to be embedded
                        output_channel (int): Output dimension of the transformed 5-mer motif
                        num_neighboring_features (int): Number of flanking positions around the target site

                Returns:
                        None
        '''
        super(KmerMultipleEmbedding, self).__init__()
        self.input_channel, self.output_channel = input_channel, output_channel
        self.embedding_layer = nn.Embedding(input_channel, output_channel)
        self.n_features = 2 * num_neighboring_features + 1

    def forward(self, x: Dict) -> Dict:
        r'''
        Instance method to apply embedding layer on sequence features, transforming them into high dimensional vector representation

                Args:
                        x (dict): Python dictionary containing signal features and sequence feature

                Returns:
                        (dict): Python dictionary containing signal features and transformed sequence features
        '''
        kmer =  self.embedding_layer(x['kmer'].long())
        return {'X': x['X'], 'kmer' :kmer.reshape(list(kmer.shape[:-2])+[-1])}


class Linear(Block):
    r"""
    Block object that applies PyTorch Linear layer, BatchNorm and Dropout
    ...

    Attributes
    -----------
    layers (nn.Module): A sequence of PyTorch Module classes
    """
    def __init__(self, input_channel, output_channel, activation='relu', batch_norm=True,n_reads_per_site= 50, dropout=0.0):
        r'''
        Initialization function for the class

                Args:
                        input_channel (int): Number of input dimension
                        output_channel (int): Number of output dimension
                        activation (str): Activation function
                        batch_norm (bool): Whether to use BatchNorm or not
                        dropout (float): Dropout value

                Returns:
                        None
        '''
        super(Linear, self).__init__()
        self.layers = self._make_layers(input_channel, output_channel, activation, batch_norm, n_reads_per_site, dropout)

    def _make_layers(self, input_channel: int, output_channel: int, activation: str, batch_norm: bool,n_reads_per_site:int, dropout: Optional[float] = 0.0):
        r'''
        Function to construct PyTorch Sequential object for this class, which comprised of a single
        Linear layer along with BatchNorm1d and possibly a dropout

                Args:
                        input_channel (int): Number of input dimension
                        output_channel (int): Number of output dimension
                        activation (str): Activation function
                        batch_norm (bool): Whether to use BatchNorm or not
                        dropout (float): Dropout value

                Returns:
                        None
        '''
        layers = [nn.Linear(input_channel, output_channel)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(num_features=n_reads_per_site))
        if activation is not None:
            layers.append(get_activation(activation))
        layers.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Instance method to apply linear layer on tensor features

                Args:
                        x (torch.Tensor): Tensor input
                Returns:
                        (torch.Tensor): Transformed tensor output
        '''
        X=self.layers(x)
        return X

####################################################################
#pooling blocks

class PoolingFilter(nn.Module):
    r"""
    The abstract class of m6Anet pooling filter layer
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def predict_read_level_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def set_num_reads(self, n_reads_per_site: int):
        self.n_reads_per_site = n_reads_per_site



class InstanceBasedPooling(PoolingFilter):
    r"""
    An abstract class for instance based pooling approach
    ...

    Attributes
    -----------
    input_channel (int): The input dimension of the read features passed to this module
    n_reads_per_site (int): Number of reads expressed for each transcriptomic site
    probability_layer (nn.Module): A PyTorch linear layer that outputs read-level probability for each read
    """
    def __init__(self, input_channel: int, n_reads_per_site: Optional[int] = 20):
        r'''
        Initialization function for the class

                Args:
                    input_channel (int): The input dimension of the read features passed to this module
                    n_reads_per_site (int): Number of reads expressed for each transcriptomic site
                    probability_layer (nn.Module): A PyTorch linear layer that outputs read-level probability for each read

                Returns:
                        None
        '''
        super(PoolingFilter, self).__init__()
        self.input_channel = input_channel
        self.n_reads_per_site = n_reads_per_site
        self.probability_layer = nn.Sequential(*[nn.Linear(input_channel, 1), get_activation('sigmoid')])

    def predict_read_level_prob(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Instance based method that takes in transform high dimensional read features and output modification probability for each read

                Args:
                    x (torch.Tensor): The input read-level tensor representation

                Returns:
                        (torch.Tensor): The output read level modification probability
        '''
        return self.probability_layer(x).view(-1, self.n_reads_per_site)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

class SigmoidProdPooling(InstanceBasedPooling):
    r"""
    A noisy-OR pooling layer that computes site probability by calculating the probability that at least one read is modified

    ...

    Attributes
    -----------
    input_channel (int): The input dimension of the read features passed to this module
    n_reads_per_site (int): Number of reads expressed for each transcriptomic site
    probability_layer (nn.Module): A PyTorch linear layer that outputs read-level probability for each read
    """
    def __init__(self, input_channel: int, n_reads_per_site: Optional[int] = 20):
        r'''
        Initialization function for the class

                Args:
                    input_channel (int): The input dimension of the read features passed to this module
                    n_reads_per_site (int): Number of reads expressed for each transcriptomic site
                    probability_layer (nn.Module): A PyTorch linear layer that outputs read-level probability for each read

                Returns:
                        None
        '''
        super(SigmoidProdPooling, self).__init__(input_channel, n_reads_per_site)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        read_level_prob = self.predict_read_level_prob(x)
        return 1 - torch.prod(1 - read_level_prob, axis=1)

