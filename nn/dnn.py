import torch
import torch.nn as nn


class DNN(nn.Module):
    ''' Setup an architecture for a deep neural network with the given settings.

    Attributes
    ----------
    layers (object): List of layers each initialized with a linear function
    activation (dict): Stores the activation function type for each layer as following:
        * hidden (str): Activation function for the hidden layers
        * output (str): Activation function for the output layer

    Methods
    --------
    forward (public): (torch.tensor) -> torch.tensor
        Feed forward with the sepcified activation functions for each layer of the network
    verbose (private): (None) -> None
        Give a brief overview of the network architecture
    '''

    def __init__(self, settings, verbose):
        print(f'---- Creating the DNN')
        super(DNN, self).__init__()
        self.processor = settings.nn['processor']
        # Number of neurons per layer
        N = settings.nn['neurons']
        # Number of inputs and outputs
        if settings.pde == 'Burger':
            N_in, N_out = 2, 1
        elif settings.pde == 'Elliptical':
            N_in, N_out = 2, 1
        elif settings.pde == 'Helmholtz':
            N_in, N_out = 2, 1
        elif settings.pde == 'Eikonal':
            N_in, N_out = 2, 1
        elif settings.pde == 'LDC':
            N_in, N_out = 2, 3

        # Input layer
        layers = [nn.Linear(N_in, N)]
        # Hidden layers
        for layer in range(settings.nn['layers']-2):
            layers.append(nn.Linear(N, N))
        # Output layer u
        layers.append(nn.Linear(N, N_out))
        self.layers = nn.ModuleList(layers)
        # Activation functions
        self.activation = dict(hidden=settings.nn['activation']['hidden'],
                               output=settings.nn['activation']['output'])
        if verbose:
            self.__verbose()

    def forward(self, x):
        x = x.to(self.processor)
        for i, l in enumerate(self.layers):
            if i == len(self.layers) - 1:
                # Activation of the output layer
                if self.activation['output'] == 'linear':
                    x = l(x)
            else:
                # Activation of the hidden layers
                if self.activation['hidden'] == 'tanh':
                    x = torch.tanh(l(x))
        return x

    def __verbose(self):
        print('\n'+15*' '+20*'-'+'  DNN Architecture  '+20*'-')
        print(15*' '+f'{"Layer":^11} {"Type":^11} {"Activation":^11} {"Input #":^11}'+
                     f' {"Output #":^11}\n'+
              15*' '+60*'=')
        for i, l in enumerate(self.layers):
            if i == 0:
                layer_type, activation = 'Input', self.activation['hidden']
            elif i == len(self.layers)-1:
                layer_type, activation = 'Output', self.activation['output']
            else:
                layer_type, activation = 'Hidden', self.activation['hidden']
            print(15*' '+f'{i+1:^11d} {layer_type:^11} {activation:^11} {l.in_features:^11d} '+
                         f'{l.out_features:^11d}\n')
        print(15*' '+60*'=')
        parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(15*' '+f'Total trainable parameters: {parameters:^11d}')
        print(15*' '+60*'-'+'\n')

