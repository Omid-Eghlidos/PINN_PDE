import torch.nn as nn
import torch.optim as optim


class Optimizer():
    ''' A class that optimizes the loss in each iteration of the training of
    a model for the given problem.

    Attributes
    ----------
    gamma      (int): The leraning rate of the optimization
    problem (object): An object that contains the loss function of the given problem
    model   (object): An object of a given model

    Methods
    -------
    Adam     (public): None -> numpy.array
        Start optimization using Adam algorithm of the pytorch
    LBFGS    (public): None -> numpy.array
        Start optimization using LBFGS algorithm of the pytorch
    closure (private): None -> torch.tensor
        An auxiliary function needed as an input for the LBFGS step function
    verbose (private): None -> None
        An auxiliary function to print the optimizer's settings on screen
    '''

    def __init__(self, settings, problem, model, verbose):
        print(f'---- Initializing the optimizer')
        self.gamma   = settings.optimizer['gamma']
        self.method  = settings.optimizer['method']
        self.epochs  = settings.optimizer['epochs']
        self.problem = problem
        self.model   = model

        if self.method == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), lr=self.gamma)
        elif self.method == 'LBFGS':
            self.optimizer = optim.LBFGS(model.parameters(), lr=self.gamma,
                                                 line_search_fn='strong_wolfe')
        if verbose:
            self.__verbose()

    def Adam(self):
        self.optimizer.zero_grad()
        loss = self.problem.Loss(self.model)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def LBFGS(self):
        self.optimizer.step(self.__closure)
        loss = self.__closure()
        return loss.detach().cpu().numpy()

    def __closure(self):
        self.optimizer.zero_grad()
        loss = self.problem.Loss(self.model)
        loss.backward()
        return loss

    def __verbose(self):
        ''' Give a brief overview of the optimizer and its settings. '''
        print('\n'+15*' '+19*'-'+'  Optimizer Settings  '+19*'-')
        print(15*' '+f'{"Method":^14}: {self.method:^11}')
        print(15*' '+f'{"Epochs":^14}: {self.epochs:^11d}')
        print(15*' '+f'{"Learning rate":^14}: {self.gamma:^11.5g}')
        print(15*' '+60*'-'+'\n')

