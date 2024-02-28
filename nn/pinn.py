import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from pde.burger import Burger
from pde.eikonal import Eikonal
from pde.elliptical import Elliptical
from pde.helmholtz import Helmholtz
from pde.ldc import LDC
from nn.dnn import DNN
from nn.optimizer import Optimizer
from in_out.progress import Progress
from in_out.outputs import output_results


def PINN(settings, verbose=False):
    ''' Initialize and train the PINN for the given PDE. '''
    # Set the problem for loss function
    if settings.pde == 'Burger':
        problem = Burger(settings, verbose)
    elif settings.pde == 'Elliptical':
        problem = Elliptical(settings, verbose)
    elif settings.pde == 'Helmholtz':
        problem = Helmholtz(settings, verbose)
    elif settings.pde == 'Eikonal':
        problem = Eikonal(settings, verbose)
    elif settings.pde == 'LDC':
        problem = LDC(settings, verbose)

    # Initialize the corresponding model
    pinn = DNN(settings, verbose).to(settings.nn['processor'])
    optimizer = Optimizer(settings, problem, pinn, verbose)
    print(f'---- PINN training')

    # Record the loss of each epoch
    lrec = []
    progress = Progress(settings.optimizer['epochs'], verbose)
    for epoch in range(settings.optimizer['epochs']+1):
        if settings.optimizer['method'] == 'Adam':
            loss = optimizer.Adam()
        elif settings.optimizer['method'] == 'LBFGS':
            loss = optimizer.LBFGS()
        if epoch % 100 == 0:
            lrec.append(loss)
            progress(epoch, loss)
    progress(total_time=True)

    output_results(settings.pde, settings.path, problem.Results(pinn), lrec)

