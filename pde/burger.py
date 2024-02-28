import numpy
import torch
from torch.autograd import grad


class Burger():
    ''' Initialize the initial and boundary conditions and the grid points
    for the Burger PDE to compute the total loss and the test results.

    Attributes
    ----------
    processor (str): Processor, i.e., cpu or cuda, to use for the training
    nu  (float): Kinematic viscosity
    ci    (int): Number of collocation points (CIs)
    tc (tensor): Combined initial and boundary values for time
    xc (tensor): Combined initial and boundary values for space
    uc (tensor): Combined initial and boundary values for velocity
    loss (object): Instance of torch MSE Loss function with mean reduction

    Methods:
    --------
    Loss (public): (object) -> torch.tensor
        Computes the loss in satisfying the ICs/BCs and the PDE.
    Results (public): (object) -> dict
        Compute the results for the given test settings and return the results of the PDE.
    PDE (private): (object) -> torch.tensor
        Computes the PDE for model prediction obtained using CIs.
    ICBC (private): (object) -> torch.tensor
        Computes the ICs/BCs for prediction obtained using combined initial boundary values.
    verbose (private): (None) -> None
        Provide details about the PDE on the secreen.
    '''

    def __init__(self, settings, verbose):
        self.processor = settings.nn['processor']
        # PDE parameters
        self.nu = settings.burger['nu']
        self.ci = settings.optimizer['ci'][0]
        self.icbc = settings.optimizer['icbc']
        self.test = settings.optimizer['ci'][1]

        # Initial conditions: x0 = [-1, 1], t0 = 0, u(x0,0) = -sin(pi*x0)
        t0 = torch.zeros(self.icbc, 1)
        x0 = torch.rand(self.icbc, 1) * 2 - 1
        u0 = -torch.sin(numpy.pi * x0)
        # Boundary conditions: xb = {-1, +1}, tb = [0, 1], ub = 0
        tb = torch.rand(2*self.icbc, 1)
        xb = torch.cat([torch.ones(self.icbc, 1), -torch.ones(self.icbc, 1)])
        ub = torch.zeros(2*self.icbc, 1)
        # Combined IC/BC conditions
        self.tc = torch.cat([t0, tb]).to(self.processor)
        self.xc = torch.cat([x0, xb]).to(self.processor)
        self.uc = torch.cat([u0, ub]).to(self.processor)

        # Define the Mean Square Error as the loss function
        self.loss = torch.nn.MSELoss(reduction='mean')

        if verbose:
            self.__verbose()

    def Loss(self, model):
        U = torch.cat([self.__ICBC(model), self.__PDE(model)])
        up = torch.zeros((self.ci, 1), requires_grad=True).to(self.processor)
        u = torch.cat([self.uc, up])
        return self.loss(U, u)

    def Results(self, model):
        t = numpy.linspace(0, 1, self.test)
        x = numpy.linspace(-1, 1, self.test)
        T, X = numpy.meshgrid(t, x)
        T_flat = torch.Tensor(T.flatten())[:, None]
        X_flat = torch.Tensor(X.flatten())[:, None]
        # Predict the velocity components using the trained model
        model.eval()
        with torch.no_grad():
            U = model(torch.cat([T_flat, X_flat], dim=1))
            U = U.cpu().numpy().reshape(X.shape)
        # Save all the results in a dictionary
        return dict(t=T, x=X, u=U)

    def __PDE(self, model):
        # Random collocation points (CI) in PDE domain of x*t: [-1,1]x[0,1]
        tp = torch.rand((self.ci, 1), requires_grad=True).to(self.processor)
        xp = (torch.rand((self.ci, 1), requires_grad=True) * 2 - 1).to(self.processor)
        # Model prediction for the CI points
        U   = model(torch.cat([tp, xp], dim=1))
        # Gradient of the velocity component
        Ut  = grad(U, tp, grad_outputs=torch.ones_like(U), create_graph=True)[0]
        Ux  = grad(U, xp, grad_outputs=torch.ones_like(U), create_graph=True)[0]
        Uxx = grad(Ux, xp, grad_outputs=torch.ones_like(Ux), create_graph=True)[0]
        # Burger's PDE prediction
        return Ut + U*Ux - self.nu*Uxx

    def __ICBC(self, model):
        # Model predictions for the IC/BC points
        return model(torch.cat([self.tc, self.xc], dim=1))

    def __verbose(self):
        print('\n'+15*' '+23*'-'+'  Burger PDE '+23*'-')
        print(15*' '+f'{"PDE":^10}: {"u_t + u*u_x - nu*u_xx = 0, x=(-1,1), t=(0,1]"}')
        print(15*' '+f'{"BCs":^10}: {"u(-1,t) = u(1,t) = 0, t=[0,1]"}')
        print(27*' '+f'{"u(x,0) = -sin(pi*x), x=[-1,1]"}')
        print(15*' '+60*'=')
        print(15*' '+f'{"Parameters":^10}: nu (kinematic viscosity) = {self.nu:^6.4f}')
        print(27*' '+f'CI (collocation points) = {self.ci}')
        print(27*' '+f'ICBC (ICs/BCs points) = {2*self.icbc}')
        print(15*' '+60*'-'+'\n')

