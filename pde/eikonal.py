import numpy
import torch
from torch.autograd import grad


class Eikonal():
    ''' Initialize the initial and boundary conditions and the grid points
    for the LDC PDEs to compute the total loss and the test results.

    Attributes
    ----------
    processor (str): Processor, i.e., cpu or cuda, to use for the training
    eps (float): Constant factor controlling the smoothing effect of regularization
    xc (tensor): Combined boundary values along x
    yc (tensor): Combined boundary values along y
    uc (tensor): Combined boundary values for velocity component u
    vc (tensor): Combined boundary values for velocity component v
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
        self.eps = settings.eikonal['eps']
        self.ci  = settings.optimizer['ci'][0]
        self.icbc = settings.optimizer['icbc']
        self.test  = settings.optimizer['ci'][1]

        # Boundaries of the 2D domain
        # Bottom and top boundaries: xbt = [0, 1], ybt = {0, 1}
        xbt = torch.rand(2*self.icbc, 1)
        ybt = torch.cat([torch.zeros(self.icbc, 1), torch.ones(self.icbc, 1)])
        # Left and right boundaries: xlr = {0, 1}, ylr = [0, 1]
        xlr = torch.cat([torch.zeros(self.icbc, 1), torch.ones(self.icbc, 1)])
        ylr = torch.rand(2*self.icbc, 1)
        # Boundary conditions for u: u(x,0) = u(x,1) = 0 and u(0,y) = u(1,y) = 0
        ubt = torch.zeros(2*self.icbc, 1)
        ulr = torch.zeros(2*self.icbc, 1)
        # Combine the boundaries
        self.xc = torch.cat([xbt, xlr]).to(self.processor)
        self.yc = torch.cat([ybt, ylr]).to(self.processor)
        self.uc = torch.cat([ubt, ulr]).to(self.processor)

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
        x = numpy.linspace(0, 1, self.test)
        y = numpy.linspace(0, 1, self.test)
        X, Y = numpy.meshgrid(x, y)
        X_flat = torch.Tensor(X.flatten())[:, None]
        Y_flat = torch.Tensor(Y.flatten())[:, None]
        # Predict the velocity components using the trained model
        model.eval()
        with torch.no_grad():
            U = model(torch.cat([X_flat, Y_flat], dim=1)).cpu().numpy().reshape(X.shape)
        # Return all the results in a dictionary
        return dict(x=X, y=Y, u=U)

    def __PDE(self, model):
        # Random collocation points (CI) in PDE domain of x*y: [0,1]x[0,1]
        xp  = torch.rand((self.ci, 1), requires_grad=True).to(self.processor)
        yp  = torch.rand((self.ci, 1), requires_grad=True).to(self.processor)
        # Model prediction for the CI points
        U   = model(torch.cat([xp, yp], dim=1))
        # Gradient of the velocity components and pressure
        Ux  = grad(U, xp, grad_outputs=torch.ones_like(U), create_graph=True)[0]
        Uy  = grad(U, yp, grad_outputs=torch.ones_like(U), create_graph=True)[0]
        # Second derivative of the velocity components for computing the Laplacian
        Uxx = grad(Ux, xp, grad_outputs=torch.ones_like(Ux), create_graph=True)[0]
        Uyy = grad(Uy, yp, grad_outputs=torch.ones_like(Uy), create_graph=True)[0]
        # Eikonal's PDE prediction
        return Ux**2 + Uy**2 - self.eps * (Uxx + Uyy) - 1

    def __ICBC(self, model):
        # Model predictions for the IC/BC points
        return model(torch.cat([self.xc, self.yc], dim=1))

    def __verbose(self):
        print('\n'+15*' '+22*'-'+'  Eikonal PDE '+22*'-')
        print(15*' '+f'{"PDE":^10}: {"u_x^2 + u_y^2 - eps*(u_xx + u_yy) = 1, x,y=(0,1)^2"}')
        print(15*' '+f'{"BCs":^10}: {"u(x,0) = u(x,1) = 0, x=[0,1]"}')
        print(27*' '+f'{"u(0,y) = u(1,y) = 0, y=[0,1]"}')
        print(15*' '+60*'=')
        print(15*' '+f'{"Parameters":^10}: eps (smoothing factor) = {self.eps:^6.2f}')
        print(27*' '+f'CI (collocation points) = {self.ci}')
        print(27*' '+f'ICBC (ICs/BCs points) = {2*self.icbc}')
        print(15*' '+60*'-'+'\n')

