import numpy
import torch
from torch.autograd import grad


class LDC():
    ''' Initialize the initial and boundary conditions and the grid points
    for the LDC PDEs to compute the total loss and the test results.

    Attributes
    ----------
    processor (str): Processor, i.e., cpu or cuda, to use for the training
    nu  (float): Kinematic viscosity
    rho (float): Density value
    A   (float): Constant scaling factor
    L   (float): Characteristic length of the flow
    u_hat (float): Characteristic speed of the flow
    Re  (float): Reynolds number of the flow
    ci    (int): Number of collocation points (CIs)
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
        self.nu   = settings.ldc['nu']
        self.rho  = settings.ldc['rho']
        self.A    = settings.ldc['A']
        self.L    = settings.ldc['L']
        self.u_hat= numpy.trapz(self.A*numpy.sin(numpy.pi*numpy.linspace(0,1, 1000)), dx=0.001)
        self.Re   = self.rho*self.u_hat*self.L/self.nu
        self.ci   = settings.optimizer['ci'][0]
        self.icbc = settings.optimizer['icbc']
        self.test = settings.optimizer['ci'][1]

        # Boundaries of the 2D domain
        # Bottom and top boundaries: xbt = [0, 1], ybt = {0, 1}
        xbt = torch.rand(2*self.icbc, 1)
        ybt = torch.cat([torch.zeros(self.icbc, 1), torch.ones(self.icbc, 1)])
        # Left and right boundaries: xlr = {0, 1}, ylr = [0, 1]
        xlr = torch.cat([torch.zeros(self.icbc, 1), torch.ones(self.icbc, 1)])
        ylr = torch.rand(2*self.icbc, 1)
        # Boundary conditions for u: u = A*sin(pi*x) at the top, u = 0 on other boundaries
        ubt = torch.cat([torch.zeros(self.icbc, 1),
                         self.A*torch.sin(numpy.pi*torch.rand(self.icbc,1))])
        ulr = torch.zeros(2*self.icbc, 1)
        # Boundary conditions for v: v = 0 on all boundaries
        vbt = torch.zeros(2*self.icbc, 1)
        vlr = torch.zeros(2*self.icbc, 1)
        # Combine the boundaries
        self.xc = torch.cat([xbt, xlr]).to(self.processor)
        self.yc = torch.cat([ybt, ylr]).to(self.processor)
        self.uc = torch.cat([ubt, ulr]).to(self.processor)
        self.vc = torch.cat([ubt, ulr]).to(self.processor)
        # Initial condition for the p
        self.p0  = torch.zeros(1, 1).to(self.processor)

        # Define the Mean Square Error as the loss function
        self.loss = torch.nn.MSELoss(reduction='mean')

        if verbose:
            self.__verbose()

    def Loss(self, model):
        cp = torch.zeros((self.ci, 1), requires_grad=True).to(self.processor)
        mup = torch.zeros((self.ci, 1), requires_grad=True).to(self.processor)
        vup = torch.zeros((self.ci, 1), requires_grad=True).to(self.processor)
        uvp = torch.cat([self.p0, self.uc, self.vc, cp, mup, vup]).view(-1, 1)
        UVP = torch.cat([self.__ICBC(model), self.__PDE(model)]).to(self.processor)
        return self.loss(UVP, uvp)

    def Results(self, model):
        x = numpy.linspace(0, 1, self.test)
        y = numpy.linspace(0, 1, self.test)
        X, Y = numpy.meshgrid(x, y)
        X_flat = torch.Tensor(X.flatten())[:, None]
        Y_flat = torch.Tensor(Y.flatten())[:, None]
        # Predict the velocity components using the trained model
        model.eval()
        with torch.no_grad():
            uvp = model(torch.cat([X_flat, Y_flat], dim=1).to(self.processor))
            U = uvp[:, 0].cpu().numpy().reshape(X.shape)
            V = uvp[:, 1].cpu().numpy().reshape(X.shape)
            P = uvp[:, 2].cpu().numpy().reshape(X.shape)
        # Compute the velocity magnitude
        Velocity = numpy.sqrt(U**2 + V**2)
        return dict(x=X, y=Y, u=U, v=V, p=P, vel=Velocity)

    def __PDE(self, model):
        # Random collocation points (CI) in PDE domain of x*y: [0,1]x[0,1]
        xp = torch.rand((self.ci, 1), requires_grad=True).to(self.processor)
        yp = torch.rand((self.ci, 1), requires_grad=True).to(self.processor)
        # Model prediction for the CI points
        UVP = model(torch.cat([xp, yp], dim=1))
        U, V, P = UVP[:, 0].view(self.ci, 1), UVP[:, 1].view(self.ci, 1), UVP[:, 2].view(self.ci, 1)
        # Gradient of the velocity components and pressure
        Ux  = grad(U, xp, grad_outputs=torch.ones_like(U), create_graph=True)[0]
        Uy  = grad(U, yp, grad_outputs=torch.ones_like(U), create_graph=True)[0]
        Vx  = grad(V, xp, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        Vy  = grad(V, yp, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        Px  = grad(P, xp, grad_outputs=torch.ones_like(P), create_graph=True)[0]
        Py  = grad(P, yp, grad_outputs=torch.ones_like(P), create_graph=True)[0]
        # Second derivative of the velocity components for computing the Laplacian
        Uxx = grad(Ux, xp, grad_outputs=torch.ones_like(Ux), create_graph=True)[0]
        Uyy = grad(Uy, yp, grad_outputs=torch.ones_like(Uy), create_graph=True)[0]
        Vxx = grad(Vx, xp, grad_outputs=torch.ones_like(Vx), create_graph=True)[0]
        Vyy = grad(Vy, yp, grad_outputs=torch.ones_like(Vy), create_graph=True)[0]
        # Navier-Stoke's PDE prediction: Continuity (C), Momentum of u (Mu), and Momentum of v (Mv)
        C  = Ux + Vy
        Mu = (U*Ux + V*Uy) + Px/self.rho - self.nu*(Uxx + Uyy)
        Mv = (U*Vx + V*Vy) + Py/self.rho - self.nu*(Vxx + Vyy)
        return torch.cat([C, Mu, Mv])

    def __ICBC(self, model):
        # Predictions for u and v on the boundary
        UVP = model(torch.cat([self.xc, self.yc], dim=1))
        U, V, P = UVP[:, 0].view(-1,1), UVP[:, 1].view(-1,1), UVP[:, 2].view(-1,1)
        P0 = P[0]*torch.ones(self.p0.shape[0], 1).to(self.processor)
        return torch.cat([P0, U, V])

    def __verbose(self):
        print('\n'+15*' '+16*'-'+'  Lid-Driven Cavity PDE  '+16*'-')
        print(15*' '+f'{"PDE":^10}: {"u_x + u_y = 0.0, x,y=(0,1)^2"}')
        print(27*' '+f'{"u*u_x + v*u_y = -p_x/rho + nu*(u_xx + u_yy), x,y=(0,1)^2"}')
        print(27*' '+f'{"u*v_x + v*v_y = -p_y/rho + nu*(v_xx + v_yy), x,y=(0,1)^2"}')
        print(15*' '+f'{"BCs":^10}: {"v(x,0) = v(x,1) = v(0,y) = v(1,y) = 0, x,y=[0,1]"}')
        print(27*' '+f'{"u(x,0) = u(0,y) = u(1,y) = 0, x,y=[0,1]"}')
        print(27*' '+f'{"u(x,1) = A*sin(pi*x), x=[0,1]"}')
        print(15*' '+f'{"ICs":^10}: {"p(0,0) = 0"}')
        print(15*' '+60*'=')
        print(15*' '+f'{"Parameters":^10}: nu (kinematic viscosity) = {self.nu:^6.2f}')
        print(27*' '+f'rho (density) = {self.rho:^6.1f}')
        print(27*' '+f'A (scaling factor) = {self.A:^6.1f}')
        print(27*' '+f'L (Characteristic Length) = {self.L:^6.1f}')
        print(27*' '+f'u^ (Characteristic speed of the flow) = {self.u_hat:^6.2f}')
        print(27*' '+f'Re (Reynolds Number) = rho*u^*L/nu = {self.Re:^6.0f}')
        print(27*' '+f'CI (collocation points) = {self.ci}')
        print(27*' '+f'ICBC (ICs/BCs points) = {2*self.icbc}')
        print(15*' '+60*'-'+'\n')

