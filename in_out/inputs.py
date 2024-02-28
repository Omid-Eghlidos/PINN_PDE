import os
import sys
import numpy


class Inputs():
    ''' Class to keep the input settings for running the program.

        Attributes
        ----------
        nn (dict): Stores the following parameters for running the neural network
            * processor  (str): Determine to run the training on CPU or GPU (if available)
            * layers     (int): Number of layers to be used for the deep neural network
            * neurons    (int): Number of neurons per layers to be used
            * activation (dict): Stores the type of activation functions of specified layers
                ** hidden (str): Activation function to be used for hidden layers
                ** output (str): Activation function to be used for output layer
        optimizer (dict): Stores the following parameters for optimization
            * method (str): Optimization method to be used
            * epoch  (int): Number of epochs to run the optimization
            * gamma(float): Learning rate for convergence of the optimization
            * ci     (int): Number of collocation points to compute the loss within the domain
            * icbc   (int): Number of points to sample on initial/boundary conditions (ICs/BCs)
        pde (dict): Stores the type of partial differential equations to be solved
        burger (dict): Stores the following parameters for Burger pde:
            * nu (float): Kinematic viscosity
        elliptic (dict): Stores the following parameters for nonlinearized Elliptic pde:
            * alpha (float): Constant factor controlling degree of nonlinearity
        helmholtz (dict): Stores the following parameters for Helmholtz pde:
            * ax (float): Constant factor controlling the frequency along x-direction
            * ay (float): Constant factor controlling the frequency along y-direction
        eikonal (dict): Stores the following parameters for Eikonal pde:
            * eps (float): Constant factor controlling smoothing effect of the regularization
        ldc (dict): Stores the following parameters for Eikonal pde:
            * A   (float): Constant scaling factor
            * nu  (float): Kinematic viscosity
            * rho (float): Density
            * L   (float): Characteristic length of the flow
        path (str): The output path to the folder to save the results.

        Methods
        -------
        read_inputs (private): (str) -> None
            Read and store the parameters from an input file.
        make_folder (private): (str) -> str
            Creates a folder with the given name and return its path.
    '''

    def __init__(self, input_file):
        ''' Initialize the variables, then read and store their value from input file. '''
        self.nn = dict(processor='cpu', layers=1, neurons=10,
                       activation=dict(hidden='tanh', output='linear'))
        self.optimizer = dict(method='Adam', epochs=1000, gamma=1e-4, ci=[1000,1000], icbc=10)
        self.pde = 'LDC'
        self.burger     = dict(nu=0.01/numpy.pi)
        self.elliptical = dict(alpha=20.0)
        self.helmholtz  = dict(ax=1.0, ay=4.0)
        self.eikonal    = dict(eps=0.01)
        self.ldc        = dict(A=5.0, nu=0.01, rho=1.0, L=1.0)
        self.path		= self.__make_folder('results')

        self.__read_inputs(input_file)

    def __read_inputs(self, input_file):
        ''' Read and store the variables from the input file. '''
        print('-- Read the inputs')
        fid = open(input_file, 'r')
        for line in fid:
            args = line.split()
            if line.startswith('#') or len(args) == 0:
                continue
            # Nural network settings
            elif args[0] == 'neural_network':
                if args[1] == 'processor':
                    self.nn['processor'] = args[2]
                    if args[2] == 'gpu':
                        self.nn['processor'] = 'cuda'
                    print(f'---- Running on {args[2].upper()}')
                elif args[1] in ['layers', 'neurons']:
                    self.nn[args[1]] = int(args[2])
                elif args[1] == 'activation':
                    if args[2] in ['hidden', 'output']:
                        self.nn['activation'][args[2]] = args[3]
                else:
                    print(f'Error: Wrong input for the neural network: {line}')
                    sys.exit()
            elif args[0] == 'optimizer':
                if args[1] == 'method':
                    self.optimizer['method'] = args[2]
                elif args[1] in self.optimizer:
                    if args[1] == 'gamma':
                        self.optimizer[args[1]] = float(args[2])
                    elif args[1] == 'ci':
                        for i in range(2, len(args), 2):
                            if args[i] == 'train':
                                self.optimizer['ci'][0] = int(args[i+1])
                            else:
                                self.optimizer['ci'][1] = int(args[i+1])
                    else:
                        self.optimizer[args[1]] = int(args[2])
                else:
                    print(f'Error: Wrong input for the optimization: {line}')
                    sys.exit()
            elif args[0] == 'pde':
                if args[1] == 'solve':
                    self.pde = args[2]
                elif args[1] == 'Burger':
                    if args[2] in self.burger:
                        self.burger['nu'] = float(args[3])/numpy.pi
                    else:
                        print(f'Error: Wrong input for the pde {args[1]}: {line}')
                        sys.exit()
                elif args[1] == 'Elliptical':
                    if args[2] in self.elliptical:
                        self.elliptical['alpha'] = float(args[3])
                    else:
                        print(f'Error: Wrong input for the pde {args[1]}: {line}')
                        sys.exit()
                elif args[1] == 'Helmholtz':
                    if args[2] in self.helmholtz:
                        self.helmholtz['ax'] = float(args[3])
                    elif args[4] in self.helmholtz:
                        self.helmholtz['ax'] = float(args[5])
                    else:
                        print(f'Error: Wrong input for the pde {args[1]}: {line}')
                        sys.exit()
                elif args[1] == 'Eikonal':
                    if args[2] in self.eikonal:
                        self.eikonal['eps'] = float(args[3])
                    else:
                        print(f'Error: Wrong input for the pde {args[1]}: {line}')
                        sys.exit()
                elif args[1] == 'LDC':
                    for i in range(2, len(args), 2):
                        if args[i] in self.ldc:
                            self.ldc[args[i]] = float(args[i+1])
                        else:
                            print(f'Error: Wrong input for the pde {args[1]}: {line}')
                            sys.exit()
                else:
                    print(f'Error: Wrong type for the pde: {line}')
                    sys.exit()
            else:
                print(f'Wrong input: {line}')
                sys.exit()

    def __make_folder(self, folder_name):
        ''' make directory if it does not exist '''
        folder_path = os.path.join(os.getcwd(), folder_name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        return folder_path

