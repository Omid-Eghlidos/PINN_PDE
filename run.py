#!/usr/bin/env python3
"""
Goal
----
This program employs neural networks to solve partial differential equations (PDEs)
of lid-driven cavity (LDC) problem.

Inputs
-------
1) NN type: Either physics-informed NN (PINN) or convolutional NN (CNN)
2) Input file: A file containing the settings for running the NN

Deployment
-----------
In the folder containing the input file run:
    ./run.py -t [type] -i [input file] -v
        * -t [type]: Network type, currently only pinn
        * -i [input file]: a .txt file containing all the settings required for runnin NN
        * -v: Changing writing details on the screen to verbose (for more details).
"""


import argparse
from argparse import BooleanOptionalAction
from in_out.inputs import Inputs
from nn.pinn import PINN


def run():
    ''' Read the input file, then run PINN to solve the LDC-PDEs. '''
    parser = argparse.ArgumentParser(description='NN for solving LDC PDEs.')
    parser.add_argument('-t', dest='NN_type', default=None, type=str, nargs=1,
                         help='Type of NN to employ: either PINN or CNN')
    parser.add_argument('-i', dest='input_file', default=None, type=str, nargs=1,
                        help='Input file containig settings for selected NN.')
    parser.add_argument('-v', dest='verbose', default=False, action=BooleanOptionalAction,
                        help='Write more details on the screen.')
    options = parser.parse_args()

    if not options.NN_type:
        print('Warning: NN type was not entered, will use PINN as the default.')
        options.NN_type = 'pinn'
    if not options.input_file:
        raise Exception('Error: Input file is not entered.')

    settings = Inputs(options.input_file[0])
    print(f'-- Start solving {settings.pde} PDE')
    if 'pinn' in options.NN_type:
        PINN(settings, options.verbose)
    else:
        raise NotImplementedError(f'Error: Method "{options.NN_type[0]}" is not recognized.')


if __name__ == '__main__':
    run()

