#!/usr/bin/env python3
"""
Goal
----
Generate analytical references for the Elliptical and Helmholtz PDEs.

Deployment
----------
In the root folder try:
    ./generate_reference.py [PDE]
        * PDE: either Elliptical or Helmholtz
"""

import sys
import numpy


def generate_reference():
    ''' Compare the results for the specified problem with a reference. '''
    if len(sys.argv) != 2:
        print('You need to specify the PDE!')
        print('Try: ./generate_reference.py [PDE], PDE = Elliptical or Helmholtz')
        sys.exit(0)
    else:
        pde = sys.argv[1]

    # Number of grid points
    np = 1000
    if pde in ['E', 'Elliptical']:
        pde = 'Elliptical'
        reference = Elliptical(np)
    else:
        pde = 'Helmholtz'
        reference = Helmholtz(np)
    write_reference(pde, reference)


def Elliptical(np):
    ''' Generate reference results for Elliptical PDE using its analytical solution. '''
    # Points in PDE domain of x*y: [0,1]x[0,1]
    x = numpy.linspace(0, 1, np)
    y = numpy.linspace(0, 1, np)
    X, Y = numpy.meshgrid(x, y)
    # u(x,y) = sin(pi*x)*sin(pi*y) + 2*sin(4*pi*x)*sin(4*pi*y)
    u = numpy.zeros((np, np))
    for i in range(np):
        for j in range(np):
            u[i,j] = numpy.sin(numpy.pi*x[i]) * numpy.sin(numpy.pi*y[j]) +\
                     2 * numpy.sin(4*numpy.pi*x[i]) * numpy.sin(4*numpy.pi*y[j])
    return dict(x=X, y=Y, u=u)


def Helmholtz(np):
    ''' Generate reference results for Helmholtz PDE using its analytical solution. '''
    # Points in PDE domain of x*y: [-1,1]x[-1,1]
    x = numpy.linspace(-1, 1, np)
    y = numpy.linspace(-1, 1, np)
    X, Y = numpy.meshgrid(x, y)
    ax, ay = 1.0, 4.0
    # u(x,y) = sin(ax*pi*x)*sin(ay*pi*y)
    u = numpy.zeros((np, np))
    for i in range(np):
        for j in range(np):
            u[i,j] = -numpy.sin(ax*numpy.pi*x[j]) * numpy.sin(ay*numpy.pi*y[i])
    return dict(x=X, y=Y, u=u)


def write_reference(pde, reference):
    ''' Write the reference of a PDE into a text file. '''
    print(f'Writing {pde} results into file ...')
    fid = open(f'{pde}_reference.txt', 'w')
    for i in reference:
        fid.write(f'{i:^12}')
    fid.write('\n')
    for j in range(len(reference['x'])):
        for k in range(len(reference['x'][j])):
            for i in reference:
                fid.write(f'{reference[i][j][k]:^12.6f}')
            fid.write('\n')


if __name__ == '__main__':
    generate_reference()


