#!/usr/bin/env python3
"""
Goal
----
Compare the results obtained from the model with the references.

Inputs
------
1) Results obtained from running the PINN for each PDE that are saved into a
text file with the same name as the PDE in the "results/" folder.
2) References obtained from other sources or generated using the analytical
solution that are saved into a text file in the "references/" folder.

Deployment
-----------
In the root folder try:
    ./compare.py [PDE]
        * PDE: is the name of the problem to be solved.
               It can be Burger/Elliptical/Helmholtz/Eikonal/LDC
"""


import sys
import numpy
from matplotlib import pyplot


pyplot.rc('font', family='Times New Roman', size=8)
pyplot.rc('text', usetex=True)
pyplot.rc('mathtext', fontset='cm')
normalize = False


def compare_results():
    ''' Compare the results for the specified problem with a reference. '''
    if len(sys.argv) != 2:
        print('You need to specify the PDE!')
        print('Try: ./compare_results.py [PDE], PDE = {Burger}')
        pde = 'Burger'
    else:
        pde = sys.argv[1]

    path = 'results'
    results = read_results('results', pde)
    if pde == 'LDC':
        plot_ldc_results(pde, path, results)
    else:
        reference = read_results('reference', pde)
        plot_results(pde, path, results)
        plot_comparison(pde, path, results, reference)


def read_results(path, pde):
    fid = open(f'{path}/{pde}_{path}.txt', 'r')
    results = dict()
    parameters = next(fid).split()
    for parameter in parameters:
        results[parameter] = []
    for line in fid:
        args = line.split()
        for i in range(len(args)):
            results[parameters[i]].append(float(args[i]))
    for p in results:
        n = int(numpy.sqrt(len(results[p])))
        results[p] = numpy.array(results[p]).reshape(n, n)
    return results


def plot_results(pde, path, results):
    ''' Plot velocity contour for a the specified grid of (x, y). '''
    pyplot.clf()
    fig, ax = pyplot.subplots(1, 1, figsize=(3.5, 2.8))
    ax.set_title(f'{pde} PDE')
    if 't' in results:
        x, y, u = results['t'], results['x'], results['u']
        output = f'{pde}_utx.png'
    else:
        x, y, u = results['x'], results['y'], results['u']
        output = f'{pde}_uxy.png'
    vmin, vmax = adjust_range(pde)
    if normalize:
        u = vmin + ((u - numpy.min(u))*(vmax - vmin)) / (numpy.max(u) - numpy.min(u))
    im = ax.pcolormesh(x, y, u, vmin=vmin, vmax=vmax,
                                rasterized=True, shading='gouraud', cmap='jet')
    adjust_axes(pde, ax)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(r'$u$(\textbf{x})', labelpad=1.0)
    cbar.ax.set_yticks(numpy.round(numpy.linspace(vmin, vmax, 5), 3))
    pyplot.tight_layout()
    pyplot.savefig(f'{path}/{output}', dpi=300, bbox_inches='tight')
    pyplot.close()


def plot_comparison(pde, path, results, reference):
    ''' Plot velocity contour map for a given grid of (x,y) or (t,x). '''
    pyplot.clf()
    fig, axes = pyplot.subplots(1, 3, figsize=(6.5, 2.0))
    if 't' in results:
        x, y, u = results['t'], results['x'], results['u']
        xr, yr, ur = reference['t'], reference['x'], reference['u']
    else:
        x, y, u = results['x'], results['y'], results['u']
        xr, yr, ur = reference['x'], reference['y'], reference['u']
    error = numpy.abs(u - ur)
    print(f'Accuracy = {numpy.linalg.norm(error):^6.2f}')
    # Normalize the result and reference to the same scale
    vmin, vmax = adjust_range(pde)
    if normalize:
        u = vmin + ((u - numpy.min(u))*(vmax - vmin)) / (numpy.max(u) - numpy.min(u))
        ur = vmin + ((ur - numpy.min(ur))*(vmax - vmin)) / (numpy.max(ur) - numpy.min(ur))
    for i in range(3):
        if i == 0:
            axes[i].set_title('Reference')
            im = axes[i].pcolormesh(xr, yr, ur, vmin=vmin, vmax=vmax,
            								rasterized=True, shading='gouraud', cmap='jet')
        elif i == 1:
            axes[i].set_title('Results')
            im = axes[i].pcolormesh(x, y, u, vmin=vmin, vmax=vmax,
                                             rasterized=True, shading='gouraud', cmap='jet')
        else:
            axes[i].set_title('Absolute Error')
            im = axes[i].pcolormesh(x, y, error, vmin=numpy.min(error), vmax=numpy.max(error),
            							rasterized=True, shading='gouraud', cmap='jet')
        adjust_axes(pde, axes[i])
        cbar = fig.colorbar(im, ax=axes[i], fraction=0.045, pad=0.05)
        cbar.set_label(r'$u$(\textbf{x})', labelpad=1.0)
        s = 'abcdefghijkl'
        axes[i].text(-0.2, -0.2, f'({s[i]})', transform=axes[i].transAxes)
    pyplot.tight_layout()
    pyplot.savefig(f'{path}/{pde}_comparison.png', dpi=300, bbox_inches='tight')
    pyplot.close()


def plot_ldc_results(pde, path, results):
    ''' Plot velocity magnitude and u, v, p, and streamline contour maps for a set of (x, y). '''
    # Plot the U, V, and P on the same figure
    pyplot.clf()
    fig, axes = pyplot.subplots(1, 4, figsize=(6.5, 1.7), constrained_layout=True)
    lims = {'u': [-1.0, 4.0], 'v': [-2.5, 1.25], 'p': [-1.0, 4.5]}
    for i, j in enumerate(lims):
        axes[i].set_title(f'${j}$' + r'(\textbf{x})')
        im = axes[i].pcolormesh(results['x'], results['y'], results[j],
                                vmin=lims[j][0], vmax=lims[j][1],
                                rasterized=True, shading='gouraud', cmap='jet')
        adjust_axes(pde, axes[i])
        cbar = fig.colorbar(im, ax=axes[i], fraction=0.05, pad=0.005)
        s = 'abcdefghijkl'
        axes[i].text(-0.2, -0.2, f'({s[i]})', transform=axes[i].transAxes)
        #cbar.ax.set_yticks(numpy.round(numpy.linspace(vmin, vmax, 6), 2))
    axes[i+1].set_title('Streamlines')
    x = numpy.linspace(0,1, len(results['x']))
    y = numpy.linspace(0,1, len(results['y']))
    axes[i+1].streamplot(x, y, results['u'], results['v'], color=results['vel'],
                                                    linewidth=0.2, cmap ='jet')
    adjust_axes(pde, axes[i+1])
    #cbar = fig.colorbar(im, ax=axes[i+1])
    pyplot.savefig(f'{path}/LDC_uvp.png', dpi=300, bbox_inches='tight')
    pyplot.close()

    # Plot the velocity magnitude
    pyplot.clf()
    fig, ax = pyplot.subplots(1, 1, figsize=(3.5, 2.8))
    im = ax.pcolormesh(results['x'], results['y'], results['vel'],
                       rasterized=True, shading='gouraud', cmap='jet')
    adjust_axes(pde, ax)
    cbar.set_label(r'$||u$(\textbf{x})+$v$(\textbf{x})$||_2$', labelpad=1.0)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    #cbar.ax.set_yticks(numpy.round(numpy.linspace(vmin, vmax, 6), 2))
    pyplot.tight_layout()
    pyplot.savefig(f'{path}/LDC_velocity.png', dpi=300, bbox_inches='tight')
    pyplot.close()


def adjust_range(pde):
    ''' Adjust the range of the contour plots of each PDE according to their references. '''
    # Return corresponding vmin and vmax for each PDE
    if pde == 'Burger':
        vmin, vmax = -1.0, 1.0
    elif pde == 'Eikonal':
        vmin, vmax = 0.0, 0.4
    elif pde == 'Elliptical':
        vmin, vmax = -2.0, 2.0
    elif pde == 'Helmholtz':
        vmin, vmax = -1.0, 1.0
    return vmin, vmax


def adjust_axes(pde, ax):
    ''' Adjust the axes label and tickmarks with respect to the problem conditions. '''
    if pde == 'Burger':
        ax.set_xlabel('$t$', labelpad=1.0)
        xmin, xmax = 0.0, 1.0
        xticks = numpy.round(numpy.linspace(xmin, xmax, 6), 2)
        ax.set_ylabel('$x$', labelpad=1.0)
        ymin, ymax = -1.0, 1.0
        yticks = numpy.round(numpy.linspace(ymin, ymax, 5), 2)
    elif pde == 'Helmholtz':
        ax.set_xlabel('$x$', labelpad=1.0)
        xmin, xmax = -1.0, 1.0
        xticks = numpy.round(numpy.linspace(xmin, xmax, 5), 2)
        ax.set_ylabel('$y$', labelpad=1.0)
        ymin, ymax = -1.0, 1.0
        yticks = numpy.round(numpy.linspace(ymin, ymax, 5), 2)
        ax.set_aspect('equal')
    else:
        ax.set_xlabel('$x$', labelpad=1.0)
        xmin, xmax = 0.0, 1.0
        xticks = numpy.round(numpy.linspace(xmin, xmax, 6), 2)
        ax.set_ylabel('$y$', labelpad=1.0)
        ymin, ymax = 0.0, 1.0
        yticks = numpy.round(numpy.linspace(ymin, ymax, 6), 2)
        ax.set_aspect('equal')

    ax.set_xlim(xmin, xmax)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.tick_params(axis='both', which='major', pad=1)
    #ax.minorticks_on()


if __name__ == '__main__':
    compare_results()

