import numpy
from matplotlib import pyplot


pyplot.rc('font', family='Times New Roman', size=8)
pyplot.rc('text', usetex=True)
pyplot.rc('mathtext', fontset='cm')


def output_results(pde, path, results, lrec):
    ''' First write the results into a text file in the given path and then
    plot and save them in the same folder. '''
    write_results(pde, path, results)
    plot_results(pde, path, results, lrec)


def write_results(pde, path, results):
    ''' Write the results of solving each PDE into a text file. '''
    print(f'---- Writing {pde} results into file')
    fid = open(f'{path}/{pde}_results.txt', 'w')
    for i in results:
        fid.write(f'{i:^12}')
    fid.write('\n')
    for j in range(len(results['x'])):
        for k in range(len(results['x'][j])):
            for i in results:
                fid.write(f'{results[i][j][k]:^12.6f}')
            fid.write('\n')


def plot_results(pde, path, results, lrec):
    ''' Plot the results and loss record for the specified problem. '''
    print(f'---- Plotting {pde} results')
    plot_loss(pde, path, lrec)
    if pde == 'LDC':
        plot_ldc_results(pde, path, results)
    else:
        plot_pde_results(pde, path, results)


def plot_loss(pde, path, lrec):
    ''' Plot loss vs. epochs. '''
    pyplot.clf()
    fig, ax = pyplot.subplots(1, 1, figsize=(3.5, 2.8))
    epochs = numpy.arange(len(lrec))*100
    ax.plot(epochs, lrec, marker = 'o', ms=2, ls='None', color ='red')
    ax.set_xlabel(r'Epoch')
    #ax.set_xlim(min(epochs)-10, max(epochs)+10)
    ax.set_ylabel(r'Loss')
    ax.set_yscale('log')
    ax.set_ylim(top=1.0)
    #ax.legend(frameon=False)
    ax.minorticks_on()
    pyplot.tight_layout()
    pyplot.savefig(f'{path}/{pde}_loss.png', dpi=300)
    pyplot.close()


def plot_pde_results(pde, path, results):
    ''' Plot velocity contour for a the specified grid of (x, y). '''
    pyplot.clf()
    fig, ax = pyplot.subplots(1, 1, figsize=(3.5, 2.8))
    ax.set_title(f'{pde} PDE')
    if 't' in results:
        x, y, u = results['t'], results['x'], results['u']
    else:
        x, y, u = results['x'], results['y'], results['u']
    vmin, vmax = round(numpy.min(u), 3), round(numpy.max(u), 3)
    im = ax.pcolormesh(x, y, u, vmin=vmin, vmax=vmax, rasterized=True,
                                                 shading='gouraud', cmap='jet')
    adjust_axes(pde, ax)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(r'$u$(\textbf{x})', labelpad=1.0)
    cbar.ax.set_yticks(numpy.round(numpy.linspace(vmin, vmax, 5), 3))
    pyplot.tight_layout()
    pyplot.savefig(f'{path}/{pde}_uxy.png', dpi=300, bbox_inches='tight')
    pyplot.close()


def plot_ldc_results(pde, path, results):
    ''' Plot velocity magnitude and u, v, p, and streamline contour maps for a set of (x, y). '''
    # Plot the U, V, and P on the same figure
    pyplot.clf()
    fig, axes = pyplot.subplots(1, 4, figsize=(6.5, 1.625), constrained_layout=True)
    lims = {'u': [-1.0, 5.0], 'v': [-3.0, 1.0], 'p': [-1.0, 5.0]}
    for i, j in enumerate(lims):
        axes[i].set_title(f'${j}$' + r'(\textbf{x})')
        im = axes[i].pcolormesh(results['x'], results['y'], results[j],
                                vmin=lims[j][0], vmax=lims[j][1],
                                rasterized=True, shading='gouraud', cmap='jet')
        adjust_axes(pde, axes[i])
        cbar = fig.colorbar(im, ax=axes[i], fraction=0.05, pad=0.005)
        #cbar.ax.set_yticks(numpy.round(numpy.linspace(vmin, vmax, 6), 2))
    axes[i+1].set_title('Streamlines')
    axes[i+1].streamplot(results['x'], results['y'], results['u'], results['v'],
                         color=results['vel'], linewidth=0.5, density=0.5, cmap ='jet')
    adjust_axes(pde, axes[i+1])
    pyplot.savefig(f'{path}/LDC_uvp.png', dpi=300, bbox_inches='tight')
    pyplot.close()

    # Plot the velocity magnitude
    pyplot.clf()
    fig, ax = pyplot.subplots(1, 1, figsize=(3.5, 2.8))
    im = ax.pcolormesh(results['x'], results['y'], results['vel'], vmin=0.0, vmax=5.0,
                       rasterized=True, shading='gouraud', cmap='jet')
    adjust_axes(pde, ax)
    cbar.set_label(r'$||u$(\textbf{x})+$v$(\textbf{x})$||_2$', labelpad=1.0)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    #cbar.ax.set_yticks(numpy.round(numpy.linspace(vmin, vmax, 6), 2))
    pyplot.tight_layout()
    pyplot.savefig(f'{path}/LDC_velocity.png', dpi=300, bbox_inches='tight')
    pyplot.close()


def adjust_axes(pde, ax):
    ''' Adjust the axes label and tickmarks with respect to the problem conditions. '''
    if pde == 'Burger':
        ax.set_xlabel('$t$', labelpad=1.0)
        xmin, xmax = 0.0, 1.0
        xticks = numpy.round(numpy.linspace(xmin, xmax, 6), 2)
        ax.set_ylabel('$x$', labelpad=1.0)
        ymin, ymax = -1.0, 1.0
        yticks = numpy.round(numpy.linspace(ymin, ymax, 5), 2)
    if pde == 'Helmholtz':
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

