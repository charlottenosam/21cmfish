import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import seaborn as sns


def Fij(dObs_dtheta_i, dObs_dtheta_j,
        sigma_obs=1, sigma_mod=0, axis=None):
    """
    Make fisher matrix elements

    Parameters
    ----------
    dObs_dtheta_i : array_like
        derivative wrt theta_i
    dObs_dtheta_j : array_like
        derivative wrt theta_j
    sigma_obs : array_like
        measurement uncertainties in the observations
    sigma_mod :
        modelling uncertainty
    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed. The default, axis=None,
        will sum all of the elements of the input array. If axis is negative
        it counts from the last to the first axis.

    Return
    ------
        F_ij fisher matrix element : float
    """
    return np.sum(dObs_dtheta_i * dObs_dtheta_j/(sigma_obs**2. + sigma_mod**2.)**2., axis=axis)


def fisher_correlations(Fij_matrix, fisher_params, plot=True):
    """
    Fisher correlation matrix

    Parameters
    ----------
    Fij_matrix : array_like
        Fisher information matrix

    fisher_params: list
        ordered list of parameters in the fisher matrix

    plot: bool
        heatmap plot

    Return
    ------
        R_ij_fisher correlation matrix
    """

    R_ij_fisher = np.zeros((len(fisher_params), len(fisher_params)))
    for i,p1 in enumerate(fisher_params):
        for j,p2 in enumerate(fisher_params):
            R_ij_fisher[i,j] = Fij_matrix[i,j]/np.sqrt(Fij_matrix[i,i]*Fij_matrix[j,j])

    if plot:
        mask = np.triu(np.ones_like(R_ij_fisher, dtype=bool))
        sns.heatmap(R_ij_fisher, mask=mask,
                    cmap='RdBu',
                    xticklabels=fisher_params,yticklabels=fisher_params,
                    square=True, linewidths=.5, cbar_kws={"shrink": 1, 'label':'Correlation $r_{ij}$'})

    return R_ij_fisher




# TODO generalize
alpha_std = {1.:1.52, 2.:2.48, 3.:3.44}

def get_ellipse_params(i: int, j: int, cov: np.array):
    """
    Extract ellipse parameters from covariance matrix.
    Based on Coe 2009

    Parameters
    ----------
        i : int
            index of parameter 1

        j : int
            index of parameter 2

        cov : array_like
            covariance matrix

    Return
    ------
        ellipse a, b, angle in degrees
    """

    # equations 1-4 Coe 2009. returns in degrees
    def length(cov, sign=1):
        return np.sqrt(0.5*(cov[i,i] + cov[j,j]) + sign*np.sqrt(0.25*(cov[i,i] - cov[j,j])**2. + cov[i,j]*cov[j,i]))

    def angle_deg(cov):
        return np.degrees(0.5*np.arctan(2*cov[i,j]/(cov[i,i] - cov[j,j])))

    a = length(cov, sign=1)
    b = length(cov, sign=-1)
    t = angle_deg(cov)

    if (cov[i,i] < cov[j,j]):
        a, b = b, a

    return a, b, t


def plot_ellipse(ax, par1, par2, parameters, fiducial, cov,
                 resize_lims=True, positive_definite=[],
                 N_std=[1.,2.,3.], plot_rescale = 4.,
                 kwargs=[{'ls': '-'}],
                 default_kwargs={'facecolor':'tab:blue', 'lw':0}):
    """
    Plot N-sigma ellipses, from Coe 2009.

    Parameters
    ----------
        ax : matpotlib axis
            axis upon which the ellipses will be drawn

        par1 : string
            parameter 1 name

        par2 : string
            parameter 2 name

        parameters : list
            list of parameter names

        fiducial : array_like(ndim,)
            fiducial values of parameters

        cov : array_like(ndim,ndim,)
            covariance matrix

        color : string
            color to plot ellipse with

        positive_definite : list of string
            convenience input, parameter names passed in this list
            will be cut off at 0 in plots.

    Returns
    -------
        list of float : sigma_x, sigma_y, sigma_xy

    """

    # Find parameters in list
    params = parameters
    pind = dict(zip(params, list(range(len(params)))))
    i = pind[par1]
    j = pind[par2]

    sigma_x  = np.sqrt(cov[i,i])
    sigma_y  = np.sqrt(cov[j,j])
    sigma_xy = cov[i,j]

    a, b, theta = get_ellipse_params(i, j, cov=cov)

    # Plot for each N sigma
    for nn, N in enumerate(N_std):
        # use defaults and then override with other kwargs
        kwargs_temp = default_kwargs.copy()
        if len(kwargs) > 1:
            kwargs_temp.update(kwargs[nn])
        else:
            kwargs_temp.update(kwargs[0])
        kwargs_n = kwargs_temp

        e = Ellipse(
            xy=(fiducial[i], fiducial[j]),
            width=a * 2 * alpha_std[N], height=b * 2 * alpha_std[N],
            angle=theta, zorder=0, **kwargs_n)

        ax.add_artist(e)
        e.set_clip_box(ax.bbox)

    # Rescale the axes a bit
    if resize_lims:
        if par1 in positive_definite:
            ax.set_xlim(max(0.0, -plot_rescale*sigma_x),
                        fiducial[i]+plot_rescale*sigma_x)
        else:
            ax.set_xlim(fiducial[i] - plot_rescale * sigma_x,
                        fiducial[i] + plot_rescale * sigma_x)

        if par2 in positive_definite:
            ax.set_ylim(max(0.0, fiducial[j] - plot_rescale * sigma_y),
                        fiducial[j] + plot_rescale * sigma_y)
        else:
            ax.set_ylim(fiducial[j] - plot_rescale * sigma_y,
                        fiducial[j] + plot_rescale * sigma_y)

    return sigma_x, sigma_y, sigma_xy


def plot_triangle(params, fiducial, cov, fig=None, ax=None,
                   positive_definite=[],
                   labels=None,
                   N_std=[1.,2.], plot_rescale = 4.,
                   ellipse_kwargs=[{},
                                  {'alpha':0.5}],
                   xlabel_kwargs={'labelpad': 5, 'fontsize':18},
                   ylabel_kwargs={'labelpad': 5, 'fontsize':18},
                   fig_kwargs={'figsize': (8, 8)},
                   plot1D_kwargs={'c':'black', 'lw':1}):
    """
    Make a triangle plot from a covariance matrix

    Based on https://github.com/xzackli/fishchips-public/blob/master/fishchips/util.py

    Parameters
    ----------
        params : list of strings
            List of parameter strings

        fiducial : array
            Numpy array consisting of where the centers of ellipses should be

        cov : numpy array
            Covariance matrix to plot

        fig : optional, matplotlib figure
            Pass this if you already have a figure

        ax : array containing matplotlib axes
            Pass this if you already have a set of matplotlib axes

        positive_definite: list
            List of parameter strings which are positive definite

        N_std : list
            List of number of standard deviations to plot

        labels : list
            List of labels corresponding to each dimension of the covariance matrix

        ellipse_kwargs : dict
            Keyword arguments for passing to the 1-sigma Matplotlib Ellipse call. You
            can change this to change the color of your ellipses, for example.

        xlabel_kwargs : dict
            Keyword arguments which are passed to `ax.set_xlabel()`. You can change the
            color and font-size of the x-labels, for example. By default, it includes
            a little bit of label padding.

        ylabel_kwargs : dict
            Keyword arguments which are passed to `ax.set_ylabel()`. You can change the
            color and font-size of the y-labels, for example. By default, it includes
            a little bit of label padding.

        fig_kwargs : dict
            Keyword arguments which are passed to `figure`. E.g. figsize

        plot1D_kwargs : dict
            Keyword arguments which are passed to `plt.plot()` for 1D gauss plot

    Returns
    -------
        fig, ax
            matplotlib figure and axis array
    """

    nparams = len(params)

    if ax is None or fig is None:
        print('generating new axis')
        fig, ax = plt.subplots(nparams, nparams, **fig_kwargs)

    if labels is None:
        labels = [(r'$\mathrm{' + p.replace('_', r'\_') + r'}$')
                  for p in params]

    # stitch together axes to row=nparams-1 and col=0
    # and turn off non-edge
    for ii in range(nparams):
        for jj in range(nparams):
            if ii == jj:
                ax[jj, ii].get_yaxis().set_visible(False)
                if ii < nparams-1:
                    ax[jj, ii].get_xaxis().set_ticks([])

            if ax[jj, ii] is not None:
                if ii < jj:
                    if jj < nparams-1:
                        ax[jj, ii].set_xticklabels([])
                    if ii > 0:
                        ax[jj, ii].set_yticklabels([])

                    if jj > 0:
                        # stitch axis to the one above it
                        if ax[0, ii] is not None:
                            ax[jj, ii].get_shared_x_axes().join(ax[jj, ii], ax[0, ii])
                    elif ii < nparams-1:
                        if ax[jj, nparams-1] is not None:
                            ax[jj, ii].get_shared_y_axes().join(ax[jj, ii], ax[jj, nparams-1])

    # call plot_ellipse
    for ii in range(nparams):
        for jj in range(nparams):
            if ax[jj, ii] is not None:

                if ii < jj:
                    plot_ellipse(ax[jj, ii], params[ii],
                                 params[jj], params, fiducial, cov,
                                 positive_definite=positive_definite,
                                 N_std=N_std, plot_rescale=plot_rescale,
                                 kwargs=ellipse_kwargs)
                    if jj == nparams-1:
                        ax[jj, ii].set_xlabel(labels[ii], **xlabel_kwargs)
                        ax[jj, ii].xaxis.set_major_locator(plt.MaxNLocator(5))
                        for tick in ax[jj, ii].get_xticklabels():
                            tick.set_rotation(45)
                    if ii == 0:
                        ax[jj, ii].set_ylabel(labels[jj], **ylabel_kwargs)

                elif ii == jj:
                    # plot a gaussian if we're on the diagonal
                    sig = np.sqrt(cov[ii, ii])
                    if params[ii] in positive_definite:
                        x = np.linspace(fiducial[ii], fiducial[ii] + plot_rescale * sig, 100)
                    else:
                        x = np.linspace(fiducial[ii] - plot_rescale*sig, fiducial[ii] + plot_rescale*sig, 100)

                    # Need to rescale if positive definite
                    rescale = 1.0
                    if params[ii] in positive_definite:
                        rescale = 2.0

                    gauss = rescale * np.exp(-(x-fiducial[ii])**2 / (2 * sig**2)) / (sig * np.sqrt(2*np.pi))
                    ax[jj, ii].plot(x, gauss, **plot1D_kwargs)
                    ax[jj, ii].set_title(f'{labels[ii]}$={fiducial[ii]:.2f}\pm{sig:.2f}$')
                    if ii == nparams-1:
                        ax[jj, ii].set_xlabel(labels[ii], **xlabel_kwargs)
                else:
                    ax[jj, ii].axis('off')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    return fig, ax
