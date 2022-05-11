import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt


def make_fisher_matrix(params_dict, fisher_params, hpeak=0.0, obs='GS',
                        sigma=None, sigma_mod_frac=0.,
                        k_min=None, k_max=None,
                        z_min=None, z_max=None,
                        axis_PS=None, cosmo_key='CDM',
                        add_sigma_poisson=False):
    """
    Make Fisher matrix and its inverse from global signal or powerspectra

    Parameters
    ----------
    params_dict : dict
        Dictionary of parameter objects
    fisher_params : list
        List of parameter strings to use for Fisher matrix (these strings must be the keys to params_dict)
    hpeak : float
        TODO
    obs : str
        'GS' - global signal, 'PS' - power spectrum
    sigma : None,array
        TODO
    sigma_mod_frac : float
        Fraction of modelling error in PS e.g. 0.2 adds a 20% error on the PS in quadrature to the 21cmsense error
    k_min : None,float
        Minimum k to use for PS [1/Mpc]
    k_max : None,float
        Maximum k to use for PS [1/Mpc]
    z_min : None,float
        Minimum redshift to use for PS
    z_max : None,float
        Maximum redshift to use for PS
    axis_PS : None,int
        TODO
    cosmo_key : None,str
        TODO
    add_sigma_poisson : bool
        TODO

    Return
    -------
    Fisher matrix, Finv matrix
    """

    Fij_matrix = np.zeros((len(fisher_params), len(fisher_params)))

    for i,p1 in enumerate(fisher_params):

        if i == 0 and obs == 'PS':
            k_where = np.arange(len(params_dict[p1].PS_err[0]['k']))
            if k_min is not None and k_max is not None:  # k range in 1/Mpc
                k_where  = np.where((params_dict[p1].PS_err[0]['k'] <= k_max) & (params_dict[p1].PS_err[0]['k'] >= k_min))[0]

            z_where = np.arange(len(params_dict[p1].PS_z_HERA))
            if z_min is not None and z_max is not None:
                z_where  = np.where((params_dict[p1].PS_z_HERA <= z_max) & (params_dict[p1].PS_z_HERA >= z_min))[0]

            # Model error (e.g. 20%)
            sigma_mod = sigma_mod_frac * params_dict[p1].PS_fid[z_where][:,k_where]
            # if cosmo_key is None:
            # cosmo_key = params_dict[p1].deriv_PS.keys()[0]
            PS0 = params_dict[p1].deriv_PS[cosmo_key][z_where][:,k_where]

            # Poisson error
            if add_sigma_poisson:
                sigma_poisson = params_dict[p1].PS_err_Poisson[z_where][:,k_where]
            else:
                sigma_poisson = 0.

            # Fisher as a function of redshift or k?
            if axis_PS is not None:
                Fij_matrix = np.zeros((PS0.shape[axis_PS-1], len(fisher_params), len(fisher_params)))

        for j,p2 in enumerate(fisher_params):
            if obs == 'GS':
                if i == 0 and j == 0:
                    print('GS shape:',params_dict[p1].deriv_GS[cosmo_key].shape)

                Fij_matrix[i,j] = Fij(params_dict[p1].deriv_GS[cosmo_key],
                                      params_dict[p2].deriv_GS[cosmo_key],
                                      sigma_obs=1, sigma_mod=0.)
            elif obs == 'PS':
                if sigma is None:
                    sigma_PS = params_dict[p1].PS_sigma[z_where][:,k_where]
                else:
                    sigma_PS = sigma

                if i==0 and j==0:
                    print('PS shape:',params_dict[p1].deriv_PS[cosmo_key][z_where][:,k_where].shape)

                if axis_PS is not None:
                    Fij_matrix[:,i,j] = Fij(params_dict[p1].deriv_PS[cosmo_key][z_where][:,k_where],
                                          params_dict[p2].deriv_PS[cosmo_key][z_where][:,k_where],
                                          sigma_obs=sigma_PS, sigma_mod=sigma_mod, sigma_poisson=sigma_poisson, axis=axis_PS)
                else:
                    Fij_matrix[i,j] = Fij(params_dict[p1].deriv_PS[cosmo_key][z_where][:,k_where],
                                          params_dict[p2].deriv_PS[cosmo_key][z_where][:,k_where],
                                          sigma_obs=sigma_PS, sigma_mod=sigma_mod, sigma_poisson=sigma_poisson, axis=axis_PS)

    Finv = np.linalg.inv(Fij_matrix)
    return Fij_matrix, Finv


def Fij(dObs_dtheta_i, dObs_dtheta_j,
        sigma_obs=1., sigma_mod=0., sigma_poisson=0., axis=None):
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
    sigma_sq = sigma_obs**2. + sigma_mod**2. + sigma_poisson**2.
    return np.nansum(dObs_dtheta_i * dObs_dtheta_j/sigma_sq, axis=axis)


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
        import seaborn as sns
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
        """
        Calculate length of the ellipse semi-major/semi-minor axes

        Aka the eigenvalues of the covariance matrix
        """
        return np.sqrt(0.5*(cov[i,i] + cov[j,j]) + sign*np.sqrt(0.25*(cov[i,i] - cov[j,j])**2. + cov[i,j]*cov[j,i]))

    def angle_deg(cov):
        """
        Calculate angle of ellipse in degrees (anti-clockwise from x axis)

        Gets the quadrant right!
        """
        return np.degrees(0.5*np.arctan2(2*cov[i,j],(cov[i,i] - cov[j,j])))

    a = length(cov, sign=1)
    b = length(cov, sign=-1)
    t = angle_deg(cov)

    return a, b, t


def plot_ellipse(ax, par1, par2, parameters, fiducial, cov,
                 resize_lims=True, positive_definite=[],
                 N_std=[1.,2.,3.], plot_rescale = 4.,
                 kwargs=[{'ls': '-'}],
                 color='tab:blue',
                 default_kwargs={'lw':0}):
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
            angle=theta, zorder=0, facecolor=color, **kwargs_n)

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


def title_double_ellipses(axes, labels,
                           chain=None,
                           med=None, sigma=None,
                           title_fontsize=18, title_pad=58,
                           vspace=0.,
                           color='k'
                           ):
    """
    Plot title with parameter constraints from 2 covariance matrixes/chains

    Parameters
    ----------
        axes : matpotlib axess
            axes upon which the titles will be added

        labels : list(ndim,)
            list of parameter names

        chain : array_like(ndim,), optional
            MCMC chain of parameters

        med : array_like(ndim,), optional
            list of median values

        sigma : array_like(ndim,)
            list of sigmas

        color : string
            color to plot ellipse with


    Returns
    -------
        None

    """

    if chain is not None:
        l, med, u = np.percentile(chain, [16,50,84], axis=0)
        q_m, q_p  = med - l, u - med

    for i in range(len(labels)):
        if med[i] < 100:
            fmt = "{{0:{0}}}".format('.2f').format
        else:
            fmt = "{{0:{0}}}".format('.0f').format

        if chain is not None:
            CI = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
            CI = CI.format(fmt(med[i]), fmt(q_m[i]), fmt(q_p[i]))
        else:
            CI = r"${{{0}}} \pm {{{1}}}$"
            CI = CI.format(fmt(med[i]), fmt(sigma[i]))

        axes[i,i].set_title(f'{labels[i]}', fontsize=title_fontsize, pad=title_pad)
        axes[i,i].annotate(f'{CI}',
                           xy=(0.5,1.05+vspace), ha='center',
                           xycoords='axes fraction', color=color)

    return


def plot_triangle(params, fiducial, cov, fig=None, ax=None,
                   positive_definite=[],
                   labels=None,
                   resize_lims=True,
                   N_std=[1.,2.], plot_rescale = 4.,
                   ellipse_color='tab:blue',
                   ellipse_kwargs=[{},
                                  {'alpha':0.5}],
                   title_fontsize=20,
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

        resize_lims : bool
            Resize ellipse limits to scale of the errors [default = True]

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
                                 resize_lims=resize_lims,
                                 color=ellipse_color,
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
                    ax[jj, ii].set_title(f'{labels[ii]}$={fiducial[ii]:.2f} \pm {sig:.2f}$', fontsize=title_fontsize)
                    if ii == nparams-1:
                        ax[jj, ii].set_xlabel(labels[ii], **xlabel_kwargs)
                else:
                    ax[jj, ii].axis('off')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    return fig, ax
