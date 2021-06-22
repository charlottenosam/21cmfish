# define functions to calculate PS, following py21cmmc
import py21cmfast as p21c
from py21cmfast import plotting
import os, sys
import numpy as np
from powerbox.tools import get_power

def get_k_min_max(lightcone, n_chunks=24):
    """
    Get the minimum and maximum k to calculate powerspectra for
    given size of box and number of chunks
    """

    BOX_LEN = lightcone.user_params.pystruct['BOX_LEN']
    HII_DIM = lightcone.user_params.pystruct['HII_DIM']

    k_fundamental = 2*np.pi/BOX_LEN*max(1,len(lightcone.lightcone_distances)/n_chunks/HII_DIM) #either kpar or kperp sets the min
    k_max         = k_fundamental * HII_DIM
    Nk            = np.floor(HII_DIM/1).astype(int)
    return k_fundamental, k_max, Nk


def compute_power(box,
                   length,
                   n_psbins,
                   log_bins=True,
                   ignore_kperp_zero=True,
                   ignore_kpar_zero=False,
                   ignore_k_zero=False):

    # Determine the weighting function required from ignoring k's.
    k_weights = np.ones(box.shape, dtype=int)
    n0 = k_weights.shape[0]
    n1 = k_weights.shape[-1]

    if ignore_kperp_zero:
        k_weights[n0 // 2, n0 // 2, :] = 0
    if ignore_kpar_zero:
        k_weights[:, :, n1 // 2] = 0
    if ignore_k_zero:
        k_weights[n0 // 2, n0 // 2, n1 // 2] = 0

    res = get_power(
        box,
        boxlength=length,
        bins=n_psbins,
        bin_ave=False,
        get_variance=True,
        log_bins=log_bins,
        k_weights=k_weights,
    )

    res = list(res)
    k = res[1]
    if log_bins:
        k = np.exp((np.log(k[1:]) + np.log(k[:-1])) / 2)
    else:
        k = (k[1:] + k[:-1]) / 2

    res[1] = k

    return res

def powerspectra(brightness_temp, n_psbins=50, nchunks=10, min_k=0.1, max_k=1.0, logk=True):
    """
    Make power spectra for given number of equally spaced chunks
    """
    data = []
    chunk_indices = list(range(0,brightness_temp.n_slices,round(brightness_temp.n_slices / nchunks),))

    if len(chunk_indices) > nchunks:
        chunk_indices = chunk_indices[:-1]
    chunk_indices.append(brightness_temp.n_slices)

    for i in range(nchunks):
        start = chunk_indices[i]
        end = chunk_indices[i + 1]
        chunklen = (end - start) * brightness_temp.cell_size

        power, k = compute_power(
            brightness_temp.brightness_temp[:, :, start:end],
            (BOX_LEN, BOX_LEN, chunklen),
            n_psbins,
            log_bins=logk,
        )
        data.append({"k": k, "delta": power * k ** 3 / (2 * np.pi ** 2)})
    return data


def powerspectra_np(brightness_temp, n_psbins=50, nchunks=10, min_k=0.1, max_k=1.0, logk=True):
    """
    Make power spectra for given number of equally spaced chunks

    JBM: same as powerspectra but for input in numpy format. Also outputs errors in delta
    """
    data = []

    mapshape = brightness_temp.shape
    chunk_indices = np.linspace(0, mapshape[-1], nchunks+1, endpoint=True,dtype=int)

    for i in range(nchunks):
        start = chunk_indices[i]
        end = chunk_indices[i + 1]
        chunklen = (end - start) * BOX_LEN/HII_DIM

        power, k, variance = compute_power(
            brightness_temp[:, :, start:end],
            (BOX_LEN, BOX_LEN, chunklen),
            n_psbins,
            log_bins=logk,
        )

        data.append({"k": k, "delta": power * k ** 3 / (2 * np.pi ** 2), "err_delta": np.sqrt(variance) * k ** 3 / (2 * np.pi ** 2)})
    return data


def powerspectra_chunks(lightcone, nchunks=10,
                        chunk_indices=None,
                        n_psbins=50,
                        min_k=0.1,
                        max_k=1.0,
                        logk=True,
                        model_uncertainty=0.15,
                        error_on_model=True,
                        ignore_kperp_zero=True,
                        ignore_kpar_zero=False,
                        ignore_k_zero=False,
                        vb=False):

    """
    Make power spectra for given number of equally spaced chunks OR list of chunk indices
    """
    data = []
    if chunk_indices is None:
        chunk_indices = list(range(0,lightcone.n_slices,round(lightcone.n_slices / nchunks),))

        if len(chunk_indices) > nchunks:
            chunk_indices = chunk_indices[:-1]

        chunk_indices.append(lightcone.n_slices)

    else:
        nchunks = len(chunk_indices) - 1

    chunk_redshift = np.zeros(nchunks)

    lc_redshifts = lightcone.lightcone_redshifts
    for i in range(nchunks):
        if vb:
            print(f'Chunk {i}/{nchunks}...')
        start    = chunk_indices[i]
        end      = chunk_indices[i + 1]
        chunklen = (end - start) * lightcone.cell_size

        chunk_redshift[i] = np.median(lc_redshifts[start:end])

        if chunklen == 0:
            print(f'Chunk size = 0 for z = {lc_redshifts[start]}-{lc_redshifts[end]}')
        else:
            power, k, variance = compute_power(
                    lightcone.brightness_temp[:, :, start:end],
                    (lightcone.user_params.BOX_LEN, lightcone.user_params.BOX_LEN, chunklen),
                    n_psbins,
                    log_bins=logk,
                    ignore_kperp_zero=ignore_kperp_zero,
                    ignore_kpar_zero=ignore_kpar_zero,
                    ignore_k_zero=ignore_k_zero,)

            power, k, variance = power[~np.isnan(power)], k[~np.isnan(power)], variance[~np.isnan(power)]

            data.append({"k": k, "delta": power * k ** 3 / (2 * np.pi ** 2), "err_delta": np.sqrt(variance) * k ** 3 / (2 * np.pi ** 2)})

    return chunk_redshift, data
