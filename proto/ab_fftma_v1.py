'''
@author: Faizan-Uni-Stuttgart

Dec 6, 2021

11:37:54 AM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt; plt.ioff()

from aa_sampled_covariance_ftn import get_ft_corr_ftn, get_rft_corr_ftn

DEBUG_FLAG = False


def sph_vg(h_arr, arg):

    # arg = (range, sill)
    a = (1.5 * h_arr) / arg[0]
    b = h_arr ** 3 / (2 * arg[0] ** 3)
    sph_vg = (arg[1] * (a - b))
    sph_vg[h_arr > arg[0]] = arg[1]
    return sph_vg


def get_fft_ma_deviates(zs, corr_ftn):

    norms_ft = np.fft.fft(zs)
    corr_ftn_ft = np.fft.fft(corr_ftn)

    norms_corr_ftn_mags_prod = norms_ft * (np.abs(corr_ftn_ft) ** 0.5)

    norms_corr_ftn_mags_prod_inv = np.fft.ifft(norms_corr_ftn_mags_prod)

    imag_vals_sum = (norms_corr_ftn_mags_prod_inv.imag ** 2).sum()
    assert np.isclose(imag_vals_sum, 0.0), imag_vals_sum

    return norms_corr_ftn_mags_prod_inv.real


def get_rfft_ma_deviates(zs, corr_ftn_rft):

    norms_ft = np.fft.rfft(zs)
    corr_ftn_ft = np.fft.rfft(corr_ftn_rft)

    norms_corr_ftn_mags_prod = norms_ft * (np.abs(corr_ftn_ft) ** 0.5)

    norms_corr_ftn_mags_prod_inv = np.fft.irfft(norms_corr_ftn_mags_prod)

    imag_vals_sum = (norms_corr_ftn_mags_prod_inv.imag ** 2).sum()
    assert np.isclose(imag_vals_sum, 0.0), imag_vals_sum

    return norms_corr_ftn_mags_prod_inv.real


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\fftma')
    os.chdir(main_dir)

    in_data_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    sep = ';'

    col = '420'

    beg_time = '1990-01-01'
    end_time = '1990-12-31'

    time_fmt = '%Y-%m-%d'

    idx_perturb = 100
    #==========================================================================

    in_data_df = pd.read_csv(in_data_file, sep=sep, index_col=0)
    in_data_df.index = pd.to_datetime(in_data_df.index, format=time_fmt)

    in_data_ser = in_data_df.loc[beg_time:end_time, col]

    if (in_data_ser.shape[0] % 2):
        in_data_ser = in_data_ser.iloc[:-1]

    in_data_ser[:] = norm.ppf(in_data_ser.rank() / (in_data_ser.shape[0] + 1.0))

    if True:
        corr_ftn = get_ft_corr_ftn(in_data_ser.values)
        corr_ftn_rft = get_rft_corr_ftn(in_data_ser.values)

    else:
        h_arr = np.arange(in_data_ser.shape[0] * 2)

        h_arr = np.concatenate((h_arr, h_arr[::-1][1:]))

        corr_ftn = sph_vg(h_arr, [100, 1])

    norms_a = np.random.normal(
        loc=in_data_ser.mean(), scale=in_data_ser.std(), size=corr_ftn.shape[0])

    norms_b = norms_a.copy()

    norms_b[idx_perturb] += -5

    norms_a_rft = np.random.normal(
        loc=in_data_ser.mean(),
        scale=in_data_ser.std(),
        size=corr_ftn_rft.shape[0])

    norms_a_fft_ma = get_fft_ma_deviates(norms_a, corr_ftn)
    norms_b_fft_ma = get_fft_ma_deviates(norms_b, corr_ftn)

    norms_a_rfft_ma = get_fft_ma_deviates(norms_a_rft, corr_ftn_rft)

    # Plot.
    plt.figure(figsize=(7, 5))
    plt.plot(norms_a_fft_ma, label='ref', alpha=0.75, ls='--', c='r', lw=3)
    plt.plot(norms_b_fft_ma, label='sim', alpha=0.75, ls='dotted', c='k', lw=2)
    # plt.plot(norms_a_rfft_ma, label='ra', alpha=0.75)
    # plt.plot(in_data_ser.values, label='orig', alpha=0.75)
    # plt.plot(corr_ftn, label='corr_ftn', alpha=0.75)

    plt.xlim(70, 130)
    plt.xlabel('Time step')
    plt.ylabel('Signal')

    plt.legend()

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.savefig('example_fftma.png', bbox_inches='tight')

    # plt.show()

    return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
