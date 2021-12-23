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
# from scipy.stats import norm
import matplotlib.pyplot as plt; plt.ioff()

from aa_sampled_covariance_ftn import roll_real_2arrs

DEBUG_FLAG = False


def sph_vg(h_arr, arg):

    # arg = (range, sill)
    a = (1.5 * h_arr) / arg[0]
    b = h_arr ** 3 / (2 * arg[0] ** 3)
    sph_vg = (arg[1] * (a - b))
    sph_vg[h_arr > arg[0]] = arg[1]
    return sph_vg


def get_rfft_ma_deviates_padded(zs, corr_ftn_rft, corr_ftn_range):

    norms_ft = np.fft.rfft(zs)
    corr_ftn_ft = np.fft.rfft(corr_ftn_rft)

    norms_corr_ftn_mags_prod = norms_ft * (np.abs(corr_ftn_ft) ** 0.5)

    norms_corr_ftn_mags_prod_inv = np.fft.irfft(norms_corr_ftn_mags_prod)

    imag_vals_sum = (norms_corr_ftn_mags_prod_inv.imag ** 2).sum()
    assert np.isclose(imag_vals_sum, 0.0), imag_vals_sum

    return norms_corr_ftn_mags_prod_inv.real[corr_ftn_range:-corr_ftn_range]


def get_lagged_corr_ftn(data):

    assert not (data.shape[0] % 2), data.shape[0]

    n_corrs = data.shape[0] // 2

    corrs = [1.0]
    for lag in range(1, n_corrs):
        corr = np.corrcoef(*roll_real_2arrs(data, data, lag, False))[0, 1]

        corrs.append(corr)

    corrs = np.concatenate((corrs, corrs[::-1]))

    assert corrs.size == data.shape[0]

    return corrs


def get_corr_ftn_range(corr_ftn):

    assert (corr_ftn[0] == 1.0) and (corr_ftn[-1] == 1.0)

    # idxs_below_zero = np.where(corr_ftn < 0)[0]
    #
    # corr_ftn_range = idxs_below_zero[0]

    corr_ftn_range = corr_ftn.size // 2

    return corr_ftn_range


def pad_corr_ftn(corr_ftn, corr_ftn_range):

    zeros_arr = np.zeros(corr_ftn.shape[0] + 1)

    padded_corr_ftn = np.concatenate((
        corr_ftn[:corr_ftn_range],
        zeros_arr,
        corr_ftn[-corr_ftn_range + 1:]))

    assert padded_corr_ftn.size == (corr_ftn.size + (2 * corr_ftn_range))

    return padded_corr_ftn

# def pad_corr_ftn(corr_ftn, corr_ftn_range):
#
#     zeros_arr = np.zeros((corr_ftn_range * 2))
#
#     padded_corr_ftn = np.concatenate((
#         corr_ftn[:(corr_ftn.size // 2)],
#         zeros_arr,
#         corr_ftn[(corr_ftn.size // 2):]))
#
#     assert padded_corr_ftn.size == (corr_ftn.size + (2 * corr_ftn_range))
#
#     return padded_corr_ftn


def main():

    '''
    Instead of the FT rolled correlation function, we use the lagged one
    and pad it with a length in the middle such that the final simulated
    series has no correlation between the first and the last element.
    '''

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\fftma')

    os.chdir(main_dir)

    in_data_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    sep = ';'

    col = '420'

    beg_time = '2000-01-01'
    end_time = '2001-12-31'

    time_fmt = '%Y-%m-%d'

    # idx_perturb = 100

    out_dir = Path(r'test_fftma_v2_02')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    in_data_df = pd.read_csv(in_data_file, sep=sep, index_col=0)
    in_data_df.index = pd.to_datetime(in_data_df.index, format=time_fmt)

    in_data_ser = in_data_df.loc[beg_time:end_time, col]

    if (in_data_ser.shape[0] % 2):
        in_data_ser = in_data_ser.iloc[:-1]

    # in_data_ser[:] = norm.ppf(
    #     in_data_ser.rank() / (in_data_ser.shape[0] + 1.0))

    corr_ftn = get_lagged_corr_ftn(in_data_ser.values)

    corr_ftn_range = corr_ftn.size // 2  # get_corr_ftn_range(corr_ftn)

    corr_ftn = pad_corr_ftn(corr_ftn, corr_ftn_range)

    norms_a = np.random.normal(
        loc=in_data_ser.mean(),
        scale=in_data_ser.std(),
        size=corr_ftn.shape[0])

    norms_a_rfft_ma = get_rfft_ma_deviates_padded(
        norms_a, corr_ftn, corr_ftn_range)

    assert norms_a_rfft_ma.size == in_data_ser.shape[0]

    # Plot.
    plt.figure(figsize=(7, 5))
    plt.plot(norms_a_rfft_ma, label='ra', alpha=0.75)
    # plt.plot(in_data_ser.values, label='orig', alpha=0.75)
    # plt.plot(corr_ftn, label='corr_ftn', alpha=0.75)

    # plt.xlim(idx_perturb - 30, idx_perturb + 30)
    plt.xlabel('Time step')
    plt.ylabel('Signal')

    plt.legend()

    plt.grid()
    plt.gca().set_axisbelow(True)

    # plt.savefig(out_dir / 'example_fftma.png', bbox_inches='tight')

    plt.show()

    plt.close()
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
