'''
@author: Faizan-Uni-Stuttgart

Dec 7, 2021

2:47:43 PM

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

from ab_fftma_v2 import (
    get_lagged_corr_ftn,
    # get_corr_ftn_range,
    pad_corr_ftn,
    get_rfft_ma_deviates_padded)

DEBUG_FLAG = False


def get_rfft_ma_white_noise(data, corr_ftn_rft):

    norms_cov_ftn_mags_prod = np.fft.rfft(data)

    norms_ft = norms_cov_ftn_mags_prod / (
        np.abs(np.fft.rfft(corr_ftn_rft)) ** 0.5)

    norms = np.fft.irfft(norms_ft)

    imag_vals_sum = (norms.imag ** 2).sum()
    assert np.abs(norms.imag).max() <= 1e-12, imag_vals_sum

    return norms.real


def get_rfft_ma_white_noise_padded(data, corr_ftn_rft, corr_ftn_range):

    norms_cov_ftn_mags_prod = np.fft.rfft(data - data.mean())

    norms_ft = norms_cov_ftn_mags_prod / (
        np.abs(np.fft.rfft(corr_ftn_rft)) ** 0.5)

    norms_ft[-1] = 1.0

    norms = np.fft.irfft(norms_ft)

    imag_vals_sum = (norms.imag ** 2).sum()
    assert np.abs(norms.imag).max() <= 1e-12, imag_vals_sum

    return norms.real[corr_ftn_range:-corr_ftn_range]


def pad_data(data, corr_ftn_range):

    means_arr = np.full(corr_ftn_range, data.mean())

    padded_data = np.concatenate((
        means_arr, data, means_arr))

    assert padded_data.size == (data.size + (2 * corr_ftn_range))

    return padded_data


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\fftma')
    os.chdir(main_dir)

    in_data_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    # in_data_file = Path(
    #     r'fftma_sa_v3_06/sims.csv')

    beg_time = '2000-01-01'
    end_time = '2001-12-30'
    time_fmt = '%Y-%m-%d'

    # beg_time = '2001-06-01'
    # end_time = '2002-06-29'
    # time_fmt = '%Y-%m-%d'

    # in_data_file = Path(
    #     r'hourly_bw_discharge__2008__2019.csv')
    #
    # beg_time = '2009-01-01-00'
    # end_time = '2018-12-31-23'
    # time_fmt = '%Y-%m-%d-%H'

    sep = ';'

    col = '420'
    # col = 'ref'
    # col = 'norms_calib_1'

    out_dir = Path(r'test_fftma_v2_02')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    in_data_df = pd.read_csv(in_data_file, sep=sep, index_col=0)
    in_data_df.index = pd.to_datetime(in_data_df.index, format=time_fmt)

    in_data_ser = in_data_df.loc[beg_time:end_time, col]

    assert np.all(np.isfinite(in_data_ser.values))

    in_data_ser_orig = in_data_ser.copy()

    in_data_ser_orig_srtd = np.sort(in_data_ser_orig)

    # if (in_data_ser.shape[0] % 2):
    #     in_data_ser = in_data_ser.iloc[:-1]

    # in_data_ser[:] = norm.ppf(
    #     in_data_ser.rank() / (in_data_ser.shape[0] + 1.0))

    corr_ftn = get_lagged_corr_ftn(in_data_ser.values)

    corr_ftn_range = corr_ftn.size // 2  # get_corr_ftn_range(corr_ftn)

    corr_ftn = pad_corr_ftn(corr_ftn, corr_ftn_range)

    data = pad_data(in_data_ser.values, corr_ftn_range)

    # DatetimeIndex is taken from ref.
    # sims = {'ref': in_data_ser_orig.copy()}

    norms_rft = get_rfft_ma_white_noise(data, corr_ftn)

    sim_rft = get_rfft_ma_deviates_padded(norms_rft, corr_ftn, corr_ftn_range)
    #
    # assert sim_rft.size == in_data_ser_orig.size, (
    #     sim_rft.size, in_data_ser_orig.size)
    #
    sim_rft = in_data_ser_orig_srtd[np.argsort(np.argsort(sim_rft))]

    # sims[f'sim'] = sim
    # sims[f'norms'] = norms

    # sims[f'sim_rft'] = sim_rft
    # sims[f'norms_rft'] = norms_rft

    # pd.DataFrame(sims).to_csv(
    #     out_dir / 'sims.csv', sep=';', float_format='%0.6f')

    # probs = np.arange(norms_rft.shape[0]) / (norms_rft.shape[0] + 1.0)
    #
    # norms_rft_srtd = np.sort(norms_rft)

    plt.figure(figsize=(10, 7))

    # plt.plot(
    #     norms_srtd[norms.shape[0] // 2:],
    #     probs[norms.shape[0] // 2:],
    #     alpha=0.75)

    # plt.plot(
    #     norms_srtd[:norms.shape[0] // 2],
    #     probs[:norms.shape[0] // 2],
    #     alpha=0.75)

    # plt.plot(
    #     norms_srtd,
    #     probs,
    #     alpha=0.75)

    # plt.plot(
    #     norms_rft_srtd,
    #     probs,
    #     alpha=0.75,
    #     c='k',
    #     lw=1)

    # plt.hist(
    #     norms_rft_srtd,
    #     bins=50,
    #     alpha=0.75)

    plt.plot(
        norms_rft[corr_ftn_range:-corr_ftn_range],
        alpha=0.75,
        c='k',
        lw=1,
        label='noise')

    plt.plot(
        sim_rft,
        alpha=0.75,
        c='r',
        lw=1.5,
        label='sim')

    plt.plot(
        in_data_ser.values,
        alpha=0.75,
        c='r',
        lw=1.5,
        label='ref')

    plt.legend()

    plt.xlabel('Time step')
    plt.ylabel('Signal')

    # plt.xlabel('z')
    # plt.ylabel('F(z)')

    plt.grid()
    plt.gca().set_axisbelow(True)
#
#     plt.savefig('example_noise_dist.png', bbox_inches='tight')

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
