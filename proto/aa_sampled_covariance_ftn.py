'''
@author: Faizan-Uni-Stuttgart

Dec 6, 2021

10:11:49 AM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False


def roll_real_2arrs(arr1, arr2, lag, rerank_flag=False):

    assert isinstance(arr1, np.ndarray)
    assert isinstance(arr2, np.ndarray)

    assert arr1.ndim == 1
    assert arr2.ndim == 1

    assert arr1.size == arr2.size

    assert isinstance(lag, (int, np.int64))
    assert abs(lag) < arr1.size

    if lag > 0:
        # arr2 is shifted ahead
        arr1 = arr1[:-lag].copy()
        arr2 = arr2[+lag:].copy()

    elif lag < 0:
        # arr1 is shifted ahead
        arr1 = arr1[-lag:].copy()
        arr2 = arr2[:+lag].copy()

    else:
        pass

    assert arr1.size == arr2.size

    if rerank_flag:
#         assert np.all(arr1 > 0) and np.all(arr2 > 0)
#         assert np.all(arr1 < 1) and np.all(arr2 < 1)

        arr1 = rankdata(arr1) / (arr1.size + 1.0)
        arr2 = rankdata(arr2) / (arr2.size + 1.0)

    return arr1, arr2


def get_ft_corr_ftn(data):

    # Divding by variance and N did not change anything.
    data_ft = np.fft.fft(data - data.mean())

    data_mag = np.abs(data_ft)

    # Wiener-Khintchin theorem.
    pwr_ft = np.fft.ifft(data_mag ** 2)

    pwr_ft_pcorr = np.sign(pwr_ft.real) * np.abs(pwr_ft)

    pwr_ft_pcorr /= pwr_ft_pcorr[0]

    pwr_ft_pcorr = np.concatenate([pwr_ft_pcorr, [pwr_ft_pcorr[0]]])

    return pwr_ft_pcorr


def get_rft_corr_ftn(data):

    # Divding by variance and N did not change anything.
    data_ft = np.fft.rfft(data - data.mean())

    data_mag = np.abs(data_ft)

    # Wiener-Khintchin theorem.
    pwr_ft = np.fft.irfft(data_mag ** 2)

    pwr_ft_pcorr = np.sign(pwr_ft.real) * np.abs(pwr_ft)

    pwr_ft_pcorr /= pwr_ft_pcorr[0]

    pwr_ft_pcorr = np.concatenate([pwr_ft_pcorr, [pwr_ft_pcorr[0]]])

    return pwr_ft_pcorr


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\fftma')
    os.chdir(main_dir)

    in_data_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    beg_time = '2000-01-01'
    end_time = '2001-12-31'
    time_fmt = '%Y-%m-%d'

    # in_data_file = Path(
    #     r'fftma_sa_v3_06/sims.csv')
    #
    # beg_time = '2000-01-01'
    # end_time = '2001-12-30'
    # time_fmt = '%Y-%m-%d'

    # in_data_file = Path(
    #     r'hourly_bw_discharge__2008__2019.csv')
    #
    # beg_time = '2009-01-01-00'
    # end_time = '2018-12-31-23'
    # time_fmt = '%Y-%m-%d-%H'

    sep = ';'

    col = '420'
    # col = 'norms_calib_0

    lags = np.arange(366 * 3, dtype=np.int64)
    #==========================================================================

    in_data_df = pd.read_csv(in_data_file, sep=sep, index_col=0)
    in_data_df.index = pd.to_datetime(in_data_df.index, format=time_fmt)

    in_data_ser = in_data_df.loc[beg_time:end_time, col]

    if (in_data_ser.shape[0] % 2):
        in_data_ser = in_data_ser.iloc[:-1]

    # Create cov_ftn from pwr_spec.
    corr_ftn = get_ft_corr_ftn(in_data_ser.values)
    corr_ftn_rft = get_rft_corr_ftn(in_data_ser.values)

    # Create corr_ftn from data.
    ref_pcorrs = []
    for lag in lags:
        # pcorr = np.corrcoef(*roll_real_2arrs(
        #     in_data_ser.values, in_data_ser.values, lag, False))[0, 1]

        pcorr = np.corrcoef(
            in_data_ser.values, np.roll(in_data_ser.values, lag))[0, 1]

        # pcorr = np.corrcoef(*roll_real_2arrs(
        #     in_data_ser.values, in_data_ser.values, lag, False))[0, 1]

        ref_pcorrs.append(round(pcorr, 6))

    ref_pcorrs = np.array(ref_pcorrs)

    print(in_data_ser.shape, corr_ftn.shape, corr_ftn_rft.shape)

    # Plot.
    plt.figure(figsize=(10, 7))
    # plt.plot(data_mag[1:])
    # plt.plot(corr_ftn, label='ft', alpha=0.75)
    plt.plot(corr_ftn_rft, label='rft', alpha=0.75, lw=3, c='r')
    # plt.plot(ref_pcorrs, label='cov', alpha=0.75)

    plt.legend()

    plt.xlabel('Time step')
    plt.ylabel('Pearson correlation')

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.savefig('example_corr_ftn.png', bbox_inches='tight')

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
