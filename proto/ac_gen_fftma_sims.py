'''
@author: Faizan-Uni-Stuttgart

Dec 6, 2021

2:27:05 PM

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

from aa_sampled_covariance_ftn import get_ft_corr_ftn
from ab_fftma_v1 import get_fft_ma_deviates, sph_vg

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\fftma')
    os.chdir(main_dir)

    in_data_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    beg_time = '2000-01-01'
    end_time = '2001-12-30'
    time_fmt = '%Y-%m-%d'

    # in_data_file = Path(
    #     r'hourly_bw_discharge__2008__2019.csv')
    #
    # beg_time = '2009-01-01-00'
    # end_time = '2018-12-31-23'
    # time_fmt = '%Y-%m-%d-%H'

    sep = ';'

    col = '420'

    n_sims = 300

    out_dir = Path(r'fftma_v1_sims_07_daily')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    in_data_df = pd.read_csv(in_data_file, sep=sep, index_col=0)
    in_data_df.index = pd.to_datetime(in_data_df.index, format=time_fmt)

    in_data_ser = in_data_df.loc[beg_time:end_time, col]

    assert np.all(np.isfinite(in_data_ser.values))

    in_data_ser_orig = in_data_ser.copy()

    in_data_ser_orig_srtd = np.sort(in_data_ser_orig)

    if (in_data_ser.shape[0] % 2):
        in_data_ser = in_data_ser.iloc[:-1]

    in_data_ser[:] = norm.ppf(
        in_data_ser.rank() / (in_data_ser.shape[0] + 1.0))

    if True:
        corr_ftn = get_ft_corr_ftn(in_data_ser.values)

    else:
        h_arr = np.arange(in_data_ser.shape[0] // 2)

        h_arr = np.concatenate((h_arr, h_arr[::-1][1:]))

        corr_ftn = sph_vg(h_arr, [100, 1])

    if in_data_ser.shape[0] > corr_ftn.shape[0]:
        in_data_ser = in_data_ser.iloc[:corr_ftn.shape[0]]
        in_data_ser_orig = in_data_ser_orig.iloc[:corr_ftn.shape[0]]

    elif in_data_ser.shape[0] < corr_ftn.shape[0]:
        corr_ftn = corr_ftn[:in_data_ser.shape[0]]

    assert corr_ftn.size == in_data_ser.size

    mean = in_data_ser.mean()
    std = in_data_ser.std()

    # DatetimeIndex is taken from ref.
    sims = {'ref': in_data_ser_orig.copy()}

    for i in range(n_sims):
        norms = np.random.normal(loc=mean, scale=std, size=corr_ftn.shape[0])

        sim = get_fft_ma_deviates(norms, corr_ftn)

        assert sim.size == in_data_ser_orig.size, (
            sim.size, in_data_ser_orig.size)

        sim = in_data_ser_orig_srtd[np.argsort(np.argsort(sim))]

        sims[f'sims_{i:04d}'] = sim

    pd.DataFrame(sims).to_csv(
        out_dir / 'sims.csv', sep=';', float_format='%0.6f')

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
