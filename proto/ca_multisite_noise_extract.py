'''
@author: Faizan-Uni-Stuttgart

Dec 20, 2021

9:38:59 AM

'''
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

from aa_sampled_covariance_ftn import get_rft_corr_ftn
from ab_fftma_v1 import get_fft_ma_deviates, get_rfft_ma_deviates

DEBUG_FLAG = False


def get_fft_ma_white_noise(data, corr_ftn):

    norms_cov_ftn_mags_prod = np.fft.fft(data)

    norms_ft = norms_cov_ftn_mags_prod / (np.abs(np.fft.fft(corr_ftn)) ** 0.5)

    norms = np.fft.ifft(norms_ft)

    imag_vals_sum = (norms.imag ** 2).sum()
    assert imag_vals_sum <= 1e-7, imag_vals_sum

    return norms.real


def get_rfft_ma_white_noise(data, corr_ftn_rft):

    norms_cov_ftn_mags_prod = np.fft.rfft(data)

    norms_ft = norms_cov_ftn_mags_prod / (
        np.abs(np.fft.rfft(corr_ftn_rft)) ** 0.5)

    norms = np.fft.irfft(norms_ft)

    imag_vals_sum = (norms.imag ** 2).sum()
    assert np.abs(norms.imag).max() <= 1e-12, imag_vals_sum

    return norms.real


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\fftma')
    os.chdir(main_dir)

    in_data_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    # in_data_file = Path(
    #     r'fftma_sa_v3_06/sims.csv')

    beg_time = '2000-01-01'
    end_time = '2001-12-31'
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

    # col = '420'
    # col = 'ref'
    # col = 'norms_calib_1'

    cols = ['420', '3421']  # , '3465', '3470']
    # cols = 'all'

    win_size = (180) - 1  # For annual cycle, an odd value.
    no_ann_cyc_flag = False

    out_dir = Path(r'test_fftma_ms_05')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    in_data_df = pd.read_csv(in_data_file, sep=sep, index_col=0)

    if cols == 'all':
        cols = in_data_df.columns.tolist()

    else:
        in_data_df = in_data_df[cols]

    in_data_df.index = pd.to_datetime(in_data_df.index, format=time_fmt)

    in_data_df = in_data_df.loc[beg_time:end_time]

    if (in_data_df.shape[0] % 2):
        in_data_df = in_data_df.iloc[:-1]

    assert np.all(np.isfinite(in_data_df.values))

    if no_ann_cyc_flag:
        ann_cyc_df = in_data_df.rolling(
            win_size,
            min_periods=win_size,
            center=True,
            win_type='triang').mean()

        in_data_df -= ann_cyc_df

        ann_cyc_df.to_csv(out_dir / 'anncyc.csv', sep=';')

        in_data_df = in_data_df.loc[~in_data_df.isnull().any(axis=1)]

    in_data_df_orig = in_data_df.copy()

    in_data_df_orig_srtd = np.sort(in_data_df_orig, axis=0)

    corr_ftns = np.empty((in_data_df.shape[0] + 1, len(cols)))
    norms = np.empty((in_data_df.shape[0], len(cols)))
    deviates = np.empty((in_data_df.shape[0], len(cols)))
    sims = np.empty((in_data_df.shape[0], len(cols)))

    for i, col in enumerate(cols):
        corr_ftns[:, i] = get_rft_corr_ftn(in_data_df[col].values)

        norms[:, i] = get_rfft_ma_white_noise(
            in_data_df.iloc[:, i].values, corr_ftns[:, i])

        deviates[:, i] = get_rfft_ma_deviates(norms[:, i], corr_ftns[:, i])

        sims[:, i] = in_data_df_orig_srtd[
            np.argsort(np.argsort(deviates[:, i])), i]

    corr_ftns = pd.DataFrame(data=corr_ftns, columns=in_data_df.columns)

    norms = pd.DataFrame(
        data=norms, columns=in_data_df.columns, index=in_data_df.index)

    deviates = pd.DataFrame(
        data=deviates, columns=in_data_df.columns, index=in_data_df.index)

    sims = pd.DataFrame(
        data=sims, columns=in_data_df.columns, index=in_data_df.index)

    print('Pearson')
    print(in_data_df.corr('pearson'))
    print(corr_ftns.corr('pearson'))
    print(norms.corr('pearson'))
    print(deviates.corr('pearson'))
    print(sims.corr('pearson'))

    print('Spearman')
    print(in_data_df.corr('spearman'))
    print(corr_ftns.corr('spearman'))
    print(norms.corr('spearman'))
    print(deviates.corr('spearman'))
    print(sims.corr('spearman'))

    in_data_df.to_csv(out_dir / 'data.csv', sep=';')
    corr_ftns.to_csv(out_dir / 'corr_ftns.csv', sep=';')
    norms.to_csv(out_dir / 'norms.csv', sep=';')
    deviates.to_csv(out_dir / 'deviates.csv', sep=';')
    sims.to_csv(out_dir / 'sims.csv', sep=';')

    plt.figure(figsize=(10, 7))

    # for col in cols:
    #     plt.scatter(
    #         norms[col],
    #         norms[col].rank() / (norms.shape[0] + 1.0),
    #         label=col,
    #         alpha=0.5)

    # for col in cols:
    #     plt.plot(
    #         ann_cyc_df[col],
    #         # norms[col].rank() / (norms.shape[0] + 1.0),
    #         label=col,
    #         alpha=0.5)

    # for col in cols:
    #     plt.plot(
    #         corr_ftns[col],
    #         # norms[col].rank() / (norms.shape[0] + 1.0),
    #         label=col,
    #         alpha=0.5)

    # for col in cols:
    #     plt.plot(
    #         norms[col],
    #         # norms[col].rank() / (norms.shape[0] + 1.0),
    #         label=col,
    #         alpha=0.5)

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
    #
    # plt.plot(
    #     norms_rft,
    #     alpha=0.75,
    #     c='k',
    #     lw=1,
    #     label='noise')
    #
    # plt.plot(
    #     in_data_ser.values,
    #     alpha=0.75,
    #     c='r',
    #     lw=1.5,
    #     label='ref')
    #
    # plt.legend()
    #
    # plt.xlabel('Time step')
    # plt.ylabel('Signal')

#     plt.xlabel('z')
#     plt.ylabel('F(z)')
#
    plt.legend()

    plt.grid()
    plt.gca().set_axisbelow(True)
#
#     plt.savefig('example_noise_dist.png', bbox_inches='tight')

    # plt.show()

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
