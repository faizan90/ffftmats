'''
@author: Faizan-Uni-Stuttgart

Mar 30, 2022

2:25:47 PM

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
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt; plt.ioff()

from ab_fftma_v2 import (
    get_lagged_corr_ftn,
    # pad_corr_ftn,
    get_rfft_ma_deviates_padded,
    # get_corr_ftn_range,
    )

from ad_fftma_reverse_white_noise_v2 import (
    get_rfft_ma_white_noise_padded,
    # pad_data,
    )

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\fftma')
    os.chdir(main_dir)

    # in_file_path = Path(
    #     r'neckar_1hr_ppt_data_20km_buff_Y2004_2020.pkl')

    in_file_path = Path(r'BW_dwd_stns_60min_1995_2020_data.csv')

    beg_time = '2010-01-01-00'
    end_time = '2014-12-31-23'

    sep = ';'

    time_fmt = '%Y-%m-%dT%H:%M:%S'

    col = 'P00071'

    evt_idx = 5000

    evt_buff_steps = 200

    fig_size = (18, 7)

    out_dir = Path(
        r'P:\Synchronize\IWS\Projects\2016_DFG_SPATE\Meetings_Second_Phase\20220404_Hannover')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    if in_file_path.suffix == '.csv':
        in_df = pd.read_csv(in_file_path, sep=sep, index_col=0)
        in_df.index = pd.to_datetime(in_df.index, format=time_fmt)

    elif in_file_path.suffix == '.pkl':
        in_df = pd.read_pickle(in_file_path)

    else:
        raise NotImplementedError(
            f'Unknown extension of in_data_file: {in_file_path.suffix}!')

    in_data_ser = in_df.loc[beg_time:end_time, col]

    assert np.all(np.isfinite(in_data_ser.values))

    if (in_data_ser.shape[0] % 2):
        in_data_ser = in_data_ser.iloc[:-1]

    in_data_ser_orig = in_data_ser.copy()

    in_data_ser_orig_srtd = np.sort(in_data_ser_orig)

    corr_ftn = get_lagged_corr_ftn(in_data_ser.values)

    corr_ftn_range = 0

    data = in_data_ser.values.copy()

    norms_white_noise = get_rfft_ma_white_noise_padded(
        data, corr_ftn, corr_ftn_range)

    norms_white_noise_dist_ftn = interp1d(
        np.sort(rankdata(norms_white_noise) / (norms_white_noise.size + 1.0)),
        np.sort(norms_white_noise),
        bounds_error=False,
        fill_value=(norms_white_noise.min(), norms_white_noise.max()))

    if False:
        norms = norms_white_noise_dist_ftn(np.random.random(corr_ftn.shape[0]))

    else:
        norms = norms_white_noise.copy()
        norms[evt_idx] += norms.max() * 0.5

    sim = get_rfft_ma_deviates_padded(norms, corr_ftn, corr_ftn_range)

    assert sim.size == in_data_ser_orig.size, (
        sim.size, in_data_ser_orig.size)

    sim = in_data_ser_orig_srtd[np.argsort(np.argsort(sim))]

    # Plot.
    if True:
        # Corr ftn.
        plt.figure(figsize=fig_size)
        plt.plot(corr_ftn, label='corr_ftn', alpha=0.75)

        plt.xlim(0, 60)

        plt.xlabel('Time [hour]')
        plt.ylabel('Pearson correlation [-]')

        plt.legend()

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.savefig(
            str(out_dir / 'fftma_demonst_cftn.png'),
            bbox_inches='tight',
            dpi=150)
        plt.close()

    if True:
        # Noise.
        plt.figure(figsize=fig_size)
        plt.plot(norms_white_noise, label='ref', alpha=0.75, ls='--', c='r', lw=3)
        plt.plot(norms, label='sim', alpha=0.75, ls='-', c='k', lw=1)

        plt.xlim(evt_idx - evt_buff_steps , evt_idx + evt_buff_steps)

        y_lim_min = np.floor(min(
            [norms_white_noise[evt_idx - evt_buff_steps: evt_idx + evt_buff_steps].min(),
             norms[evt_idx - evt_buff_steps: evt_idx + evt_buff_steps].min()]))

        y_lim_max = np.ceil(max([
            norms_white_noise[evt_idx - evt_buff_steps: evt_idx + evt_buff_steps].max(),
            norms[evt_idx - evt_buff_steps: evt_idx + evt_buff_steps].max()]))

        plt.ylim(y_lim_min, y_lim_max)

        plt.xlabel('Time step [hour]')
        plt.ylabel('Noise [-]')

        plt.legend()

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.savefig(
            str(out_dir / 'fftma_demonst_noise.png'),
            bbox_inches='tight',
            dpi=150)

        plt.close()

    if True:
        # Series.
        plt.figure(figsize=fig_size)
        plt.plot(data, label='ref', alpha=0.75, ls='--', c='r', lw=3)
        plt.plot(sim, label='sim', alpha=0.75, ls='-', c='k', lw=1)

        plt.xlim(evt_idx - evt_buff_steps , evt_idx + evt_buff_steps)

        y_lim_min = np.floor(min(
            [data[evt_idx - evt_buff_steps: evt_idx + evt_buff_steps].min(),
             sim[evt_idx - evt_buff_steps: evt_idx + evt_buff_steps].min()]))

        y_lim_max = np.ceil(max([
            data[evt_idx - evt_buff_steps: evt_idx + evt_buff_steps].max(),
            sim[evt_idx - evt_buff_steps: evt_idx + evt_buff_steps].max()]))

        plt.ylim(y_lim_min, y_lim_max)

        plt.xlabel('Time step [hour]')
        plt.ylabel('Precipitation [mm]')

        plt.legend()

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.savefig(
            str(out_dir / 'fftma_demonst_cmpr.png'),
            bbox_inches='tight',
            dpi=150)

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
