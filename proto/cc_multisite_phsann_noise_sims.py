'''
@author: Faizan-Uni-Stuttgart

Dec 21, 2021

11:30:06 AM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd

from ab_fftma_v1 import get_rfft_ma_deviates

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\fftma')

    main_dir /= r'test_fftma_ms_05'

    os.chdir(main_dir)
    #==========================================================================

    out_dir = Path('phsann_noise_sims')

    out_dir.mkdir(exist_ok=True)

    data_df = pd.read_csv('data.csv', sep=';', index_col=0)

    data_df_srtd = data_df.copy()

    data_df_srtd.values[:] = np.sort(data_df, axis=0)

    corr_ftns_df = pd.read_csv(r'corr_ftns.csv', sep=';')

    sim_noise_dfs = []
    for sim_noise_file in (
        main_dir / r'phsann_noise').glob('./sim_data_*.csv'):

        sim_noise_df = pd.read_csv(sim_noise_file, sep=';')
        sim_noise_df.index = data_df.index

        sim_noise_dfs.append(sim_noise_df)

    assert len(sim_noise_dfs)

    sim_dfs = [sim_noise_df.copy() for sim_noise_df in sim_noise_dfs]

    for col in data_df.columns:
        corr_ftn = corr_ftns_df[col].values

        for sim_noise_df, sim_df in zip(sim_noise_dfs, sim_dfs):
            fftma_deviates = get_rfft_ma_deviates(
                sim_noise_df[col].values, corr_ftn)

            sim_df[col] = fftma_deviates

    for col in data_df.columns:
        out_wo_shuff_df = pd.DataFrame(
            index=data_df.index,
            columns=['ref'] + [f'sim_calib_{i}' for i in range(len(sim_dfs))])

        out_wi_shuff_df = out_wo_shuff_df.copy()

        out_wo_shuff_df['ref'] = data_df[col]
        out_wi_shuff_df['ref'] = data_df[col]

        for i in range(len(sim_dfs)):
            out_wo_shuff_df[f'sim_calib_{i}'] = sim_dfs[i][col]

            out_wi_shuff_df[f'sim_calib_{i}'] = data_df_srtd[col].values[
                np.argsort(np.argsort(sim_dfs[i][col].values))]

        out_wo_shuff_df.to_csv(out_dir / f'sims_{col}_wo_shuff.csv', sep=';')
        out_wi_shuff_df.to_csv(out_dir / f'sims_{col}_wi_shuff.csv', sep=';')

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
