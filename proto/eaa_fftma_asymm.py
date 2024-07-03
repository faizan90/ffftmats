'''
@author: Faizan-Uni-Stuttgart

Jul 27, 2022

9:14:42 AM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.ioff()

from ab_fftma_v1 import sph_vg, get_fft_ma_deviates

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\fftma_asymm')

    os.chdir(main_dir)

    n_vals = 1000

    n_levels = 100

    sills = np.linspace(0.01, 3, n_levels, dtype=float)
    # sills = np.full(n_levels, 0.1)
    vranges = np.linspace(1, 500, n_levels, dtype=float)

    # sills, vranges = np.meshgrid(sills, vranges)
    # sills, vranges = sills.ravel(), vranges.ravel()

    out_dir = Path(r'test_theo_vgs_18')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    h_arr = np.arange(n_vals // 2)
    h_arr = np.concatenate((h_arr, h_arr[::-1][1:]))

    corr_ftns = gen_corr_ftns(h_arr, sills, vranges)

    gau_noise = np.random.normal(loc=0.0, scale=1.0, size=h_arr.shape)

    plt.figure(figsize=(10, 5))

    out_df = pd.DataFrame(
        index=np.arange(h_arr.size),
        columns=[f'sim_{i}' for i in range(sills.size)] + ['sum'],
        dtype=float)

    out_df['sum'] = 0.0

    surrogates = []
    leg_flag = True
    for i, corr_ftn in enumerate(corr_ftns):
        surrogate = get_fft_ma_deviates(gau_noise, corr_ftn)
        surrogates.append(surrogate)

        if leg_flag:
            label = 'sim'
            leg_flag = False

        else:
            label = None

        plt.plot(surrogate, c='k', alpha=0.25, label=label)

        out_df.loc[:, f'sim_{i}'][:] = surrogate

        out_df['sum'] += surrogate

    mean = out_df['sum'].mean()
    std = out_df['sum'].std()

    out_df['sum'] -= mean
    out_df['sum'] /= std

    out_df.to_csv(out_dir / 'sims.csv', sep=';')

    plt.plot(out_df['sum'].values, c='r', alpha=0.5, label='sum')

    plt.legend()

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.xlabel('Time step')
    plt.ylabel('Amplitude')

    plt.savefig(out_dir / 'sims.png', bbox_inches='tight', dpi=150)
    plt.close()
    return


def gen_corr_ftns(h_arr, sills, vranges):

    corr_ftns = []
    for sill, vrange in zip(sills, vranges):
        corr_ftn = sill - sph_vg(h_arr, [vrange, sill])

        corr_ftns.append(corr_ftn)

    return corr_ftns


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
