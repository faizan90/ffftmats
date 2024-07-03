'''
@author: Faizan-Uni-Stuttgart

Jul 27, 2022

10:50:56 AM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path
from fnmatch import fnmatch

import numpy as np
import pandas as pd
from scipy.stats import rankdata, expon
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False


def main():

    '''
    Using the Professor's idea to shift values in time based on their
    magnitudes. Smoothing and control of the amount of shift are applied here.
    '''

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\fftma_asymm')

    main_dir /= r'test_theo_vgs_16'

    os.chdir(main_dir)

    # Proportional to directional asymmetry.
    n_levels = 100

    # Same as n_levels.
    max_shift_exp = 5
    max_shift = 6

    pre_vals_ratio = 0.5

    sims_file = Path(r'sims.csv')
    #==========================================================================

    sims_df = pd.read_csv(sims_file, sep=';', index_col=0)

    sims_df_cols = [fnmatch(col, 'sim_*') for col in sims_df.columns]

    sims_df = sims_df.loc[:, sims_df_cols]

    asymms_df = pd.DataFrame(
        index=sims_df.index, columns=sims_df.columns, dtype=float)

    plt.figure(figsize=(10, 5))

    for i in range(sims_df.shape[1]):

        vals = sims_df.iloc[:, i].values.copy()
        vals_gau = vals.copy()

        vals_sort = np.sort(vals)

        probs = rankdata(vals) / (vals.size + 1.0)

        levels = (probs * n_levels).astype(int)

        rand_epss = ((-1e-6) + (2e-6) * np.random.random(size=vals.size))

        asymm_vals = vals.copy()
        for level in range(n_levels):
            asymm_vals_i = asymm_vals.copy()

            # max_shift_level = int(
            #     round(max_shift - (max_shift * ((level / n_levels) ** max_shift_exp))))

            max_shift_level = int(
                (max_shift - (max_shift * ((level / n_levels) ** max_shift_exp))))

            if max_shift_level <= 0:
                shifts = range(max_shift_level, 0, 1)

            else:
                shifts = range(1, max_shift_level + 1)

            for shift in shifts:
                asymm_vals_i = np.roll(asymm_vals_i, shift)

                idxs_to_shift = levels <= level

                asymm_vals[idxs_to_shift] = (
                    asymm_vals[idxs_to_shift] * pre_vals_ratio +
                    asymm_vals_i[idxs_to_shift] * (1 - pre_vals_ratio))

        asymm_vals += rand_epss
        asymm_vals = vals_sort[np.argsort(np.argsort(asymm_vals))]

        asymms_df.iloc[:, i] = asymm_vals
        #======================================================================

        vals = asymm_vals.copy()

        vals_sort = np.sort(vals)

        probs = rankdata(vals) / (vals.size + 1.0)

        levels = (probs * n_levels).astype(int)

        asymm_vals_2 = vals.copy()
        for level in range(n_levels):
            asymm_vals_i = asymm_vals_2.copy()

            max_shift_level = int(
                round(max_shift - (max_shift * ((level / n_levels) ** max_shift_exp))))

            if max_shift_level <= 0:
                shifts = range(max_shift_level, 0, 1)

            else:
                shifts = range(1, max_shift_level + 1)

            for shift in shifts:
                asymm_vals_i = np.roll(asymm_vals_i, shift)

                idxs_to_shift = levels <= level

                asymm_vals_2[idxs_to_shift] = (
                    asymm_vals_2[idxs_to_shift] * pre_vals_ratio +
                    asymm_vals_i[idxs_to_shift] * (1 - pre_vals_ratio))

        asymm_vals_2 += rand_epss
        asymm_vals_2 = vals_sort[np.argsort(np.argsort(asymm_vals_2))]
        #======================================================================

        asymms_df.iloc[:, i] = asymm_vals

        # plt.plot(vals_gau, c='r', alpha=0.5, label='gau')
        # plt.plot(asymm_vals, c='b', alpha=0.5, label='asymm2')

        plt.plot(
            expon.ppf(rankdata(vals_gau) / (vals_gau.size + 1.0), scale=10),
            c='r',
            alpha=0.5,
            label='gau')

        plt.plot(
            expon.ppf(rankdata(asymm_vals) / (asymm_vals.size + 1.0), scale=10),
            c='b',
            alpha=0.5,
            label='asymm2')

        plt.plot(
            expon.ppf(rankdata(asymm_vals_2) / (asymm_vals_2.size + 1.0), scale=10),
            c='g',
            alpha=0.5,
            label='asymm2_2')

        plt.legend()

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.xlabel('Time step')
        plt.ylabel('Amplitude')

        plt.savefig(f'sims_asymm23_{i}.png', bbox_inches='tight', dpi=150)

        plt.clf()

    asymms_df.to_csv('sims_asymm23.csv', sep=';')

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
