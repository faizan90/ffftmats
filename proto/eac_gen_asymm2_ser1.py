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
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\fftma_asymm')

    main_dir /= r'test_theo_vgs_07'

    os.chdir(main_dir)

    plot_flag = True
    plot_flag = False

    sims_file = Path(r'sims.csv')
    #==========================================================================

    sims_df = pd.read_csv(sims_file, sep=';', index_col=0)

    sims_df_cols = [fnmatch(col, 'sim_*') for col in sims_df.columns]

    sims_df = sims_df.loc[:, sims_df_cols]

    if plot_flag:
        plt.figure()

    sims_min = sims_df.values.min() - 0.01
    sims_max = sims_df.values.max() + 0.01

    diag_vals = np.linspace(sims_min, sims_max, sims_df.shape[1])

    step_vals_pre = None
    asymm_ser = []
    for i in range(sims_df.shape[0]):

        step_vals = sims_df.iloc[i,:].values.copy()

        step_vals.sort()

        # x_min_idx = ((diag_vals - step_vals) ** 2).argmin()
        if i == 0:
            x_min_idx = 0

        else:
            if step_vals_pre > step_vals[x_min_idx]:
                x_min_idx = ((diag_vals - step_vals) ** 2).argmin()

            else:
                x_min_idx = 0

        step_vals_pre = step_vals[x_min_idx]

        asymm_ser.append(step_vals[x_min_idx])

        if plot_flag:
            plt.plot(diag_vals, diag_vals, alpha=0.75, c='k')
            plt.plot(diag_vals, step_vals, alpha=0.75, c='r')

            plt.scatter([diag_vals[x_min_idx]], [step_vals[x_min_idx]], c='g')

            plt.savefig(f'z{i:03d}.png', bbox_inches='tight')

            plt.clf()

    if plot_flag:
        plt.close()

    asymm_ser = np.array(asymm_ser)

    plt.figure(figsize=(10, 5))

    leg_flag = True
    for i in range(sims_df.shape[1]):
        surrogate = sims_df.iloc[:, i]

        if leg_flag:
            label = 'sim'
            leg_flag = False

        else:
            label = None

        plt.plot(surrogate, c='k', alpha=0.25, label=label)

    plt.plot(asymm_ser, c='r', alpha=0.5, label='asymm2')

    plt.legend()

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.xlabel('Time step')
    plt.ylabel('Amplitude')

    plt.savefig('sims_asymm2.png', bbox_inches='tight', dpi=150)
    plt.close()

    sims_df['asymm2'] = asymm_ser

    sims_df.to_csv(r'sims_asymm2.csv', sep=';')
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
