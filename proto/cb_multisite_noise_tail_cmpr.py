'''
@author: Faizan-Uni-Stuttgart

Dec 20, 2021

4:32:01 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\fftma')

    main_dir /= r'test_fftma_ms_04__all/phsann_noise'

    os.chdir(main_dir)

    tail_dist_thresh = 0.05
    #==========================================================================

    # norms_df = pd.read_csv('norms.csv', sep=';', index_col=0)
    norms_df = pd.read_csv('sim_data_5.csv', sep=';')

    probs_df = norms_df.rank(axis=0) / (norms_df.shape[0] + 1.0)

    print('norms pearson - overall:')
    print(norms_df.corr('pearson'))

    print('')
    print('norms spearman - overall:')
    print(norms_df.corr('spearman'))

    probs_ltail_df = probs_df.loc[
        (probs_df < tail_dist_thresh).any(axis=1).values]

    probs_utail_df = probs_df.loc[
        (probs_df > (1 - tail_dist_thresh)).any(axis=1).values]

    norms_ltail_df = norms_df.loc[probs_ltail_df.index]
    norms_utail_df = norms_df.loc[probs_utail_df.index]

    print('')
    print('norms pearson - lower tail:')
    print(norms_ltail_df.corr('pearson'))

    print('')
    print('norms pearson - upper tail:')
    print(norms_utail_df.corr('pearson'))

    print('')
    print('norms spearman - lower tail:')
    print(norms_ltail_df.corr('spearman'))

    print('')
    print('norms spearman - upper tail:')
    print(norms_utail_df.corr('spearman'))

    print('')
    print('probs pearson - lower tail:')
    print(probs_ltail_df.corr('pearson'))

    print('')
    print('probs pearson - upper tail:')
    print(probs_utail_df.corr('pearson'))

    # ax_ltail, ax_utail = plt.subplots(
    #     1, 2, sharex=True, sharey=True)[1].ravel()

    ax_full, ax_ltail, ax_utail = plt.subplots(1, 3)[1].ravel()

    ax_full.scatter(
        probs_df.iloc[:, 0],
        probs_df.iloc[:, 1],
        alpha=0.1,
        c='r')

    ax_full.grid()
    ax_full.set_axisbelow(True)

    ax_full.set_aspect('equal')

    ax_full.set_xlabel(probs_df.columns[0])
    ax_full.set_ylabel(probs_df.columns[1])

    ax_full.set_title(f'Full')

    # Lower tail.
    ax_ltail.scatter(
        probs_ltail_df.iloc[:, 0].rank() / (probs_ltail_df.shape[0] + 1.0),
        probs_ltail_df.iloc[:, 1].rank() / (probs_ltail_df.shape[0] + 1.0),
        alpha=0.5,
        c='r')

    # ax_ltail.scatter(
    #     probs_ltail_df.iloc[:, 0],
    #     probs_ltail_df.iloc[:, 1],
    #     alpha=0.5,
    #     c='r')

    ax_ltail.grid()
    ax_ltail.set_axisbelow(True)

    ax_ltail.set_aspect('equal')

    ax_ltail.set_xlabel(probs_ltail_df.columns[0])
    # ax_ltail.set_ylabel(probs_ltail_df.columns[1])

    ax_ltail.set_title(f'Lower tail (p < {tail_dist_thresh})')

    # Upper tail.
    ax_utail.scatter(
        probs_utail_df.iloc[:, 0].rank() / (probs_utail_df.shape[0] + 1.0),
        probs_utail_df.iloc[:, 1].rank() / (probs_utail_df.shape[0] + 1.0),
        alpha=0.5,
        c='r')

    # ax_utail.scatter(
    #     probs_utail_df.iloc[:, 0],
    #     probs_utail_df.iloc[:, 1],
    #     alpha=0.5,
    #     c='r')

    ax_utail.grid()
    ax_utail.set_axisbelow(True)

    ax_utail.set_aspect('equal')

    ax_utail.set_xlabel(probs_utail_df.columns[0])
    # ax_utail.set_ylabel(probs_utail_df.columns[1])

    ax_utail.set_title(f'Upper tail (p > {1 - tail_dist_thresh})')

    plt.show()

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
