'''
@author: Faizan-Uni-Stuttgart

Dec 6, 2021

2:35:27 PM

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

from fcopulas import (
    get_asymms_sample,
    fill_bi_var_cop_dens,
    get_asymm_1_max,
    get_asymm_2_max,
    get_etpy_min,
    get_etpy_max,)

from aa_sampled_covariance_ftn import roll_real_2arrs

DEBUG_FLAG = True


def set_mpl_prms(prms_dict):

    plt.rcParams.update(prms_dict)

    return


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\fftma')

    main_dir /= r'fftma_sa_v2_v2_03'

    os.chdir(main_dir)

    in_file = Path(r'sims.csv')

    lag_steps = np.arange(1, 100, dtype=np.int64)

    ecop_bins = 20

    prms_dict = {
        'figure.figsize': (15, 10),
        'figure.dpi': 150,
        'font.size': 16,
        }

    sep = ';'

    if True:
    # if False:
        patt_ref = 'ref'
        # patt_sim = 'sim_calib_*'
        patt_sim = 'sim*'

        out_fig_name_pecop = 'ecop_props.png'
        out_fig_name_pwr = 'cumm_pwr.png'

    else:
        patt_ref = 'norms_init'
        patt_sim = 'norms_calib_*'

        out_fig_name_pecop = 'ecop_props_norms.png'
        out_fig_name_pwr = 'cumm_pwr_norms.png'
    #==========================================================================

    set_mpl_prms(prms_dict)

    in_df = pd.read_csv(in_file, sep=sep, index_col=0)

    if in_df.shape[0] % 2:
        in_df = in_df.iloc[:-1,:]

    etpy_min = get_etpy_min(ecop_bins)
    etpy_max = get_etpy_max(ecop_bins)

    ecop_dens_arrs = np.full((ecop_bins, ecop_bins), np.nan, dtype=np.float64)

    axes = plt.subplots(2, 3, squeeze=False)[1]

    clrs = ['r', 'k']

    leg_flag = True
    for i in range(in_df.shape[1]):
        data = in_df.iloc[:, i].values.copy()

        if (fnmatch(in_df.columns[i], patt_ref) or
            fnmatch(in_df.columns[i], patt_sim)):

            pass

        else:
            continue

        if fnmatch(in_df.columns[i], patt_ref):
            clr = clrs[0]

            lab = 'ref'

            zorder = 2

            plt_alpha = 0.6
            lw = 3.0

        else:
            clr = clrs[1]

            if leg_flag and fnmatch(in_df.columns[i], patt_sim):
                leg_flag = False
                lab = 'sim'

            else:
                lab = None

            plt_alpha = 0.35
            lw = 2.0

            zorder = 1

        scorrs = []
        asymms_1 = []
        asymms_2 = []
        etpys = []
        pcorrs = []
        for lag_step in lag_steps:
            probs_i, rolled_probs_i = roll_real_2arrs(
                data, data, lag_step, True)

            data_i, rolled_data_i = roll_real_2arrs(
                data, data, lag_step, False)

            # scorr.
            scorr = np.corrcoef(probs_i, rolled_probs_i)[0, 1]
            scorrs.append(scorr)

            # asymms.
            asymm_1, asymm_2 = get_asymms_sample(probs_i, rolled_probs_i)

            asymm_1 /= get_asymm_1_max(scorr)

            asymm_2 /= get_asymm_2_max(scorr)

            asymms_1.append(asymm_1)
            asymms_2.append(asymm_2)

            # ecop etpy.
            fill_bi_var_cop_dens(probs_i, rolled_probs_i, ecop_dens_arrs)

            non_zero_idxs = ecop_dens_arrs > 0

            dens = ecop_dens_arrs[non_zero_idxs]

            etpy_arr = -(dens * np.log(dens))

            etpy = etpy_arr.sum()

            etpy = (etpy - etpy_min) / (etpy_max - etpy_min)

            etpys.append(etpy)

            # pcorr.
            pcorr = np.corrcoef(data_i, rolled_data_i)[0, 1]
            pcorrs.append(pcorr)

        # plot
        axes[0, 0].plot(
            lag_steps,
            scorrs,
            alpha=plt_alpha,
            color=clr,
            label=lab,
            lw=lw,
            zorder=zorder)

        axes[1, 0].plot(
            lag_steps,
            asymms_1,
            alpha=plt_alpha,
            color=clr,
            label=lab,
            lw=lw,
            zorder=zorder)

        axes[1, 1].plot(
            lag_steps,
            asymms_2,
            alpha=plt_alpha,
            color=clr,
            label=lab,
            lw=lw,
            zorder=zorder)

        axes[0, 1].plot(
            lag_steps,
            etpys,
            alpha=plt_alpha,
            color=clr,
            label=lab,
            lw=lw,
            zorder=zorder)

        axes[0, 2].plot(
            lag_steps,
            pcorrs,
            alpha=plt_alpha,
            color=clr,
            label=lab,
            lw=lw,
            zorder=zorder)

    axes[0, 0].grid()
    axes[1, 0].grid()
    axes[1, 1].grid()
    axes[0, 1].grid()
    axes[0, 2].grid()
    axes[1, 2].set_axis_off()

    axes[0, 0].set_axisbelow(True)
    axes[1, 0].set_axisbelow(True)
    axes[1, 1].set_axisbelow(True)
    axes[0, 1].set_axisbelow(True)
    axes[0, 2].set_axisbelow(True)
    # axes[1, 2].set_axisbelow(True)

    axes[0, 0].legend()
    axes[1, 0].legend()
    axes[1, 1].legend()
    axes[0, 1].legend()
    axes[0, 2].legend()
#     axes[1, 2].legend()

    axes[0, 0].set_ylabel('Spearman correlation')

    axes[1, 0].set_xlabel('Lag steps')
    axes[1, 0].set_ylabel('Asymmetry (Type - 1)')

    axes[1, 1].set_xlabel('Lag steps')
    axes[1, 1].set_ylabel('Asymmetry (Type - 2)')

    axes[0, 1].set_ylabel('Entropy')

    axes[0, 2].set_xlabel('Lag steps')
    axes[0, 2].set_ylabel('Pearson correlation')

    plt.tight_layout()

    plt.savefig(out_fig_name_pecop, bbox_inches='tight')
    plt.close()
    #==========================================================================

    ref_pwr = None
    leg_flag = True
    for i in range(in_df.shape[1]):
        data = in_df.iloc[:, i].values.copy()

        # if i not in (0, 3, 5):
        #     continue

        if (fnmatch(in_df.columns[i], patt_ref) or
            fnmatch(in_df.columns[i], patt_sim)):

            pass

        else:
            continue

        if fnmatch(in_df.columns[i], patt_ref):
            clr = clrs[0]

            lab = 'ref'

            zorder = 2

            plt_alpha = 0.6
            lw = 3.0

        else:
            clr = clrs[1]

            if leg_flag and fnmatch(in_df.columns[i], patt_sim):
                leg_flag = False
                lab = 'sim'

            else:
                lab = None

            plt_alpha = 0.35
            lw = 2.0

            zorder = 1

        ft = np.fft.rfft(data)[1:]

        pwr = np.abs(ft) ** 2

        pwr = pwr.cumsum()

        if fnmatch(in_df.columns[i], patt_ref):
            ref_pwr = pwr[-1]

        pwr /= ref_pwr
        # pwr /= pwr[-1]

        periods = (pwr.size * 2) / (
            np.arange(1, pwr.size + 1))

        assert periods.size == pwr.shape[0]

        plt.semilogx(
            periods,
            pwr,
            alpha=plt_alpha,
            color=clr,
            label=lab,
            lw=lw,
            zorder=zorder)

    plt.legend()

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.xlabel('Period')
    plt.ylabel('Cummulative power')

    plt.xlim(plt.xlim()[::-1])

    plt.savefig(out_fig_name_pwr, bbox_inches='tight')
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
