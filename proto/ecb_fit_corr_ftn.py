'''
@author: Faizan-Uni-Stuttgart

Jul 29, 2022

11:53:05 AM

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
from scipy.optimize import differential_evolution

from aa_sampled_covariance_ftn import roll_real_2arrs
import ecc_covariancefunction as covfun

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\fftma_asymm')

    os.chdir(main_dir)

    in_data_file = Path(r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    lags = np.arange(1, 31, dtype=np.int64)

    out_dir = Path('fit_corr_ftns')
    #==========================================================================

    fit_auto_corr_ftns(in_data_file, out_dir, lags)
    return


def fit_auto_corr_ftns(in_data_file, out_dir, lags):

    def get_cov_mod_string(x):

        (sill_n,
         sill_a,
         sill_b,
         sill_c,
         # sill_d,
         range_a,
         range_b,
         range_c,
         # range_d
        ) = x

        covmod = (
            f'{sill_n:0.6f} Nug({max(range_a, range_b, range_c):0.3f}) + '
            f'{sill_a:0.6f} Exp({range_a:0.3f}) + '
            f'{sill_b:0.6f} Sph({range_b:0.3f}) + '
            f'{sill_c:0.6f} Sph({range_c:0.3f})'
            # f'{sill_d:0.6f} Hol({range_d:0.3f})'
            )

        return covmod

    def obj_ftn(x, emp_corrs, lags, obj_wts):

        covmod = get_cov_mod_string(x)

        theo_corrs = covfun.Covariogram(lags, covmod)

        return ((obj_wts * (emp_corrs - theo_corrs)) ** 2).sum()

    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    df_data = pd.read_csv(in_data_file, sep=';', index_col=0)

    stn_lag_corrs = {}

    fig = plt.figure()
    for stn in df_data.columns:
        print(stn)
        data_stn = df_data[stn].values.copy('c')

        lag_corrs = []
        for lag in lags:
            data_stn_a, data_stn_b = roll_real_2arrs(
                data_stn, data_stn, lag)

            not_nan_idxs = np.isfinite(data_stn_a) & np.isfinite(data_stn_b)

            assert not_nan_idxs.sum() > 1, (stn, not_nan_idxs.sum())

            data_stn_a, data_stn_b = (
                data_stn_a[not_nan_idxs], data_stn_b[not_nan_idxs])

            lag_corr = np.corrcoef(data_stn_a, data_stn_b)[0, 1]

            lag_corrs.append(lag_corr)

        lag_corrs = np.array(lag_corrs)

        stn_lag_corrs[stn] = lag_corrs

        plt.plot(lags, lag_corrs, alpha=0.5, c='k')

        # break

    plt.xlabel('Lag step')

    plt.ylabel('Pearson correlation')

    plt.grid()

    plt.gca().set_axisbelow(True)

    # plt.show()

    plt.savefig(
        str(out_dir / 'autopcorrs_empirical.png'),
        bbox_inches='tight',
        dpi=150)

    plt.close(fig)
    #==========================================================================

    print('Fitting...')

    bds = np.array([
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        # [0, 2],
        [1e-4, 1e3],
        [1e-4, 1e3],
        [1e-4, 1e3],
        # [1e-4, 5e3],
        ])

    obj_wts = np.arange(lags.size, 0, -1) ** 1

    out_prms_arr = np.full((len(stn_lag_corrs), bds.shape[0]), np.nan)

    for i, stn in enumerate(stn_lag_corrs.keys()):
        print(stn)

        opt_res = differential_evolution(
            obj_ftn, bds, popsize=100, args=(stn_lag_corrs[stn], lags, obj_wts))

        opt_prms = opt_res.x
        sq_diff = opt_res.fun

        print(np.round(opt_prms, 3))
        print(sq_diff)

        out_prms_arr[i,:] = opt_prms

        covmod = get_cov_mod_string(opt_prms)

        theo_corrs = covfun.Covariogram(lags, covmod)

        fig = plt.figure()

        plt.plot(lags, stn_lag_corrs[stn], alpha=0.5, c='k')
        plt.plot(lags, theo_corrs, alpha=0.5, c='b')

        plt.xlabel('Lag step')

        plt.ylabel('Pearson correlation')

        plt.grid()

        plt.gca().set_axisbelow(True)

        plt.title(covmod)

        # plt.show()

        plt.savefig(
            str(out_dir / f'autopcorrs_fitted_{stn}.png'),
            bbox_inches='tight',
            dpi=150)

        plt.close(fig)

    np.savetxt(
        str(out_dir / 'auto_pcorrs.dat'), out_prms_arr, delimiter=',')

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
