'''
@author: Faizan-Uni-Stuttgart

Dec 7, 2021

11:14:49 AM

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

from fcopulas import (
    get_asymms_sample,
    fill_bi_var_cop_dens,
    get_asymm_1_max,
    get_asymm_2_max,
    get_etpy_min,
    get_etpy_max,)

from aa_sampled_covariance_ftn import roll_real_2arrs

from aa_sampled_covariance_ftn import get_rft_corr_ftn
from ab_fftma_v1 import get_rfft_ma_deviates, sph_vg
from ad_fftma_reverse_white_noise import get_rfft_ma_white_noise

DEBUG_FLAG = False


def get_props(
        data,
        lag_steps,
        ecop_dens_arrs,
        etpy_min,
        etpy_max):

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

        # SCorr.
        scorr = np.corrcoef(probs_i, rolled_probs_i)[0, 1]
        scorrs.append(scorr)

        # Asymms.
        asymm_1, asymm_2 = get_asymms_sample(probs_i, rolled_probs_i)

        asymm_1 /= get_asymm_1_max(scorr)

        asymm_2 /= get_asymm_2_max(scorr)

        asymms_1.append(asymm_1)
        asymms_2.append(asymm_2)

        # ECop etpy.
        fill_bi_var_cop_dens(probs_i, rolled_probs_i, ecop_dens_arrs)

        non_zero_idxs = ecop_dens_arrs > 0

        dens = ecop_dens_arrs[non_zero_idxs]

        etpy_arr = -(dens * np.log(dens))

        etpy = etpy_arr.sum()

        etpy = (etpy - etpy_min) / (etpy_max - etpy_min)

        etpys.append(etpy)

        # PCorr.
        pcorr = np.corrcoef(data_i, rolled_data_i)[0, 1]
        pcorrs.append(pcorr)

    props = np.array([scorrs, asymms_1, asymms_2, etpys, pcorrs])
    return props


def get_fftma_sim(norms, corr_ftn, data_srtd):

    sim = get_rfft_ma_deviates(norms, corr_ftn)
    sim = data_srtd[np.argsort(np.argsort(sim))]
    return sim


def obj_ftn(
        sim_props,
        ref_props,
        wts):

    obj_val = (wts * (ref_props - sim_props) ** 2).sum()

    return obj_val


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
    # end_time = '2009-12-31-23'
    # time_fmt = '%Y-%m-%d-%H'

    sep = ';'

    col = '420'

    lag_steps = np.arange(1, 40, dtype=np.int64)
    ecop_bins = 20

    n_iters = int(1e3)

    break_opt_iters = 5000

    out_dir = Path(r'fftma_v1_real_calibrate_sim_01_daily')
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

    if True:
        corr_ftn = get_rft_corr_ftn(in_data_ser.values)

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

    norms_white_noise = get_rfft_ma_white_noise(in_data_ser.values, corr_ftn)

    mean = norms_white_noise.mean()
    std = norms_white_noise.std()

    etpy_min = get_etpy_min(ecop_bins)
    etpy_max = get_etpy_max(ecop_bins)

    ecop_dens_arrs = np.full((ecop_bins, ecop_bins), np.nan, dtype=np.float64)

    # Optimization start.
    ref_props = get_props(
        in_data_ser_orig.values,
        lag_steps,
        ecop_dens_arrs,
        etpy_min,
        etpy_max)

    wts = np.ones(ref_props.shape[0]).reshape(-1, 1)
    # wts[:] = 0
    # wts[2] = 1

    obj_val_global = None
    sim_calib = None

    acpt_iters = 0
    not_acpt_iters = 0
    act_iters = 0
    for i in range(n_iters):
        if not i:
            norms = np.random.normal(
                loc=mean, scale=std, size=corr_ftn.shape[0])

            norms_iter = norms.copy()

        else:
            norms_iter = norms.copy()

            idx_rnd = np.random.randint(norms_iter.shape[0])

            norms_iter[idx_rnd] = np.random.normal(loc=mean, scale=std)

            if i > 5000:
                sclr = (acpt_iters / i) ** 6

            else:
                sclr = 1.0

            norms_iter[idx_rnd] *= sclr

        sim = get_fftma_sim(norms_iter, corr_ftn, in_data_ser_orig_srtd)

        sim_props = get_props(
            sim,
            lag_steps,
            ecop_dens_arrs,
            etpy_min,
            etpy_max)

        obj_val_iter = obj_ftn(sim_props, ref_props, wts)

        act_iters += 1

        if not i:
            obj_val_global = obj_val_iter

            norms = norms_iter.copy()
            sim_calib = sim.copy()

            sim_init = sim.copy()

            acpt_iters += 1
            not_acpt_iters = 0

        else:
            if obj_val_iter < obj_val_global:
                obj_val_global = obj_val_iter

                norms = norms_iter.copy()

                sim_calib = sim.copy()

                acpt_iters += 1
                not_acpt_iters = 0

                print('Accept iter:', acpt_iters, i)

            else:
                not_acpt_iters += 1

                if not_acpt_iters > break_opt_iters:
                    break
                # print('Reject iter:', i)

    print(f'Accepted {acpt_iters} out of {act_iters}.')
    # Optimization end.

    # DatetimeIndex is taken from ref.
    sims = {'ref': in_data_ser_orig.copy()}
    sims[f'sim_calib'] = sim_calib
    sims[f'sim_init'] = sim_init

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
