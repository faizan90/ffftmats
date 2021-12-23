'''
Created on Dec 7, 2021

@author: Faizan3800X-Uni
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

from aa_sampled_covariance_ftn import get_ft_corr_ftn
from ab_fftma_v1 import get_fft_ma_deviates
from ad_fftma_reverse_white_noise_v1 import get_fft_ma_white_noise

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

    sim = get_fft_ma_deviates(norms, corr_ftn)
    sim = data_srtd[np.argsort(np.argsort(sim))]
    return sim


def obj_ftn(
        sim_props,
        ref_props,
        wts):

    obj_val = (wts * (ref_props - sim_props) ** 2).sum()

    return obj_val


def run_sa(
        n_iters,
        corr_ftn,
        in_data_ser_orig_srtd,
        lag_steps,
        ecop_dens_arrs,
        etpy_min,
        etpy_max,
        ref_props,
        wts,
        break_opt_iters,
        temp_init,
        temp_updt_iters,
        temp_red_rate,
        sim_type,
        norms_init):

    obj_val_global = None
    sim_calib = None

    acpt_iters = 0
    not_acpt_iters = 0
    act_iters = 0
    temp = temp_init
    for i in range(n_iters):
        if not i:
            norms = norms_init.copy()

            norms_iter = norms.copy()

        else:
            norms_iter = norms.copy()

            idx_rnd = np.random.randint(norms_iter.shape[0] - 1)

            rnd_val = np.random.normal(scale=50)

            norms_iter[idx_rnd + 0] += rnd_val
            norms_iter[idx_rnd + 1] -= rnd_val

        sim = get_fftma_sim(norms_iter, corr_ftn, in_data_ser_orig_srtd)

        sim_props = get_props(
            sim,
            lag_steps,
            ecop_dens_arrs,
            etpy_min,
            etpy_max)

        obj_val_iter = obj_ftn(sim_props, ref_props, wts)

        if np.isclose(obj_val_iter, 0):
            print('obj_val is zero!')
            break

        act_iters += 1

        if not i:
            obj_val_global = obj_val_iter
            obj_val_old = obj_val_global

            norms = norms_iter.copy()
            sim_calib = sim.copy()

            sim_init = sim.copy()

            acpt_iters += 1
            not_acpt_iters = 0

        else:

            if sim_type == 0:
                pass

            elif sim_type == 1:
                if i and (not (i % temp_updt_iters)):
                    temp *= temp_red_rate

                    if temp <= 1e-10:
                        break

            else:
                raise ValueError(sim_type)

            if obj_val_iter < obj_val_global:
                obj_val_global = obj_val_iter

                sim_calib = sim.copy()

            if obj_val_iter < obj_val_old:
                obj_val_old = obj_val_iter

                norms = norms_iter.copy()

                acpt_iters += 1
                not_acpt_iters = 0

                if sim_type == 1:
                    print('Accept iter:', acpt_iters, i)

            else:
                rand_p = np.random.random()
                boltz_p = np.exp((obj_val_old - obj_val_iter) / temp)

                if rand_p < boltz_p:
                    obj_val_old = obj_val_iter

                    norms = norms_iter.copy()

                    acpt_iters += 1
                    not_acpt_iters = 0

                    if sim_type == 1:
                        print('Accept iter:', acpt_iters, i)

                else:
                    not_acpt_iters += 1

                    if not_acpt_iters > break_opt_iters:
                        break

    return (
        sim_calib,
        sim_init,
        act_iters,
        acpt_iters,
        not_acpt_iters,
        obj_val_global,
        temp,
        norms)


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

    n_iters = int(1e5)

    break_opt_iters = 5000

    temp_inits = [0.0001, 0.001, 0.01, 0.05, 0.1]
    temp_updt_iters = 100
    temp_red_rate = 0.95
    temp_init_iters = 1000

    out_dir = Path(r'fftma_v1_sa_v2_sim_07_daily')
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

    corr_ftn = get_ft_corr_ftn(in_data_ser.values)

    if in_data_ser.shape[0] > corr_ftn.shape[0]:
        in_data_ser = in_data_ser.iloc[:corr_ftn.shape[0]]
        in_data_ser_orig = in_data_ser_orig.iloc[:corr_ftn.shape[0]]

    elif in_data_ser.shape[0] < corr_ftn.shape[0]:
        corr_ftn = corr_ftn[:in_data_ser.shape[0]]

    assert corr_ftn.size == in_data_ser.size

    norms_white_noise = get_fft_ma_white_noise(in_data_ser.values, corr_ftn)

    norms_init = norms_white_noise.copy()

    # Offset less than zero and greater than zero differently.
    idxs_le_zero = norms_init < 0
    # norms_le_zero_mean = norms_init[idxs_le_zero].mean()
    # norms_le_zero_stdv = norms_init[idxs_le_zero].std()

    idxs_ge_zero = norms_init >= 0
    # norms_ge_zero_mean = norms_init[idxs_ge_zero].mean()
    # norms_ge_zero_stdv = norms_init[idxs_ge_zero].std()

    # norms_init[idxs_le_zero] += np.random.normal(
    #     loc=0,
    #     scale=norms_le_zero_stdv,
    #     size=idxs_le_zero.sum())
    #
    # norms_init[idxs_ge_zero] += np.random.normal(
    #     loc=0,
    #     scale=norms_ge_zero_stdv,
    #     size=idxs_ge_zero.sum())

    assert idxs_le_zero.size == idxs_ge_zero.size, (
        idxs_le_zero.size, idxs_ge_zero.size)

    le_vals = norms_init[idxs_le_zero]
    ge_vals = norms_init[idxs_ge_zero]

    rand_idxs = np.arange(idxs_le_zero.sum())

    np.random.shuffle(rand_idxs)

    norms_init[idxs_le_zero] = le_vals[rand_idxs]
    norms_init[idxs_ge_zero] = ge_vals[rand_idxs]

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

    sim_type = 0

    temp_fin = None
    for temp_init in temp_inits:

        print('Init. temp.:', temp_init)

        (_,
         _,
         act_iters,
         acpt_iters,
         _,
         _,
         _,
         _) = run_sa(
            temp_init_iters,
            corr_ftn,
            in_data_ser_orig_srtd,
            lag_steps,
            ecop_dens_arrs,
            etpy_min,
            etpy_max,
            ref_props,
            wts,
            break_opt_iters,
            temp_init,
            temp_updt_iters,
            temp_red_rate,
            sim_type,
            norms_init)

        acpt_rate = acpt_iters / act_iters
        print('acpt_rate:', acpt_rate)

        if 0.6 < acpt_rate < 0.85:
            temp_fin = temp_init
            break

        if acpt_rate > 0.85:
            break

    if temp_fin is None:
        raise RuntimeError

    print('Optimizing...')
    sim_type = 1

    (sim_calib,
     sim_init,
     act_iters,
     acpt_iters,
     not_acpt_iters,
     obj_val_global,
     temp,
     norms) = run_sa(
        n_iters,
        corr_ftn,
        in_data_ser_orig_srtd,
        lag_steps,
        ecop_dens_arrs,
        etpy_min,
        etpy_max,
        ref_props,
        wts,
        break_opt_iters,
        temp_init,
        temp_updt_iters,
        temp_red_rate,
        sim_type,
        norms_init)

    acpt_rate = acpt_iters / act_iters

    print(f'Accepted {acpt_iters} out of {act_iters}.')
    print('acpt_rate:', acpt_rate)
    print('not_acpt_iters:', not_acpt_iters)
    print('obj_val_global:', obj_val_global)
    print('temp:', temp)
    # Optimization end.

    # DatetimeIndex is taken from ref.
    sims = {'ref': in_data_ser_orig.copy()}
    sims[f'sim_calib'] = sim_calib
    sims[f'sim_init'] = sim_init
    sims[f'norms'] = norms
    sims[f'norms_white_noise'] = norms_white_noise

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
