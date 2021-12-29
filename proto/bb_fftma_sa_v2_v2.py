'''
Created on Dec 7, 2021

@author: Faizan3800X-Uni
'''
import os
import sys
import time
import timeit
import winsound
import traceback as tb
from pathlib import Path
from collections import deque
from multiprocessing import Pool

os.environ[str('MKL_NUM_THREADS')] = str(1)
os.environ[str('NUMEXPR_NUM_THREADS')] = str(1)
os.environ[str('OMP_NUM_THREADS')] = str(1)

import psutil
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt; plt.ioff()

from fcopulas import (
    get_asymms_sample,
    fill_bi_var_cop_dens,
    get_asymm_1_max,
    get_asymm_2_max,
    get_etpy_min,
    get_etpy_max,
    get_asymm_1_var,
    get_asymm_1_skew,
    get_asymm_2_var,
    get_asymm_2_skew,
    )

from aa_sampled_covariance_ftn import roll_real_2arrs

from ab_fftma_v2 import (
    get_rfft_ma_deviates_padded,
    get_lagged_corr_ftn,
    get_corr_ftn_range,
    pad_corr_ftn)
from ad_fftma_reverse_white_noise_v1 import (
    get_rfft_ma_white_noise,)
from ad_fftma_reverse_white_noise_v2 import (
    get_rfft_ma_white_noise_padded,
    pad_data)

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

    asymms_1_var = []
    asymms_1_skew = []
    asymms_2_var = []
    asymms_2_skew = []

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

        asymm_1_max = get_asymm_1_max(scorr)
        asymm_2_max = get_asymm_2_max(scorr)

        asymms_1.append(asymm_1 / asymm_1_max)
        asymms_2.append(asymm_2 / asymm_2_max)

        # Additional asymmetry properties.
        asymm_1_var = get_asymm_1_var(probs_i, rolled_probs_i, asymm_1)

        # asymm_1_skew = get_asymm_1_skew(
        #     probs_i, rolled_probs_i, asymm_1, asymm_1_var)

        asymm_2_var = get_asymm_2_var(probs_i, rolled_probs_i, asymm_2)

        # asymm_2_skew = get_asymm_2_skew(
        #     probs_i, rolled_probs_i, asymm_2, asymm_2_var)

        # asymm_1_var = 0.0
        asymm_1_skew = 0.0
        #
        # asymm_2_var = 0.0
        asymm_2_skew = 0.0

        asymms_1_var.append(asymm_1_var)
        asymms_1_skew.append(asymm_1_skew)

        asymms_2_var.append(asymm_2_var)
        asymms_2_skew.append(asymm_2_skew)

        # asymms_1_var.append(asymm_1_var / asymm_1_max)
        # asymms_1_skew.append(asymm_1_skew / asymm_1_max)
        #
        # asymms_2_var.append(asymm_2_var / asymm_2_max)
        # asymms_2_skew.append(asymm_2_skew / asymm_2_max)

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

    props = np.array([
        scorrs,
        asymms_1,
        asymms_2,
        etpys,
        pcorrs,
        asymms_1_var,
        asymms_1_skew,
        asymms_2_var,
        asymms_2_skew])

    return props


def get_fftma_sim(norms, corr_ftn, data_srtd, corr_ftn_range):

    sim = get_rfft_ma_deviates_padded(norms, corr_ftn, corr_ftn_range)
    sim = data_srtd[np.argsort(np.argsort(sim))]
    return sim


def obj_ftn(
        sim_props,
        ref_props,
        wts):

    obj_val = (wts * ((ref_props - sim_props)) ** 2).sum()

    return obj_val


def run_sa(args):

    (n_iters,
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
     sim_idx,
     noise_dist_ftn,
     corr_ftn_range) = args

    obj_val_global = None
    sim_calib = None

    acpt_rate_len = 1000
    acpts_rjts_dfrntl = deque(maxlen=acpt_rate_len)

    corr_ftn_range_ratio = 0.75

    acpt_rate = 0.5

    idxs_ctr_max = 100

    acpt_iters = 0
    not_acpt_iters = 0
    act_iters = 0
    temp = temp_init
    for i in range(n_iters):
        if not i:
            norms = noise_dist_ftn(np.random.random(size=corr_ftn.shape[0]))

            norms_iter = norms.copy()

        else:
            norms_iter = norms.copy()

            idxs_ctr = 0
            while True:
                beg_idx = np.random.randint(
                    low=int(corr_ftn_range * corr_ftn_range_ratio),
                    high=norms_iter.shape[0] -
                        int(corr_ftn_range * corr_ftn_range_ratio))

                end_idx = beg_idx + int(
                    acpt_rate * (norms_iter.shape[0] - (2 * corr_ftn_range)))

                end_idx = min(end_idx, norms_iter.shape[0])

                idxs_ctr += 1

                if idxs_ctr >= idxs_ctr_max:
                    break

                if beg_idx != end_idx:
                    break

            if idxs_ctr >= idxs_ctr_max:
                break

            old_idxs = np.arange(beg_idx, end_idx)
            shuff_idxs = old_idxs.copy()

            np.random.shuffle(shuff_idxs)

            norms_iter[old_idxs] = norms_iter[shuff_idxs]

        sim = get_fftma_sim(
            norms_iter, corr_ftn, in_data_ser_orig_srtd, corr_ftn_range)

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
            obj_val_old = obj_val_global

            norms = norms_iter.copy()
            sim_calib = sim.copy()
            norms_calib = norms_iter.copy()

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

                    if i > acpt_rate_len:
                        acpt_rate = (
                            sum(acpts_rjts_dfrntl) / float(acpt_rate_len))

                        if acpt_rate <= 1e-5:
                            break

            else:
                raise ValueError(sim_type)

            if obj_val_iter < obj_val_global:
                obj_val_global = obj_val_iter

                sim_calib = sim.copy()
                norms_calib = norms_iter.copy()

            if obj_val_iter < obj_val_old:
                obj_val_old = obj_val_iter

                norms = norms_iter.copy()

                acpt_iters += 1
                not_acpt_iters = 0

                acpts_rjts_dfrntl.append(1)

                if sim_type == 1 and (not (acpt_iters % temp_updt_iters)):
                    print('Accept iter:', acpt_iters, i, sim_idx, acpt_rate)

            else:
                rand_p = np.random.random()
                boltz_p = np.exp((obj_val_old - obj_val_iter) / temp)

                if rand_p < boltz_p:
                    obj_val_old = obj_val_iter

                    norms = norms_iter.copy()

                    acpt_iters += 1
                    not_acpt_iters = 0

                    acpts_rjts_dfrntl.append(1)

                    if sim_type == 1 and (not (acpt_iters % temp_updt_iters)):
                        print(
                            'Accept iter:', acpt_iters, i, sim_idx, acpt_rate)

                else:
                    not_acpt_iters += 1

                    acpts_rjts_dfrntl.append(0)

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
        norms_calib[corr_ftn_range:-corr_ftn_range])


def get_n_cpus():

    phy_cores = psutil.cpu_count(logical=False)
    log_cores = psutil.cpu_count()

    if phy_cores < log_cores:
        n_cpus = phy_cores

    else:
        n_cpus = log_cores - 1

    n_cpus = max(n_cpus, 1)

    return n_cpus


def main():

    '''
    Instead of swapping two values, this one shuffles values in a
    window that is proportional to the acceptance rate.
    '''

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

    n_iters = int(3e5)

    break_opt_iters = 5000

    # temp_inits = [0.01, 0.05, 0.1]  # When variance and skew of asymms is not considered.
    temp_inits = [1e0, 3e0, 5e0, 1e1, 1e2]  # When variance and skew of asymms is considered.
    temp_updt_iters = 100
    temp_red_rate = 0.985
    temp_init_iters = 200

    n_sims = 8

    n_cpus = 'auto'

    out_dir = Path(r'fftma_sa_v2_v2_03')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    if n_cpus == 'auto':
        n_cpus = get_n_cpus()

    elif isinstance(n_cpus, int):
        assert n_cpus > 0, n_cpus

    else:
        raise ValueError(f'n_cpus {n_cpus} invalid!')

    in_data_df = pd.read_csv(in_data_file, sep=sep, index_col=0)
    in_data_df.index = pd.to_datetime(in_data_df.index, format=time_fmt)

    in_data_ser = in_data_df.loc[beg_time:end_time, col]

    if (in_data_ser.shape[0] % 2):
        in_data_ser = in_data_ser.iloc[:-1]

    assert np.all(np.isfinite(in_data_ser.values))

    in_data_ser_orig = in_data_ser.copy()

    in_data_ser_orig_srtd = np.sort(in_data_ser_orig)

    corr_ftn = get_lagged_corr_ftn(in_data_ser.values)

    corr_ftn_range = get_corr_ftn_range(corr_ftn)

    corr_ftn = pad_corr_ftn(corr_ftn, corr_ftn_range)

    norms_white_noise_unpadded = get_rfft_ma_white_noise(
        pad_data(in_data_ser.values, corr_ftn_range),
        corr_ftn)

    norms_white_noise = get_rfft_ma_white_noise_padded(
        pad_data(in_data_ser.values, corr_ftn_range),
        corr_ftn,
        corr_ftn_range)

    # norms_white_noise = get_rfft_ma_white_noise(in_data_ser.values, corr_ftn)

    norms_white_noise_dist_ftn = interp1d(
        np.sort(rankdata(norms_white_noise) / (norms_white_noise.size + 1.0)),
        np.sort(norms_white_noise),
        bounds_error=False,
        fill_value=(norms_white_noise.min(), norms_white_noise.max()))

    # mean = norms_white_noise.mean()
    # std = norms_white_noise.std()

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
    # wts[0] = 50  # Because other properties depend on it.
    wts[6] = 0.01
    wts[8] = 0.01

    #==========================================================================

    mp_pool = Pool(n_cpus)

    sim_type = 0

    temp_fin = None
    for temp_init in temp_inits:

        print('Init. temp.:', temp_init)

        args = (
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
            0,
            norms_white_noise_dist_ftn,
            corr_ftn_range)

        (_,
         _,
         act_iters,
         acpt_iters,
         _,
         _,
         _,
         _) = run_sa(args)

        acpt_rate = acpt_iters / act_iters
        print('acpt_rate:', acpt_rate)

        if 0.65 <= acpt_rate <= 0.85:
            temp_fin = temp_init
            break

        if acpt_rate > 0.85:
            break

    if temp_fin is None:
        # winsound.PlaySound('SystemExclamation', winsound.SND_ALIAS)
        winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
        winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
        winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
        # winsound.PlaySound('SystemExit', winsound.SND_ALIAS)
        raise RuntimeError

    # DatetimeIndex is taken from ref.
    sims = {}

    sims['ref'] = in_data_ser_orig.copy()

    sims[f'norms_init'] = norms_white_noise

    sims[f'sim_init'] = get_fftma_sim(
        norms_white_noise_unpadded,
        corr_ftn,
        in_data_ser_orig_srtd,
        corr_ftn_range)

    print('Optimizing...')
    sim_type = 1

    args_gen = ((
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
        sim_no,
        norms_white_noise_dist_ftn,
        corr_ftn_range) for sim_no in range(n_sims))

    ress = list(mp_pool.imap_unordered(run_sa, args_gen))

    for i, res in enumerate(ress):
        (sim_calib,
         _,
         act_iters,
         acpt_iters,
         not_acpt_iters,
         obj_val_global,
         temp,
         norms_calib) = res

        acpt_rate = acpt_iters / act_iters

        print(f'Accepted {acpt_iters} out of {act_iters}.')
        print('acpt_rate:', acpt_rate)
        print('not_acpt_iters:', not_acpt_iters)
        print('obj_val_global:', obj_val_global)
        print('temp:', temp)
        # Optimization end.

        sims[f'sim_calib_{i}'] = sim_calib
        sims[f'norms_calib_{i}'] = norms_calib

    mp_pool.close()
    mp_pool.join()

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
