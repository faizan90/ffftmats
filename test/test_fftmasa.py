'''
Created on Jan 3, 2022

@author: Faizan3800X-Uni
'''

import os
import sys

if ('P:\\Synchronize\\Python3Codes' not in sys.path):
    sys.path.append('P:\\Synchronize\\Python3Codes')

import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd

from ffftmats import FFTMASAMain, FFTMASAPlot

# raise Exception

DEBUG_FLAG = False


def get_unit_peak(n_vals, beg_index, peak_index, end_index):

    rising_exp = 1.5
    recession_exp = 9

    assert beg_index <= peak_index <= end_index
    assert n_vals > end_index
    assert beg_index >= 0

    unit_peak_arr = np.zeros(n_vals)

    rising_limb = np.linspace(
        0.0, 1.0, peak_index - beg_index, endpoint=False) ** rising_exp

    recessing_limb = np.linspace(
        1.0, 0.0, end_index - peak_index) ** recession_exp

    unit_peak_arr[beg_index:peak_index] = rising_limb
    unit_peak_arr[peak_index:end_index] = recessing_limb

    return unit_peak_arr


def main():

    # TODO: Formally implement the constants used in the _set_cdf_wts method.
    # TODO: Move computations of running variables under the temp update
    # section.
    # TODO: Decide which ftn connected to dist obj ftns computation.
    # TODO: Show elapsed time in days, hours, minutes and second.
    # Write a ftn in misc for this.
    # TODO: Communicate with running threads through a text file.
    # TODO: Ecop containment figures for single-site.
    # TODO: Find out lags at which asymms are insignificant and leave them out.
    # TODO: Label wts for pairs in multisite obj ftns.
    # TODO: FT of the cross-power spectrum.
    # TODO: Scaling exp for auto obj wts.

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\fftmasa')
    os.chdir(main_dir)

    test_unit_peak_flag = False

#==============================================================================
#    Daily HBV sim
#==============================================================================
#     in_file_path = Path(r'hbv_sim__1963_2015.csv')
#
#     sim_label = 'test_label_wts_09'  # next:
#
#     labels = 'prec;q_obs'.split(';')  # pet;temp;
#
#     time_fmt = '%Y-%m-%d'
#
#     beg_time = '1996-01-01'
#     end_time = '2000-12-31'

#==============================================================================
#    Daily ppt.
#==============================================================================
    # in_file_path = Path(r'precipitation_bw_1961_2015.csv')
    #
    # sim_label = ''  # next:
    #
    # labels = ['P1162', 'P1197', 'P4259'], 'P5229'
    # # labels = ['P1162']
    #
    # time_fmt = '%Y-%m-%d'
    #
    # beg_time = '1961-01-01'
    # end_time = '2015-12-31'

#==============================================================================
#    Hourly ppt.
#==============================================================================
    # in_file_path = Path(r'neckar_1hr_ppt_data_20km_buff_Y2004_2020.pkl')
    #
    # sim_label = 'phd_sims__ppt_pair__asymms_dist_03'  # next:
    #
    # labels = ['P1176', 'P1290']  # , 'P13674', 'P13698', 'P1937', 'P2159', 'P2292', ]
    #
    # time_fmt = '%Y-%m-%d'
    #
    # beg_time = '2009-01-01'
    # end_time = '2009-12-31'

#==============================================================================
#    Daily discharge.
#==============================================================================
    # in_file_path = Path(
    #     r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    in_file_path = Path(r'neckar_q_data_combined_20180713.csv')

    sim_label = 'test_noise_09'  # next:

    labels = ['420' , '427']  # , '3470']  # , '3421' , '3465'

    time_fmt = '%Y-%m-%d'

    beg_time = '1961-01-01'
    end_time = '1963-12-31'

#==============================================================================

#==============================================================================
#    Hourly
#==============================================================================
    # in_file_path = Path(r'hourly_bw_discharge__2008__2019.csv')
    #
    # sim_label = 'cmpr_with_fftm1_06_hourly'
    #
    # labels = ['420']  # '3470', '3465']
    #
    # time_fmt = '%Y-%m-%d-%H'
    #
    # beg_time = '2009-01-01-00'
    # end_time = '2009-12-31-23'

#==============================================================================

#==============================================================================
#    FFTMA - Noise
#==============================================================================
    # in_file_path = Path(
    #     r'neckar_norm_cop_infill_discharge_1961_2015_20190118__fftma_noise.csv')
    #
    # sim_label = 'test_gnrctsgenr_05'
    #
    # labels = ['420', '3421']  # '3470', '3465']
    #
    # time_fmt = '%Y-%m-%d'
    #
    # beg_time = '2000-01-01'
    # end_time = '2001-12-31'

#==============================================================================

    sep = ';'

    n_vals = 200

    beg_idx = 0
    cen_idx = 50
    end_idx = 199

    verbose = True
#     verbose = False

    h5_name = 'fftmasa.h5'

    gen_rltzns_flag = True
    # gen_rltzns_flag = False

    plt_flag = True
    # plt_flag = False

    long_test_flag = True
    # long_test_flag = False

    auto_init_temperature_flag = True
    # auto_init_temperature_flag = False

    scorr_flag = True
    asymm_type_1_flag = True
    asymm_type_2_flag = True
    ecop_dens_flag = True
    ecop_etpy_flag = True
    nth_order_diffs_flag = True
    cos_sin_dist_flag = True
    pcorr_flag = True
    asymm_type_1_ms_flag = True
    asymm_type_2_ms_flag = True
    ecop_dens_ms_flag = True
    match_data_ft_flag = True
    match_probs_ft_flag = True
    asymm_type_1_ft_flag = True
    asymm_type_2_ft_flag = True
    nth_order_ft_flag = True
    asymm_type_1_ms_ft_flag = True
    asymm_type_2_ms_ft_flag = True
    etpy_ft_flag = True
    etpy_ms_ft_flag = True
    scorr_ms_flag = True
    etpy_ms_flag = True
    match_data_ms_ft_flag = True
    match_probs_ms_ft_flag = True
    match_data_ms_pair_ft_flag = True
    match_probs_ms_pair_ft_flag = True

    scorr_flag = False
    asymm_type_1_flag = False
    asymm_type_2_flag = False
    ecop_dens_flag = False
    ecop_etpy_flag = False
    nth_order_diffs_flag = False
    cos_sin_dist_flag = False
    pcorr_flag = False
    asymm_type_1_ms_flag = False
    asymm_type_2_ms_flag = False
    ecop_dens_ms_flag = False
    # match_data_ft_flag = False
    # match_probs_ft_flag = False
    asymm_type_1_ft_flag = False
    asymm_type_2_ft_flag = False
    nth_order_ft_flag = False
    asymm_type_1_ms_ft_flag = False
    asymm_type_2_ms_ft_flag = False
    etpy_ft_flag = False
    etpy_ms_ft_flag = False
    scorr_ms_flag = False
    etpy_ms_flag = False
    match_data_ms_ft_flag = False
    match_probs_ms_ft_flag = False
    match_data_ms_pair_ft_flag = False
    match_probs_ms_pair_ft_flag = False

    n_reals = 4  # A multiple of n_cpus.
    outputs_dir = main_dir / sim_label
    n_cpus = 'auto'

    # transform_type = 'probs'
    # transform_type = 'data'
    transform_type = 'norm'

    lag_steps = np.arange(1, 9)
    # lag_steps = np.concatenate((np.arange(1, 10), [16, 20, 25, 30]))
    ecop_bins = 20
    nth_ords = np.arange(1, 3)
#     nth_ords = np.array([1, 5])
    lag_steps_vld = np.arange(1, 16)
    nth_ords_vld = np.arange(1, 3)

    use_dists_in_obj_flag = True
    use_dists_in_obj_flag = False

    use_dens_ftn_flag = True
    use_dens_ftn_flag = False

    ratio_per_dens_bin = 0.01

    n_beg_idxs, n_end_idxs = 1, 500
    idxs_sample_type = 3
    idxs_number_reduction_rate = 0.999
    mult_idxs_flag = True
    mult_idxs_flag = False

    wts_flag = True
    # wts_flag = False

#     weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.005], dtype=np.float64)
#     auto_wts_set_flag = False
#     wts_n_iters = None

    weights = None
    auto_wts_set_flag = True
    wts_n_iters = 200

    lags_nths_wts_flag = True
    lags_nths_wts_flag = False
    lags_nths_exp = 2.5
    lags_nths_n_iters = 500
    lags_nths_cumm_wts_contrib = 0.9999
    lags_nths_n_thresh = max(lag_steps.size, nth_ords.size)

    label_wts_flag = True
    label_wts_flag = False
    label_exp = 2.0
    label_n_iters = 500

    cdf_penalt_flag = True
    cdf_penalt_flag = False
    n_vals_thresh = 1
    n_vals_penlt = 6

    prt_cdf_calib_flag = True
    prt_cdf_calib_flag = False
    lower_threshold = 0.05
    upper_threshold = 0.95
    inside_flag = False

    stop_criteria_labels = (
        'Iteration completion',
        'Iterations without acceptance',
        'Running objective function tolerance',
        'Annealing temperature',
        'Running acceptance rate',
        'Iterations without updating the global minimum')

    plt_osv_flag = True
    plt_ss_flag = True
    plt_ms_flag = True
    plt_qq_flag = True

    # plt_osv_flag = False
    # plt_ss_flag = False
    # plt_ms_flag = False
    plt_qq_flag = False

    max_sims_to_plot = 2

    if long_test_flag:
        initial_annealing_temperature = 0.0001
        temperature_reduction_ratio = 0.993
        update_at_every_iteration_no = 100
        maximum_iterations = int(1e6)
        maximum_without_change_iterations = int(maximum_iterations * 0.1)
        objective_tolerance = 1e-5
        objective_tolerance_iterations = 2000
        stop_acpt_rate = 3e-4
        maximum_iterations_without_updating_best = int(
            maximum_iterations * 0.05)

        temperature_lower_bound = 1e1
        temperature_upper_bound = 5e9
        n_iterations_per_attempt = int(update_at_every_iteration_no * 3)
        acceptance_lower_bound = 0.35
        acceptance_upper_bound = 0.45
        target_acpt_rate = 0.40
        ramp_rate = 1.2

        acceptance_rate_iterations = 5000

        acceptance_threshold_ratio = 1e-3

    else:
        initial_annealing_temperature = 0.0001
        temperature_reduction_ratio = 0.99
        update_at_every_iteration_no = 20
        maximum_iterations = 2
        maximum_without_change_iterations = maximum_iterations
        objective_tolerance = 1e-15
        objective_tolerance_iterations = 20
        stop_acpt_rate = 1e-15
        maximum_iterations_without_updating_best = int(1e2)

        temperature_lower_bound = 1e2
        temperature_upper_bound = 5e9
        n_iterations_per_attempt = update_at_every_iteration_no
        acceptance_lower_bound = 0.6
        acceptance_upper_bound = 0.7
        target_acpt_rate = 0.65
        ramp_rate = 1.2

        acceptance_rate_iterations = 50

        acceptance_threshold_ratio = 1e-3

    if gen_rltzns_flag:
        if test_unit_peak_flag:
            np.random.seed(234324234)

            in_vals_1 = get_unit_peak(
                n_vals, beg_idx, cen_idx - 20, end_idx) + (
                    np.random.random(n_vals) * 0.1)

            in_vals_2 = get_unit_peak(
                n_vals, beg_idx, cen_idx - 30, end_idx) + (
                    np.random.random(n_vals) * 0.1)

#             in_vals_1 = get_unit_peak(
#                 n_vals, beg_idx, cen_idx + 20, end_idx)
#
#             in_vals_2 = get_unit_peak(n_vals, beg_idx, cen_idx, end_idx)

            in_vals = np.concatenate((in_vals_1, in_vals_2)).reshape(-1, 1)

#             import matplotlib.pyplot as plt
#             plt.plot(in_vals)
#             plt.show()

        else:
#             in_df = pd.read_csv(in_file_path, index_col=0, sep=sep)
#             in_df.index = pd.to_datetime(in_df.index, format=time_fmt)

            if in_file_path.suffix == '.csv':
                in_df = pd.read_csv(in_file_path, sep=sep, index_col=0)
                in_df.index = pd.to_datetime(in_df.index, format=time_fmt)

            elif in_file_path.suffix == '.pkl':
                in_df = pd.read_pickle(in_file_path)

            else:
                raise NotImplementedError(
                    f'Unknown extension of in_data_file: '
                    f'{in_file_path.suffix}!')

            sub_df = in_df.loc[beg_time:end_time, labels]

            in_vals = sub_df.values

        fftmasa_cls = FFTMASAMain(verbose)

        fftmasa_cls.set_reference_data(in_vals, list(labels))

        fftmasa_cls.set_objective_settings(
            scorr_flag,
            asymm_type_1_flag,
            asymm_type_2_flag,
            ecop_dens_flag,
            ecop_etpy_flag,
            nth_order_diffs_flag,
            cos_sin_dist_flag,
            lag_steps,
            ecop_bins,
            nth_ords,
            use_dists_in_obj_flag,
            pcorr_flag,
            lag_steps_vld,
            nth_ords_vld,
            asymm_type_1_ms_flag,
            asymm_type_2_ms_flag,
            ecop_dens_ms_flag,
            match_data_ft_flag,
            match_probs_ft_flag,
            asymm_type_1_ft_flag,
            asymm_type_2_ft_flag,
            nth_order_ft_flag,
            asymm_type_1_ms_ft_flag,
            asymm_type_2_ms_ft_flag,
            etpy_ft_flag,
            use_dens_ftn_flag,
            ratio_per_dens_bin,
            etpy_ms_ft_flag,
            scorr_ms_flag,
            etpy_ms_flag,
            match_data_ms_ft_flag,
            match_probs_ms_ft_flag,
            match_data_ms_pair_ft_flag,
            match_probs_ms_pair_ft_flag,
            )

        fftmasa_cls.set_annealing_settings(
            initial_annealing_temperature,
            temperature_reduction_ratio,
            update_at_every_iteration_no,
            maximum_iterations,
            maximum_without_change_iterations,
            objective_tolerance,
            objective_tolerance_iterations,
            acceptance_rate_iterations,
            stop_acpt_rate,
            maximum_iterations_without_updating_best,
            acceptance_threshold_ratio)

        if auto_init_temperature_flag:
            fftmasa_cls.set_annealing_auto_temperature_settings(
                temperature_lower_bound,
                temperature_upper_bound,
                n_iterations_per_attempt,
                acceptance_lower_bound,
                acceptance_upper_bound,
                target_acpt_rate,
                ramp_rate)

        if mult_idxs_flag:
            fftmasa_cls.set_mult_indices_annealing_settings(
            n_beg_idxs,
            n_end_idxs,
            idxs_sample_type,
            idxs_number_reduction_rate)

        if wts_flag:
            fftmasa_cls.set_objective_weights_settings(
                weights, auto_wts_set_flag, wts_n_iters)

        if lags_nths_wts_flag:
            fftmasa_cls.set_lags_nths_weights_settings(
                lags_nths_exp,
                lags_nths_n_iters,
                lags_nths_cumm_wts_contrib,
                lags_nths_n_thresh)

        if label_wts_flag:
            fftmasa_cls.set_label_weights_settings(label_exp, label_n_iters)

        if cdf_penalt_flag:
            fftmasa_cls.set_cdf_penalty_settings(n_vals_thresh, n_vals_penlt)

        if prt_cdf_calib_flag:
            fftmasa_cls.set_partial_cdf_calibration_settings(
                lower_threshold, upper_threshold, inside_flag)

        fftmasa_cls.set_stop_criteria_labels(stop_criteria_labels)

        fftmasa_cls.set_internal_data_transform_to_use_settings(transform_type)

        fftmasa_cls.set_misc_settings(n_reals, outputs_dir, n_cpus)

        fftmasa_cls.update_h5_file_name(h5_name)

        fftmasa_cls.verify()

        fftmasa_cls.prepare()

        fftmasa_cls.simulate()

    if plt_flag:
        fftmasa_plt_cls = FFTMASAPlot(verbose)

        fftmasa_plt_cls.set_input(
            outputs_dir / h5_name,
            n_cpus,
            plt_osv_flag,
            plt_ss_flag,
            plt_ms_flag,
            plt_qq_flag,
            max_sims_to_plot)

        fftmasa_plt_cls.set_output(outputs_dir)

        fftmasa_plt_cls.verify()

        fftmasa_plt_cls.plot()

    return


if __name__ == '__main__':

    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(
            r'P:\Synchronize\python_script_logs\\%s_log_%s.log' % (
            os.path.basename(__file__),
            datetime.now().strftime('%Y%m%d%H%M%S')))

        log_link = StdFileLoggerCtrl(out_log_file)

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack
    # 2. "up" move the stack up to an older frame
    # 3. "down" move the stack down to a newer frame
    # 4. "interact" start an interactive interpreter
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

    if _save_log_:
        log_link.stop()
