'''
Created on Dec 29, 2021

@author: Faizan3800X-Uni
'''

import numpy as np
import matplotlib as mpl
# Has to be big enough to accomodate all plotted values.
mpl.rcParams['agg.path.chunksize'] = 50000
from scipy.stats import rankdata

from timeit import default_timer
from multiprocessing import Pool
from itertools import combinations

import h5py
import matplotlib.pyplot as plt;plt.ioff()

from gnrctsgenr import (
    get_mpl_prms,
    set_mpl_prms,
    GTGPlotBase,
    GTGPlotOSV,
    GTGPlotSingleSite,
    GTGPlotMultiSite,
    GTGPlotSingleSiteQQ,
    GenericTimeSeriesGeneratorPlot,
    )

from gnrctsgenr.misc import print_sl, print_el


class FFTMASAPlot(
        GTGPlotBase,
        GTGPlotOSV,
        GTGPlotSingleSite,
        GTGPlotMultiSite,
        GTGPlotSingleSiteQQ,
        GenericTimeSeriesGeneratorPlot):

    def __init__(self, verbose):

        GTGPlotBase.__init__(self, verbose)
        GTGPlotOSV.__init__(self)
        GTGPlotSingleSite.__init__(self)
        GTGPlotMultiSite.__init__(self)
        GTGPlotSingleSiteQQ.__init__(self)
        GenericTimeSeriesGeneratorPlot.__init__(self)

        self._plt_sett_idx_red_rates = self._default_line_sett
        self._plt_sett_noise = self._default_line_sett
        self._plt_sett_corr_ftn = self._default_line_sett
        self._plt_sett_noise_cdfs = self._plt_sett_gnrc_cdfs
        self._plt_sett_cross_noise_sctr = self._plt_sett_ecops_sctr
        return

    @staticmethod
    def _plot_noise_scatter_ms_base(args):

        (data_a,
         data_b,
         fig_suff,
         out_dir,
         plt_sett,
         dl_a,
         dl_b) = args

        axes = plt.subplots(1, 1, squeeze=False)[1]

        row, col = 0, 0

        axes[row, col].scatter(
            data_a,
            data_b,
            alpha=plt_sett.alpha_1,
            c='b')

        axes[row, col].grid()

        axes[row, col].set_axisbelow(True)

        axes[row, col].set_aspect('equal')

        axes[row, col].set_xlabel(f'Noise - {dl_a}')
        axes[row, col].set_ylabel(f'Noise - {dl_b}')

        min_crd = min([data_a.min(), data_b.min()]) * 0.95
        max_crd = max([data_a.max(), data_b.max()]) * 1.05

        axes[row, col].set_xlim(min_crd, max_crd)
        axes[row, col].set_ylim(min_crd, max_crd)

        pcorr = np.corrcoef(data_a, data_b)[0, 1]

        scorr = np.corrcoef(
            np.argsort(np.argsort(data_a)),
            np.argsort(np.argsort(data_b)))[0, 1]

        plt.suptitle(f'pcorr: {pcorr:+0.3f}, scorr: {scorr:+0.3f}')

        plt.tight_layout()

        plt.savefig(
            str(out_dir / f'ms__cross_noise_scatter_{fig_suff}.png'),
            bbox_inches='tight')

        plt.close()
        return

    def _plot_noise_scatter_ms(self):

        '''
        Meant for pairs only.
        '''

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_cross_noise_sctr

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        data_labels = tuple(h5_hdl['data_ref'].attrs['data_ref_labels'])

        data_label_idx_combs = combinations(enumerate(data_labels), 2)

        loop_prod = data_label_idx_combs

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for ((di_a, dl_a), (di_b, dl_b)) in loop_prod:

            ref_noise_a = h5_hdl[f'data_ref_rltzn/data_tfm_noise'][:, di_a]

            ref_noise_b = h5_hdl[f'data_ref_rltzn/data_tfm_noise'][:, di_b]

            fig_suff = f'ref_{dl_a}_{dl_b}'

            args = (
                ref_noise_a,
                ref_noise_b,
                fig_suff,
                self._ms_dir,
                plt_sett,
                dl_a,
                dl_b)

            self._plot_noise_scatter_ms_base(args)

            plot_ctr = 0
            for rltzn_lab in sim_grp_main:

                sim_noise_a = sim_grp_main[f'{rltzn_lab}/noise'][:, di_a]
                sim_noise_b = sim_grp_main[f'{rltzn_lab}/noise'][:, di_b]

                fig_suff = f'sim_{dl_a}_{dl_b}_{rltzn_lab}'

                args = (
                    sim_noise_a,
                    sim_noise_b,
                    fig_suff,
                    self._ms_dir,
                    plt_sett,
                    dl_a,
                    dl_b)

                self._plot_noise_scatter_ms_base(args)

                plot_ctr += 1

                if plot_ctr == self._plt_max_n_sim_plots:
                    break

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting multi-site pairwise noise scatter '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_noise_cdfs(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_noise_cdfs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        out_name_pref = 'ss__noise_data_tfm_dist'

        data_labels = tuple(h5_hdl['data_ref'].attrs['data_ref_labels'])

        n_data_labels = h5_hdl['data_ref'].attrs['data_ref_n_labels']

        loop_prod = np.arange(n_data_labels)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for data_lab_idx in loop_prod:
            ref_noise = np.sort(h5_hdl[
                f'data_ref_rltzn/data_tfm_noise'][:, data_lab_idx])

            ref_probs = rankdata(ref_noise) / (ref_noise.size + 1)

            plt.figure()

            plt.plot(
                ref_noise,
                ref_probs,
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim'

                else:
                    label = None

                sim_noise = np.sort(
                    sim_grp_main[f'{rltzn_lab}/noise'][:, data_lab_idx])

                sim_probs = rankdata(sim_noise) / (sim_noise.size + 1)

                plt.plot(
                    sim_noise,
                    sim_probs,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.gca().set_axisbelow(True)

            plt.legend(framealpha=0.7)

            plt.ylabel('F(Noise)')

            plt.xlabel(f'Noise')

            fig_name = f'{out_name_pref}_{data_labels[data_lab_idx]}.png'

            plt.savefig(str(self._ss_dir / fig_name), bbox_inches='tight')

            plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site noise distribution function '
                f'took {end_tm - beg_tm:0.2f} seconds.')
            
        return

    def _plot_corr_ftn(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_corr_ftn

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        out_name_pref = 'ss__data_corr_ftn'

        data_labels = tuple(h5_hdl['data_ref'].attrs['data_ref_labels'])

        n_data_labels = h5_hdl['data_ref'].attrs['data_ref_n_labels']

        loop_prod = np.arange(n_data_labels)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for data_lab_idx in loop_prod:
            ref_corr_ftn = h5_hdl[
                f'data_ref_rltzn/data_corr_ftn'][:, data_lab_idx]

            plt.figure()

            plt.plot(
                ref_corr_ftn[:ref_corr_ftn.size // 2],
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim'

                else:
                    label = None

                sim_corr_ftn = sim_grp_main[
                    f'{rltzn_lab}/data_corr_ftn'][:, data_lab_idx]

                plt.plot(
                    sim_corr_ftn[:sim_corr_ftn.size // 2],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.gca().set_axisbelow(True)

            plt.legend(framealpha=0.7)

            plt.ylabel('Correlation')

            plt.xlabel(f'Time lag')

            fig_name = f'{out_name_pref}_{data_labels[data_lab_idx]}.png'

            plt.savefig(str(self._ss_dir / fig_name), bbox_inches='tight')

            plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site correlation function '
                f'took {end_tm - beg_tm:0.2f} seconds.')

        return

    def _plot_noise(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_noise

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        out_name_pref = 'ss__noise_data_tfm'

        data_labels = tuple(h5_hdl['data_ref'].attrs['data_ref_labels'])

        n_data_labels = h5_hdl['data_ref'].attrs['data_ref_n_labels']

        loop_prod = np.arange(n_data_labels)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for data_lab_idx in loop_prod:
            ref_noise = h5_hdl[
                f'data_ref_rltzn/data_tfm_noise'][:, data_lab_idx]

            plt.figure()

            plt.plot(
                ref_noise,
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim'

                else:
                    label = None

                sim_noise = sim_grp_main[
                    f'{rltzn_lab}/noise'][:, data_lab_idx]

                plt.plot(
                    sim_noise,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.gca().set_axisbelow(True)

            plt.legend(framealpha=0.7)

            plt.ylabel('Noise')

            plt.xlabel(f'Time step')

            fig_name = f'{out_name_pref}_{data_labels[data_lab_idx]}.png'

            plt.savefig(str(self._ss_dir / fig_name), bbox_inches='tight')

            plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site noise time series '
                f'took {end_tm - beg_tm:0.2f} seconds.')

        return

    def _plot_noise_idxs_sclrs(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_idx_red_rates

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        plt.figure()

        for rltzn_lab in sim_grp_main:
            noise_idxs_sclrs_all = sim_grp_main[f'{rltzn_lab}/idxs_sclrs']

            plt.plot(
                noise_idxs_sclrs_all[:, 0],
                noise_idxs_sclrs_all[:, 1],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1)

        plt.ylim(0, 1)

        plt.xlabel('Iteration')

        plt.ylabel(f'Noise indices reduction rate')

        plt.grid()

        plt.gca().set_axisbelow(True)

        plt.savefig(
            str(self._osv_dir / f'osv__noise_idxs_sclrs.png'),
            bbox_inches='tight')

        plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting optimization indices scaler rates '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def plot(self):

        if self._vb:
            print_sl()

            print('Plotting...')

        assert self._plt_verify_flag, 'Plot in an unverified state!'

        ftns_args = []

        self._fill_osv_args_gnrc(ftns_args)

        # Variables specific to FFTMASA.
        if self._plt_osv_flag:

            h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

            mult_idx_flag = h5_hdl['flags'].attrs['sett_mult_idx_flag']

            h5_hdl.close()

            if mult_idx_flag:
                ftns_args.append((self._plot_noise_idxs_sclrs, []))

            # ftns_args.extend([
            #     ])

        self._fill_ss_args_gnrc(ftns_args)

        if self._plt_ss_flag:
            ftns_args.extend([
                (self._plot_noise, []),
                (self._plot_corr_ftn, []),
                (self._plot_noise_cdfs, []),
                ])

        self._fill_ms_args_gnrc(ftns_args)

        if self._plt_ms_flag:
            h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

            n_data_labels = h5_hdl['data_ref'].attrs['data_ref_n_labels']

            h5_hdl.close()

            if n_data_labels >= 2:
                ftns_args.extend([
                    (self._plot_noise_scatter_ms, []),
                    ])

        self._fill_qq_args_gnrc(ftns_args)

        assert ftns_args, 'Huh!'

        n_cpus = min(self._n_cpus, len(ftns_args))

        if n_cpus == 1:
            for ftn_arg in ftns_args:
                self._exec(ftn_arg)

        else:
            mp_pool = Pool(n_cpus)

            # NOTE:
            # imap_unordered does not show exceptions, map does.

            # mp_pool.imap_unordered(self._exec, ftns_args)

            mp_pool.map(self._exec, ftns_args, chunksize=1)

            mp_pool.close()
            mp_pool.join()

            mp_pool = None

        if self._vb:
            print('Done plotting.')

            print_el()

        return
