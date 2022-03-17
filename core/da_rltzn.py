'''
Created on Dec 29, 2021

@author: Faizan3800X-Uni
'''

from time import asctime
from collections import deque
from timeit import default_timer

import numpy as np

from gnrctsgenr import (
    GTGBase,
    GTGAlgRealization,
    )

from gnrctsgenr.misc import print_sl, print_el


class FFTMASARealization(GTGAlgRealization):

    def __init__(self):

        GTGAlgRealization.__init__(self)
        return

    def _update_wts(self):

        self._update_wts_fftma(1.0)
        return

    def _update_wts_fftma(self, idxs_sclr):

        if self._sett_wts_lags_nths_set_flag:
            if self._vb:
                print_sl()

                print(f'Computing lag and nths weights...')

            self._set_lag_nth_wts(idxs_sclr)

            if self._vb:

                self._show_lag_nth_wts()

                print(f'Done computing lag and nths weights.')

                print_el()

        if self._sett_wts_label_set_flag:
            if self._vb:
                print_sl()

                print(f'Computing label weights...')

            self._set_label_wts(idxs_sclr)

            if self._vb:
                self._show_label_wts()

                print(f'Done computing label weights.')

                print_el()

        if self._sett_wts_obj_auto_set_flag:
            if self._vb:
                print_sl()

                print(f'Computing individual objective function weights...')

            self._set_auto_obj_wts(idxs_sclr)

            if self._vb:
                self._show_obj_wts()

                print(f'Done computing individual objective function weights.')

                print_el()

        return

    def _show_rltzn_situ(
            self,
            iter_ctr,
            rltzn_iter,
            iters_wo_acpt,
            tol,
            temp,
            acpt_rate,
            new_obj_val,
            obj_val_min,
            iter_wo_min_updt):

        c1 = self._sett_ann_max_iters >= 10000
        c2 = not (iter_ctr % (0.05 * self._sett_ann_max_iters))

        if (c1 and c2) or (iter_ctr == 1):
            with self._lock:
                print_sl()

                print(
                    f'Realization {rltzn_iter} finished {iter_ctr} out of '
                    f'{self._sett_ann_max_iters} iterations on {asctime()}.')

                print(f'Current objective function value: {new_obj_val:9.2E}')

                print(
                    f'Running minimum objective function value: '
                    f'{obj_val_min:9.2E}\n')

                iter_wo_min_updt_ratio = (
                    iter_wo_min_updt / self._sett_ann_max_iter_wo_min_updt)

                print(
                    f'Stopping criteria variables:\n'
                    f'{self._alg_cnsts_stp_crit_labs[0]}: '
                    f'{iter_ctr/self._sett_ann_max_iters:6.2%}\n'
                    f'{self._alg_cnsts_stp_crit_labs[1]}: '
                    f'{iters_wo_acpt/self._sett_ann_max_iter_wo_chng:6.2%}\n'
                    f'{self._alg_cnsts_stp_crit_labs[2]}: {tol:9.2E}\n'
                    f'{self._alg_cnsts_stp_crit_labs[3]}: {temp:9.2E}\n'
                    f'{self._alg_cnsts_stp_crit_labs[4]}: {acpt_rate:6.3%}\n'
                    f'{self._alg_cnsts_stp_crit_labs[5]}: '
                    f'{iter_wo_min_updt_ratio:6.2%}')

                print_el()
        return

    def _get_stopp_criteria(self, test_vars):

        (iter_ctr,
         iters_wo_acpt,
         tol,
         temp,
         acpt_rate,
         iter_wo_min_updt) = test_vars

        stopp_criteria = (
            (iter_ctr < self._sett_ann_max_iters),
            (iters_wo_acpt < self._sett_ann_max_iter_wo_chng),
            (tol > self._sett_ann_obj_tol),
            (temp > self._alg_cnsts_almost_zero),
            (acpt_rate > self._sett_ann_stop_acpt_rate),
            (iter_wo_min_updt < self._sett_ann_max_iter_wo_min_updt),
            )

        if iter_ctr <= 1:
            assert len(self._alg_cnsts_stp_crit_labs) == len(stopp_criteria), (
                'stopp_criteria and its labels are not of the '
                'same length!')

        return stopp_criteria

    def _get_noise_idxs_sclr(self, iter_ctr, acpt_rate, old_idxs_sclr):

        _ = old_idxs_sclr  # To avoid the annoying unused warning.

        if not self._sett_mult_idx_flag:
            idxs_sclr = np.nan

        else:
            if self._sett_mult_idx_sample_type == 0:
                idxs_sclr = 1.0

            elif self._sett_mult_idx_sample_type == 1:
                idxs_sclr = 1.0 - (iter_ctr / self._sett_ann_max_iters)

            elif self._sett_mult_idx_sample_type == 2:
                idxs_sclr = float((
                    self._sett_mult_idxs_red_rate **
                    (iter_ctr // self._sett_ann_upt_evry_iter)))

            elif self._sett_mult_idx_sample_type == 3:
                # An unstable mean of acpts_rjts_dfrntl is a
                # problem. So, it has to be long enough.

                # Why the min(acpt_rate, old_phs_red_rate) was used?

                # Not using min might result in instability as acpt_rate will
                # oscillate when phs_red_rate oscillates but this is taken
                # care of by the maximum iterations without updating the
                # global minimum.

                # Also, it might get stuck in a local minimum by taking min.

                # Normally, it moves very slowly if min is used after some
                # iterations. The accpt_rate stays high due to this slow
                # movement after hitting a low. This becomes a substantial
                # part of the time taken to finish annealing which doesn't
                # bring much improvement to the global minimum.

                idxs_sclr = acpt_rate

            else:
                raise NotImplementedError

            assert np.isfinite(idxs_sclr), f'Invalid idxs_sclr ({idxs_sclr})!'

        return idxs_sclr

    def _get_next_idxs(self, idxs_sclr):

        idxs_diff = self._data_ref_shape[0]

        assert idxs_diff > 0, idxs_diff

        # if any([
        #     self._alg_wts_lag_nth_search_flag,
        #     self._alg_wts_label_search_flag,
        #     self._alg_wts_obj_search_flag,
        #     self._alg_ann_runn_auto_init_temp_search_flag,
        #     ]):
        #
        #     if self._sett_mult_idx_flag:
        #         # Full spectrum randomization during search.
        #         new_idxs = self._rr.noise_idxs
        #
        #     else:
        #         new_idxs = np.random.choice(
        #             self._rr.noise_idxs, size=(1,), replace=False)
        #
        # else:
        if self._sett_mult_idx_flag:
            min_idxs_to_gen = self._sett_mult_idx_n_beg_idxs
            max_idxs_to_gen = self._sett_mult_idx_n_end_idxs

        else:
            min_idxs_to_gen = 1
            max_idxs_to_gen = 2

        # Inclusive.
        min_idxs_to_gen = min([min_idxs_to_gen, idxs_diff])

        # Inclusive.
        max_idxs_to_gen = min([max_idxs_to_gen, idxs_diff])

        if np.isnan(idxs_sclr):
            idxs_to_gen = np.random.randint(min_idxs_to_gen, max_idxs_to_gen)

        else:
            idxs_to_gen = min_idxs_to_gen + (
                int(round(idxs_sclr *
                    (max_idxs_to_gen - min_idxs_to_gen))))

        assert min_idxs_to_gen >= 1, 'This shouldn\'t have happend!'
        assert idxs_to_gen >= 1, 'This shouldn\'t have happend!'

        if min_idxs_to_gen == idxs_diff:
            new_idxs = np.arange(1, min_idxs_to_gen + 1)

        else:
            new_idxs = []
            sample = self._rr.noise_idxs

            new_idxs = np.random.choice(
                sample,
                idxs_to_gen,
                replace=False)

        assert np.all(0 <= new_idxs)

        assert np.all(
            new_idxs < (self._data_ref_shape[0] + (self._sett_padd_steps * 2))), (
                (self._data_ref_shape[0] + (self._sett_padd_steps * 2)), new_idxs)

        return new_idxs

    @GTGBase._timer_wrap
    def _get_next_iter_vars(self, idxs_sclr):

        new_idxs = self._get_next_idxs(idxs_sclr)

        old_noise = self._rs.noise[new_idxs,:].copy()

        if True:
            # Sample new noise values.
            new_noise = self._sample_from_noise_dist(new_idxs.shape[0])

        else:
            # Shuffle existing values.

            # For shuffling values instead of generating new ones.
            assert new_idxs.size > 1, new_idxs

            new_noise = old_noise.copy()

            np.random.shuffle(new_noise)

        return old_noise, new_noise, new_idxs

    @GTGBase._timer_wrap
    def _update_sim(self, idxs, noise, load_snapshot_flag):

        self._rs.noise[idxs,:] = noise

        self._rs.data_tfm = self._get_fftma_ts(self._rs.noise)

        ft = np.fft.rfft(self._rs.data_tfm[
            self._sett_padd_steps:
            self._data_ref_shape[0] + self._sett_padd_steps], axis=0)

        self._rs.ft = ft
        self._rs.phs_spec = np.angle(ft)
        self._rs.mag_spec = np.abs(ft)

        if load_snapshot_flag:
            self._load_snapshot()

        else:
            self._update_sim_no_prms()

        return

    def _update_sim_no_prms(self):

        data = self._rs.data_tfm[
            self._sett_padd_steps:
            self._data_ref_shape[0] + self._sett_padd_steps]

        probs = self._get_probs(data, True)

        self._rs.data = np.empty_like(
            self._data_ref_rltzn_srtd, dtype=np.float64)

        for i in range(self._data_ref_n_labels):
            self._rs.data[:, i] = self._data_ref_rltzn_srtd[
                np.argsort(np.argsort(probs[:, i])), i]

        self._rs.probs = probs

        if any([self._sett_obj_match_data_ft_flag,
                self._sett_obj_match_data_ms_ft_flag,
                self._sett_obj_match_data_ms_pair_ft_flag]):

            self._rs.data_ft_coeffs = np.fft.rfft(self._rs.data, axis=0)
            self._rs.data_ft_coeffs_mags = np.abs(self._rs.data_ft_coeffs)

            if self._sett_obj_match_data_ms_pair_ft_flag:
                self._rs.data_ft_coeffs_phss = np.angle(
                    self._rs.data_ft_coeffs)

        if any([self._sett_obj_match_probs_ft_flag,
                self._sett_obj_match_probs_ms_ft_flag,
                self._sett_obj_match_probs_ms_pair_ft_flag]):

            self._rs.probs_ft_coeffs = np.fft.rfft(self._rs.probs, axis=0)
            self._rs.probs_ft_coeffs_mags = np.abs(self._rs.probs_ft_coeffs)

            if self._sett_obj_match_probs_ms_pair_ft_flag:
                self._rs.probs_ft_coeffs_phss = np.angle(
                    self._rs.probs_ft_coeffs)

        self._update_obj_vars('sim')
        return

    def _gen_gnrc_rltzn(self, args):

        (rltzn_iter,
         init_temp,
        ) = args

        assert self._alg_verify_flag, 'Call verify first!'

        beg_time = default_timer()

        assert isinstance(rltzn_iter, int), 'rltzn_iter not integer!'

        if self._alg_ann_runn_auto_init_temp_search_flag:
            temp = init_temp

        else:
            # _alg_rltzn_iter should be only set when annealing is started.
            self._alg_rltzn_iter = rltzn_iter

            assert 0 <= rltzn_iter < self._sett_misc_n_rltzns, (
                    'Invalid rltzn_iter!')

            temp = self._sett_ann_init_temp

        if self._data_ref_rltzn.ndim != 2:
            raise NotImplementedError('Implemention for 2D only!')

        # Randomize all phases before starting.
        self._gen_sim_aux_data()

        # Initialize sim anneal variables.
        iter_ctr = 0

        if self._sett_ann_auto_init_temp_trgt_acpt_rate is not None:
            acpt_rate = self._sett_ann_auto_init_temp_trgt_acpt_rate

        else:
            acpt_rate = 1.0

        # FFTMA Annealing variable.
        idxs_sclr = self._get_noise_idxs_sclr(iter_ctr, acpt_rate, 1.0)

        if self._alg_ann_runn_auto_init_temp_search_flag:
            stopp_criteria = (
                (iter_ctr <= self._sett_ann_auto_init_temp_niters),
                )

        else:
            iters_wo_acpt = 0
            tol = np.inf

            iter_wo_min_updt = 0

            tols_dfrntl = deque(maxlen=self._sett_ann_obj_tol_iters)

            acpts_rjts_dfrntl = deque(maxlen=self._sett_ann_acpt_rate_iters)

            stopp_criteria = self._get_stopp_criteria(
                (iter_ctr,
                 iters_wo_acpt,
                 tol,
                 temp,
                 acpt_rate,
                 iter_wo_min_updt))

        old_idxs = self._get_next_idxs(idxs_sclr)
        new_idxs = old_idxs

        old_obj_val = self._get_obj_ftn_val().sum()

        self._update_snapshot()

        # Initialize diagnostic variables.
        acpts_rjts_all = []

        if not self._alg_ann_runn_auto_init_temp_search_flag:
            tols = []

            obj_vals_all = []

            obj_val_min = old_obj_val
            obj_vals_min = []

            temps = [[iter_ctr, temp]]

            acpt_rates_dfrntl = [[iter_ctr, acpt_rate]]

            # Phase Annealing variable.
            idxs_sclrs = [[iter_ctr, idxs_sclr]]

            obj_vals_all_indiv = []

        else:
            pass

        self._rs.ft_best = self._rs.ft.copy()

        while all(stopp_criteria):

            #==============================================================
            # Simulated annealing start.
            #==============================================================

            # FFTMA variables.
            (old_noise,
             new_noise,
             new_idxs) = self._get_next_iter_vars(idxs_sclr)

            self._update_sim(new_idxs, new_noise, False)

            new_obj_val_indiv = self._get_obj_ftn_val()
            new_obj_val = new_obj_val_indiv.sum()

            old_new_diff = old_obj_val - new_obj_val

            old_new_adj_diff = old_new_diff - (
                self._sett_ann_acpt_thresh * old_obj_val)

            if old_new_adj_diff > 0:
                accept_flag = True

            else:
                rand_p = np.random.random()

                boltz_p = np.exp(old_new_adj_diff / temp)

                if rand_p < boltz_p:
                    accept_flag = True

                else:
                    accept_flag = False

            if self._alg_force_acpt_flag:
                accept_flag = True
                self._alg_force_acpt_flag = False

            if accept_flag:
                old_idxs = new_idxs

                old_obj_val = new_obj_val

                self._update_snapshot()

            else:
                self._update_sim(new_idxs, old_noise, True)

            iter_ctr += 1

            #==============================================================
            # Simulated annealing end.
            #==============================================================

            acpts_rjts_all.append(accept_flag)

            if self._alg_ann_runn_auto_init_temp_search_flag:
                stopp_criteria = (
                    (iter_ctr <= self._sett_ann_auto_init_temp_niters),
                    )

            else:
                obj_vals_all_indiv.append(new_obj_val_indiv)

                if new_obj_val < obj_val_min:
                    iter_wo_min_updt = 0

                    self._rs.ft_best = self._rs.ft.copy()

                else:
                    iter_wo_min_updt += 1

                tols_dfrntl.append(abs(old_new_diff))

                obj_val_min = min(obj_val_min, new_obj_val)

                obj_vals_min.append(obj_val_min)
                obj_vals_all.append(new_obj_val)

                acpts_rjts_dfrntl.append(accept_flag)

                self._rs.n_idxs_all_cts[new_idxs] += 1

                if iter_ctr >= tols_dfrntl.maxlen:
                    tol = sum(tols_dfrntl) / float(tols_dfrntl.maxlen)

                    assert np.isfinite(tol), 'Invalid tol!'

                    tols.append(tol)

                if accept_flag:
                    self._rs.n_idxs_acpt_cts[new_idxs] += 1
                    iters_wo_acpt = 0

                else:
                    iters_wo_acpt += 1

                if iter_ctr >= acpts_rjts_dfrntl.maxlen:
                    acpt_rates_dfrntl.append([iter_ctr - 1, acpt_rate])

                    acpt_rate = (
                        sum(acpts_rjts_dfrntl) /
                        float(acpts_rjts_dfrntl.maxlen))

                    acpt_rates_dfrntl.append([iter_ctr, acpt_rate])

                if (iter_ctr % self._sett_ann_upt_evry_iter) == 0:

                    # Temperature.
                    temps.append([iter_ctr - 1, temp])

                    temp *= self._sett_ann_temp_red_rate

                    assert temp >= 0.0, 'Invalid temp!'

                    temps.append([iter_ctr, temp])

                    # FFTMA Annealing variable.
                    # Noise indices reduction rate.
                    idxs_sclrs.append([iter_ctr - 1, idxs_sclr])

                    idxs_sclr = self._get_noise_idxs_sclr(
                        iter_ctr, acpt_rate, idxs_sclr)

                    idxs_sclrs.append([iter_ctr, idxs_sclr])

                if self._vb:
                    self._show_rltzn_situ(
                        iter_ctr,
                        rltzn_iter,
                        iters_wo_acpt,
                        tol,
                        temp,
                        acpt_rate,
                        new_obj_val,
                        obj_val_min,
                        iter_wo_min_updt)

                stopp_criteria = self._get_stopp_criteria(
                    (iter_ctr,
                     iters_wo_acpt,
                     tol,
                     temp,
                     acpt_rate,
                     iter_wo_min_updt))

        # Manual update of timer because this function writes timings
        # to the HDF5 file before it returns.
        if '_gen_gnrc_rltzn' not in self._dur_tmr_cumm_call_times:
            self._dur_tmr_cumm_call_times['_gen_gnrc_rltzn'] = 0.0
            self._dur_tmr_cumm_n_calls['_gen_gnrc_rltzn'] = 0.0

        self._dur_tmr_cumm_call_times['_gen_gnrc_rltzn'] += (
            default_timer() - beg_time)

        self._dur_tmr_cumm_n_calls['_gen_gnrc_rltzn'] += 1

        if self._alg_ann_runn_auto_init_temp_search_flag:

            ret = sum(acpts_rjts_all) / len(acpts_rjts_all), temp

        else:
            # _sim_ft set to _sim_ft_best in _update_sim_at_end.
            self._update_ref_at_end()
            self._update_sim_at_end()

            self._rs.data_corr_ftn = self._get_corr_ftn_auto(
                self._rs.data)

            self._rs.label = (
                f'{rltzn_iter:0{len(str(self._sett_misc_n_rltzns))}d}')

            self._rs.iter_ctr = iter_ctr
            self._rs.iters_wo_acpt = iters_wo_acpt
            self._rs.tol = tol
            self._rs.temp = temp
            self._rs.stopp_criteria = np.array(stopp_criteria)
            self._rs.tols = np.array(tols, dtype=np.float64)
            self._rs.obj_vals_all = np.array(obj_vals_all, dtype=np.float64)

            self._rs.acpts_rjts_all = np.array(acpts_rjts_all, dtype=bool)

            self._rs.acpt_rates_all = (
                np.cumsum(self._rs.acpts_rjts_all) /
                np.arange(1, self._rs.acpts_rjts_all.size + 1, dtype=float))

            self._rs.obj_vals_min = np.array(obj_vals_min, dtype=np.float64)

            self._rs.temps = np.array(temps, dtype=np.float64)

            self._rs.acpt_rates_dfrntl = np.array(
                acpt_rates_dfrntl, dtype=np.float64)

            self._rr.ft_cumm_corr = self._get_cumm_ft_corr(
                self._rr.ft, self._rr.ft)

            self._rs.ref_sim_ft_corr = self._get_cumm_ft_corr(
                self._rr.ft, self._rs.ft).astype(np.float64)

            self._rs.sim_sim_ft_corr = self._get_cumm_ft_corr(
                    self._rs.ft, self._rs.ft).astype(np.float64)

            self._rs.obj_vals_all_indiv = np.array(
                obj_vals_all_indiv, dtype=np.float64)

            # FFTMA-SA variable.
            self._rs.idxs_sclrs = np.array(idxs_sclrs, dtype=np.float64)

            self._rs.cumm_call_durations = self._dur_tmr_cumm_call_times
            self._rs.cumm_n_calls = self._dur_tmr_cumm_n_calls

            self._write_cls_rltzn()

            ret = stopp_criteria

        self._alg_snapshot = None
        return ret
