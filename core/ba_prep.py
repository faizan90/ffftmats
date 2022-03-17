'''
Created on Dec 29, 2021

@author: Faizan3800X-Uni
'''

import numpy as np
from scipy.stats import rankdata, norm
from scipy.interpolate import interp1d

from gnrctsgenr import (
    GTGPrepareRltznRef,
    GTGPrepareRltznSim,
    GTGPrepareTfms,
    GTGPrepare,
    roll_real_2arrs,
    )

# def gen_symm_corr_matrix(n_dims, corr_min, corr_max):
#
#     corrs = np.zeros((n_dims, n_dims))
#
#     corrs.ravel()[::(n_dims + 1)] = 1
#     for i in range(n_dims):
#         corrs[i, i + 1:] = corr_min + (
#             (corr_max - corr_min) * np.random.random(n_dims - i - 1))
#
#         corrs[i + 1:, i] = corrs[i, i + 1:]
#
#     # To check if the matrix is symmetric.
#     if True:
#         for i in range(corrs.shape[0]):
#             assert np.all(corrs[i,:] == corrs[:, i]), (i,
#                 corrs[i,:], corrs[:, i])
#
#     return corrs


class FFTMASAPrepareRltznRef(GTGPrepareRltznRef):

    def __init__(self):

        GTGPrepareRltznRef.__init__(self)

        self.data_tfm_corr_ftn = None
        self.data_tfm_corr_ftn_ft_mag_sqrt = None

        self.data_tfm_noise = None

        self.data_corr_ftn = None

        # Which indices to sample.
        self.noise_idxs = None

        self.noise_dist_ftns = None

        # These have to be dealt with in the _write_ref_rltzn_extra method.
        self.skip_io_vars.append('noise_dist_ftns')
        return


class FFTMASAPrepareRltznSim(GTGPrepareRltznSim):

    def __init__(self):

        GTGPrepareRltznSim.__init__(self)

        self.data_tfm = None

        self.data_corr_ftn = None

        self.idxs_sclrs = None

        self.noise = None
        return


class FFTMASAPrepareTfms(GTGPrepareTfms):

    def __init__(self):

        GTGPrepareTfms.__init__(self)
        return

    def _get_corr_ftn_auto(self, *args):

        method = 'emp'
        data = args[0]

        if method == 'emp':
            corr_ftn = self._get_corr_ftn_auto_emp(data)

        elif method == 'wk':
            corr_ftn = self._get_corr_ftn_auto_wk(data)

        else:
            raise NotImplementedError(f'Unknown method for computing correlation function: {method}')

        return corr_ftn

    def _get_corr_ftn_auto_wk(self, data):

        '''
        Correlation function computation based on the Wiener-khintchin
        theorem.
        '''

        # This one computes through rolling the series.

        # data_ft = np.fft.rfft(data - data.mean(axis=0), axis=0)
        data_ft = np.fft.rfft(data, axis=0)

        data_ft[0,:] = 0.0  # Equivalent to having a mean of zero.

        data_mag = np.abs(data_ft)

        # Wiener-Khintchin theorem.
        pwr_ft = np.fft.irfft(data_mag ** 2, axis=0)

        corr_ftn = np.sign(pwr_ft.real) * np.abs(pwr_ft)

        # corr_ftn /= corr_ftn[0,:]

        if self._sett_padd_steps:
            zeros = np.zeros((self._sett_padd_steps * 2, data.shape[1]))

            corr_ftn = np.concatenate(
                (corr_ftn[:1 + (corr_ftn.shape[0] // 2),:],
                 zeros,
                 # corr_ftn[1:1 + (corr_ftn.shape[0] // 2), :][::-1,:]),
                 corr_ftn[(corr_ftn.shape[0] // 2):,:]),
                axis=0)

        else:
            pass
            # corr_ftn = np.concatenate(
            #     (corr_ftn, [corr_ftn[0,:]]), axis=0)

        return corr_ftn

    def _get_corr_ftn_auto_emp(self, data):

        '''
        Empirical correlation function.
        '''

        assert data.ndim == 2, data.ndim

        corr_ftn = np.zeros(
            (data.shape[0] + (self._sett_padd_steps * 2), data.shape[1]),
            order='f',
            dtype=np.float64)

        corr_ftn[+0,:] = data.var(axis=0)
        # corr_ftn[+0,:] = 1.0
        # corr_ftn[-1,:] = corr_ftn[+0,:]
        for col in range(corr_ftn.shape[1]):
            for lag in range(1, data.shape[0] // 2):
                pcorr = np.cov(
                    *roll_real_2arrs(data[:, col], data[:, col], lag, False))

                # Using this results in 0.0 ft values.
                # I think, the corr ftn doesn't have to be symmetric at the
                # first step. This could be due rfft.

                # corr_ftn[+lag - 0, col] = pcorr[0, 1]
                # corr_ftn[-lag - 1, col] = pcorr[0, 1]

                corr_ftn[+lag, col] = pcorr[0, 1]
                corr_ftn[-lag, col] = pcorr[0, 1]

        # assert corr_ftn.size == data.size, (corr_ftn.size, data.size)

        return corr_ftn

    def _get_fftma_noise(self, data):

        if self._sett_padd_steps:
            padd_data = np.full(
                (self._sett_padd_steps, data.shape[1]), data.mean())

            data = np.concatenate((padd_data, data, padd_data), axis=0)

        data_ft = np.fft.rfft(data, axis=0)

        data_ft[0,:] = 0.0  # Equivalent to having a mean of zero.

        # FIXME: If first value of self._rr.data_tfm_corr_ftn_ft_mag_sqrt
        # is zero, then we have a problem. For now, making it slighlty
        # larger to avoid the division by zero error.
        for i in range(self._data_ref_shape[1]):
            if self._rr.data_tfm_corr_ftn_ft_mag_sqrt[0, i] < 1e-14:
                self._rr.data_tfm_corr_ftn_ft_mag_sqrt[0, i] = 1e-14

        noise_ft = data_ft / self._rr.data_tfm_corr_ftn_ft_mag_sqrt

        noise = np.fft.irfft(noise_ft, axis=0)

        assert np.all(np.isfinite(noise)), 'Invalid values in noise!'

        imag_vals_err = (noise.imag ** 2).mean()
        assert np.isclose(imag_vals_err, 0.0), imag_vals_err

        return noise.real

    def _get_fftma_ts(self, noise):

        noise_ft = np.fft.rfft(noise, axis=0)

        noise_corr_ftn_prod = noise_ft * (
            self._rr.data_tfm_corr_ftn_ft_mag_sqrt)

        noise_corr_ftn_prod_inv = np.fft.irfft(noise_corr_ftn_prod, axis=0)

        imag_vals_err = (noise_corr_ftn_prod_inv.imag ** 2).mean()
        assert np.isclose(imag_vals_err, 0.0), imag_vals_err

        return noise_corr_ftn_prod_inv.real

    def _get_fftma_noise_dists(self):

        # TODO: The fill_value? Provide options.

        dist_ftns = {}
        for col in range(self._data_ref_shape[1]):

            if True:
                fill_value = 'extrapolate'

            else:
                fill_value = (
                    self._rr.data_tfm_noise[:, col].min(),
                    self._rr.data_tfm_noise[:, col].max())

            probs = np.sort(
                rankdata(self._rr.data_tfm_noise[:, col]) / (
                    self._rr.data_tfm_noise[:, col].size + 1.0))

            dist_ftn = interp1d(
                probs,
                np.sort(self._rr.data_tfm_noise[:, col]),
                bounds_error=False,
                fill_value=fill_value)

            dist_ftns[self._data_ref_labels[col]] = dist_ftn

        return dist_ftns

    def _sample_from_noise_dist(self, n_samples):

        if self._data_ref_shape[1] == 1:
            sample = np.empty(
                (n_samples, self._data_ref_shape[1]),
                order='f',
                dtype=np.float64)

            for i in range(self._data_ref_shape[1]):
                sample[:, i] = self._rr.noise_dist_ftns[
                    self._data_ref_labels[i]](np.random.random(size=n_samples))

        else:
            sample = self._mvn_rng(
                self._prep_mvn_means, self._prep_mvn_covs, size=n_samples)

            for i in range(self._data_ref_shape[1]):
                sample[:, i] = self._rr.noise_dist_ftns[
                    self._data_ref_labels[i]](norm.cdf(sample[:, i]))

        return sample


class FFTMASAPrepare(GTGPrepare):

    def __init__(self):

        GTGPrepare.__init__(self)

        self._mvn_rng = None
        self._prep_mvn_means = None
        self._prep_mvn_covs = None
        self._prep_mvn_corrs = None
        return

    def _init_mvn_rng_prms(self):

        assert self._data_ref_shape[1] > 1, 'For multisite only!'

        self._mvn_rng = np.random.default_rng().multivariate_normal

        # self._prep_mvn_means = self._rr.data_tfm_noise.mean(axis=0)
        # self._prep_mvn_covs = np.cov(self._rr.data_tfm_noise.T)
        # self._prep_mvn_corrs = np.corrcoef(self._rr.data_tfm_noise.T)
        #
        # eig_vals = np.linalg.eigvals(self._prep_mvn_covs)
        #
        # if np.any(eig_vals < 0):
        #     raise RuntimeError(eig_vals)
        #======================================================================

        probs = rankdata(self._rr.data_tfm_noise, axis=0) / (
            self._rr.data_tfm_noise.shape[0] + 1.0)

        norms = norm.ppf(probs)

        self._prep_mvn_means = norms.mean(axis=0)
        self._prep_mvn_covs = np.cov(norms.T)
        self._prep_mvn_corrs = np.corrcoef(norms.T)

        eig_vals = np.linalg.eigvals(self._prep_mvn_covs)

        if np.any(eig_vals < 0):
            raise RuntimeError(eig_vals)

        return

    def _gen_ref_aux_data(self):

        self._gen_ref_aux_data_gnrc()

        self._rr.noise_idxs = np.arange(
            self._data_ref_shape[0] + (self._sett_padd_steps * 2))

        self._rr.data_corr_ftn = self._get_corr_ftn_auto(self._rr.data)

        self._rr.data_tfm_corr_ftn = self._get_corr_ftn_auto(self._rr.data_tfm)

        assert np.all(np.isfinite(self._rr.data_tfm_corr_ftn)), (
            'Invalid values in data_tfm_corr_ftn!')

        corr_ftn_ft = np.round(
            np.fft.rfft(self._rr.data_tfm_corr_ftn, axis=0), 14)

        self._rr.data_tfm_corr_ftn_ft_mag_sqrt = np.abs(corr_ftn_ft) ** 0.5

        assert np.all(np.isfinite(self._rr.data_tfm_corr_ftn_ft_mag_sqrt)), (
            'Invalid values in data_tfm_corr_ftn_ft_mag_sqrt!')

        # NOTE: The choice of the threshold for small values is arbitrary,
        # for now. The first value does not matter, as 0 is divided by it.
        assert np.all(self._rr.data_tfm_corr_ftn_ft_mag_sqrt[1:] >= 1e-6), (
            'Too small values in data_tfm_corr_ftn_ft_mag_sqrt!')

        self._rr.data_tfm_noise = self._get_fftma_noise(self._rr.data_tfm)

        assert np.all(np.isfinite(self._rr.data_tfm_noise)), (
            'Invalid values in data_tfm_noise!')

        self._rr.noise_dist_ftns = self._get_fftma_noise_dists()

        if self._data_ref_shape[1] > 1:
            self._init_mvn_rng_prms()

        self._prep_ref_aux_flag = True
        return

    def _gen_sim_aux_data(self):

        assert self._prep_ref_aux_flag, 'Call _gen_ref_aux_data first!'

        if self._data_ref_rltzn.ndim != 2:
            raise NotImplementedError('Implementation for 2D only!')

        self._rs.shape = (self._data_ref_shape[0], self._data_ref_n_labels)

        self._rs.n_idxs_all_cts = np.zeros(
            self._rs.shape[0] + (self._sett_padd_steps * 2), dtype=np.uint64)

        self._rs.n_idxs_acpt_cts = np.zeros(
            self._rs.shape[0] + (self._sett_padd_steps * 2), dtype=np.uint64)

        self._rs.noise = self._sample_from_noise_dist(
            self._data_ref_shape[0] + (self._sett_padd_steps * 2))

        assert np.all(np.isfinite(self._rs.noise)), 'Invalid values in noise!'

        data = self._get_fftma_ts(self._rs.noise)

        assert np.all(np.isfinite(data)), 'Invalid values in data!'

        self._rs.data_tfm = data

        ft = np.fft.rfft(data[
            self._sett_padd_steps:
            self._data_ref_shape[0] + self._sett_padd_steps], axis=0)

        assert np.all(np.isfinite(ft)), 'Invalid values in ft!'

        probs = self._get_probs(data[
            self._sett_padd_steps:
            self._data_ref_shape[0] + self._sett_padd_steps], True)

        self._rs.data = np.empty_like(
            self._data_ref_rltzn_srtd, dtype=np.float64)

        for i in range(self._data_ref_n_labels):
            self._rs.data[:, i] = self._data_ref_rltzn_srtd[
                np.argsort(np.argsort(probs[:, i])), i]

        self._rs.probs = probs

        self._rs.ft = ft
        self._rs.phs_spec = np.angle(ft)
        self._rs.mag_spec = np.abs(ft)

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

        self._prep_sim_aux_flag = True
        return

    def verify(self):

        GTGPrepare._GTGPrepare__verify(self)
        return

    __verify = verify
