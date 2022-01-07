'''
Created on Dec 29, 2021

@author: Faizan3800X-Uni
'''

from gnrctsgenr import GTGSettings

from gnrctsgenr.misc import print_sl, print_el


class FFTMASASettings(GTGSettings):

    def __init__(self):

        GTGSettings.__init__(self)

        # Multiple index annealing.
        self._sett_mult_idx_n_beg_idxs = None
        self._sett_mult_idx_n_end_idxs = None
        self._sett_mult_idx_sample_type = None
        self._sett_mult_idxs_red_rate = None

        # Offset steps for decorrelation of beginning from the end of
        # the time series because FFTMA implicitly rolls the entire
        # array while computing the correlation instead of discarding the
        # beginning and end time steps to compute correlation for a given
        # shift in time.
        self._sett_padd_steps = 500

        # Flags.
        self._sett_mult_idx_flag = False
        self._sett_ann_fftma_sa_sett_verify_flag = False
        return

    def set_mult_indices_annealing_settings(
            self,
            n_beg_idxs,
            n_end_idxs,
            sample_type,
            number_reduction_rate):

        '''
        Randomize multiple values of noise instead of just one.

        A random number of indices are generated for each iteration between
        n_beg_idxs and n_end_idxs (both inclusive). These values are adjusted
        if available number of steps are not enough internally, but these
        values are kept.

        Parameters
        ----------
        n_beg_idxs : integer
            Minimum noise values to randomize per iteration.
            Should be > 0.
        n_end_idxs : integer
            Maximum number of noise values to randomize per iteration.
            Should be >= n_beg_idxs.
        sample_type : integer
            How to sample the number of indices generated for each iteration.
            0:  New noise indices are generated randomly between
                n_beg_idxs and n_end_idxs, regardless of the state of
                optimization for each iteration.
            1:  The number of newly generated indices depends on the ratio
                of current iteration number and maximum_iterations.
            2:  The number of newly generated noise indices is reduced
                by multiplying with number_reduction_rate at every
                temperature update iteration.
            3:  The number of newly generated noise indices is proportional
                to the acceptance rate.
        number_reduction_rate : float
            Generated noise indices reduction rate. A value between > 0 and
            <= 1. The same as temperature reduction schedule. Required
            to have a valid value only is sample_type == 2.
        '''

        if self._vb:
            print_sl()

            print('Setting multiple noise indices annealing parameters...\n')

        assert isinstance(n_beg_idxs, int), 'n_beg_idxs not an integer!'
        assert isinstance(n_end_idxs, int), 'n_end_idxs not an integer!'
        assert isinstance(sample_type, int), 'sample_type is not an integer!'

        assert n_beg_idxs > 0, 'Invalid n_beg_idxs!'
        assert n_end_idxs >= n_beg_idxs, 'Invalid n_end_idxs!'

        assert sample_type in (0, 1, 2, 3), 'Invalid sample_type!'

        if sample_type > 0:
            assert n_beg_idxs < n_end_idxs, (
                'n_beg_idxs and n_end_idxs cannot be equal for sample_type '
                '> 0!')

        if sample_type == 2:
            assert isinstance(number_reduction_rate, float), (
                'number_reduction_rate not a float!')

            assert 0 < number_reduction_rate <= 1, (
                'Invalid number_reduction_rate!')

        elif sample_type in (0, 1, 3):
            pass

        else:
            raise NotImplementedError

        self._sett_mult_idx_n_beg_idxs = n_beg_idxs
        self._sett_mult_idx_n_end_idxs = n_end_idxs
        self._sett_mult_idx_sample_type = sample_type

        if sample_type == 2:
            self._sett_mult_idxs_red_rate = number_reduction_rate

        if self._vb:
            print(
                f'Starting multiple noise indices: '
                f'{self._sett_mult_idx_n_beg_idxs}')

            print(
                f'Ending multiple noise indices: '
                f'{self._sett_mult_idx_n_end_idxs}')

            print(
                f'Multiple noise indices sampling type: '
                f'{self._sett_mult_idx_sample_type}')

            print(
                f'Multiple noise indices number reduction rate: '
                f'{self._sett_mult_idxs_red_rate}')

            print_el()

        self._sett_mult_idx_flag = True
        return

    def verify(self):

        GTGSettings._GTGSettings__verify(self)

        self._sett_ann_fftma_sa_sett_verify_flag = True
        return

    __verify = verify
