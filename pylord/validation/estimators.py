"""Estimators of pi0 (the fraction of incorrect PSMs in among all top-scoring PSMs)"""

# MIT License

# Copyright (C) 2023 Dominik Madej

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import scipy.stats as st


class PiZeroEstimator(ABC):

    @abstractmethod
    def calculate_pi_zero(self):
        pass


class BootstrapPiZero(PiZeroEstimator):
    """estimate pi0 for given set of p-values"""

    def __init__(self):
        pass


    def calculate_pi_zero(self, pvs, n_reps):
        pi0_estimates = np.array(self.get_all_pi0s(pvs))
        pi0_ave = np.mean(pi0_estimates)
        b_set = [5, 10, 20, 50, 100]

        mses = [self.get_mse(self.get_bootstrap_pi0s(pvs, n_reps, b_val), pi0_ave) for b_val in b_set]

        optimal_idx = np.argmin(mses)
        return pi0_estimates[optimal_idx]

    @staticmethod
    def get_pi0_b(pvs, b_val):
        i = 1

        while True:
            t_i = (i - 1) / b_val
            t_iplus = i / b_val
            ns_i = np.sum((t_i <= pvs) & (pvs < t_iplus))
            nb_i = np.sum(pvs >= t_i)

            if ns_i <= nb_i / (b_val - i + 1):
                break

            i += 1

        i -= 1
        t_values = [(j - 1) / b_val for j in range(i, b_val + 1)]
        pi_0 = np.sum([np.sum(pvs >= t) / ((1 - t) * len(pvs)) for t in t_values]) / (b_val - i + 2)

        return pi_0


    def get_all_pi0s(self, pvs):
        b_set = [5, 10, 20, 50, 100]
        return [self.get_pi0_b(pvs, b_val) for b_val in b_set]


    def get_bootstrap_pi0s(self, pvs, no_reps, b_val):
        return np.array([self.get_pi0_b(np.random.choice(pvs, size=len(pvs)), b_val) for _ in range(no_reps)])


    @staticmethod
    def get_mse(pi0_bootstrap, pi0_true):
        return np.mean((pi0_bootstrap - pi0_true)**2)


class TruePiZero(PiZeroEstimator):

    @staticmethod
    def calculate_pi_zero(df):
        pi_0 = df['gt_label'].value_counts().get(0, 0) / len(df['gt_label'])
        return pi_0


class CoutePiZero(PiZeroEstimator):
    
    @staticmethod
    def calculate_pi_zero(coute_pvs):
        """Get pi0 estimate using the graphical method describe in Coute et al."""
        compl_pvs = 1 - coute_pvs
        sorted_compls = np.sort(compl_pvs)
        dfs = pd.DataFrame(sorted_compls, columns=['score'])
        dfs.index += 1
        dfs['cdf'] = dfs.index/len(dfs)

        l_lim = int(0.4*len(dfs))
        u_lim = int(0.6*len(dfs))
        lr_ = st.linregress(dfs['score'][l_lim:u_lim], dfs['cdf'][l_lim:u_lim])
        return lr_.slope
    
