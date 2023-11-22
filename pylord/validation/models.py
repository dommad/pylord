"""Null models for calculation of p-values in hypothesis testing"""

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
import scipy.stats as st
import numpy as np
from .. import stat


class PValueCalculator(ABC):

    @abstractmethod
    def calculate_and_add_p_values(self, df_to_modify, score_column, param_dict):
        pass


class SidakCorrectionMixin:

    @staticmethod
    def sidak_correction(df, p_val_column):
        df[f"sidak_{p_val_column}"] = 1 - pow(1 - df[p_val_column].values, df["num_candidates"].values)
        return df


class CDDModel(PValueCalculator, SidakCorrectionMixin):

    @staticmethod
    def calculate_and_add_p_values(df_to_modify, score_column, param_dict):

        pv_column_name = 'CDD_p_value'
        group = df_to_modify[df_to_modify['hit_rank'] == 1].groupby('charge')[score_column]
        for label, items in group:
            df_to_modify.loc[items.index, pv_column_name] = 1 - st.gumbel_r.cdf(items.values, *param_dict[label])
        return df_to_modify, pv_column_name
    

class DecoyModel(PValueCalculator, SidakCorrectionMixin):

    @staticmethod
    def calculate_and_add_p_values(df_to_modify, score_column, param_dict):
        
        pv_column_name = 'Decoy_p_value'
        group = df_to_modify[df_to_modify['hit_rank'] == 1].groupby('charge')[score_column]
        for label, items in group:
            df_to_modify.loc[items.index, pv_column_name] = 1 - st.gumbel_r.cdf(items.values, *param_dict[label])
        
        return df_to_modify, pv_column_name
    

class LowerOrderModel(PValueCalculator, SidakCorrectionMixin):

    @staticmethod
    def calculate_and_add_p_values(df_to_modify, score_column, param_dict):
        
        pv_column_name = 'LowerOrder_p_value'
        group = df_to_modify[df_to_modify['hit_rank'] == 1].groupby('charge')[score_column]
        for label, items in group:
            df_to_modify.loc[items.index, pv_column_name] = 1 - st.gumbel_r.cdf(items.values, *param_dict[label])
        
        return df_to_modify, pv_column_name
    

class EMPosteriorModel:



    def add_posterior_error_prob(self, dfs, params, all_charges, colname='pep_em'):
        """add PEPs from different EM variants to dataframe"""
        # TODO: adding support for PeptideProphet results with CDD or for PEPs obtained from my EM algorithm
        pvs = np.zeros(len(dfs))

        for pos, idx in enumerate(dfs.index):
            cur_tev = dfs.loc[idx, 'tev']
            charge = int(dfs.loc[idx, 'charge'])
            old_mu1, old_beta, old_mu2, old_sigma, old_pi0 = params[charge]

            if charge in all_charges:
                neg = stat.TEVDistribution().pdf(cur_tev, old_mu1, old_beta, 0)
                posit = st.norm.pdf(cur_tev, old_mu2, old_sigma)
                pep = old_pi0*neg/(old_pi0*neg + (1-old_pi0)*posit)
            else:
                pep = 1

            #to prevent inaccurate fit of positive model to mess up the results
            if cur_tev <=0.15:
                pep = 1

            pvs[pos] = pep

        dfs[colname] = pvs

        return dfs
