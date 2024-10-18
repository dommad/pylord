"""Plotting results of all analyses"""


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



from typing import Tuple
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from KDEpy import FFTKDE
from .. import stat
from ..utils import largest_factors
from ..constants import TH_BETA, TH_MU
from .optimization_modes import LinearRegressionMode

matplotlib.use('Agg')



class PlotEstimationResults:
    """Plotting functionalities for the analysis of lower-order models"""

    def __init__(self, config, optimal_parameters, parameters_data, results_df, decoy_df) -> None:

        self.filter_score = config.get('general', 'filter_score').strip()
        self.out_name = config.get('general', 'output_path').strip()
        self.file_format = config.get('general.plotting', 'plot_format').strip()
        self.dpi = int(config.get('general.plotting', 'plot_dpi').strip())

        self.optimal_parameters = optimal_parameters
        self.parameters_data = parameters_data
        self.df = results_df
        self.decoy_df = decoy_df

        plt.style.use('default')
        plt.rcParams.update({'font.size': 12,
                             'xtick.labelsize': 10,
                             'ytick.labelsize': 10})
        

    def save_figure(self, fig, core_name):
        """general function for saving the figures"""
        fig.savefig(f"{self.out_name}{core_name}.{self.file_format}", dpi=self.dpi, bbox_inches='tight')



    def plot_mu_beta_scatter(self: Tuple, **kwargs):
        
        expand = lambda df: (df.loc['location', :], df.loc['scale', :])

        plot_kwargs = {'marker': 'o', 'edgecolors': 'k', 'linewidth': 0.5}
        colors = ('#2D58B8', '#D65215')

        charges = self.parameters_data.available_charges

        n_row, n_col, idx_combinations = self.get_optimal_subplot_settings(len(charges))
        fig, axs = plt.subplots(n_row, n_col, figsize=(n_col * 5, n_row * 5), constrained_layout=True)

        for idx, charge in enumerate(charges):
            this_charge_params_dict = self.parameters_data.output_dict[charge]
            ax = axs[idx_combinations[idx]]

            for p_idx, item in enumerate(this_charge_params_dict.items()):
                p_estimator, param_df = item

                if kwargs.get('selected_hits'):
                    param_df = param_df.loc[:, kwargs.get('selected_hits')]

                mu_vals, beta_vals = expand(param_df)
                ax.scatter(mu_vals, beta_vals, color=colors[p_idx], label=p_estimator, **plot_kwargs)

                if kwargs.get('annotation'):
                    self.annotation(ax, mu_vals, beta_vals, colors[p_idx])

                if kwargs.get('linear_regression'):
                    linreg = LinearRegressionMode(param_df).find_best_linear_regression()
                    self.add_linear_regression(ax, mu_vals, linreg, color=colors[p_idx])


            ax.set_xlabel(r"$\mu$")
            ax.set_ylabel(r"$\beta$")
            ax.set_title(f"charge {charge}")

        self.save_figure(fig, f"mubeta_params_annot_{bool(kwargs.get('annotation'))}_lr_{bool(kwargs.get('linear_regression'))}")
    
    @staticmethod
    def add_linear_regression(axes, xs, linreg, color):
        """
        Add fitted linear regression to the mu-beta plot and show the 
        starting parameters as an asterisk
        """
        x_range = np.array([min(TH_MU, *xs) * 0.99, max(TH_MU, *xs) * 1.01])
        y_range = x_range * linreg.slope + linreg.intercept
        axes.plot(x_range, y_range, color=color)
        axes.scatter([TH_MU], [TH_BETA], marker='*', s=100, color='green')


    @staticmethod
    def annotation(axes, x_text, y_text, color):
        """Add text annotation with the hit rank"""
        offset = 2
        for idx, pair in enumerate(zip(x_text, y_text)):
            axes.annotate(idx + offset, (pair[0], pair[1]-0.0002), color=color)

    
    @staticmethod
    def add_axis_labels(axs, n_col, n_row, mode='density'):

        if mode == 'density':
            ylab = 'density'
            xlab = 'TEV'
        elif mode == 'PP':
            ylab = 'empirical CDF'
            xlab = 'theoretical CDF'

        for idx in range(n_col * n_row):
            if idx % n_col == 0:
                axs[divmod(idx, n_col)].set_ylabel(ylab)

            if divmod(idx, n_col)[0] == n_row-1:
                axs[divmod(idx, n_col)].set_xlabel(xlab)

    @staticmethod
    def get_number_lower_hits(param_dict):

        first_key = list(param_dict.keys())[0]
        sample_df = param_dict[first_key]
        if isinstance(sample_df, pd.DataFrame):
            return param_dict[first_key].shape[1]
        else:
            raise TypeError(f"The value in parameters dictionary should be pd.DataFrame, but it is {sample_df}")

    @staticmethod
    def get_optimal_subplot_settings(num_entries):
        """Get the number of rows, columns, and dimensions of the whole figure
        to get the subplots nicely distributed"""

        n_col, n_row = largest_factors(num_entries)
        idx_combinations = [(i, j) for i in range(n_row) for j in range(n_col)]
        if n_row == 1 or n_col == 1:
            return n_row, n_col, [x[1] for x in idx_combinations]
        return n_row, n_col, idx_combinations


    def plot_lower_models(self):
        """Plot density and PP plots for models of lower order TEV distributions"""

        charges = self.parameters_data.available_charges

        for charge in charges:
            this_charge_params_dict = self.parameters_data.output_dict[charge]
            this_charge_df = self.df[self.df['charge'] == charge]

            num_lower_hits = self.get_number_lower_hits(this_charge_params_dict)
            n_row, n_col, idx_combinations = self.get_optimal_subplot_settings(num_lower_hits)
            fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 3, n_row * 3), constrained_layout=True)
           
            for hit_idx in range(num_lower_hits):
                hit_rank = hit_idx + 2 # we skip hit number 1 as it's a mixture
                this_hit_scores = this_charge_df.loc[this_charge_df['hit_rank'] == hit_rank, self.filter_score].values
        
                self.add_lower_model_plot(axes[idx_combinations[hit_idx]], this_hit_scores, this_charge_params_dict, hit_rank)
            
            self.add_axis_labels(axes, n_col, n_row, mode='density')
            self.save_figure(fig, f"lower_models_charge_{charge}")


    @staticmethod
    def add_pp_plot(axs, cur_tevs, parameters_dict, hit_rank):
        """Add P-P plot of the model with rank equal to "hit_rank" to plt.subplots"""

        mle_pars = parameters_dict['mle'][0].loc[:, hit_rank]
        mm_pars = parameters_dict['mm'][0].loc[:, hit_rank]

        mm_pp = stat.TEVDistribution().cdf_asymptotic(cur_tevs, mm_pars[0], mm_pars[1], hit_rank)
        mle_pp = stat.TEVDistribution().cdf_asymptotic(cur_tevs, mle_pars[0], mle_pars[1], hit_rank)
        emp_pp = np.arange(1, len(cur_tevs) + 1) / len(cur_tevs)

        axs.scatter(mle_pp, emp_pp, color='#D65215', s=1)
        axs.scatter(mm_pp, emp_pp, color='#2CB199', s=1)
        axs.plot([0,1], [0,1], color='k')


    @staticmethod
    def add_lower_model_plot(axs, scores, estimator_params: dict, hit_rank):
        """Plotting KDE for all estimation methods for given hit_rank"""

        colors = ('#2D58B8', '#D65215', '#2CB199')

        kde_xs, kde_ys_observed = FFTKDE(bw=0.0005, kernel='gaussian').fit(scores).evaluate(2**8)
        axs.plot(kde_xs, kde_ys_observed, color='grey')
        

        if len(kde_ys_observed) == 0:
            return 0
        
        for idx, (p_estimator, param_df) in enumerate(estimator_params.items()):
            parameters = param_df.loc[:, hit_rank].values
            mu, beta = parameters
            pdf_vals = stat.TEVDistribution().pdf(kde_xs, mu, beta, hit_rank)
            axs.plot(kde_xs, pdf_vals, color=colors[idx], label=p_estimator)

        axs.set_ylim(0,)
        axs.set_title(f"hit {hit_rank}", fontsize=10)


    @staticmethod
    def get_rough_pi0_estimate(scores, mu, beta, hit_rank):

        #pi0 = len(scores[scores < 0.2]) / len(scores)
        xs, kde_observed = FFTKDE(bw=0.01, kernel='gaussian').fit(scores).evaluate(2**8)
        pdf_fitted = stat.TEVDistribution().pdf(xs, mu, beta, hit_rank=hit_rank)
        pvals = 1-st.gumbel_r.cdf(scores, 0.138, 0.02)
        
        pi0s = []
        #for i in np.linspace(0.6, 0.8, 50):
        #    pi0s.append(len(pvals[pvals > i]) / ((1 - i) * len(pvals)))
        
        #pi0 = np.mean(pi0s)
        pi0 = len(pvals[pvals > 0.8]) / ((1 - 0.8) * len(pvals))


        return pi0, xs, kde_observed, pdf_fitted



    def plot_top_model_with_pi0(self):
        """find pi0 estimates for plotting the final models"""

        num_charges = len(self.optimal_parameters)
        colors = ('#2CB199', '#2CB199', '#D65215')

        n_row, n_col, idx_combinations = self.get_optimal_subplot_settings(num_charges)
        fig, axs = plt.subplots(n_row, n_col, figsize=(n_col * 3, n_row * 3), constrained_layout=True)


        for idx, charge in enumerate(self.optimal_parameters):
            top_mask = (self.df['charge'] == charge) & (self.df['hit_rank'] == 1)
            top_scores = self.df.loc[top_mask, self.filter_score].values
            _, (mu, beta) = self.optimal_parameters[charge]

            pi0, xs, kde_observed, pdf_fitted = self.get_rough_pi0_estimate(top_scores, mu, beta, 1) # we only work on top hits

            if self.decoy_df is not None:
                top_mask_d = (self.decoy_df['charge'] == charge) & (self.decoy_df['hit_rank'] == 1)
                top_scores_d = self.decoy_df.loc[top_mask_d, self.filter_score].values
                mu_d, beta_d = st.gumbel_r.fit(top_scores_d)
    
                _, xs_d, kde_observed_d, pdf_fitted_d = self.get_rough_pi0_estimate(top_scores_d, mu_d, beta_d, 1) # we only work on top hits
                axs[idx_combinations[idx]].plot(xs_d,  pi0 * pdf_fitted_d, color='magenta', linestyle='-', label='fitted')
                
                self.plot_qq(charge, top_scores, mu, beta, mu_d, beta_d)


          

            axs[idx_combinations[idx]].fill_between(xs, kde_observed, alpha=0.2, color=colors[0], label='observed')
            axs[idx_combinations[idx]].plot(xs, kde_observed, color=colors[1])
            axs[idx_combinations[idx]].plot(xs,  pi0 * pdf_fitted, color=colors[2], linestyle='-', label='fitted')
            axs[idx_combinations[idx]].set_xlim(0.0, 0.8)
            axs[idx_combinations[idx]].set_ylim(0,)
            axs[idx_combinations[idx]].set_xlabel("TEV")
            axs[idx_combinations[idx]].set_ylabel("density")
            # axs[idx_combinations[idx]].set_title(f"charge {charge}+")

            self.save_figure(fig, f"fitted_top_models_{charge}")



    def plot_decoy_model_with_pi0(self):
        """overlay decoy models on the empirical mixture distributions"""

        num_charges = len(self.optimal_parameters)
        colors = ('#2CB199', '#2CB199', '#D65215')

        n_row, n_col, idx_combinations = self.get_optimal_subplot_settings(num_charges)
        fig, axs = plt.subplots(n_row, n_col, figsize=(n_col * 3, n_row * 3), constrained_layout=True)


        for idx, charge in enumerate(self.optimal_parameters):
            top_mask_d = (self.decoy_df['charge'] == charge) & (self.decoy_df['hit_rank'] == 1)
            top_scores_d = self.decoy_df.loc[top_mask_d, self.filter_score].values
            mu, beta = st.gumbel_r.fit(top_scores_d)

            top_mask_t = (self.df['charge'] == charge) & (self.df['hit_rank'] == 1)
            top_scores_t = self.df.loc[top_mask_t, self.filter_score].values

            

            pi0, xs, kde_observed, pdf_fitted = self.get_rough_pi0_estimate(top_scores_t, mu, beta, 1) # we only work on top hits
            pi0, xs, kde_observed, pdf_fitted = self.get_rough_pi0_estimate(top_scores_t, mu, beta, 1) # we only work on top hits

            axs[idx_combinations[idx]].fill_between(xs, kde_observed, alpha=0.2, color=colors[0], label='observed')
            axs[idx_combinations[idx]].plot(xs, kde_observed, color=colors[1])
            axs[idx_combinations[idx]].plot(xs,  pi0 * pdf_fitted, color=colors[2], linestyle='-', label='decoy')
            axs[idx_combinations[idx]].set_xlim(0.0, 0.6)
            axs[idx_combinations[idx]].set_ylim(0,)
            axs[idx_combinations[idx]].set_xlabel("TEV")
            axs[idx_combinations[idx]].set_ylabel("density")

        self.save_figure(fig, "fitted_top_models_decoy")


    def plot_qq(self, charge, observed, lower_mu, lower_beta, decoy_mu, decoy_beta):

    
        (osr_l, osm_l), (slope, intercept, r) = st.probplot(observed, dist=st.gumbel_r, sparams=(lower_mu, lower_beta))
        (osr_d, osm_d), (slope, intercept, r) = st.probplot(observed, dist=st.gumbel_r, sparams=(decoy_mu, decoy_beta))

        fig, ax = plt.subplots(figsize=(4, 4))
        # Plot the data points with custom marker size and color
        ax.plot(osm_l, osr_l, color='blue', label='PyLord')
        ax.plot(osm_d, osr_d, color='orange', label='decoy')

        # Plot the reference line
        #ax.plot(osm, slope*osm + intercept, 'r--', lw=2, label='Fit Line')
        ax.plot([min(osm_l[0], osr_l[0], osm_d[0], osr_d[0]), max(osm_l[-1], osr_l[-1], osm_d[-1], osr_d[-1])], 
                [min(osm_l[0], osr_l[0], osm_d[0], osr_d[0]), max(osm_l[-1], osr_l[-1], osm_d[-1], osr_d[-1])],
                color='grey', label='x=y')
        
        ax.set_ylabel('tested quantiles (null model)')
        ax.set_xlabel('reference quantiles (observed data)')
        ax.set_xlim(0.9*min(osm_l[0], osm_d[0]), 0.3)
        ax.set_ylim(0.9*min(osr_l[0], osr_d[0]), 0.3)
        ax.legend()
        self.save_figure(fig, f"qq_plot_models_charge_{charge}")
