"""Plotting results of all analyses"""


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


from typing import Tuple, List
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class PlotValidationResults:

    def __init__(self, config):
        self.out_name = config.get('general', 'output_path').strip()
        self.file_format = config.get('general.plotting', 'plot_format').strip()
        self.dpi = int(config.get('general.plotting', 'plot_dpi').strip())
        self.labels = config.get('validation.general', 'p_value_type').strip().split("_")


    def plot_validation_results(self, all_bootstrap_results):
        """Plot validation results"""

        cs_ = ['#2D58B8', '#D65215', '#2CB199', '#7600bc']
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

        for idx, bootstrap_stats in enumerate(all_bootstrap_results):

            fdps, tprs = bootstrap_stats
            self.plot_fdp_fdr_cis(axs[0], fdps, cs_[idx])
            self.plot_tpr_fdr_cis(axs[1], tprs, cs_[idx])
        
        self.save_figure(fig, "fdp_tpr_fdr_validation")

    @staticmethod
    def plot_fdp_fdr_cis(axs, fdps: List[Tuple], color: str):
        """Plotting the FDR vs. estimated FDP + bootstrapped CIs"""

        plt.style.use('ggplot')
        plt.rcParams.update({'font.size': 12, 'font.family': 'Arial',
                                'xtick.labelsize': 10, 'ytick.labelsize': 10})

        support = np.linspace(0.001, 0.1, 100) # TODO: abstract out
        means, upper_lims, lower_lims = (np.array(x) for x in zip(*fdps))

        sns.lineplot(x=support, y=means, color=color,ax=axs)
        sns.lineplot(x=[0, 0.1], y=[0, 0.1], linestyle='--', color='gray', ax=axs)
        axs.fill_between(support, upper_lims, lower_lims, color=color, alpha=0.3)

        axs.set_xlim(0, 0.1)
        axs.set_ylim(0, 0.1)
        # Customize the plot
        axs.set_xlabel('Estimated False Discovery Rate')
        axs.set_ylabel('False Discovery Proportion')



    @staticmethod
    def plot_tpr_fdr_cis(axs, tprs: List[Tuple], color):
        """Plotting the FDR vs. estimated FDP + bootstrapped CIs"""


        plt.style.use('ggplot')
        plt.rcParams.update({'font.size': 12, 'font.family': 'Arial',
                                'xtick.labelsize': 10, 'ytick.labelsize': 10})

        support = np.linspace(0.001, 0.1, 100) # TODO: abstract out
        means, upper_lims, lower_lims = list(np.array(x) for x in zip(*tprs))

        sns.lineplot(x=support, y=means, color=color, ax=axs)
        #sns.lineplot(x=[0, 0.1], y=[0, 0.1], linestyle='--', color='gray', ax=axs)
        axs.fill_between(support, upper_lims, lower_lims, color=color, alpha=0.3)

        # Customize the plot
        axs.set_ylabel('True Positive Rate')
        axs.set_xlabel('False Discovery Rate')


    def save_figure(self, fig, core_name):
        """general function for saving the figures"""
        fig.savefig(f"{self.out_name}{core_name}.{self.file_format}", dpi=self.dpi)




