"""Validation module"""

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


from typing import Generator, List
import numpy as np
import pandas as pd
from .. import parsers
from ..constants import GROUND_TRUTH_TAGS
from .models import PValueCalculator, SidakCorrectionMixin
from ..utils import ParserError, fetch_instance
from .. import cythonized as cyt
from . import estimators, fdr, models, initializer


class PreProcessing:

    def __init__(self, config) -> None:
        self.engine = config.get('general', 'engine', fallback='Tide')

    def parse_all_files(self, input_files: List[str]) -> pd.DataFrame:
        """Helper function to parse all files"""

        parser_instance = fetch_instance(parsers, f"{self.engine}Parser")()

        all_dfs = []

        for file in input_files:

            file_tag = self.get_ground_truth_label(file, GROUND_TRUTH_TAGS)

            try:
                psms_df = parser_instance.parse(file)
            except ParserError as err:
                print(f"Error parsing {file} with {self.engine} parser: {err}")
                continue

            psms_df = self.add_ground_truth_label(psms_df, file_tag)
            all_dfs.append(psms_df)

        master_df = pd.concat(all_dfs, axis=0, ignore_index=True)

        return master_df[master_df['hit_rank'] == 1]


    @staticmethod
    def get_ground_truth_label(file_name, file_tags):
        """Extracting ground truth label based on the name of the file"""
        label = next((k for k in file_tags.keys() if k in file_name.lower()), 'unidentified')
        return file_tags[label]

    @staticmethod
    def add_ground_truth_label(df: pd.DataFrame, file_tag: int) -> pd.DataFrame:
        """Adding ground truth label to the dataframe"""
        df['gt_label'] = file_tag * np.ones(len(df))
        return df



class BootstrapProcessing:

    def __init__(self, config) -> None:

        self.config = config

        pi_zero_name = config.get('validation.general', 'pi_zero_method').strip()

        if pi_zero_name == "":
            self.pi_zero_calculator = None
        else:
            self.pi_zero_calculator = pi_zero_name + "PiZero"


    def process_bootstrap(self, master_df):

        bootstrap_results = self.initialize_and_run_bootstrap(master_df)
        tprs, fdps = self.extract_bootstrap_fdps_tprs(bootstrap_results)

        if self.pi_zero_calculator is not None:
            fdps = self.adjust_with_pi_zero(master_df, fdps)
        
        bootstrap_stats = ConfidenceInterval(self.config).calculate_all_confidence_intervals(fdps, tprs)
        
        return bootstrap_stats


    @staticmethod
    def extract_bootstrap_fdps_tprs(results: Generator):
        """extract FDP and TPR values from the bootstrapping output"""

        fdps_tprs = []

        while True:
            try:
                result = next(results)
                fdps_tprs.append(result)
            except StopIteration:
                break

        tprs, fdps = list(zip(*fdps_tprs))

        return np.array(fdps), np.array(tprs)


    def initialize_and_run_bootstrap(self, df):
        """Initialize the Bootstrap instance and run the bootstrap"""
        bootstrap_instance = Bootstrap(self.config, initializer.BootstrapInitializer)
        
        return bootstrap_instance.run_bootstrap(df)


    def adjust_with_pi_zero(self, master_df, fdps):
        """Adjust the FDP values for pi_0, e.g. to compensate for conservative
        nature of Benjamini-Hochberg procedure"""
        pi_zero_calculator_instance = fetch_instance(estimators, self.pi_zero_calculator)
        pi_zero = pi_zero_calculator_instance.calculate_pi_zero(master_df)
        fdps *= (1 / pi_zero)
        
        return fdps



class PValueProcessing:

    def __init__(self, config, parameters_files: List[str] = None):
        self.filter_score = config.get('general', 'filter_score', fallback='tev').strip()
        self.p_value_calculators = config.get('validation.general', 'null_model').strip().split('_')

        if parameters_files is not None:
            self.parameters_files_dict = dict((x.split('/')[-1].split('_')[0], x) for x in parameters_files)


    def add_p_values_from_all_models(self, master_df):
        """Processing all selected p-value calculation models and appending the
        resulting p-values to the original dataframe"""

        for model in self.p_value_calculators:
            
            param_df = None

            if model == 'Decoy':
                param_df = master_df[master_df['gt_label'] == GROUND_TRUTH_TAGS['decoy']]

            parameters_file = self.parameters_files_dict.get(model, None)
            master_df = self.generate_and_append_p_values(master_df, model, param_df=param_df, parameters_file=parameters_file)

        return master_df
    

    def generate_and_append_p_values(self, df_to_modify: pd.DataFrame, model: PValueCalculator, sidak: bool = False, param_df: pd.DataFrame = None, parameters_file: str = None):
        """Initialize model and use it to calculate p-values for the provided dataframe"""
        
        if param_df is not None:
            init_instance = fetch_instance(initializer, f"{model}ModelInitializer")(param_df, self.filter_score)
        elif parameters_file is not None:
            init_instance = fetch_instance(initializer, f"{model}ModelInitializer")(parameters_file)
        else:
            raise ValueError("No dataframe provided or parameter file is not found, so cannot initialize the model for p-value calculation.")
        
        init_instance.initialize()
        model_instance = fetch_instance(models, f"{model}Model")
        df_with_pvs, pv_column_name = model_instance.calculate_and_add_p_values(df_to_modify, self.filter_score, init_instance.param_dict)

        if issubclass(model_instance.__class__, SidakCorrectionMixin) and sidak:
            df_with_pvs = model_instance.sidak_correction(df_with_pvs, pv_column_name)

        return df_with_pvs



class Bootstrap:
    """Bootstrap"""
    
    def __init__(self, config, init: initializer.Initializer) -> None:

        self.init = init(config)
        self.n_rep = int(config.get('validation.bootstrap', 'num_rep').strip())
       
        self.p_value_column = config.get('validation.general', 'p_value_type').strip() + "_p_value"
        self.fdr_calculator = fetch_instance(fdr, config.get('validation.general', 'fdr_method').strip())

    def run_bootstrap(self, df):
        """main method to run bootstrap"""
        df_sorted, critical_array, pos_label, neg_label = self.init.initialize(df)
        return (self.fdr_calculator.calculate_fdp_tpr(next(df_sorted), self.p_value_column, critical_array, pos_label, neg_label) for _ in range(self.n_rep))



class ConfidenceInterval:
    """Calculating confidence intervals on the bootstrapped FDR and FDP"""

    def __init__(self, config) -> None:

        self.ci_alpha = float(config.get('validation.bootstrap', 'confidence_interval_alpha').strip())
        fdr_num_points = int(config.get('validation.bootstrap', 'fdr_thresholds_array').strip().split('_')[-1])
        self.fdr_num_range = range(fdr_num_points)


    def get_confidence_interval(self, data, idx):
        """obtain CIs from empirical bootstrap method"""

        mean = np.mean(data[:, idx])
        diff = sorted([el - mean for el in data[:, idx]])
        ci_upper_bound = mean - diff[int(len(diff) * self.ci_alpha / 2)]
        ci_lower_bound = mean - diff[int(len(diff) * (1 - self.ci_alpha / 2))]

        return mean, ci_lower_bound, ci_upper_bound


    def get_confidence_interval_cython(self, data, idx):
        """obtain CIs from empirical bootstrap method"""
        return cyt.get_confidence_intervals(data, idx, self.ci_alpha)
    

    def calculate_all_confidence_intervals(self, fdps, tprs):
        """calculate CIs for both FDP and TPR"""

        fdp_cis = list(self.get_confidence_interval_cython(fdps, x) for x in self.fdr_num_range)
        tpr_cis = list(self.get_confidence_interval_cython(tprs, x) for x in self.fdr_num_range)

        return np.array(fdp_cis), np.array(tpr_cis)
