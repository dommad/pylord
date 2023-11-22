"""Initializers for various objects in estimation and validation frameworks"""


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
from typing import Union
from configparser import ConfigParser
import pandas as pd
import scipy.stats as st
import numpy as np
from . import exporter
from ..constants import GROUND_TRUTH_TAGS
from .plot import PlotValidationResults
from ..utils import fetch_instance
from .. import parsers


class Initializer:

    @abstractmethod
    def initialize(self):
        pass


class ValidationInitializer:

    def __init__(self, config: ConfigParser, df_processor) -> None:
        self.config = config
        self.engine = config.get('general', 'engine', fallback='Tide')
        self.filter_score = config.get('general', 'filter_score', fallback='tev')
        self.df_processor = df_processor



class ExporterInitializer:

    def __init__(self, config) -> None:
        self.exporter_name = f"{config.get('estimation', 'exporter', fallback='PeptideProphet').strip()}Exporter"
        self.out_name = config.get('general', 'outname', fallback='test').strip()

    def initialize(self):
        try:
            return fetch_instance(exporter, self.exporter_name)(self.out_name)
        except AttributeError as exc:
            raise ValueError("Unsupported or invalid exporter.") from exc


class ModelInitializer(ABC):

    @abstractmethod
    def initialize(self):
        pass


class CDDModelInitializer(ModelInitializer):

    def __init__(self, parameters: Union[str, pd.DataFrame]) -> None:
        self.param_input = parameters
        self.param_dict = {}

    def initialize(self):
        if isinstance(self.param_input, str):
            self.param_dict = parsers.ParamFileParser(self.param_input).parse()
        elif isinstance(self.param_input, pd.DataFrame):
            self.param_dict = self.param_input
        else:
            raise TypeError("Parameters input format is unsupported.")
        

class DecoyModelInitializer(ModelInitializer):

    def __init__(self, decoy_df: pd.DataFrame, score_column):
        self.decoy_df = decoy_df
        self.score_column = score_column
        self.param_dict = {}

    def initialize(self):
        if not isinstance(self.decoy_df, pd.DataFrame):
            raise TypeError("The input provided is not a pandas DataFrame.")
            
        self.param_dict = self.decoy_df[self.decoy_df['hit_rank'] == 1].groupby('charge')[self.score_column].apply(st.gumbel_r.fit).to_dict()


class LowerOrderModelInitializer(ModelInitializer):

    def __init__(self, parameters: Union[str, pd.DataFrame]) -> None:
        self.param_input = parameters
        self.param_dict = {}

    def initialize(self):
        if isinstance(self.param_input, str):
            self.param_dict = parsers.ParamFileParser(self.param_input).parse()
        # it's possible for the user to provide parameters directly as pandas dataframe
        elif isinstance(self.param_input, pd.DataFrame):
            self.param_dict = self.param_input
        else:
            raise TypeError("Parameters input format is unsupported.")
        

class BootstrapInitializer(Initializer):

    def __init__(self, config):

        self.n_rep = int(config.get('validation.bootstrap', 'num_rep').strip())
        self.p_value_column = config.get('validation.general', 'p_value_type').strip() + "_p_value"
        
        fdr_threshold_data = config.get('validation.bootstrap', 'fdr_thresholds_array').strip().split('_')
        fdr_llim, fdr_ulim, fdr_num_points = [float(x) for x in fdr_threshold_data]
        fdr_num_points = int(fdr_num_points)
        self.fdr_threshold_array = np.linspace(fdr_llim, fdr_ulim, fdr_num_points)


    def initialize(self, df: pd.DataFrame):

        length_df = len(df)

        bootstrap_idxs = [np.random.choice(np.arange(length_df), size=length_df, replace=True) for _ in range(self.n_rep)]
        critical_vals = np.arange(1, length_df + 1) / length_df
        critical_list = [critical_vals * x for x in self.fdr_threshold_array]
        bootstrapped = [df.iloc[x, :] for x in bootstrap_idxs]
        sorted_dfs = (x.iloc[np.argsort(x.p_value.values)] for x in bootstrapped)

        return sorted_dfs, critical_list, GROUND_TRUTH_TAGS['positive'], GROUND_TRUTH_TAGS['negative']

 
class PlotInitializer(Initializer):

    def __init__(self, config) -> None:
        self.config = config
        self.bool_mapping = {"True": True, "False": False}
        self.config_plot = config.get('general.plotting', 'plot_results').strip()


    def initialize(self):

        if self.config_plot not in self.bool_mapping:
            raise ValueError("'plot_results' options has invalid value. Use 'True' or 'False'.")
        
        if self.bool_mapping.get(self.config_plot, True):
            return PlotValidationResults(self.config)
        
        return None
            