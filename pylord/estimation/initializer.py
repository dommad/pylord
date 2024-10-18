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
from .processing import OptimalModelsFinder, ParametersProcessing
from . import estimators, optimization_modes, exporter
from .. import parsers
from ..utils import fetch_instance
from .plot import PlotEstimationResults


class Initializer:

    @abstractmethod
    def initialize(self):
        pass
    

class EstimationInitializer:

    def __init__(self, config: ConfigParser, df_processor) -> None:
        self.config = config
        self.engine = config.get('general', 'engine', fallback='Tide')
        self.filter_score = config.get('general', 'filter_score', fallback='tev')
        self.df_processor = df_processor


    def initialize_parser(self):

        return fetch_instance(parsers, f"{self.engine}Parser")


    def initialize_parameter_estimators(self):
        estimator_names = self.config.get('estimation', 'estimators').strip().split(', ')
        estimator_objects = [getattr(estimators, x) if hasattr(estimators, x) else AttributeError(f"{x} not found in estimators module!") for x in estimator_names]
        return estimator_objects


    def initialize_param_processing(self, df):
        return ParametersProcessing(self.config, df, self.df_processor, self.filter_score)


    def initialize_optimal_models_finder(self, param_dict):
        return OptimalModelsFinder(self.config, param_dict, self.filter_score)


    def initialize_optimization_modes(self):
        opt_mode_names = self.config.get('estimation', 'optimization_modes').strip().split(', ')
        optimization_objects = [fetch_instance(optimization_modes, f"{x}Mode") for x in opt_mode_names]
        return optimization_objects
    


class ValidationInitializer:

    def __init__(self, config: ConfigParser, df_processor) -> None:
        self.config = config
        self.engine = config.get('general', 'engine', fallback='Tide')
        self.filter_score = config.get('general', 'filter_score', fallback='tev')
        self.df_processor = df_processor



class ExporterInitializer:

    def __init__(self, config) -> None:
        self.exporter_name = f"{config.get('estimation', 'exporter', fallback='PeptideProphet').strip()}Exporter"
        self.out_name = config.get('general', 'output_path', fallback='./').strip()

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
 
 
class PlotInitializer(Initializer):

    def __init__(self, config) -> None:
        self.config = config
        self.bool_mapping = {"True": True, "False": False}
        self.config_plot = config.get('general.plotting', 'plot_results').strip()


    def initialize(self, optimal_parameters, parameters_data, df, decoy_df):

        if self.config_plot not in self.bool_mapping:
            raise ValueError("'plot_results' options has invalid value. Use 'True' or 'False'.")
        
        if self.bool_mapping.get(self.config_plot, True):
            return PlotEstimationResults(self.config, optimal_parameters, parameters_data, df, decoy_df)
        
        return None
            

