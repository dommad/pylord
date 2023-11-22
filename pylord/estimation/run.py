"""Full analysis of pepxml file using lower order statistics"""


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


from argparse import Namespace
import logging
from ..utils import open_config
from .processing import DataFrameProcessor
from .initializer import ExporterInitializer, EstimationInitializer, PlotInitializer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def run_estimation(args: Namespace = None, config_file_path: str = "configuration.ini", input_file: str = "example.pep.xml"):
    """The main function running estimation of distribution parameters
    for top-scoring target PSMs using lower-order statistics"""
    logging.info("Starting the analysis...")

    if args:
        config_file_path = args.configuration_file
        input_file = args.input_file

    config = open_config(config_file_path)

    df_processor = DataFrameProcessor
    init = EstimationInitializer(config, df_processor)

    parser = init.initialize_parser()
    df = parser().parse(input_file)

    parameter_estimators = init.initialize_parameter_estimators()
    para_init = init.initialize_param_processing(df)
    parameters_data = para_init.process_parameters_into_charge_dicts(parameter_estimators)

    opt_finder = init.initialize_optimal_models_finder(parameters_data)
    opt_modes = init.initialize_optimization_modes()

    optimal_results = opt_finder.find_parameters_for_best_estimation_optimization(df, df_processor, opt_modes)
    optimal_parameters = opt_finder.get_charge_best_combination_dict(optimal_results)

    exporter_object = ExporterInitializer(config).initialize()
    exporter_object.export_parameters(optimal_parameters)

    run_plots(config, optimal_parameters, parameters_data, df)

    logging.info("Analysis finished!")

    return optimal_parameters, parameters_data, df


def run_plots(config, optimal_parameters, parameters_data, df):

    logging.info("Plotting started...")

    plotter = PlotInitializer(config).initialize(optimal_parameters, parameters_data, df)

    if plotter is None:
        return

    plotter.plot_lower_models()
    plotter.plot_top_model_with_pi0()
    plotter.plot_mu_beta_scatter()
    plotter.plot_mu_beta_scatter(linear_regression=True, annotation=True)

    logging.info("Plotting finished!")
