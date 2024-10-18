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


import logging
from typing import List
from argparse import Namespace
from ..utils import open_config
from .processing import PreProcessing, PValueProcessing, BootstrapProcessing
from .initializer import PlotInitializer


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def run_validation(args: Namespace = None, config_file_path: str = "", input_file_paths: List = [str], parameters_file_paths: List = [str]):
    
    if args:
        config_file_path = args.configuration_file
        input_file_paths = args.input_file
        parameters_file_paths = args.parameters_file

    config = open_config(config_file_path)

    logging.info("Analysis started...")

    master_df = PreProcessing(config).parse_all_files(input_file_paths)
    master_df.loc[(master_df.gt_label == 1) & (master_df.tev < 0.3), 'gt_label'] = 2 # just for testing

    master_df = PValueProcessing(config, parameters_file_paths).add_p_values_from_all_models(master_df)
    all_bootstrap_results = BootstrapProcessing(config).process_bootstrap(master_df)

    logging.info("Analysis finished!")

    run_plots(config, all_bootstrap_results)

    return all_bootstrap_results, master_df



def run_plots(config, all_boostrap_results):

    logging.info("Plotting started...")
    plotter = PlotInitializer(config).initialize()

    if plotter is None:
        return

    plotter.plot_validation_results(all_boostrap_results)
    logging.info("Plotting finished!")

    