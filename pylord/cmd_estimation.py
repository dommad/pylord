"""Script for command-line estimation of FDR using PyLord"""

# MIT License

# Copyright (c) 2023 Dominik Madej

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


import argparse
from pylord.estimation.run import run_estimation


def display_info():
    """Basic info about the tool"""
    software_name = "pylord"
    description = "Estimation and validation of top null models in shotgun proteomics using lower-order statistics"
    author = "Dominik Madej"
    university = "HKUST"


    # Create a formatted string with the software name, author, and university
    info_text = f"\n{software_name} - {description}\n\nAuthor: {author}\n\nDeveloped at {university}\n"


    # Create a box around the information
    box_width = len(max(info_text.split('\n'), key=len)) + 4
    box_top_bottom = '+' + '-' * (box_width - 2) + '+'
    formatted_info = f"\n{box_top_bottom}\n{info_text}\n{box_top_bottom}\n"

    print(formatted_info)


if __name__ == "__main__":

    display_info()

    arg_parser = argparse.ArgumentParser(description='Estimation of parameters for top null models using lower-order statistics')
    arg_parser.add_argument('-c', '--configuration_file', required=True, type=str,
                            help="Configuration file in TOML format")
    arg_parser.add_argument('-i',  '--input_file', nargs="*", required=True,
                        type=str, help='input file with multiple lower-ranking hits to estimate the parameters of top null models by charge')

    arguments = arg_parser.parse_args()
    print(arguments)

    run_estimation(args=arguments)
