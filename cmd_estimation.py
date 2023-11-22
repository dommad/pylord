

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


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description='Estimation of parameters for top null models using lower-order statistics')
    arg_parser.add_argument('-c', '--configuration_file', required=True, type=str,
                            help="Configuration file in TOML format")
    arg_parser.add_argument('-i',  '--input_file', required=True,
                        type=str, help='input file with multiple lower-ranking hits to estimate the parameters of top null models by charge')

    arguments = arg_parser.parse_args()
    run_estimation(args=arguments)