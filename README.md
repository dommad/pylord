
# PyLord

Lower-Order Statistics for FDR Estimation in Proteomics



## Introduction

PyLord (Python Implementation of Lower-Order Statistics for Proteomics) is a Python package for constructing null models to estimate false discovery rates (FDR) in shotgun proteomics, without the need for decoys. The process is supported by the abundance of information extracted from the score distributions of non-top-scoring target peptide-spectrum matches (PSMs). The theoretical connection between lower-order models combined with empirical optimization enables generating null models for top-scoring PSMs and calculation of p-values that can be used as an input to FDR control frameworks such as Benjamini-Hochberg procedure and its variants. More details can be found in this [paper](https://pubs.acs.org/doi/full/10.1021/acs.jproteome.2c00604).
## Installation

You may clone this repo with the source code and necessary scripts:

```bash
git clone github.com/dommad/pylord.git
```

and install the essential dependencies in a virtual environment using pip (with requirements_pip.txt):

```bash
conda create --name <env> python=3.10
conda activate <env>
pip install -r requirements.txt
```

Alternatively, you can pull the Docker image:

```bash
docker pull dominikmadej/pylord
```



    
## Input

PyLord accepts search results from Comet (pep.xml), MSFragger (pepXML) or Tide (txt). The file with PSMs should contain at least 10 top hits for each MS2 spectrum to ensure sufficient accuracy of the lower-order models.

Configuration file (example [here](https://github.com/dommad/pylord/blob/main/configuration.ini)) also needs to be provided.

Optional: To validate the accuracy of FDR estimated using PyLord models, the input must include files with positive, negative, and decoy PSMs, parameter files specifying parameters of null models to be compared (e.g., PyLord and CDD). Details of the procedure can be found in this [paper](https://pubs.acs.org/doi/full/10.1021/acs.jproteome.2c00604).
## Usage

PyLord can be used to generate parameters of top null models via command-line mode:

```bash
python -m pylord.cmd_estimation -c configuration.ini -i sample_data.pep.xml
```

in Python environment:

```python
from pylord.estimation import run_estimation

psms = "./sample_data.pep.xml"
config = "./configuration.ini"

results = run_estimation(config_file_path=config, input_file=psms)
```

or using Docker:

```bash
docker run --rm -v /local/dir/:/output_dir -it pylord pylord.cmd_estimation -c /output_dir/configuration.ini -i /output_dir/sample_data.pep.xml
```

If you use this approach, remember to specify the “output_path” parameter in the configuration file as “/usr/src/app/output/” to make sure the output files will be placed in your “/local/dir/” directory.


Comparison of FDR estimates produced by PyLord models + Benajmini-Hochberg procedure can be validated in a command-line mode:

```bash
python -m pylord.cmd_validation -c configuration.ini -i positive.pep.xml negative.pep.xml decoy.pep.xml -p lower_order_params.txt cdd_params.txt
```

in Python environment:

```python
from pylord.validation import run_validation

input_files = ['./positive.txt', './negative.txt', './decoy.txt']
param_files = ['./lower_order_params.txt', './cdd_params.txt']
config_file = './configuration.ini'

results = run_validation(config_file_path=config_file,
                         input_file_paths=input_files,
                         parameters_file_paths=param_files)
```

or using Docker:

```bash
docker run --rm -v /local/dir/:/output_dir -it pylord pylord.cmd_validation -c /output_dir/configuration.ini -i /output_dir/positive.pep.xml /output_dir/negative.pep.xml /output_dir/decoy.pep.xml -p /output_dir/lower_order_params.txt /output_dir/cdd_params.txt
```
