<!-- # Project description and context -->

# Setup - Installation

[Anaconda](https://www.anaconda.com/download) is used as the package manager (running python Python 3.10.6). To get set up
with an environment, install Anaconda from the link above, and (from this directory) run

```bash
conda env create -f environment.yml
```

This will create an environment named `pasterquality` with all the necessary packages to run the code. To
activate this environment, run

```bash
conda activate pasterquality
```

then to install the src package, run

```bash
python -m pip install -e .
```

# How to get started

1. Place the provided 'data' folder in the root directory of the project. 
2. ```cd src/experiments/examples/```
   - In that folder each dataset variant has a respective boilerplate script for loading the data etc.

### Reproduce RTDL results 
- This part uses the excellent public repository [Revisiting Deep Learning Models for Tabular Data (NeurIPS 2021)](https://github.com/Yura52/tabular-dl-revisiting-models), by Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko.
- The code and data of the results are placex in `src/experiments/revisiting_models/`
- The script files are in `src/experiments/revisiting_models/paster_scripts/` are offered in this repo to automate the process of tuning and evaluating the models
  - For more details on how the tuning and evaluation works, please refer to the comprehensive documentation of the original repository, [here](src/experiments/revisiting_models/README.md)


1. place the provided `data/tabular/numpy` files and `info.json` in `src/experiments/revisiting_models/data/tabular_100_trials_32_batch_size/`
2. Rename `X_train|val|test` into `N_train|val|test`
3. The results can be found in `original_report.ipynb` and `extended_pasterquality_report.ipynb`
4. To run again all models and experiments
    - Vrtual environment creation for the models
        - ```cd src/revisiting_models/paster_scripts```
        - run ```./install_torch_env.sh```
        - run ```./install_tf_env.sh```
    - delete the `tuned` and `tuned_ensemble` folders in `src/experiments/revisiting_models/output_pasterquality/tabular_100_trials_32_batch_size/`
    - run ```./run_all.sh```




<!-- # Project structure -->


<!-- # Cite -->

<!-- If you find this code or papers useful in your research, please consider citing:

```bibtex
TBA
``` -->
# Contact
We are happy to receive any issues, pull requests, questions, or comments you may have. Please feel free to open an issue or contact us at spanagiotou@ceid.upatras.gr

<!-- # License

# Authors -->
