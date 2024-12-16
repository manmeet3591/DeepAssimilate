# DeepAssimilate

This package integrates station data into gridded data using deep learning techniques.

## Installation
```bash
pip install .
```
## Usage
```
from deepassimilate import DataAssimilation

assimilator = DataAssimilation()
result = assimilator.assimilate(gridded_data, station_data)
```
## Literature Review

FUXI-DA (https://arxiv.org/pdf/2404.08522) is a fully DL-native DA framework targeting a simplified, end-to-end pipeline for assimilating satellite data into ML forecasting models.

FENGWU-4DVAR (https://arxiv.org/pdf/2312.12455) merges ML models with a classical 4D-Var setup, preserving the variational optimization approach and its theoretical foundations.

Manshausen et al. (https://arxiv.org/pdf/2406.16947) demonstrate “score-based data assimilation” using a diffusion generative model trained purely from analysis states and then integrating sparse station observations at inference time to produce km-scale analyses, showcasing a zero-shot, generative approach distinct from both purely ML-native end-to-end frameworks and classical variational methods.
