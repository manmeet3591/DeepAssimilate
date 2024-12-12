# DeepAssimilate

This package integrates station data into gridded data using deep learning techniques.

## Installation
```bash
pip install .

## Usage

from data_assimilation_package import DataAssimilation

assimilator = DataAssimilation()
result = assimilator.assimilate(gridded_data, station_data)
