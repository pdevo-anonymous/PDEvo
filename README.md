# PDEvo

This repository provides the anonymous review version of **PDEvo**, including the core implementation of the proposed model.

## Overview

PDEvo is designed for long-term time series forecasting. The model captures local dynamic variations through differential patch representations and further enhances the forecasting representation with PDE-inspired evolution and variable-level interaction.

## Repository Structure

```text
PDEvo/
├── Config/
├── data_provider/
├── exp/
├── layers/
├── models/
│   └── PDEvoNet.py
├── utils/
├── README.md
├── requirements.txt
└── run.py
```

## Core Implementation

The main model is implemented in:

```text
models/PDEvoNet.py
```

The repository also includes the training entry, data provider, basic layers, and utility functions required by the model.

## Hyperparameter Settings

The `Config/` directory provides the main hyperparameter settings for different datasets and prediction lengths.

## Usage

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the model:

```bash
python run.py
```

## Acknowledgement

We sincerely acknowledge the THUML Time-Series-Library for its valuable codebase and experimental framework.
