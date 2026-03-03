# GRN Inference Pipeline

A modular, reproducible pipeline for Gene Regulatory Network inference using XGBoost with various objectives and distributions.

## Features
- **Modular design**: Separated data loading, preprocessing, modeling, and evaluation
- **Hyperparameter tuning**: Multi-stage exploration (single, pairs, triples, refined)
- **Multiple objectives**: Support for both standard XGBoost objectives and XGBoost-Distribution
- **Parallel execution**: Efficient parallel processing of gene targets
- **Reproducible**: Full configuration management and logging

## Installation

### Option 1: Using pip
1. Clone the repository:
```bash
git clone https://github.com/Jubaer-Shondhi/grn-inference-pipeline
cd grn-inference-pipeline
```

2. Create and activate virtual environment:

Linux:
```bash
python -m venv venv
source venv/bin/activate  
```
Windows (Gitbash)
```bash
python -m venv venv
source venv/Scripts/activate
```

3. Install dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

## Usage

### Data Options

#### Option A: Use Provided Datasets (Quick Start)
The repository includes example datasets in the `data/` folder for testing. **Each complexity folder contains 1 dataset**:
**Note:** By default, the configuration is set to run only the **5_sources** dataset (1 dataset) for testing. This allows you to run the complete pipeline, to verify everything works.

Run immediately:

```bash
python scripts/run_experiment.py
```

### Generate Plots
After running the experiment, generate visualization plots:

```bash
python scripts/generate_plots.py
```

The script generates:
- Precision curves for all objectives
- Stage-wise precision comparisons
- Best configurations tables (Stages 1-3)
- Stage 4 configurations by network complexity
- Top objectives bar charts at key thresholds

All plots are saved in `results/figures/`.

## Project Structure
```
├── configs/                    # Configuration files (.yaml)
│   ├── config.yaml             # Main configuration
│   ├── objectives.yaml         # Objectives and distributions
│   └── simulation_config.yaml   # Simulation parameters
├── data/                       # Simulated/example datasets
│   ├── 5_sources/
│   ├── 10_sources/
│   ├── 20_sources/
│   └── simulated/ 
├── r_scripts/                  # R code for data simulation
│   ├── install_dependencies.R
│   └── simulate_data.R
├── results/                    # Generated outputs (populated after running)
│   └── .gitkeep                # (placeholder to keep empty folder in Git)
├── scripts/                    # Main executable scripts
│   ├── generate_plots.py
│   ├── generate_simulated_data.py
│   └── run_experiment.py
├── src/                        # Source code modules
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── simulation_bridge.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── models/
│   │   └── inference.py
│   ├── pipeline/
│   │   └── stages.py
│   └── utils/
│       ├── config_manager.py
│       ├── logger.py
│       └── parallel.py
├── .gitignore
├── environment.yml
├── requirements.txt
└── README.md
```                  

## Output Structure
After running the pipeline, results are organized as:
```
results/
├── precision_metrics.csv           # All precision results
├── GRN_Inference_RESULTS.csv       # Raw results
└── figures/                        # Generated plots
    ├── precision_curves.pdf
    ├── precision_curves_stagewise.pdf
    ├── stages_1to3_configs.pdf
    ├── stage4_by_complexity.pdf
    └── top_objectives_bar.pdf
```
**Note:** The `.gitkeep` file is only to keep the empty folder in Git. It does not affect the pipeline and will remain after generating results.

## Large-Scale Datasets (Optional)

For users who want to run experiments with larger datasets, I have pre-simulated **1000 datasets** available on FAUBOX.

### How to Use

1. **Download** the datasets from FAUBOX
2. **Extract** to the `data/simulated/` folder:
```bash
# Create simulated folder if it doesn't exist
mkdir -p data/simulated
```

3. **Uploaded** the datasets in data/simulated folder for 5, 10 and 20_sources. The structure should look like:
```
data/simulated/
├── 5_sources/
│   ├── data/
│   │   ├── data_1.tsv
│   │   ├── data_2.tsv
│   │   └── ... (333 files)
│   └── nets/
│       ├── network_1.tsv
│       ├── network_2.tsv
│       └── ... (333 files)
├── 10_sources/
└── 20_sources/
```

4. **Update** config to use the large datasets:

```yaml
# In configs/config.yaml
paths:
  base_data: "data/simulated"  # Change from "data" to "data/simulated"
```
5. **Run** the pipeline as usual:

```bash
python scripts/run_experiment.py
python scripts/generate_plots.py
```

## Configuration
Edit configs/config.yaml to modify:
- Data paths: Input data and output directories
- Model hyperparameters: Base parameters and grids
- Pipeline settings: Batch size, workers, thresholds
- Logging options: Level and save frequency

## Reproducibility
- All parameters stored in YAML configuration files
- R scripts for complete data simulation
- Full pipeline from data generation to visualization
- Version-controlled with Git
- Fixed random seeds for all stochastic processes

## Requirements
- Python: 3.10 or higher
- R: 4.0 or higher (optional, for data simulation)
- Operating System: Windows/Linux/macOS

Main Python packages:
- pandas, numpy, scikit-learn
- xgboost, xgboost-distribution
- scanpy, matplotlib, seaborn
- pyyaml

