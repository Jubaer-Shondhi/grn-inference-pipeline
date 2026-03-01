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

4. Install the package in development mode:
```bash
pip install -e .
```

5. (Optional) For R data simulation:
```
Rscript r_scripts/install_dependencies.R
```

### Option 2: Using Conda
1. Clone the repository:
```bash
git clone https://github.com/Jubaer-Shondhi/grn-inference-pipeline
cd grn-inference-pipeline
```

2. Create and activate conda environment using environment.yml:
```bash
conda env create -f environment.yml
conda activate grn-inference
```

3. Install the package:
```bash
pip install -e .
```

4. (Optional) For R data simulation:
```
Rscript r_scripts/install_dependencies.R
```

## Usage

### Data Options

#### Option A: Use Provided Datasets (Quick Start)
The repository includes example datasets in the `data/` folder for testing. **Each complexity folder contains 1 dataset**:
**Note:** By default, the configuration is set to run only the **5_sources** dataset (1 dataset) for testing. This allows you to run the complete pipeline, to verify everything works.

Run immediately:

```bash
python scripts/run_experiment.py
python scripts/generate_plots.py
```

#### Option B: Generate and Use Simulated Data (Optional)
1. Generate new simulated data:

```bash
python scripts/generate_simulated_data.py --config configs/simulation_config.yaml
```

2. Update config to use simulated data:
   - Edit `configs/config.yaml`

3. Run experiment on simulated data:
```bash
python scripts/run_experiment.py
```

```bash
python scripts/generate_plots.py
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

### Custom Configuration
To use a custom config file:

```bash
python scripts/run_experiment.py --config configs/your_config.yaml
```

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

Simulated Datasets (after running `generate_simulated_data.py`)
```
data/simulated/
├── 5_sources/      
├── 10_sources/     
└── 20_sources/   
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
