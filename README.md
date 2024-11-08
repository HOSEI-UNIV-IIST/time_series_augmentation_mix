
# Time Series Forecasting Project

This project focuses on time series forecasting using neural networks. The approach involves data augmentation, custom deep learning models, and hyperparameter optimization. The main objective is to forecast multiple days ahead by analyzing several prior days of data.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Techniques](#models-and-techniques)
- [Datasets](#datasets)
- [Acknowledgments](#acknowledgments)
- [Author](#author)

---

## Overview

### Key Components
1. **Data Augmentation**:
   - Techniques include time and magnitude transformations based on:
     > B. K. Iwana and S. Uchida, "An Empirical Survey of Data Augmentation for Time Series Classification with Neural Networks," arXiv, 2020.

2. **Custom Models**:
   - Several deep learning architectures are defined in `custom_models.py`, including:
     - CNN, LSTM, GRU
     - Hybrid CNN-RNN models
     - Attention-based models

3. **Hyperparameter Tuning**:
   - Using `tuning.py`, Optuna explores a defined search space to find optimal model parameters.

4. **Training and Evaluation**:
   - The `training.py` script trains and tests models with customizable configurations, managed via **ArgumentParser**.

5. **Datasets**:
   - Data is sourced from **GNOME Project 2** and organized into `cleaned` and `final` versions for ease of use.

### Forecasting Objective
- Forecasting future time steps based on a specified lookback period (e.g., given 7 days of past data, forecast the next 6 days).

---

## Project Structure

The project directory includes the following key components:

```plaintext
├── README.md               # Project documentation
├── config                  # Configuration files for hyperparameters and run parameters
├── data                    # Dataset files in 'cleaned' and 'final' folders
├── docs                    # Documentation files
├── models                  # Custom model definitions, training, and tuning scripts
├── notebooks               # Jupyter notebooks for data exploration
├── utils                   # Utility scripts for data processing and augmentation
├── logs                    # Log files for model training
├── output                  # Model outputs and results
├── run_gnome.sh            # Script for running the project
└── requirements.txt        # Required dependencies
```

For a detailed view of the directory structure, see `docs/project_structure.txt`.

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### Step 2: Install Python Packages

To set up the environment, you can install all dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Or manually install the key dependencies as listed below.

#### Core Libraries

```bash
# PyTorch Installation (adjust for your CUDA version if using a GPU)
# For CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.4
pip3 install torch torchvision torchaudio

# Other Required Libraries
pip install numpy==1.26.4 matplotlib~=3.8.0 tqdm==4.66.4 einops==0.8.0
pip install plotly~=5.22.0 kaleido==0.2.1 scipy==1.14.0 scikit-learn~=1.4.1.post1 pandas
pip install optuna~=3.6.0 shap~=0.46.0 lime~=0.2.0.1 pyyaml
```

---

## Usage

1. **Configuration**:
   - Adjust configurations in `config/hyperparameters.yml` and `config/run_params.yaml` to set model parameters and run settings.

2. **Running the Project**:
   - Use `run_gnome.sh` to train models with specified configurations:
     ```bash
     bash run_gnome.sh
     ```

3. **Training and Evaluation**:
   - `training.py` handles both training and testing, allowing dynamic model selection and parameter customization via command-line arguments.

4. **Hyperparameter Tuning**:
   - Use `tuning.py` to perform hyperparameter tuning with Optuna:
     ```bash
     python tuning.py
     ```

---

## Models and Techniques

- **Data Augmentation**: Implemented in `augmentation.py` with various time and magnitude transformations.
- **Neural Network Architectures**:
  - CNN, LSTM, GRU models, and hybrid configurations such as `cnn_lstm`, `cnn_attention_bigru`, and others.
- **Optimization**:
  - Optuna is used in `tuning.py` to search for the best hyperparameters within a specified space.
- **Forecasting Objective**:
  - Predicts future time steps based on a configurable lookback window, such as 7 days back for a 6-day forecast.

---

## Datasets

Data is sourced from the **GNOME Project 2**, which includes `cleaned` and `final` versions organized in the `data/` directory. Notebooks for dataset exploration are available in the `notebooks/` directory.

---

## Acknowledgments

This project references techniques and methods from:
- B. K. Iwana and S. Uchida, "An Empirical Survey of Data Augmentation for Time Series Classification with Neural Networks," arXiv, 2020.

---

## Author

Created on 11/05/2024  
🚀 Welcome to the Awesome Python Script 🚀

- **User**: messou
- **Email**: mesabo18@gmail.com / messouaboya17@gmail.com
- **GitHub**: [https://github.com/mesabo](https://github.com/mesabo)
- **University**: Hosei University  
- **Department**: Science and Engineering  
- **Lab**: Prof YU Keping's Lab  
