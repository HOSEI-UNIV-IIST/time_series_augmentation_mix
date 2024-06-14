# Time Series Augmentation

This is a collection of time series data augmentation methods and an example use using PyTorch.

## News

- 2024/06/08: Creating Repository

## Requires

This code was developed in Python 3.11.0. and requires PyTorch 2.x.x

### Normal Install

```
pip install torch==2.3.1 keras==2.13.1 numpy1.26.4' matplotlib==3.9.0 scikit-image==0.22.0 tqdm==4.66.4
```

### Dataset

`main.py` is designed to use the UCR Time Series Archive 2018 datasets. To install the datasets, download the .zip file from https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ and extract the contents into the `data` folder.

## Usage

### Description of Time Series Augmentation Methods

[Augmentation description](./docs/AugmentationMethods.md)

### Jupyter Example

[Jupyter Notebook](./example.ipynb)

### Keras Example

Example: 
To **train** a 1D **VGG** on the **FiftyWords** dataset from the **UCR Time Series Archive 2018** with **4x** the training dataset in **Jittering**, use:

```
python3 main.py --gpus=0 --dataset=CBF --preset_files --ucr2018 --normalize_input --train --save --jitter --augmentation_ratio=4 --model=vgg
```

## Citation

B. K. Iwana and S. Uchida, "An Empirical Survey of Data Augmentation for Time Series Classification with Neural Networks," arXiv, 2020.

```
@article{Iwana_2021,
	doi = {10.1371/journal.pone.0254841},
	url = {https://doi.org/10.1371%2Fjournal.pone.0254841},
	year = 2021,
	month = {jul},
	publisher = {Public Library of Science ({PLoS})},
	volume = {16},
	number = {7},
	pages = {e0254841},
	author = {Brian Kenji Iwana and Seiichi Uchida},
	title = {An empirical survey of data augmentation for time series classification with neural networks},
	journal = {{PLOS} {ONE}}
}
```
