# Laplacian Pyramid Networks: A new Approach for Multispectral Pansharpening

Homepage: 

https://chengjin.netlify.app/

https://liangjiandeng.github.io/

- Code for paper: "Laplacian Pyramid Networks: A new Approach for Multispectral Pansharpening, Information Fusion"
- State-of-the-art pansharpening performance


# Dependencies and Installation
- Python 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/))
- TensorFlow 1.14.0
- NVIDIA GPU + CUDA
- Python packages: `pip install numpy scipy h5py`
- TensorBoard


# Dataset Preparation
The datasets used in this paper is WorldView-3 (can be downloaded [here](https://www.maxar.com/product-samples/)), QuickBird (can be downloaded [here](https://earth.esa.int/eogateway/catalog/quickbird-full-archive)) and GaoFen-2 (can be downloaded [here](http://www.rscloudmart.com/dataProduct/sample)). Due to the copyright of dataset, we can not upload the datasets, you may download the data and simulate them according to the paper.


# Get Started
Training and testing codes are in '[codes/](./codes)'. Pretrained model can be found in '[codes/pretrained/](./codes/pretrained)'. All codes will be presented after the paper is completed published. Please refer to `codes/how-to-run.md` for detail description.

# LPPN Architecture
![LPPN_architecture](./figures/LPPN_architecture.png)

FCNN architecture is presented below:

![FCNN_architecture](./figures/FCNN.png)

# Results
### Quantitative results
The following quantitative results is generated from WorldView-3 datasets. A.T. is short for Average running Time for saving spaces in the paper.

![Quantitative_WV3](./results/Quantitative_WorldView3.png)

All quantitative results can be found in '[results/](./results)'.

### Visual Results
The following visual results is generated from WorldView-3 datasets.

![Visual_WV3](./results/Visual_WorldView3.png)

All visual results can be also found in '[results/](./results)'.

# Citation
Citation information will be presented after the paper is fully published.