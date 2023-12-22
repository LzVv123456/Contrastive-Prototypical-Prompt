# Steering Prototypes with Prompt-tuning for Rehearsal-free Continual Learning

## 1. Introduction

*This repository contains the official source code for the research paper titled "Steering Prototypes with Prompt-tuning for Rehearsal-free Continual Learning". Please refer to the paper https://arxiv.org/abs/2303.09447 for detailed methods*

## 2. Setting Up the Environment

To ensure reproducibility and smooth execution of the code, we recommend setting up a dedicated environment using `conda`.

### Steps:

1. First, make sure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.
2. Create a new conda environment:

    ```bash
    conda create --name your_env_name python=3.9
    ```

3. Activate the environment:

    ```bash
    conda activate your_env_name
    ```

4. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## 3. Preparing and Downloading Datasets
1. For CIFAR-100 and 5datasets, they should download automatically when running the code.
2. For ImageNet-R, download and unzip the original data from https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar. Please also download the data split from  https://drive.google.com/drive/folders/1D5ADrPs9OweevMNA-ZjuAGdbilYfA1io?usp=sharing.
3. For ImageNet-Sub, download and unzip from https://drive.google.com/file/d/1n5Xg7Iye_wkzVKc0MTBao5adhYSUlMCL/view?usp=sharing
4. Save all dowmloaded contents in the `data/` folder.


## 4. How to Run the Code

1. Navigate to the script directory:

    ```bash
    cd script/
    ```

2. Run the bash scripts:

    ```bash
    bash cifar100.sh   # for CIFAR-100
    ````
    ```bash
    bash imagenet.sh   # for ImageNet-R
    ```
    ```bash
    bash imagenet_sub.sh   # for ImageNet-Sub
    ```
    ```bash
    bash 5datasets.sh   # for 5datasets
    ```
3. How to run other settings:
   
    Please refer to the "main.py" for the detailed arguments.
    


## 5. How to Cite the Paper

If you find our research or this repository useful, please consider citing our work:

```bibtex
@misc{li2023steering,
      title={Steering Prototypes with Prompt-tuning for Rehearsal-free Continual Learning}, 
      author={Zhuowei Li and Long Zhao and Zizhao Zhang and Han Zhang and Di Liu and Ting Liu and Dimitris N. Metaxas},
      year={2023},
      eprint={2303.09447},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


## 6. Acknowledgments

We would like to extend our gratitude to [dino](https://github.com/facebookresearch/dino), [mae](https://github.com/facebookresearch/mae) and [SupContrast](https://github.com/HobbitLong/SupContrast) which we use partial of their code in our project.
