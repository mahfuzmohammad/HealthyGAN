# HealthyGAN
This repository provides the official PyTorch implementation of HealthyGAN.

## Paper
**HealthyGAN: Learning from Unannotated Medical Images to Detect Anomalies Associated with Human Disease**

[Md Mahfuzur Rahman Siddiquee](https://github.com/mahfuzmohammad)<sup>1,2</sup>, [Jay Shah]<sup>1,2</sup>, [Teresa Wu](https://search.asu.edu/profile/342678)<sup>1,2</sup>, [Catherine Chong](https://www.mayo.edu/research/faculty/chong-catherine-cat-ph-d/bio-20146998?_ga=2.180486759.1523948133.1540410816-481891997.1538001956)<sup>2,3</sup>, [Todd Schwedt](https://www.mayo.edu/research/faculty/schwedt-todd-j-m-d/bio-20091053)<sup>2,3</sup>, and [Baoxin Li](https://search.asu.edu/profile/747601)<sup>1,2</sup><br/>

<sup>1</sup>Arizona State University; <sup>2</sup>ASU-Mayo Center for Innovative Imaging; <sup>3</sup>Mayo Clinic<br/>
Simulation and Synthesis in Medical Imaging ([SASHIMI](https://2022.sashimi-workshop.org/)), 2022

## Abstract
Automated anomaly detection from medical images, such as MRIs and X-rays, can significantly reduce human effort in disease diagnosis. Owing to the complexity of modeling anomalies and the high cost of manual annotation by domain experts (e.g., radiologists), a typical technique in the current medical imaging literature has focused on deriving diagnostic models from healthy subjects only, assuming the model will detect the images from patients as outliers. However, in many real-world scenarios, unannotated datasets with a mix of both healthy and diseased individuals are abundant. Therefore, this paper poses the research question of how to improve unsupervised anomaly detection by utilizing (1) an unannotated set of mixed images, in addition to (2) the set of healthy images as being used in the literature. To answer the question, we propose HealthyGAN, a novel one-directional image-to-image translation method, which learns to translate the images from the mixed dataset to only healthy images. Being one-directional, HealthyGAN relaxes the requirement of cycle-consistency of existing unpaired image-to-image translation methods, which is unattainable with mixed unannotated data. Once the translation is learned, we generate a difference map for any given image by subtracting its translated output. Regions of significant responses in the difference map correspond to potential anomalies (if any). Our HealthyGAN outperforms the conventional state-of-the-art methods by significant margins on two publicly available datasets: COVID-19 and NIH ChestX-ray14, and one institutional dataset collected from Mayo Clinic.

## Usage

### 0. Cloning the repository

```bash
$ git clone https://github.com/mahfuzmohammad/Fixed-Point-GAN.git
$ cd Fixed-Point-GAN/
```

### 1. Creating python environment

```bash
$ conda env create -f conda_env/conda_env_pytorch0.2.yml
$ source activate pytorch0.2
$ cat conda_env/pip_pytorch0.2.txt | xargs -n 1 pip install
```

### 2. Downloading the dataset

- Covid: https://github.com/lindawangg/COVID-Net
Put the train and test folder under datasets/covid

### 3. Training

- Covid: `bash train_covid.sh`

### 4. Testing

- Covid: `bash test_covid.sh`

### 5. Testing using pretrained models

- Covid: `bash test_covid_pretrained.sh`
