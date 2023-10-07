## Hierarchical deep learning with Generative Adversarial Network
A PyTorch implementation of the hierarchical deep learning model as outlined in the paper "Hierarchical deep learning with Generative Adversarial Network for automatic cardiac diagnosis from ECG signals".

In this paper, we propose a two-level hierarchical deep learning framework with Generative Adversarial Network (GAN) for automatic diagnosis of ECG signals. The first-level model is composed of a Memory-Augmented Deep autoEncoder with GAN (MadeGAN), which aims to differentiate abnormal signals from normal ECGs for anomaly detection. The second-level learning aims at robust multi-class classification for different arrhythmias identification, which is achieved by integrating the transfer learning technique to transfer knowledge from the first-level learning with the multi-branching architecture to handle the data-lacking and imbalanced data issue. We evaluate the performance of the proposed framework using real-world medical data from the MIT-BIH arrhythmia database. Experimental results show that our proposed model outperforms existing methods that are commonly used in current practice.

## Dataset
The dataset used in this study is obtained from MIT-BIH database:
1. MIT-BIH Arrhythmia Database: https://www.physionet.org/content/mitdb/1.0.0/. 
2. Long Term ST Database: https://www.physionet.org/content/ltstdb/1.0.0/.

Pre-processed dataset of MBAD can be downloaded from: https://www.dropbox.com/sh/b17k2pb83obbrkn/AADzJigiIrottyTOyvAEU1LOa?dl=0. 

## Dependencies
Python3: 3.9.12

Pytorch: 1.13.1

Numpy: 1.23.4

Pandas: 2.0.3

Scikit-learn: 1.2.2

wfdb: 3.4.1

biosppy: 1.0.0

GPU: NVIDIA RTX A4500

## Citation
If you utilize this code in your research, please consider citing our paper:

```
@article{wang2023hierarchical,
  title={Hierarchical deep learning with Generative Adversarial Network for automatic cardiac diagnosis from ECG signals},
  author={Wang, Zekai and Stavrakis, Stavros and Yao, Bing},
  journal={Computers in Biology and Medicine},
  volume={155},
  pages={106641},
  year={2023},
  publisher={Elsevier}
}
```
