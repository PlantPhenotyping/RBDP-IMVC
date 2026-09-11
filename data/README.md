# Data

This directory contains the three feature datasets used for the quantitative
missingness-shift experiments:

| File | Samples | Classes | Views used by RBDP-IMVC |
| --- | ---: | ---: | --- |
| `Caltech101-20.mat` | 2,386 | 20 | HOG (1,984), GIST (512), LBP (928) |
| `Scene-15.mat` | 4,485 | 15 | GIST (20), PHOG (59), LBP (40) |
| `LandUse-21.mat` | 2,100 | 21 | PHOG (59), LBP (40), GIST (20) |

The files follow the MATLAB layout distributed with the upstream
[COMPLETER repository](https://github.com/Lin-Yijie/2021-CVPR-Completer/tree/main/data).
The loader validates sample counts, class counts, and selected view dimensions
before an experiment starts.

NoisyMNIST is intentionally not stored in this repository because of its size.
Download it from the
[link provided by the upstream project](https://drive.google.com/file/d/1b__tkQMHRrYtcCNi_LxnVVTwB-TWdj93/view?usp=sharing)
and save it as `data/NoisyMNIST.mat` when running an optional NoisyMNIST
experiment.

Dataset use remains subject to the terms and citation requirements of the
original data providers.
