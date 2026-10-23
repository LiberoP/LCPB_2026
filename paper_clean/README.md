# LCP-B — Publication-style paper on Boosting algorithms

This directory contains a publication-style research assignment (paper) and the code used to produce it. The paper compares Boosting algorithms performances, also benchmarking them against a neural network. Please refer to the paper (`assignment_group2608.pdf`) for all details.

**Main finding**: tuned boosting models were the strongest performers on the tabular tasks, with XGBoost leading on the MAGIC benchmark and the best UCI HAR results coming from a CNN + CatBoost pipeline; the image task used MNIST, where deep features were also highly competitive.

Main libraries used were `tensorflow` / `keras`, `sklearn`, `XGBoost`, `LightGBM`, `CatBoost`.

Datasets used: the [MAGIC Gamma Telescope dataset](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope), the [UCI HAR dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), and the [MNIST handwritten digits dataset](http://yann.lecun.com/exdb/mnist/).

Notebook-to-dataset mapping:
- `magic_dataset_analysis.ipynb` — MAGIC Gamma Telescope dataset
- `time_series_analysis.ipynb` — UCI HAR dataset
- `image_analysis.ipynb` — MNIST handwritten digits dataset

See `contribution.pdf` for a list of the contribution by each of the authors.


---

## Directory structure

```
.
├── README.md
├── assignment_group2608.pdf        # final product (paper)
├── contribution.pdf                # individual contributions
├── image_analysis.ipynb            # notebooks
├── magic_dataset_analysis.ipynb
└── time_series_analysis.ipynb
```

---

## Note on reproducibility
This project is for educational purposes. As such, it has not been fully optimized for reproducibility.

---

## Acknowledgements
* Prof. Ilaria Siloi provided directions on the general structure of the paper.
* All training was done using the free version of [Google Colab](https://colab.research.google.com/)

---

## Declaration of AI use

Generative Artificial Intelligence (AI) was used throughout this project mainly as a debugging and plotting aid. However, code was also AI-generated at some points of the development in order to boost productivity. All code and results were reviewed, tested, and validated by the team.