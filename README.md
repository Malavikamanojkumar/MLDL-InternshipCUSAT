# Advanced Mathematical Foundations of Machine Learning and Deep Learning Optimisation - CUSAT Summer Internship, June 2026

This repository contains the code from my one-month Summer Internship Programme (1 June – 29 June 2026, 120 hours) at the **Department of Mathematics, Cochin University of Science and Technology (CUSAT)**, under the guidance of **Dr. Linu Pinto** (Assistant Professor) and **Dr. Shery Fernandez** (Professor and Head).

The internship studied machine learning and deep learning from their mathematical foundations. Each topic was worked out on paper first (definitions, geometric intuition, derivations) and then turned into Python written from scratch. Most of the code here uses only NumPy, with no deep learning frameworks. This includes the linear algebra work (change of basis, CR, LU, QR and SVD decompositions, gradient descent versus Newton's method, and the Adam optimiser) as well as the neural network architectures (MLP, RNN, LSTM, CNN), where forward passes, loss functions, backpropagation and gradient-descent updates are all written by hand.

## Internship Details

| | |
|---|---|
| **Programme** | Summer Internship Programme: *Advanced Mathematical Foundations of Machine Learning and Deep Learning Optimization* |
| **Institution** | Department of Mathematics, Cochin University of Science and Technology (CUSAT), Kochi |
| **Duration** | 1 June 2026 – 29 June 2026 (120 hours) |
| **Guide** | Dr. Linu Pinto, Assistant Professor |
| **Certificate** | [Internship Certificate (PDF)](assets/Internship_Certificate.pdf) |
| **Notes and report** | [Google Drive: LaTeX notes and internship report](<YOUR_DRIVE_LINK>) |

## Projects

The repository is split into two sections. Click a module name to open it.

### Architecture_ScratchCodes

Deep learning architectures implemented from scratch, with the forward pass, loss, backpropagation and parameter update written out explicitly.

| # | Module | Description |
|---|--------|-------------|
| 1 | [MLP](./Architecture_ScratchCodes/MLP) | Multi-layer perceptron regression predicting next-day temperature, trained with per-sample SGD |
| 2 | [RNN](./Architecture_ScratchCodes/RNN) | Vanilla RNN regression on the same temperature data, with the daily readings treated as a $T=3$ sequence and trained with BPTT |
| 3 | [LSTM](./Architecture_ScratchCodes/LSTM) | LSTM regression over sliding windows of $T=5$ days, trained with gated BPTT |
| 4 | [CNN](./Architecture_ScratchCodes/CNN) | Convolutional network classifying 8×8 images of "X" and "O" |
| 5 | [Token Embedding and Positional Encoding](./Architecture_ScratchCodes/Token_Embedding_Positional_Encoding) | Step-by-step construction of token embeddings and sinusoidal positional encodings, as used at the input of a Transformer |

### Linear_Algebra_Foundations

The linear algebra, decompositions and optimisation topics that underpin the architectures above.

| # | Module | Description |
|---|--------|-------------|
| 1 | [ChangeOfBasis](./Linear_Algebra_Foundations/ChangeOfBasis) | Expressing the same vector and the same linear map in different bases using the change-of-basis matrix |
| 2 | [CR_Decomposition](./Linear_Algebra_Foundations/CR_Decomposition) | Factorising a matrix as $A = CR$ from its pivot columns and RREF |
| 3 | [LUvsQR_Decomposition](./Linear_Algebra_Foundations/LUvsQR_Decomposition) | LU decomposition (elimination) and QR decomposition (orthogonalisation) side by side |
| 4 | [SVD](./Linear_Algebra_Foundations/SVD) | SVD computed from the eigendecomposition of $A^\top A$, used for rank-$k$ image compression |
| 5 | [GDvsNewtons](./Linear_Algebra_Foundations/GDvsNewtons) | Comparing gradient descent and Newton's method on an ill-conditioned quadratic |
| 6 | [Adam Optimiser](./Linear_Algebra_Foundations/Adam%20Optimiser) | The Adam optimiser, built from momentum and RMSProp-style adaptive step sizes with bias correction |

Each module folder has its own README with the underlying math, code and results, and a notebook that runs end to end.

## Setup

**Requirements:** Python 3.9 or later.

```bash
# 1. Clone the repository
git clone https://github.com/Malavikamanojkumar/MLDL-InternshipCUSAT.git
cd MLDL-InternshipCUSAT

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install the dependencies
pip install numpy pandas matplotlib scikit-learn jupyter

# 4. Launch Jupyter and open any module's notebook
jupyter notebook
```

NumPy is the only library the from-scratch implementations rely on. pandas and scikit-learn are used for loading data, scaling, train/test splits and the sample image in the SVD module, and matplotlib is used for plots.

## Repository Structure

```
MLDL-InternshipCUSAT/
├── Architecture_ScratchCodes/
│   ├── MLP/
│   ├── RNN/
│   ├── LSTM/
│   ├── CNN/
│   └── Token_Embedding_Positional_Encoding/
├── Linear_Algebra_Foundations/
│   ├── Adam Optimiser/
│   ├── CR_Decomposition/
│   ├── ChangeOfBasis/
│   ├── GDvsNewtons/
│   ├── LUvsQR_Decomposition/
│   └── SVD/
├── assets/                  # internship certificate
└── README.md
```

## Acknowledgement

My thanks to Dr. Linu Pinto for her guidance throughout the internship, to Dr. Shery Fernandez and the Department of Mathematics, CUSAT, for the opportunity, and to the research scholars of the department for patiently clearing my doubts.
