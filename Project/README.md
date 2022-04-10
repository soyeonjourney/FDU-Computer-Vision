<h1 align = "center">Project: Latte</h1>

## Introduction

**Latte** (**L**et’s **A**bsorb **T**orch **T**echnology **E**legantly) is a self-designed deep learning framework working on CPU, the package name shows tribute to Caffe while the inner structure is inspired by the PyTorch framework. This project focuses on the manual implementation of the most common deep learning package modules and solves the **MNIST** dataset classification problem.

```
.
├── latte
│   ├── __init__.py
│   ├── functional.py
│   ├── nn.py
│   ├── optim.py
│   ├── tensor.py
│   └── utils
│       ├── __init__.py
│       └── data.py
└── lattevision
    ├── __init__.py
    ├── datasets.py
    └── transforms.py
```

## Usage

- [Quick Start](./code/quick_start.ipynb) : See how to build a simple network with Latte framework.
- [Hyperparameter Search](./code/hyperparam_search.ipynb) : Apply a **decreasing scheme** for learning rate (the learning rate is set to 0.01 initially, and decrease it by a factor of 0.9 when the **validation loss** does not improve), then roughly search for the best **hidden units** and **weight decay**.
- [Parameter Visualization](./code/parameter_vis.ipynb) : Visualize parameters in heatmap.
- [Test Best Model](./code/test_best_model.ipynb) : Load best model and test it.

## Result

Training process

<img src="https://raw.githubusercontent.com/Tequila-Sunrise/Image-Hosting/main/FDU-Computer-Vision/train-process.png" alt="Training Process" style="zoom:80%;" />

Parameter heatmap

<img src="https://raw.githubusercontent.com/Tequila-Sunrise/Image-Hosting/main/FDU-Computer-Vision/heatmap-512.png" alt="Heatmap" style="zoom:80%;" />

Best test accuracy: 97.94%

## Reference

1. [Latte](https://github.com/Tequila-Sunrise/Latte) : Deep learning framework implementation from scratch (using numpy/CPU only).
