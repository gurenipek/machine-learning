# ANN Image Classifier — CIFAR-10

a PyTorch implementation to explore how depth, learning rate, and activation
functions affect neural network performance on image classification.

all experiment results including performance metrics and training vs. validation loss graphs can be found in the report.

## what it does

trains and evaluates fully-connected ANNs of 1, 2, and 3 layers on the
CIFAR-10 dataset, systematically comparing:

- depth: 1-layer, 2-layer (200 hidden units), 3-layer (150-200 hidden units)
- learning rates: 1e-3, 1e-5, 1e-7
- activation functions: ReLU, HardSwish, Tanh
- regularization: optional L2 weight decay (uncomment in code)

images are converted to grayscale and normalized before training.
each configuration runs for 20 epochs with Adam optimizer and CrossEntropyLoss,
and plots training vs. validation loss curves.

## run

```bash
python cifar10_ann_classifier.py
```

or open `cifar10_ann_experiments.ipynb` in Jupyter / Google Colab (GPU needed)

to try only the best-performing configurations, uncomment the relevant
calls at the bottom of the script.

to enable L2 regularization, uncomment the marked block inside the training loop.

## stack

Python, PyTorch, torchvision, NumPy, Matplotlib

---
*CENG 499 — Machine Learning, METU, Fall 2021*