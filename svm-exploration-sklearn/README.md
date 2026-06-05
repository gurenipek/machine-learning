# support vector machines with sklearn

exploring SVMs across four tasks,
using scikit-learn on 2D and image datasets. built for a ML course at METU.

## what it does

task 1 — effect of C (regularization)
trains a linear SVM with `C ∈ {0.01, 0.1, 1, 10, 100}` and visualizes
how the decision boundary changes with regularization strength.

task 2 — kernel comparison
compares linear, RBF, polynomial, and sigmoid kernels on the same dataset,
with decision boundary plots for each.

task 3 — hyperparameter tuning
grid search over `kernel × C × gamma` on a flattened image dataset,
using k-fold cross-validation to find the best configuration.

task 4 — handling class imbalance
compares three strategies on an imbalanced dataset:
- oversampling the minority class
- undersampling the majority class
- adjusting class weights (`class_weight="balanced"`)

## run

open in Jupyter or Google Colab. dataset `.npy` files are included, though i recommend making sure paths are correct to access them.

```bash
jupyter notebook svm_exploration.ipynb
```

## stack

Python, NumPy, scikit-learn, Matplotlib

---
*CENG 499 — Machine Learning, METU, Fall 2021*