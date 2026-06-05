# ML algorithms from scratch with numpy

pure numpy implementations of three classic ML algorithms, written without
scikit-learn or any ML library. built for a ML course at METU.

## algorithms

k-means clustering (`kmeans.py`)
- cluster assignment by Euclidean distance
- iterative centroid recalculation
- returns final cluster centers and objective function value

k-nearest neighbors (`knn.py`)
- L1 (Manhattan) and L2 (Euclidean) distance metrics
- majority voting with tie-breaking
- k-fold cross-validation

hierarchical agglomerative clustering (`hac.py`)
- Four linkage criteria: single, complete, average, centroid
- Merges clusters until a target number is reached

## run

```bash
python test_kmeans.py
python test_knn.py
python test_hac.py
```
or open the notebooks in Jupyter / Google Colab (GPU needed)

## stack

Python, NumPy

---
*CENG 499 — Machine Learning, METU, Fall 2021*
