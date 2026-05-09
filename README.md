# 🕸️ MimiSBM

**mimisbm** is a Python package implementing the **Mixture of Multilayer Integrator Stochastic Block Model** proposed by the original authors. It jointly groups nodes into clusters and layers into components, providing a unified framework for identifying shared connectivity patterns across multiple network layers.

---

## ✨ Features

- **Multilayer Clustering**: Jointly identifies node communities and layer components in a single probabilistic framework.
- **Variational EM**: Efficient inference using a Variational Expectation-Maximization (VEM) algorithm for large-scale networks.
- **Bayesian Framework**: Supports flexible Dirichlet and Beta priors, allowing for robust structure discovery under different sparsity regimes.
- **Component-wise SBMs**: Groups layers sharing similar block-model structures into distinct mixture components.
- **scikit-learn API**: Native `BaseEstimator` and `ClusterMixin` integration with a familiar `fit` / `predict` interface.

---

## 🚀 Installation

```bash
pip install mimisbm
```

## 🔧 Usage

### Example

```python
import numpy as np
from mimisbm import MimiSBM

# Generate a synthetic multilayer adjacency tensor (20 nodes, 5 layers)
np.random.seed(42)
N, V = 20, 5
X = np.random.randint(0, 2, size=(N, N, V))

# Ensure the adjacency matrices are symmetric (undirected)
for v in range(V):
    X[..., v] = np.tril(X[..., v], -1) + np.tril(X[..., v], -1).T

# Initialize the model with 3 node clusters and 2 layer components
model = MimiSBM(n_clusters=3, n_components=2, random_state=42)

# Fit the model to the multilayer network
model.fit(X)

# Predict node cluster and layer component assignments
node_labels, layer_labels = model.predict()

print(f"Node clusters: {node_labels}")
print(f"Layer components: {layer_labels}")
print(f"Final ELBO: {model.elbo_:.2f}")
```

---

## 📖 Learn More

For tutorials and detailed API reference, visit the official site:  
👉 [mimisbm's documentation](https://felixlaplante0.github.io/mimisbm)

### 📚 Citation

If you use MimiSBM in your research, please cite the original authors' paper:

```bibtex
@article{de2024mixture,
  title={Mixture of multilayer stochastic block models for multiview clustering},
  author={De Santiago, Kylliann and Szafranski, Marie and Ambroise, Christophe},
  journal={arXiv preprint arXiv:2401.04682},
  year={2024}
}
```

For more details, see the corresponding Preprint: https://arxiv.org/abs/2401.04682
