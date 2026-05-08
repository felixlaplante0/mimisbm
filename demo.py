import numpy as np
from sklearn.metrics import adjusted_rand_score

from mimisbm._model import MimiSBM


def gen_data(N=100, V=20, K=2, Q=2):
    z = np.random.randint(0, K, N)
    w = np.random.randint(0, Q, V)

    # Connectivity for component 0 (assortative) and component 1 (disassortative)
    alphas = np.zeros((K, K, Q))
    alphas[:, :, 0] = [[0.8, 0.1], [0.1, 0.8]]
    alphas[:, :, 1] = [[0.1, 0.8], [0.8, 0.1]]

    A = np.zeros((N, N, V))
    for v in range(V):
        p_mat = alphas[np.ix_(z, z, [w[v]])].squeeze()
        # Generate undirected edges without self-loops
        tri_mask = np.random.rand(N, N) < p_mat
        A[:, :, v] = np.tril(tri_mask, -1).astype(float)
        A[:, :, v] += A[:, :, v].T

    return A, z, w

np.random.seed(42)

print("Generating synthetic data...")
A, true_z, true_w = gen_data()

print("Fitting MimiSBM...")
model = MimiSBM(n_clusters=2, n_subclusters=2, random_state=42).fit(A)
pred_z, pred_w = model.predict()

# 3. Evaluate
node_ari = adjusted_rand_score(true_z, pred_z)
view_ari = adjusted_rand_score(true_w, pred_w)

print("\nEvaluation Results:")
print(f"Node ARI: {node_ari:.4f}")
print(f"View ARI: {view_ari:.4f}")
