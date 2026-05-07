import numpy as np
from mimisbm2._model import MimiSBM


def generate_data(N=100, V=20, K=2, Q=2):
    """Generates synthetic multilayer network data using pure NumPy."""
    np.random.seed(42)

    z = np.random.randint(0, K, N)
    w = np.random.randint(0, Q, V)

    alphas = np.zeros((K, K, Q))
    alphas[:, :, 0] = [[0.8, 0.1], [0.1, 0.8]]
    alphas[:, :, 1] = [[0.1, 0.8], [0.8, 0.1]]

    A = np.zeros((N, N, V))
    for v in range(V):
        q = w[v]
        for i in range(N):
            for j in range(i + 1, N):
                p = alphas[z[i], z[j], q]
                if np.random.rand() < p:
                    A[i, j, v] = A[j, i, v] = 1

    return A, z, w


def test_mimisbm():
    print("Generating synthetic data...")
    n_nodes, n_layers, n_clusters, n_components = 100, 20, 2, 2
    A, true_z, true_w = generate_data(n_nodes, n_layers, n_clusters, n_components)

    print(A[..., 0])

    print(
        f"Fitting MimiSBM with n_clusters={n_clusters}, n_components={n_components}..."
    )

    # Run once as initialization is deterministic with KMeans
    model = MimiSBM(
        n_clusters=n_clusters,
        n_components=n_components,
        max_iter=100,
        tol=1e-4,
        random_state=42,
    )
    model.fit(A)

    pred_z, pred_w = model.predict()

    def cluster_accuracy(true_labels, pred_labels):
        from itertools import permutations

        n_clusters = len(np.unique(true_labels))
        best_acc = 0
        for p in permutations(range(n_clusters)):
            mapping = np.array(p)
            mapped_pred = mapping[pred_labels]
            acc = np.mean(mapped_pred == true_labels)
            best_acc = max(best_acc, acc)
        return best_acc

    acc_z = cluster_accuracy(true_z, pred_z)
    acc_w = cluster_accuracy(true_w, pred_w)

    print(f"\n======================")
    print(f"Node clustering accuracy: {acc_z:.2%}")
    print(f"View component accuracy: {acc_w:.2%}")

    if acc_z > 0.8 and acc_w > 0.8:
        print("\nDemo successful: Model efficiently recovered the block structure!")
    else:
        print(
            "\nDemo finished. Accuracy might be low due to random initialization or noise."
        )


if __name__ == "__main__":
    test_mimisbm()
