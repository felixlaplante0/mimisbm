from typing import Self

import numpy as np
from fastkmeanspp import KMeans
from scipy.special import digamma, softmax  # type: ignore
from sklearn.base import BaseEstimator, ClusterMixin


class MimiSBM(ClusterMixin, BaseEstimator):
    """Mixture of Multisubcluster Integrator Stochastic Block Model."""

    def __init__(
        self,
        n_clusters: int = 2,
        n_subclusters: int = 2,
        *,
        clusters_prior: np.ndarray | str = "jeffreys",
        subclusters_prior: np.ndarray | str = "jeffreys",
        adjacency_prior: np.ndarray | str = "jeffreys",
        max_iter: int = 100,
        tol: float = 1e-4,
        warm_start: bool = False,
        random_state: int | None = None,
    ):
        self.n_clusters = n_clusters
        self.n_subclusters = n_subclusters
        self.clusters_prior = self._init_prior(clusters_prior, n_clusters)
        self.subclusters_prior = self._init_prior(subclusters_prior, self.n_subclusters)
        self.adjacency_prior = self._init_prior(adjacency_prior, 2)
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.random_state = random_state

    @staticmethod
    def _init_prior(prior: np.ndarray | str, d: int) -> np.ndarray:
        if isinstance(prior, np.ndarray):
            prior = prior.reshape(-1)
            if len(prior) != d:
                raise ValueError(f"Prior must have {d} elements, got {len(prior)}")
            return prior
        if prior == "jeffreys":
            return np.full((d,), 0.5)
        if prior == "uniform":
            return np.full((d,), 1.0)
        raise ValueError(
            f"Prior must be either array, `uniform`, or `jeffreys`, got {prior}"
        )

    def _init_responsibilities(
        self, X: np.ndarray, n_clusters: int, axis: tuple[int, ...]
    ) -> np.ndarray:
        X_agg = X.sum(axis=axis)
        X_agg = X_agg.reshape(X_agg.shape[0], -1)

        labels = KMeans(
            n_clusters=n_clusters, random_state=self.random_state
        ).fit_predict(X_agg)

        responsibilities = np.zeros((labels.shape[0], n_clusters))
        responsibilities[np.arange(labels.shape[0]), labels] = 1
        return responsibilities

    def _m_step(self, X: np.ndarray, X_non: np.ndarray):
        self.cluster_posterior_ = (
            self.clusters_prior + self.cluster_responsibilities_.sum(axis=0)
        )
        self.subcluster_posterior_ = (
            self.subclusters_prior + self.subcluster_responsibilities_.sum(axis=0)
        )

        weighted_edges = X @ self.subcluster_responsibilities_
        weighted_non_edges = X_non @ self.subcluster_responsibilities_

        expected_edges = np.einsum(
            "ik,ijq,jl->klq",
            self.cluster_responsibilities_,
            weighted_edges,
            self.cluster_responsibilities_,
        )

        expected_non_edges = np.einsum(
            "ik,ijq,jl->klq",
            self.cluster_responsibilities_,
            weighted_non_edges,
            self.cluster_responsibilities_,
        )

        # Correct for undirected double-counting on the diagonal
        rows, cols = np.diag_indices(self.n_clusters)
        expected_edges[rows, cols, :] *= 0.5
        expected_non_edges[rows, cols, :] *= 0.5

        self.adjacency_posterior_ = np.stack(
            [
                expected_edges + self.adjacency_prior[0],
                expected_non_edges + self.adjacency_prior[1],
            ]
        )

    def _e_step(self, X: np.ndarray, X_non: np.ndarray):
        digamma_adjacency_posterior = digamma(self.adjacency_posterior_.sum(axis=0))
        log_edges = digamma(self.adjacency_posterior_[0]) - digamma_adjacency_posterior
        log_non_edges = (
            digamma(self.adjacency_posterior_[1]) - digamma_adjacency_posterior
        )

        # Correct for sum only on tril(k=-1)
        expected_pair_edges = np.einsum(
            "ik,jl,klq->ijq",
            self.cluster_responsibilities_,
            self.cluster_responsibilities_,
            log_edges,
        )
        expected_pair_non_edges = np.einsum(
            "ik,jl,klq->ijq",
            self.cluster_responsibilities_,
            self.cluster_responsibilities_,
            log_non_edges,
        )
        subcluster_posterior_evidence = digamma(self.subcluster_posterior_) - digamma(
            self.subcluster_posterior_.sum()
        )
        expected_pair_evidence = 0.5 * (
            np.einsum("ijv,ijq->vq", X, expected_pair_edges)
            + np.einsum("ijv,ijq->vq", X_non, expected_pair_non_edges)
        )
        self.subcluster_responsibilities_ = softmax(
            subcluster_posterior_evidence + expected_pair_evidence,
            axis=1,
        )

        # Sum on off-diagonal
        expected_cluster_edges = np.einsum(
            "vq,klq->vkl", self.subcluster_responsibilities_, log_edges
        )
        expected_cluster_non_edges = np.einsum(
            "vq,klq->vkl", self.subcluster_responsibilities_, log_non_edges
        )

        cluster_posterior_evidence = digamma(self.cluster_posterior_) - digamma(
            self.cluster_posterior_.sum()
        )
        cluster_evidence = np.einsum(
            "ijv,jl,vkl->ik", X, self.cluster_responsibilities_, expected_cluster_edges
        ) + np.einsum(
            "ijv,jl,vkl->ik",
            X_non,
            self.cluster_responsibilities_,
            expected_cluster_non_edges,
        )

        self.cluster_responsibilities_ = softmax(
            cluster_posterior_evidence + cluster_evidence,
            axis=1,
        )

    def fit(self, X: np.ndarray) -> Self:
        X_non = 1 - X
        X_non[np.diag_indices(X.shape[0])] = 0

        if not (self.warm_start and hasattr(self, "converged_")):
            self.cluster_responsibilities_ = self._init_responsibilities(
                X, self.n_clusters, (2,)
            )
            self.subcluster_responsibilities_ = self._init_responsibilities(
                X, self.n_subclusters, (0, 1)
            )

        for _ in range(self.max_iter):
            self._m_step(X, X_non)
            self._e_step(X, X_non)

        return self

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        return self.cluster_responsibilities_.argmax(
            axis=1
        ), self.subcluster_responsibilities_.argmax(axis=1)
