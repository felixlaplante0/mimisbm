from functools import cached_property

import numpy as np
from fastkmeanspp import KMeans
from scipy.special import digamma, softmax
from sklearn.base import BaseEstimator, ClusterMixin


class MimiSBM(ClusterMixin, BaseEstimator):
    """Mixture of Multiview Integrator Stochastic Block Model."""

    def __init__(
        self,
        n_clusters: int = 2,
        n_components: int = 2,
        *,
        cluster_prior: np.ndarray | str = "jeffrey",
        view_prior: np.ndarray | str = "jeffrey",
        adjacency_prior: np.ndarray | str = "jeffrey",
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
    ):
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.cluster_prior = self._init_prior(cluster_prior, n_clusters)
        self.view_prior = self._init_prior(view_prior, self.n_components)
        self.adjacency_prior = self._init_prior(adjacency_prior, 2)
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def _init_prior(prior: np.ndarray | str, d: int) -> np.ndarray:
        if isinstance(prior, np.ndarray):
            prior = prior.reshape(-1)
            if len(prior) != d:
                raise ValueError(f"Prior must have {d} elements, got {len(prior)}")
            return prior
        if prior == "jeffrey":
            return np.full((d,), 0.5)
        if prior == "uniform":
            return np.full((d,), 1.0)
        raise ValueError(
            f"Prior must be either array, `uniform`, or `jeffrey`, got {prior}"
        )

    def _init_responsabilities(
        self, X: np.ndarray, n_clusters: int, axis: tuple[int, ...]
    ) -> np.ndarray:
        n = X.shape[0]

        X = X.sum(axis=axis).reshape(n, -1)  # type: ignore
        labels = KMeans(
            n_clusters=n_clusters, random_state=self.random_state
        ).fit_predict(X)

        responsabilities = np.zeros((n, n_clusters))
        responsabilities[np.arange(n), labels] = 1
        return responsabilities

    def _M_step(
        self,
        X: np.ndarray,
        cluster_responsabilties: np.ndarray,
        view_responsabilities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = X.shape[0]

        X_non = 1 - X
        X_non[np.diag_indices(n)] = 0

        cluster_posterior = self.cluster_prior + cluster_responsabilties.sum(axis=0)
        view_posterior = self.view_prior + view_responsabilities.sum(axis=0)

        weighted_edges = X @ view_responsabilities
        weighted_non_edges = X_non @ view_responsabilities

        edges = np.einsum(
            "ik,ijq,jl->klq",
            cluster_responsabilties,
            weighted_edges,
            cluster_responsabilties,
        )
        nonedges = np.einsum(
            "ik,ijq,jl->klq",
            cluster_responsabilties,
            weighted_non_edges,
            cluster_responsabilties,
        )

        rows, cols = np.diag_indices(self.n_clusters)
        edges[rows, cols, :] *= 0.5
        nonedges[rows, cols, :] *= 0.5

        adjacency_posterior = self.adjacency_prior.reshape(2, 1, 1, 1) + np.stack(
            [edges, nonedges]
        )
        return cluster_posterior, view_posterior, adjacency_posterior

    def _E_step(
        self,
        X: np.ndarray,
        cluster_posterior: np.ndarray,
        view_posterior: np.ndarray,
        adjacency_posterior: np.ndarray,
        cluster_responsabilties: np.ndarray,
        view_responsabilities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]

        X_non = 1 - X
        X_non[np.diag_indices(n)] = 0

        log_edges = digamma(adjacency_posterior[0]) - digamma(
            adjacency_posterior.sum(axis=0)
        )
        log_nonedges = digamma(adjacency_posterior[1]) - digamma(
            adjacency_posterior.sum(axis=0)
        )

        log_view_responsabilities = digamma(view_posterior) - digamma(
            view_posterior.sum()
        )
        evidence_edges = np.einsum(
            "ik,jl,klq->ijq",
            cluster_responsabilties,
            cluster_responsabilties,
            log_edges,
        )
        evidence_nonedges = np.einsum(
            "ik,jl,klq->ijq",
            cluster_responsabilties,
            cluster_responsabilties,
            log_nonedges,
        )
        log_view_responsabilities = log_view_responsabilities[
            np.newaxis, :
        ] + np.einsum("ijv,ijq->vq", X, evidence_edges)
        log_view_responsabilities += np.einsum("ijv,ijq->vq", X_non, evidence_nonedges)
        view_responsabilities = softmax(log_view_responsabilities, axis=1)

        log_cluster_responsabilties = digamma(cluster_posterior) - digamma(
            cluster_posterior.sum()
        )
        evidence_v_edges = np.einsum("vq,klq->vkl", view_responsabilities, log_edges)
        evidence_v_nonedges = np.einsum(
            "vq,klq->vkl", view_responsabilities, log_nonedges
        )
        log_cluster_responsabilties = log_cluster_responsabilties[
            np.newaxis, :
        ] + np.einsum("ijv,jl,vkl->ik", X, cluster_responsabilties, evidence_v_edges)
        log_cluster_responsabilties += np.einsum(
            "ijv,jl,vkl->ik", X_non, cluster_responsabilties, evidence_v_nonedges
        )
        cluster_responsabilties = softmax(log_cluster_responsabilties, axis=1)

        return cluster_responsabilties, view_responsabilities

    def fit(self, X: np.ndarray) -> "MimiSBM":
        self.cluster_responsabilties_ = self._init_responsabilities(
            X, self.n_clusters, (2,)
        )
        self.view_responsabilities_ = self._init_responsabilities(
            X.transpose(2, 0, 1), self.n_components, (1, 2)
        )

        for _ in range(self.max_iter):
            prev_responsabilties = self.cluster_responsabilties_.copy()
            (
                self.cluster_posterior_,
                self.view_posterior_,
                self.adjacency_posterior_,
            ) = self._M_step(
                X, self.cluster_responsabilties_, self.view_responsabilities_
            )
            self.cluster_responsabilties_, self.view_responsabilities_ = self._E_step(
                X,
                self.cluster_posterior_,
                self.view_posterior_,
                self.adjacency_posterior_,
                self.cluster_responsabilties_,
                self.view_responsabilities_,
            )
            if (
                np.linalg.norm(self.cluster_responsabilties_ - prev_responsabilties)
                < self.tol
            ):
                break
        return self

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        return self.cluster_responsabilties_.argmax(
            axis=1
        ), self.view_responsabilities_.argmax(axis=1)
