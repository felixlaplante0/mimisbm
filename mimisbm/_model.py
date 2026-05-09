from numbers import Integral, Real
from typing import Self

import numpy as np
from fastkmeanspp import KMeans  # type: ignore
from scipy.special import betaln, digamma, entr, gammaln, softmax  # type: ignore
from sklearn.base import BaseEstimator, ClusterMixin  # type: ignore
from sklearn.utils._param_validation import (  # type: ignore
    Interval,  # type: ignore
    StrOptions,  # type: ignore
    validate_params,  # type: ignore
)
from sklearn.utils.validation import check_is_fitted, validate_data  # type: ignore


class MimiSBM(ClusterMixin, BaseEstimator):
    r"""Mixture of Multilayer Integrator Stochastic Block Model (MimiSBM).

    The MimiSBM is a generative model for multilayer networks that identifies mesoscale
    structures by grouping nodes into clusters and layers into components.

    Each component represents a distinct Stochastic Block Model (SBM) shared by a subset
    of layers. This model uses a Variational Expectation-Maximization (VEM) algorithm to
    perform inference and estimation of the posterior distributions.

    Model settings:
        - `n_clusters`: Number of clusters for the nodes.
        - `n_components`: Number of mixture components for the layers.

    Prior settings:
        - `clusters_prior`: Dirichlet prior for the node cluster mixing proportions.
        - `components_prior`: Dirichlet prior for the layer component mixing
          proportions.
        - `adjacency_prior`: Beta prior for the edge probabilities within and
          between clusters for each component.

    EM settings:
        - `max_iter`: Maximum number of iterations for the VEM algorithm.
        - `tol`: Convergence tolerance based on the Evidence Lower Bound (ELBO).
        - `warm_start`: If True, reuse the responsibilities from the previous fit
          as initialization.

    Attributes:
        n_clusters (int): Number of node clusters.
        n_components (int): Number of layer components.
        clusters_prior (np.ndarray): Prior parameters for node clusters.
        components_prior (np.ndarray): Prior parameters for layer components.
        adjacency_prior (np.ndarray): Prior parameters for edge connections.
        max_iter (int): Maximum number of iterations for the EM algorithm.
        tol (float): Tolerance to declare convergence based on the ELBO.
        warm_start (bool): Whether to reuse the solution of the previous call
            to fit as initialization.
        random_state (int | None): Random state for initialization.
        cluster_responsibilities_ (np.ndarray): Posterior probabilities of node cluster
            assignments (N, K).
        component_responsibilities_ (np.ndarray): Posterior probabilities of layer
            component assignments (V, Q).
        cluster_posterior_ (np.ndarray): Dirichlet posterior parameters for clusters.
        component_posterior_ (np.ndarray): Dirichlet posterior parameters for
            components.
        adjacency_posterior_ (np.ndarray): Beta posterior parameters for edge
            connections (2, K, K, Q).
        elbo_ (float): Evidence Lower Bound of the fitted model.
        converged_ (bool): True if the algorithm converged, False otherwise.

    Examples:
        >>> from mimisbm import MimiSBM
        >>> import numpy as np
        >>> X = np.random.randint(0, 2, size=(10, 10, 5))
        >>> model = MimiSBM(n_clusters=2, n_components=2)
        >>> model.fit(X)
        >>> node_labels, layer_labels = model.predict()
    """

    n_clusters: int
    n_components: int
    clusters_prior: np.ndarray
    components_prior: np.ndarray
    adjacency_prior: np.ndarray
    max_iter: int
    tol: float
    warm_start: bool
    random_state: int | None
    cluster_responsibilities_: np.ndarray
    component_responsibilities_: np.ndarray
    cluster_posterior_: np.ndarray
    component_posterior_: np.ndarray
    adjacency_posterior_: np.ndarray
    elbo_: float
    converged_: bool

    @validate_params(
        {
            "n_clusters": [Interval(Integral, 1, None, closed="left")],
            "n_components": [Interval(Integral, 1, None, closed="left")],
            "clusters_prior": [StrOptions({"jeffreys", "uniform"}), np.ndarray],
            "components_prior": [StrOptions({"jeffreys", "uniform"}), np.ndarray],
            "adjacency_prior": [StrOptions({"jeffreys", "uniform"}), np.ndarray],
            "max_iter": [Interval(Integral, 1, None, closed="left")],
            "tol": [Interval(Real, 0, None, closed="left")],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        n_clusters: int = 2,
        n_components: int = 2,
        *,
        clusters_prior: np.ndarray | str = "jeffreys",
        components_prior: np.ndarray | str = "jeffreys",
        adjacency_prior: np.ndarray | str = "jeffreys",
        max_iter: int = 100,
        tol: float = 1e-4,
        warm_start: bool = False,
        random_state: int | None = None,
    ):
        r"""Initializes the MimiSBM model with specified design and priors.

        Constructs a mixture of multilayer SBMs with user-defined priors and
        EM settings. Provides default settings for Bayesian inference and
        convergence criteria.

        Args:
            n_clusters (int, optional): Number of clusters for the nodes.
                Defaults to 2.
            n_components (int, optional): Number of mixture components for the layers.
                Defaults to 2.
            clusters_prior (np.ndarray | str, optional): Dirichlet prior for node
                clusters. Can be "jeffreys" (0.5), "uniform" (1.0), or a custom array.
                Defaults to "jeffreys".
            components_prior (np.ndarray | str, optional): Dirichlet prior for layer
                components. Defaults to "jeffreys".
            adjacency_prior (np.ndarray | str, optional): Beta prior for edge
                probabilities. Defaults to "jeffreys".
            max_iter (int, optional): Maximum number of VEM iterations. Defaults to 100.
            tol (float, optional): Convergence tolerance for ELBO. Defaults to 1e-4.
            warm_start (bool, optional): Whether to reuse responsibilities from a
                previous fit. Defaults to False.
            random_state (int | None, optional): Seed for the KMeans initialization.
                Defaults to None.
        """
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.clusters_prior = self._init_prior(clusters_prior, n_clusters)
        self.components_prior = self._init_prior(components_prior, self.n_components)
        self.adjacency_prior = self._init_prior(adjacency_prior, 2)
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.random_state = random_state

    @staticmethod
    def _init_prior(prior: np.ndarray | str, d: int) -> np.ndarray:
        r"""Initializes the prior parameters for a given dimension.

        Args:
            prior (np.ndarray | str): The prior specification.
            d (int): The dimension of the prior vector.

        Returns:
            np.ndarray: The initialized prior parameters.

        Raises:
            ValueError: If the provided prior array has an incorrect length.
        """
        if prior == "jeffreys":
            return np.full((d,), 0.5)
        if prior == "uniform":
            return np.full((d,), 1.0)

        prior = prior.reshape(-1)
        if len(prior) != d:
            raise ValueError(f"Prior must have {d} elements, got {len(prior)}")
        return prior

    def _init_responsibilities(
        self, X: np.ndarray, n_clusters: int, axis: tuple[int, ...]
    ) -> np.ndarray:
        r"""Initializes responsibilities using KMeans on aggregated adjacency data.

        Args:
            X (np.ndarray): The multilayer adjacency tensor.
            n_clusters (int): Number of clusters/components to initialize.
            axis (tuple[int, ...]): Axis over which to aggregate the tensor.

        Returns:
            np.ndarray: Initialized responsibilities.
        """
        X_agg = X.sum(axis=axis)
        X_agg = X_agg.reshape(X_agg.shape[0], -1)

        labels = KMeans(
            n_clusters=n_clusters, random_state=self.random_state
        ).fit_predict(X_agg)

        responsibilities = np.zeros((labels.shape[0], n_clusters))
        responsibilities[np.arange(labels.shape[0]), labels] = 1
        return responsibilities

    def _elbo(self) -> float:
        r"""Computes the Evidence Lower Bound (ELBO) for the current state.

        The ELBO is used to monitor convergence and as a surrogate for the
        log-likelihood in the Variational EM algorithm.

        Returns:
            float: The computed ELBO value.
        """
        cluster_entropy = entr(self.cluster_responsibilities_).sum()
        component_entropy = entr(self.component_responsibilities_).sum()

        cluster_evidence = (
            gammaln(self.cluster_posterior_).sum()
            - gammaln(self.cluster_posterior_.sum())
            - gammaln(self.clusters_prior).sum()
            + gammaln(self.clusters_prior.sum())
        )

        component_evidence = (
            gammaln(self.component_posterior_).sum()
            - gammaln(self.component_posterior_.sum())
            - gammaln(self.components_prior).sum()
            + gammaln(self.components_prior.sum())
        )

        log_adjacency_posterior = betaln(
            self.adjacency_posterior_[0], self.adjacency_posterior_[1]
        )
        log_adjacency_prior = betaln(self.adjacency_prior[0], self.adjacency_prior[1])

        # Sum over i < j
        rows, cols = np.tril_indices(self.n_clusters)
        adjacency_evidence = (
            log_adjacency_posterior[rows, cols, :] - log_adjacency_prior
        ).sum()

        evidence = cluster_evidence + component_evidence + adjacency_evidence
        entropy = cluster_entropy + component_entropy

        return evidence + entropy

    def _m_step(self, X: np.ndarray, X_non: np.ndarray):
        r"""Performs the M-step of the Variational EM algorithm.

        Updates the posterior parameters of the priors based on the current
        responsibilities.

        Args:
            X (np.ndarray): The multilayer adjacency tensor.
            X_non (np.ndarray): The complement of the adjacency tensor.
        """
        self.cluster_posterior_ = (
            self.clusters_prior + self.cluster_responsibilities_.sum(axis=0)
        )
        self.component_posterior_ = (
            self.components_prior + self.component_responsibilities_.sum(axis=0)
        )

        weighted_edges = X @ self.component_responsibilities_
        weighted_non_edges = X_non @ self.component_responsibilities_

        expected_edges = (
            self.cluster_responsibilities_.T
            @ weighted_edges.swapaxes(0, 2)
            @ self.cluster_responsibilities_
        ).swapaxes(0, 2)
        expected_non_edges = (
            self.cluster_responsibilities_.T
            @ weighted_non_edges.swapaxes(0, 2)
            @ self.cluster_responsibilities_
        ).swapaxes(0, 2)

        # Sum over i < j
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
        r"""Performs the E-step of the Variational EM algorithm.

        Updates the responsibilities for node clusters and layer components given the
        current posterior parameters.

        Args:
            X (np.ndarray): The multilayer adjacency tensor.
            X_non (np.ndarray): The complement of the adjacency tensor.
        """
        digamma_adjacency_posterior = digamma(self.adjacency_posterior_.sum(axis=0))
        log_edges = digamma(self.adjacency_posterior_[0]) - digamma_adjacency_posterior
        log_non_edges = (
            digamma(self.adjacency_posterior_[1]) - digamma_adjacency_posterior
        )

        expected_component_edges = (
            self.cluster_responsibilities_
            @ log_edges.swapaxes(0, 2)
            @ self.cluster_responsibilities_.T
        ).swapaxes(0, 2)
        expected_component_non_edges = (
            self.cluster_responsibilities_
            @ log_non_edges.swapaxes(0, 2)
            @ self.cluster_responsibilities_.T
        ).swapaxes(0, 2)

        # Sum over i < j
        component_posterior_evidence = digamma(self.component_posterior_) - digamma(
            self.component_posterior_.sum()
        )
        component_edges_evidence = 0.5 * np.tensordot(
            X, expected_component_edges, axes=([0, 1], [0, 1])
        )
        component_edges_evidence = np.nan_to_num(component_edges_evidence)
        component_non_edges_evidence = 0.5 * np.tensordot(
            X_non, expected_component_non_edges, axes=([0, 1], [0, 1])
        )
        component_non_edges_evidence = np.nan_to_num(component_non_edges_evidence)
        self.component_responsibilities_ = softmax(
            component_posterior_evidence
            + component_edges_evidence
            + component_non_edges_evidence,
            axis=1,
        )

        # Sum over i != j
        expected_cluster_edges = np.tensordot(
            self.component_responsibilities_, log_edges, axes=([1], [2])
        )
        expected_cluster_non_edges = np.tensordot(
            self.component_responsibilities_, log_non_edges, axes=([1], [2])
        )
        cluster_posterior_evidence = digamma(self.cluster_posterior_) - digamma(
            self.cluster_posterior_.sum()
        )

        cluster_edges_evidence = np.tensordot(
            X,
            self.cluster_responsibilities_ @ expected_cluster_edges.swapaxes(1, 2),
            axes=([1, 2], [1, 0]),
        )
        cluster_non_edges_evidence = np.tensordot(
            X_non,
            self.cluster_responsibilities_ @ expected_cluster_non_edges.swapaxes(1, 2),
            axes=([1, 2], [1, 0]),
        )

        self.cluster_responsibilities_ = softmax(
            cluster_posterior_evidence
            + cluster_edges_evidence
            + cluster_non_edges_evidence,
            axis=1,
        )

    def _validate(self, X: np.typing.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        r"""Validates the input data and ensures it is in the correct format.

        Checks that the input is a 3D numpy array with appropriate dimension for a
        multilayer adjacency tensor.

        Args:
            X (np.typing.ArrayLike): The input data to validate.

        Raises:
            ValueError: If the input data is not a 3D array.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the validated adjacency
                tensor and its complement.
        """
        X = np.asarray(validate_data(self, X, allow_nd=True, dtype=np.bool))  # type: ignore
        if X.ndim != 3:  # noqa: PLR2004
            raise ValueError(f"Input data must be a 3D array, got {X.ndim}D array")

        X |= X.swapaxes(0, 1)
        rows, cols = np.diag_indices(X.shape[0])
        X[rows, cols, :] = 0
        X_non = 1 - X
        X_non[rows, cols, :] = 0

        return X, X_non

    @validate_params({"X": ["array-like"]}, prefer_skip_nested_validation=True)
    def fit(self, X: np.typing.ArrayLike) -> Self:
        r"""Fits the MimiSBM model to the multilayer adjacency tensor.

        Initializes the model responsibilities and iteratively updates them using the
        VEM algorithm. The process continues until the ELBO converges or the maximum
        number of iterations is reached.

        Args:
            X (np.typing.ArrayLike): A 3D numpy array-like representing the multilayer
                adjacency tensor of shape (N, N, V).

        Returns:
            Self: The fitted model instance.
        """
        X, X_non = self._validate(X)  # type: ignore

        if not (self.warm_start and hasattr(self, "converged_")):
            self.cluster_responsibilities_ = self._init_responsibilities(
                X, self.n_clusters, (2,)
            )
            self.component_responsibilities_ = self._init_responsibilities(
                X, self.n_components, (0, 1)
            )

        old_elbo = -np.inf
        for _ in range(self.max_iter):
            self._m_step(X, X_non)
            self._e_step(X, X_non)

            self.elbo_ = self._elbo()
            if abs(self.elbo_ - old_elbo) < self.tol:
                self.converged_ = True
                return self
            old_elbo = self.elbo_

        self.converged_ = False

        return self

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Predicts the node clusters and layer components labels.

        Assigns each node and each layer to the cluster/component with the highest
        probability.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - node_labels (np.ndarray): Predicted cluster for each node (N,).
                - layer_labels (np.ndarray): Predicted component for each layer (V,).
        """
        check_is_fitted(
            self, ["cluster_responsibilities_", "component_responsibilities_"]
        )
        return self.cluster_responsibilities_.argmax(
            axis=1
        ), self.component_responsibilities_.argmax(axis=1)
