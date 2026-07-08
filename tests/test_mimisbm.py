"""Tests for the MimiSBM estimator."""

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning, NotFittedError

from mimisbm import MimiSBM


def _multilayer_graph() -> np.ndarray:
    """Returns a nondegenerate multilayer adjacency tensor."""
    rng = np.random.default_rng(42)
    A = np.zeros((8, 8, 4), dtype=int)
    for layer, probability in enumerate([0.2, 0.35, 0.5, 0.65]):
        lower = np.tril(rng.random((8, 8)) < probability, k=-1)
        A[:, :, layer] = lower + lower.T
    return A


def test_fit():
    """Fits the estimator and exposes learned state."""
    A = _multilayer_graph()
    model = MimiSBM(
        n_clusters=2,
        n_components=2,
        clusters_prior="uniform",
        components_prior=np.array([0.25, 0.75]),
        adjacency_prior=np.array([0.5, 1.5]),
        tol=1e9,
        random_state=42,
    )

    result = model.fit(A)
    node_labels, layer_labels = model.predict()

    assert result is model
    assert model.converged_
    assert node_labels.shape == (A.shape[0],)
    assert layer_labels.shape == (A.shape[2],)
    assert np.allclose(model.clusters_prior_, np.ones(2))
    assert np.allclose(model.components_prior_, [0.25, 0.75])
    assert np.allclose(model.adjacency_prior_, [0.5, 1.5])
    assert model.cluster_posterior_.shape == (2,)
    assert model.component_posterior_.shape == (2,)
    assert model.adjacency_posterior_.shape == (2, 2, 2, 2)
    assert np.isfinite(model.elbo_)


def test_fit_predict():
    """Fits and predicts through the convenience method."""
    node_labels, layer_labels = MimiSBM(
        n_clusters=2,
        n_components=2,
        tol=1e9,
        random_state=42,
    ).fit_predict(_multilayer_graph())

    assert node_labels.shape == (8,)
    assert layer_labels.shape == (4,)


def test_unfit_predict():
    """Rejects prediction before fitting."""
    with pytest.raises(NotFittedError):
        MimiSBM().predict()


def test_no_convergence():
    """Emits a scikit-learn convergence warning when VEM stops early."""
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        model = MimiSBM(max_iter=1, tol=0.0, random_state=42).fit(
            _multilayer_graph()
        )

    assert not model.converged_


def test_warm_start():
    """Reuses responsibilities on a second fit when warm start is enabled."""
    A = _multilayer_graph()
    model = MimiSBM(warm_start=True, max_iter=1, tol=1e9, random_state=42).fit(A)
    cluster_responsibilities = model.cluster_responsibilities_.copy()

    model.fit(A)

    assert not model.converged_
    assert model.cluster_responsibilities_.shape == cluster_responsibilities.shape


def test_fit_validation():
    """Defers constructor parameter errors until fitting."""
    model = MimiSBM(n_clusters=0)

    with pytest.raises(ValueError, match="n_clusters"):
        model.fit(_multilayer_graph())


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("clusters_prior", np.ones(3)),
        ("components_prior", np.array([1.0, -1.0])),
        ("adjacency_prior", "bad"),
    ],
)
def test_prior_validation(parameter, value):
    """Validates prior values and shapes during fitting."""
    model = MimiSBM(**{parameter: value})

    with pytest.raises(ValueError, match=parameter):
        model.fit(_multilayer_graph())


def test_bad_prior_string():
    """Rejects unsupported prior strings in the local prior parser."""
    with pytest.raises(ValueError, match="clusters_prior"):
        MimiSBM._init_prior("bad", 2, "clusters_prior")


def test_adjacency_cleanup():
    """Validates adjacency tensors and normalizes undirected structure."""
    A = np.zeros((3, 3, 2), dtype=int)
    A[0, 1, 0] = 1
    A[2, 2, 1] = 1
    model = MimiSBM()

    A_valid, A_non = model._validate_A(A)

    assert A_valid[0, 1, 0]
    assert A_valid[1, 0, 0]
    assert not A_valid[2, 2, 1]
    assert not A_non[2, 2, 1]


def test_adjacency_dims():
    """Rejects non-tensor inputs."""
    with pytest.raises(ValueError, match="3D"):
        MimiSBM()._validate_A(np.ones((3, 3)))
