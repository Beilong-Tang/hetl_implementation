import numpy as np
import pytest

from src.hetl import HeTLConfig, fit_hetl


def make_related_domains(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(40, 3))
    source = latent @ rng.normal(size=(3, 8)) + 0.01 * rng.normal(size=(40, 8))
    target = latent @ rng.normal(size=(3, 6)) + 0.01 * rng.normal(size=(40, 6))
    return source, target


def test_hetl_shapes_and_orthogonality() -> None:
    source, target = make_related_domains()
    result = fit_hetl(
        source,
        target,
        HeTLConfig(latent_dimension=3, beta=1.0, learning_rate=1e-4, max_steps=20),
    )

    assert result.V_source.shape == (40, 3)
    assert result.V_target.shape == (40, 3)
    assert result.P_source.shape == (3, 8)
    assert result.P_target.shape == (3, 6)
    np.testing.assert_allclose(result.V_source.T @ result.V_source, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(result.V_target.T @ result.V_target, np.eye(3), atol=1e-8)
    assert np.isfinite(result.history[-1]["loss"])


def test_hetl_is_reproducible() -> None:
    source, target = make_related_domains()
    config = HeTLConfig(
        latent_dimension=3,
        beta=0.1,
        learning_rate=1e-4,
        max_steps=5,
        initialization="random",
        seed=42,
    )
    first = fit_hetl(source, target, config)
    second = fit_hetl(source, target, config)
    np.testing.assert_allclose(first.V_source, second.V_source)
    np.testing.assert_allclose(first.V_target, second.V_target)


def test_hetl_rejects_unequal_sample_counts() -> None:
    with pytest.raises(ValueError, match="样本数相同"):
        fit_hetl(
            np.zeros((10, 4)),
            np.zeros((9, 4)),
            HeTLConfig(latent_dimension=2),
        )
