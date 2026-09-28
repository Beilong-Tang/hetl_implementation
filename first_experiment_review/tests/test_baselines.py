import numpy as np

from src.baselines import build_classifiers, positive_class_scores


def test_all_paper_baselines_are_present() -> None:
    assert set(build_classifiers(seed=42)) == {
        "CART",
        "RandomForest",
        "LinearSVM",
        "NaiveBayes",
        "KNN",
    }


def test_every_baseline_produces_attack_scores() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.9, 1.0],
            [1.0, 0.9],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    for model in build_classifiers(seed=42).values():
        model.fit(X, y)
        scores = positive_class_scores(model, X)
        assert scores.shape == (6,)
        assert np.isfinite(scores).all()
