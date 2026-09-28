import numpy as np
import pytest

from src.data_loader import VALID_TASKS, load_task


@pytest.mark.parametrize("task", VALID_TASKS)
def test_seed_42_task_contract(task: str) -> None:
    data = load_task(task, seed=42)

    assert data.X_source.shape[0] == 1990
    assert data.X_target.shape[0] == 1990
    assert data.X_source.shape[1] == data.X_target.shape[1]
    assert len(data.feature_names) == data.X_source.shape[1]
    assert np.bincount(data.y_source).tolist() == [995, 995]
    assert np.bincount(data.y_target).tolist() == [995, 995]
    assert len(data.validation_indices) == 500
    assert len(data.test_indices) == 1490
    assert set(data.validation_indices).isdisjoint(set(data.test_indices))


def test_unknown_task_is_rejected() -> None:
    with pytest.raises(ValueError, match="未知任务"):
        load_task("wrong_task", seed=42)
