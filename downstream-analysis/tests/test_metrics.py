import numpy as np

from interface_downstream.metrics import regression_metrics


def test_regression_metrics_include_mse_and_consistent_rmse():
    metrics = regression_metrics(np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 4.0, 2.0]))

    assert metrics["mse"] == 5.0 / 3.0
    assert metrics["rmse"] == np.sqrt(metrics["mse"])
