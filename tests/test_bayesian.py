from __future__ import annotations

import unittest

from backend.bayesian import (
    fit_bayesian_logistic,
    posterior_average_probability_interval,
    posterior_probability_interval,
)


class BayesianLogisticTests(unittest.TestCase):
    def test_fit_learns_a_negative_cloud_weight_and_a_posterior_range(self) -> None:
        rows = [([0.0], 1)] * 30 + [([1.0], 0)] * 30

        fit = fit_bayesian_logistic(rows, max_iterations=30)
        clear_probability, clear_low, clear_high = posterior_probability_interval(
            weights=fit.weights,
            covariance=fit.covariance,
            features=[0.0],
        )
        cloudy_probability, cloudy_low, cloudy_high = posterior_probability_interval(
            weights=fit.weights,
            covariance=fit.covariance,
            features=[1.0],
        )

        self.assertTrue(fit.converged)
        self.assertLess(fit.weights[1], 0)
        self.assertGreater(clear_probability, cloudy_probability)
        self.assertLessEqual(clear_low, clear_probability)
        self.assertGreaterEqual(clear_high, clear_probability)
        self.assertLessEqual(cloudy_low, cloudy_probability)
        self.assertGreaterEqual(cloudy_high, cloudy_probability)
        self.assertTrue(all(fit.covariance[index][index] > 0 for index in range(2)))

        average_probability, average_low, average_high = posterior_average_probability_interval(
            weights=fit.weights,
            covariance=fit.covariance,
            feature_rows=[[0.0], [1.0]],
        )
        self.assertLess(cloudy_probability, average_probability)
        self.assertLess(average_probability, clear_probability)
        self.assertLessEqual(average_low, average_probability)
        self.assertGreaterEqual(average_high, average_probability)


if __name__ == "__main__":
    unittest.main()
