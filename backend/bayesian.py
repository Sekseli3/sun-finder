"""Small dependency-free Bayesian logistic-regression helpers.

The trainer finds the maximum a posteriori weights, then uses the inverse
negative Hessian at that point as a Gaussian Laplace approximation of the
posterior. The model has only a handful of features, so plain Python linear
algebra is enough for training and serving uncertainty intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


NORMAL_90_PERCENT_Z = 1.6448536269514722


@dataclass(frozen=True)
class BayesianLogisticFit:
    """Posterior summary for a Bayesian logistic-regression fit."""

    weights: list[float]
    covariance: list[list[float]]
    prior_means: list[float]
    prior_scales: list[float]
    iterations: int
    converged: bool


def sigmoid(value: float) -> float:
    """Numerically stable logistic function."""
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def logit(probability: float) -> float:
    """Log odds with enough clipping to keep a prior finite."""
    clipped = max(0.001, min(0.999, probability))
    return math.log(clipped / (1 - clipped))


def fit_bayesian_logistic(
    rows: Sequence[tuple[Sequence[float], int]],
    *,
    coefficient_prior_scale: float = 2.5,
    intercept_prior_scale: float = 2.5,
    max_iterations: int = 50,
    tolerance: float = 1e-7,
) -> BayesianLogisticFit:
    """Fit a logistic model and return a Laplace posterior approximation.

    Priors are independent normal distributions. The intercept prior is centred
    on the training-set direct-sun rate, while every feature begins centred on
    zero. Newton steps solve the MAP estimate directly, then the inverse
    negative Hessian becomes the Gaussian posterior covariance.
    """
    if not rows:
        raise ValueError("Bayesian logistic regression needs at least one row")
    if coefficient_prior_scale <= 0 or intercept_prior_scale <= 0:
        raise ValueError("Prior scales must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    feature_count = len(rows[0][0])
    if feature_count < 1:
        raise ValueError("Bayesian logistic regression needs at least one feature")
    if any(len(features) != feature_count for features, _ in rows):
        raise ValueError("Every training row must have the same number of features")
    if any(target not in {0, 1} for _, target in rows):
        raise ValueError("Targets must be zero or one")

    positive_rate = sum(target for _, target in rows) / len(rows)
    prior_means = [logit(positive_rate), *([0.0] * feature_count)]
    prior_scales = [intercept_prior_scale, *([coefficient_prior_scale] * feature_count)]
    precisions = [1 / scale**2 for scale in prior_scales]
    weights = prior_means.copy()
    current_log_posterior = log_posterior(rows, weights, prior_means, precisions)
    converged = False

    for iteration in range(1, max_iterations + 1):
        gradient, negative_hessian = gradient_and_negative_hessian(
            rows,
            weights,
            prior_means,
            precisions,
        )
        step = solve_linear_system(negative_hessian, gradient)
        step_scale = 1.0
        candidate = [weight + delta for weight, delta in zip(weights, step)]
        candidate_log_posterior = log_posterior(rows, candidate, prior_means, precisions)
        while candidate_log_posterior < current_log_posterior and step_scale > 1 / 128:
            step_scale /= 2
            candidate = [weight + step_scale * delta for weight, delta in zip(weights, step)]
            candidate_log_posterior = log_posterior(rows, candidate, prior_means, precisions)
        if candidate_log_posterior < current_log_posterior:
            break

        weights = candidate
        current_log_posterior = candidate_log_posterior
        if max(abs(step_scale * delta) for delta in step) < tolerance:
            converged = True
            break
    else:
        iteration = max_iterations

    final_gradient, negative_hessian = gradient_and_negative_hessian(rows, weights, prior_means, precisions)
    if max(abs(value) for value in final_gradient) < tolerance:
        converged = True
    covariance = invert_matrix(negative_hessian)
    return BayesianLogisticFit(
        weights=weights,
        covariance=covariance,
        prior_means=prior_means,
        prior_scales=prior_scales,
        iterations=iteration,
        converged=converged,
    )


def posterior_probability_interval(
    *,
    weights: Sequence[float],
    covariance: Sequence[Sequence[float]],
    features: Sequence[float],
    z_score: float = NORMAL_90_PERCENT_Z,
) -> tuple[float, float, float]:
    """Return posterior median and 90% interval for a fixed feature vector."""
    vector = [1.0, *features]
    if len(weights) != len(vector) or len(covariance) != len(vector):
        raise ValueError("Posterior dimensions do not match the feature vector")
    if any(len(row) != len(vector) for row in covariance):
        raise ValueError("Posterior covariance must be square")

    mean_logit = sum(weight * value for weight, value in zip(weights, vector))
    variance = sum(
        vector[row_index] * covariance[row_index][column_index] * vector[column_index]
        for row_index in range(len(vector))
        for column_index in range(len(vector))
    )
    standard_deviation = math.sqrt(max(0.0, variance))
    return (
        sigmoid(mean_logit),
        sigmoid(mean_logit - z_score * standard_deviation),
        sigmoid(mean_logit + z_score * standard_deviation),
    )


def posterior_average_probability_interval(
    *,
    weights: Sequence[float],
    covariance: Sequence[Sequence[float]],
    feature_rows: Sequence[Sequence[float]],
    z_score: float = NORMAL_90_PERCENT_Z,
) -> tuple[float, float, float]:
    """Approximate a posterior interval for an average of probabilities.

    The same fitted weights affect every forecast row, so the uncertainty of a
    one-hour average is not the average of three separate intervals. This uses
    the delta method around the MAP weights to carry the shared covariance into
    the average probability.
    """
    if not feature_rows:
        raise ValueError("At least one feature row is needed")
    dimension = len(weights)
    if len(covariance) != dimension or any(len(row) != dimension for row in covariance):
        raise ValueError("Posterior covariance must be square")

    mean_probability = 0.0
    gradient = [0.0] * dimension
    for features in feature_rows:
        vector = [1.0, *features]
        if len(vector) != dimension:
            raise ValueError("Posterior dimensions do not match the feature vector")
        probability = sigmoid(sum(weight * value for weight, value in zip(weights, vector)))
        mean_probability += probability
        slope = probability * (1 - probability)
        for index, value in enumerate(vector):
            gradient[index] += slope * value

    row_count = len(feature_rows)
    mean_probability /= row_count
    gradient = [value / row_count for value in gradient]
    variance = sum(
        gradient[row_index] * covariance[row_index][column_index] * gradient[column_index]
        for row_index in range(dimension)
        for column_index in range(dimension)
    )
    standard_deviation = math.sqrt(max(0.0, variance))
    return (
        mean_probability,
        max(0.0, mean_probability - z_score * standard_deviation),
        min(1.0, mean_probability + z_score * standard_deviation),
    )


def log_posterior(
    rows: Sequence[tuple[Sequence[float], int]],
    weights: Sequence[float],
    prior_means: Sequence[float],
    precisions: Sequence[float],
) -> float:
    value = 0.0
    for features, target in rows:
        logit_value = weights[0] + sum(weight * feature for weight, feature in zip(weights[1:], features))
        if target:
            value += log_sigmoid(logit_value)
        else:
            value += log_sigmoid(-logit_value)
    value -= 0.5 * sum(
        precision * (weight - prior_mean) ** 2
        for weight, prior_mean, precision in zip(weights, prior_means, precisions)
    )
    return value


def gradient_and_negative_hessian(
    rows: Sequence[tuple[Sequence[float], int]],
    weights: Sequence[float],
    prior_means: Sequence[float],
    precisions: Sequence[float],
) -> tuple[list[float], list[list[float]]]:
    dimension = len(weights)
    gradient = [0.0] * dimension
    negative_hessian = [[0.0] * dimension for _ in range(dimension)]

    for features, target in rows:
        vector = [1.0, *features]
        probability = sigmoid(sum(weight * value for weight, value in zip(weights, vector)))
        error = target - probability
        curvature = probability * (1 - probability)
        for row_index, value in enumerate(vector):
            gradient[row_index] += error * value
            for column_index, other_value in enumerate(vector):
                negative_hessian[row_index][column_index] += curvature * value * other_value

    for index, precision in enumerate(precisions):
        gradient[index] -= precision * (weights[index] - prior_means[index])
        negative_hessian[index][index] += precision
    return gradient, negative_hessian


def solve_linear_system(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    """Solve a small positive-definite system with pivoted elimination."""
    size = len(vector)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("Matrix dimensions do not match the vector")
    augmented = [list(row) + [value] for row, value in zip(matrix, vector)]

    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("Posterior Hessian is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for item in range(column, size + 1):
            augmented[column][item] /= pivot_value
        for row in range(size):
            if row == column:
                continue
            multiplier = augmented[row][column]
            if multiplier == 0:
                continue
            for item in range(column, size + 1):
                augmented[row][item] -= multiplier * augmented[column][item]
    return [augmented[row][size] for row in range(size)]


def invert_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """Invert a small matrix by solving once for each identity column."""
    size = len(matrix)
    inverse_columns = [
        solve_linear_system(matrix, [1.0 if row == column else 0.0 for row in range(size)])
        for column in range(size)
    ]
    return [[inverse_columns[column][row] for column in range(size)] for row in range(size)]


def log_sigmoid(value: float) -> float:
    if value >= 0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))
