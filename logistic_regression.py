from math import exp, log
from linear_algebra import dot, vector_add
from functools import reduce, partial
from random import seed, randrange, random
from woking_with_data import rescale
from multiple_regression import estimate_beta
from mchine_learning import train_test_split
from gradient_descent import maximize_batch, maximize_stochastic
seedTest = 125


def logistic(x: float):
    return 1.0 / (1 + exp(-x))


def logistic_prime(x: float):
    return logistic(x) * (1 - logistic(x))


def logistic_log_likelihood_i(x_i, y_i, beta):
    if y_i == 1:
        return log(logistic(dot(x_i, beta)))
    else:
        return log(1 - logistic(dot(x_i, beta)))


def logistic_log_likelihood(x, y, beta):
    return sum(logistic_log_likelihood_i(x_i, y_i, beta)
               for x_i, y_i in zip(x, y))


def logistic_log_partial_ij(x_i, y_i, beta, j):
    return (y_i - logistic(dot(x_i, beta))) * x_i[j]


def logistic_log_gradient_i(x_i, y_i, beta):
    return [logistic_log_partial_ij(x_i, y_i, beta, j)
            for j, _ in enumerate(beta)]


def logistic_log_gradient(x, y, beta):
    return reduce(vector_add,
                  [logistic_log_gradient_i(x_i, y_i, beta)
                   for x_i, y_i in zip(x, y)])


if __name__ == "__main__":
    seed(seedTest)
    data = [(randrange(1, 13, 1), randrange(1, 6, 1), randrange(1, 6, 1), randrange(1, 4, 1), randrange(0, 2, 1)) for _
            in range(30)]
    data = list(map(list, data))

    x = [[1] + r[:4] for r in data]
    y = [r[4] for r in data]

    print("Линейная регрессия")
    rescaled_x = rescale(x)
    beta = estimate_beta(rescaled_x, y)
    print(beta)

    print("Логистическая регрессия")

    x_train, x_test, y_train, y_test = train_test_split(rescaled_x, y, 0.33)

    fn = partial(logistic_log_likelihood, x_train, y_train)
    gradient_fn = partial(logistic_log_gradient, x_train, y_train)

    beta_0 = [random() for _ in range(5)]
    beta_hat = maximize_batch(fn, gradient_fn, beta_0)

    print("beta_batch", beta_hat)

    beta_0 = [random() for _ in range(5)]
    beta_hat = maximize_stochastic(logistic_log_likelihood_i,
                                   logistic_log_gradient_i,
                                   x_train, y_train, beta_0)

    print("beta stochastic", beta_hat)

    true_positives = false_positives = true_negatives = false_negatives = 0

    for x_i, y_i in zip(x_test, y_test):
        predict = logistic(dot(beta_hat, x_i))

        if y_i == 1 and predict >= 0.5:  # TP: paid and we predict paid
            true_positives += 1
        elif y_i == 1:                   # FN: paid and we predict unpaid
            false_negatives += 1
        elif predict >= 0.5:             # FP: unpaid and we predict paid
            false_positives += 1
        else:                            # TN: unpaid and we predict unpaid
            true_negatives += 1

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    print("Точность", precision)
    print("Полнота", recall)
