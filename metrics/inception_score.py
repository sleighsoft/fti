import numpy as np


def inception_score(py_x, splits=10):
    """Computes the Inception Score as reported by https://arxiv.org/pdf/1606.03498.pdf.

    Can be used with any input.

    For original results use:
        - Inception-v3 Network: https://github.com/tensorflow/models/tree/master/research/slim.
        - @Node: 'InceptionV3/Predictions/Reshape:0'
    
    Note: This expects the output of a classifier network!
    
    Args:
        py_x: Conditional probabilities of y (class probabilities) given x of shape [Batch, Classes].
        splits: Number of times to split `py_x` to calculate the mean.

    Returns:
        mean: Mean IS over splits.
        std: Standard deviation of IS over splits.
    """
    scores = []
    for i in range(splits):
        start = i * py_x.shape[0] // splits
        end = (i + 1) * py_x.shape[0] // splits
        part = py_x[start:end, :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)
