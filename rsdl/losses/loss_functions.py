from rsdl import Tensor
import numpy as np

def MeanSquaredError(preds: Tensor, actual: Tensor):
    # TODO : implement mean squared error

    loss = ((preds-actual)**2).sum()
    loss.data = loss.data / preds.shape[0]
    return loss

def CategoricalCrossEntropy(preds: Tensor, actual: Tensor):
    # TODO : implement categorical cross entropy
    assert preds.data.shape == actual.data.shape, "Shape mismatch between predictions and actual values"
    preds.data.clip(1e-15, 1 - 1e-15)

    error = -(preds.log() * actual)
    out = error.data.sum()
    return out



