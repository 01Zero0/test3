import pytest
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

@pytest.fixture(scope="session")
def iris_dataset():
    """Фикстура для датасета Iris"""
    iris = load_iris()
    return iris

@pytest.fixture(scope="session")
def scaler():
    """Фикстура для scaler"""
    return StandardScaler()