import pytest
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import defaultdict


from src.main import (
    IrisLoader, balance_classes, create_tensor_with_memory_saving,
    IrisNN, validate_input_data
)



@pytest.fixture
def iris_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y


@pytest.fixture
def preprocessed_data():
    loader = IrisLoader()
    X, y, data_type, num_classes, feature_names, target_names = loader.load_iris_data()
    return X, y


@pytest.fixture
def balanced_data(preprocessed_data):
    X, y = preprocessed_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=24, stratify=y
    )
    X_balanced, y_balanced = balance_classes(X_train, y_train, 35)
    return X_balanced, y_balanced


@pytest.fixture
def sample_model():
    return IrisNN(input_size=4, num_classes=3)


# Тесты для предобработки данных
class TestDataPreprocessing:

    def test_iris_loader(self):

        loader = IrisLoader()
        X, y, data_type, num_classes, feature_names, target_names = loader.load_iris_data()

        assert X.shape[1] == 4  # 4 признака
        assert len(np.unique(y)) == 3  # 3 класса
        assert data_type == 'tabular'
        assert len(feature_names) == 4
        assert len(target_names) == 3

    def test_balance_classes(self, preprocessed_data):
        X, y = preprocessed_data
        samples_per_class = 30

        X_balanced, y_balanced = balance_classes(X, y, samples_per_class)


        unique, counts = np.unique(y_balanced, return_counts=True)
        assert all(count == samples_per_class for count in counts)


        assert X_balanced.shape[1] == X.shape[1]
        assert len(np.unique(y_balanced)) == len(np.unique(y))

    def test_data_normalization(self, preprocessed_data):

        X, y = preprocessed_data


        assert np.allclose(np.mean(X, axis=0), 0, atol=1e-1)
        assert np.allclose(np.std(X, axis=0), 1, atol=1e-1)




class TestModel:

    def test_model_architecture(self, sample_model):

        model = sample_model


        assert len(list(model.children())) == 5  # 3 линейных слоя + dropout + relu


        assert model.fc1.in_features == 4
        assert model.fc1.out_features == 64
        assert model.fc3.out_features == 3

    def test_model_forward_pass(self, sample_model, balanced_data):

        model = sample_model
        X_balanced, y_balanced = balanced_data


        test_batch = torch.FloatTensor(X_balanced[:5])


        output = model(test_batch)

        assert output.shape[0] == 5  # 5 образцов
        assert output.shape[1] == 3  # 3 класса


        assert not torch.isnan(output).any()

    def test_model_parameters(self, sample_model):

        model = sample_model


        parameters = list(model.parameters())


        assert len(parameters) > 0


        for param in parameters:
            assert not torch.all(param == 0)

    def test_model_training_mode(self, sample_model):

        model = sample_model


        assert model.training == True


        model.eval()
        assert model.training == False


        model.train()
        assert model.training == True



class TestMetrics:

    def test_accuracy_calculation(self):
        from sklearn.metrics import accuracy_score


        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        accuracy = accuracy_score(y_true, y_pred)
        assert accuracy == 1.0


        y_pred_wrong = np.array([1, 2, 0, 1, 2, 0])
        accuracy_wrong = accuracy_score(y_true, y_pred_wrong)
        assert accuracy_wrong == 0.0


        y_pred_partial = np.array([0, 1, 2, 1, 1, 2])
        accuracy_partial = accuracy_score(y_true, y_pred_partial)
        assert accuracy_partial >= 4 / 6  # 5 из 6 верных = 0.8333...

    def test_confusion_matrix(self):

        from sklearn.metrics import confusion_matrix

        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1])  # Одна ошибка

        cm = confusion_matrix(y_true, y_pred)


        assert cm.shape == (3, 3)


        assert cm[0, 0] == 2  # Класс 0: 2 верных
        assert cm[1, 1] == 2  # Класс 1: 2 верных
        assert cm[2, 2] == 1  # Класс 2: 1 верное


        assert cm[2, 1] == 1  # Класс 2 ошибочно предсказан как класс 1



class TestIntegration:

    def test_end_to_end_pipeline(self, preprocessed_data):

        X, y = preprocessed_data


        X_balanced, y_balanced = balance_classes(X, y, 30)


        x_tensor = create_tensor_with_memory_saving(X_balanced)
        y_tensor = torch.LongTensor(y_balanced)


        model = IrisNN(input_size=4, num_classes=3)


        output = model(x_tensor[:10])  # Тестируем на 10 образцах


        assert output.shape[0] == 10
        assert output.shape[1] == 3
        assert not torch.isnan(output).any()



if __name__ == "__main__":
    pytest.main([__file__, "-v"])