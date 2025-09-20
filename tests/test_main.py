import pytest
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Импортируем функции из вашего кода
from main import (
    IrisLoader, balance_classes, create_tensor_with_memory_saving,
    IrisNN, validate_input_data
)


# Фикстуры для тестов
@pytest.fixture
def iris_data():
    """Фикстура для загрузки данных Iris"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y


@pytest.fixture
def preprocessed_data():
    """Фикстура для предобработанных данных"""
    loader = IrisLoader()
    X, y, data_type, num_classes, feature_names, target_names = loader.load_iris_data()
    return X, y


@pytest.fixture
def balanced_data(preprocessed_data):
    """Фикстура для сбалансированных данных"""
    X, y = preprocessed_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=24, stratify=y
    )
    X_balanced, y_balanced = balance_classes(X_train, y_train, 35)
    return X_balanced, y_balanced


@pytest.fixture
def sample_model():
    """Фикстура для тестовой модели"""
    return IrisNN(input_size=4, num_classes=3)


# Тесты для предобработки данных
class TestDataPreprocessing:

    def test_iris_loader(self):
        """Тест загрузки данных Iris"""
        loader = IrisLoader()
        X, y, data_type, num_classes, feature_names, target_names = loader.load_iris_data()

        assert X.shape[1] == 4  # 4 признака
        assert len(np.unique(y)) == 3  # 3 класса
        assert data_type == 'tabular'
        assert len(feature_names) == 4
        assert len(target_names) == 3

    def test_balance_classes(self, preprocessed_data):
        """Тест балансировки классов"""
        X, y = preprocessed_data
        samples_per_class = 30

        X_balanced, y_balanced = balance_classes(X, y, samples_per_class)

        # Проверяем, что все классы имеют одинаковое количество образцов
        unique, counts = np.unique(y_balanced, return_counts=True)
        assert all(count == samples_per_class for count in counts)

        # Проверяем, что данные не повреждены
        assert X_balanced.shape[1] == X.shape[1]
        assert len(np.unique(y_balanced)) == len(np.unique(y))

    def test_data_normalization(self, preprocessed_data):
        """Тест нормализации данных"""
        X, y = preprocessed_data

        # Проверяем, что данные нормализованы (среднее ~0, std ~1)
        assert np.allclose(np.mean(X, axis=0), 0, atol=1e-1)
        assert np.allclose(np.std(X, axis=0), 1, atol=1e-1)



# Тесты для модели
class TestModel:

    def test_model_architecture(self, sample_model):
        """Тест архитектуры модели"""
        model = sample_model

        # Проверяем количество слоев
        assert len(list(model.children())) == 5  # 3 линейных слоя + dropout + relu

        # Проверяем размеры слоев
        assert model.fc1.in_features == 4
        assert model.fc1.out_features == 64
        assert model.fc3.out_features == 3

    def test_model_forward_pass(self, sample_model, balanced_data):
        """Тест прямого прохода модели"""
        model = sample_model
        X_balanced, y_balanced = balanced_data

        # Создаем тестовый батч
        test_batch = torch.FloatTensor(X_balanced[:5])

        # Прямой проход
        output = model(test_batch)

        # Проверяем выходные данные
        assert output.shape[0] == 5  # 5 образцов
        assert output.shape[1] == 3  # 3 класса

        # Проверяем, что нет NaN значений
        assert not torch.isnan(output).any()

    def test_model_parameters(self, sample_model):
        """Тест параметров модели"""
        model = sample_model

        # Получаем все параметры
        parameters = list(model.parameters())

        # Проверяем, что параметры существуют
        assert len(parameters) > 0

        # Проверяем, что параметры инициализированы (не все нули)
        for param in parameters:
            assert not torch.all(param == 0)

    def test_model_training_mode(self, sample_model):
        """Тест переключения режимов модели"""
        model = sample_model

        # Проверяем начальный режим
        assert model.training == True

        # Переключаем в eval режим
        model.eval()
        assert model.training == False

        # Возвращаем в train режим
        model.train()
        assert model.training == True


# Тесты для метрик качества
class TestMetrics:

    def test_accuracy_calculation(self):
        """Тест расчета accuracy"""
        from sklearn.metrics import accuracy_score

        # Идеальные предсказания
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        accuracy = accuracy_score(y_true, y_pred)
        assert accuracy == 1.0

        # Все предсказания неверные
        y_pred_wrong = np.array([1, 2, 0, 1, 2, 0])
        accuracy_wrong = accuracy_score(y_true, y_pred_wrong)
        assert accuracy_wrong == 0.0

        # Частично верные предсказания - 5 из 6 верных = 0.8333...
        y_pred_partial = np.array([0, 1, 2, 1, 1, 2])
        accuracy_partial = accuracy_score(y_true, y_pred_partial)
        assert accuracy_partial >= 4 / 6  # 5 из 6 верных = 0.8333...

    def test_confusion_matrix(self):
        """Тест создания confusion matrix"""
        from sklearn.metrics import confusion_matrix

        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1])  # Одна ошибка

        cm = confusion_matrix(y_true, y_pred)

        # Проверяем размерность
        assert cm.shape == (3, 3)

        # Проверяем диагональные элементы (верные предсказания)
        assert cm[0, 0] == 2  # Класс 0: 2 верных
        assert cm[1, 1] == 2  # Класс 1: 2 верных
        assert cm[2, 2] == 1  # Класс 2: 1 верное

        # Проверяем ошибки
        assert cm[2, 1] == 1  # Класс 2 ошибочно предсказан как класс 1


# Интеграционные тесты
class TestIntegration:

    def test_end_to_end_pipeline(self, preprocessed_data):
        """Интеграционный тест полного пайплайна"""
        X, y = preprocessed_data

        # Балансировка
        X_balanced, y_balanced = balance_classes(X, y, 30)

        # Создание тензоров
        x_tensor = create_tensor_with_memory_saving(X_balanced)
        y_tensor = torch.LongTensor(y_balanced)

        # Создание модели
        model = IrisNN(input_size=4, num_classes=3)

        # Прямой проход
        output = model(x_tensor[:10])  # Тестируем на 10 образцах

        # Проверяем результаты
        assert output.shape[0] == 10
        assert output.shape[1] == 3
        assert not torch.isnan(output).any()



if __name__ == "__main__":
    pytest.main([__file__, "-v"])