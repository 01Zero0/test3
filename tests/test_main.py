import pytest
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


from src.main import (
    IrisLoader, balance_classes, create_tensor_with_memory_saving,
    IrisNN
)


@pytest.fixture
def real_iris_data():
    loader = IrisLoader()
    X, y, data_type, num_classes, feature_names, target_names = loader.load_iris_data()


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=24, stratify=y
    )


    X_train_balanced, y_train_balanced = balance_classes(X_train, y_train, 35)
    X_test_balanced, y_test_balanced = balance_classes(X_test, y_test, 14)

    return {
        'X_train': X_train_balanced,
        'y_train': y_train_balanced,
        'X_test': X_test_balanced,
        'y_test': y_test_balanced,
        'num_classes': num_classes,
        'feature_names': feature_names,
        'target_names': target_names
    }


@pytest.fixture
def real_data_loader(real_iris_data):
    data = real_iris_data

    x_train_tensor = create_tensor_with_memory_saving(data['X_train'])
    y_train_tensor = torch.LongTensor(data['y_train'])
    x_test_tensor = create_tensor_with_memory_saving(data['X_test'])
    y_test_tensor = torch.LongTensor(data['y_test'])

    batch_size = 16
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'num_classes': data['num_classes']
    }

class TestDataPreprocessing:

    def test_iris_loader(self):
        loader = IrisLoader()
        X, y, data_type, num_classes, feature_names, target_names = loader.load_iris_data()


        assert X.shape[1] == 4
        assert len(np.unique(y)) == 3
        assert data_type == 'tabular'
        assert len(feature_names) == 4
        assert len(target_names) == 3

        assert np.allclose(np.mean(X, axis=0), 0, atol=1e-1)
        assert np.allclose(np.std(X, axis=0), 1, atol=1e-1)

    def test_train_test_balance(self):
        loader = IrisLoader()
        X, y, _, _, _, _ = loader.load_iris_data()


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=24, stratify=y
        )


        X_train_balanced, y_train_balanced = balance_classes(X_train, y_train, 35)
        X_test_balanced, y_test_balanced = balance_classes(X_test, y_test, 14)


        train_counts = np.bincount(y_train_balanced)
        test_counts = np.bincount(y_test_balanced)

        assert all(count == 35 for count in train_counts)
        assert all(count == 14 for count in test_counts)


        assert X_train_balanced.shape == (105, 4)
        assert X_test_balanced.shape == (42, 4)


class TestModelArchitecture:

    def test_model_architecture(self):
        model = IrisNN(input_size=4, num_classes=3)

        assert model.fc1.in_features == 4
        assert model.fc1.out_features == 64
        assert model.fc2.in_features == 64
        assert model.fc2.out_features == 32
        assert model.fc3.in_features == 32
        assert model.fc3.out_features == 3


        model_custom = IrisNN(input_size=10, num_classes=5)
        assert model_custom.fc1.in_features == 10
        assert model_custom.fc3.out_features == 5

    def test_model_forward_pass(self, real_iris_data):
        model = IrisNN(input_size=4, num_classes=3)

        X = real_iris_data['X_train'][:5]
        x_tensor = torch.FloatTensor(X)

        output = model(x_tensor)

        assert output.shape == (5, 3)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_model_parameters(self):
        model = IrisNN(input_size=4, num_classes=3)


        for name, param in model.named_parameters():
            assert param.requires_grad == True
            assert param.shape.numel() > 0

    def test_model_training_mode(self):

        model = IrisNN(input_size=4, num_classes=3)
        X = torch.randn(10, 4)


        model.train()
        output_train = model(X)


        model.eval()
        output_eval = model(X)


        assert not torch.equal(output_train, output_eval)


class TestIntegration:

    def test_complete_training_pipeline(self, real_data_loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader = real_data_loader['train_loader']
        num_classes = real_data_loader['num_classes']

        model = IrisNN(input_size=4, num_classes=num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()


            if batch_idx == 0:
                assert not torch.isnan(loss)
                assert loss.item() > 0
                break


        train_accuracy = 100. * correct / total
        assert 0 <= train_accuracy <= 100
        assert running_loss > 0


class TestMetricsCalculation:

    def test_accuracy_calculation (self, real_data_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_loader = real_data_loader['test_loader']
        num_classes = real_data_loader['num_classes']

        model = IrisNN(input_size=4, num_classes=num_classes).to(device)

        def get_predictions(model, test_loader, device):
            model.eval()
            all_predictions = []
            all_targets = []

            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(device)
                    output = model(data)
                    _, predicted = output.max(1)

                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())

            return np.array(all_predictions), np.array(all_targets)

        y_pred, y_true = get_predictions(model, test_loader, device)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)


        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1


        assert len(y_pred) == len(y_true) == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])