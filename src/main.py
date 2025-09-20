import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import resource
from sklearn.decomposition import PCA


def set_memory_limit(limit_gb=8j
                     ):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_limit = limit_gb * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
    print(f"Установлено ограничение памяти: {limit_gb} GB")


set_memory_limit(8)


class IrisLoader:


    @staticmethod
    def load_iris_data():

        iris = load_iris()
        X = iris.data
        y = iris.target

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, y, 'tabular', 3, iris.feature_names, iris.target_names


def balance_classes(x_data, y_data, samples_per_class=50):
    class_data = defaultdict(list)
    for x, y in zip(x_data, y_data):
        class_data[y].append(x)

    balanced_x = []
    balanced_y = []

    for class_label, samples in class_data.items():
        selected_samples = samples[:samples_per_class]
        balanced_x.extend(selected_samples)
        balanced_y.extend([class_label] * len(selected_samples))

    return np.array(balanced_x), np.array(balanced_y)



loader = IrisLoader()
X, y, data_type, num_classes, feature_names, target_names = loader.load_iris_data()

print(f"Загружен датасет: Iris Dataset")
print(f"Размер данных: {X.shape}")
print(f"Тип данных: {data_type}")
print(f"Количество классов: {num_classes}")
print(f"Названия классов: {target_names}")
print(f"Признаки: {feature_names}")


train_samples = 35
test_samples = 14

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=24, stratify=y
)

# Балансировка тренировочных данных
X_train_balanced, y_train_balanced = balance_classes(X_train, y_train, train_samples)
X_test_balanced, y_test_balanced = balance_classes(X_test, y_test, test_samples)

print(f"Сбалансированный тренировочный набор: {X_train_balanced.shape}")
print(f"Сбалансированный тестовый набор: {X_test_balanced.shape}")

# Сохранение данных
save_dir = "./data"
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, "x_train_balanced.npy"), X_train_balanced)
np.save(os.path.join(save_dir, "y_train_balanced.npy"), y_train_balanced)

save_dir = "./tests"
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, "x_test_balanced.npy"), X_test_balanced)
np.save(os.path.join(save_dir, "y_test_balanced.npy"), y_test_balanced)

# Визуализация распределения классов
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

train_balanced_counts = np.bincount(y_train_balanced)
test_balanced_counts = np.bincount(y_test_balanced)

bars1 = ax1.bar(range(len(train_balanced_counts)), train_balanced_counts, color='blue', alpha=0.7)
ax1.set_title('Сбалансированное распределение (тренировочный набор)')
ax1.set_xlabel('Класс')
ax1.set_ylabel('Количество')
ax1.set_xticks(range(len(train_balanced_counts)))
ax1.set_xticklabels(target_names, rotation=45)

bars2 = ax2.bar(range(len(test_balanced_counts)), test_balanced_counts, color='red', alpha=0.7)
ax2.set_title('Сбалансированное распределение (тестовый набор)')
ax2.set_xlabel('Класс')
ax2.set_ylabel('Количество')
ax2.set_xticks(range(len(test_balanced_counts)))
ax2.set_xticklabels(target_names, rotation=45)

save_dir = "./notebooks"
os.makedirs(save_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'balans_iris.png'), dpi=300, bbox_inches='tight')
plt.close()


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
for i, target_name in enumerate(target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], alpha=0.8, label=target_name)
plt.title('Визуализация Iris Dataset (PCA)')
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.legend()
plt.savefig(os.path.join(save_dir, 'pca_iris.png'), dpi=300, bbox_inches='tight')
plt.close()



def create_tensor_with_memory_saving(data):
    tensor = torch.from_numpy(data.astype(np.float32))
    return tensor


x_train_tensor = create_tensor_with_memory_saving(X_train_balanced)
y_train_tensor = torch.LongTensor(y_train_balanced)

x_test_tensor = create_tensor_with_memory_saving(X_test_balanced)
y_test_tensor = torch.LongTensor(y_test_balanced)


del X, y, X_train, y_train, X_test, y_test, X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced
gc.collect()

batch_size = 16
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def validate_input_data(loader, name="train"):
    print(f"\nВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ ({name})")

    data_iter = iter(loader)
    features, labels = next(data_iter)

    print(f"Размер батча: {features.shape}")
    print(f"Тип данных признаков: {features.dtype}")
    print(f"Тип данных меток: {labels.dtype}")

    if torch.isnan(features).any():
        print("Обнаружены NaN значения в признаках!")
    else:
        print("NaN значения в признаках отсутствуют")

    if torch.isnan(labels).any():
        print("Обнаружены NaN значения в метках!")
    else:
        print("NaN значения в метках отсутствуют")

    return features, labels


train_features, train_labels = validate_input_data(train_loader, "train")
test_features, test_labels = validate_input_data(test_loader, "test")


class IrisNN(nn.Module):


    def __init__(self, input_size=4, num_classes=3):
        super(IrisNN, self).__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nИспользуемое устройство: {device}")

model = IrisNN(input_size=4, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 100
train_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if batch_idx % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        if torch.isnan(data).any() or torch.isinf(data).any():
            print(f"Обнаружены NaN/Inf значения в батче {batch_idx}")
            continue

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Обнаружены NaN/Inf потери в батче {batch_idx}")
            continue

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_accuracy = 100. * correct / total
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)


    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            if torch.isnan(data).any() or torch.isinf(data).any():
                print("Обнаружены NaN/Inf значения в тестовых данных")
                continue

            output = model(data)
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()

    test_accuracy = 100. * test_correct / test_total
    test_accuracies.append(test_accuracy)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')


    torch.cuda.empty_cache()
    gc.collect()


models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)
model_save_path = os.path.join(models_dir, 'model_iris.pth')
torch.save(model.state_dict(), model_save_path)


torch.cuda.empty_cache()
gc.collect()


print("ОЦЕНКА КАЧЕСТВА МОДЕЛИ")


def get_all_predictions(model, test_loader, device):
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



y_pred, y_true = get_all_predictions(model, test_loader, device)


accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print(f"\nОБЩИЕ МЕТРИКИ:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


cm = confusion_matrix(y_true, y_pred)


plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.title('Матрица ошибок Iris Dataset')
plt.xlabel('Предсказанные метки')
plt.ylabel('Истинные метки')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'confusion_matrix_iris.png'), dpi=300, bbox_inches='tight')
plt.close()


class_accuracy = []
for i in range(num_classes):
    class_mask = (y_true == i)
    if np.sum(class_mask) > 0:
        class_acc = accuracy_score(y_true[class_mask], y_pred[class_mask])
    else:
        class_acc = 0
    class_accuracy.append(class_acc)

plt.figure(figsize=(12, 6))
bars = plt.bar(range(num_classes), class_accuracy, color='green', alpha=0.7)
plt.title('Точность по классам Iris Dataset')
plt.xlabel('Класс')
plt.ylabel('Accuracy')
plt.xticks(range(num_classes), target_names, rotation=45)
plt.ylim(0, 1.0)


for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'class_accuracy_iris.png'), dpi=300, bbox_inches='tight')
plt.close()


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'learning_curves_iris.png'), dpi=300, bbox_inches='tight')
plt.close()

def visualize_predictions_table(model, test_loader, device, target_names, num_examples=10):
    model.eval()
    features_list = []
    true_labels = []
    pred_labels = []
    probabilities = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = output.max(1)
            prob = torch.softmax(output, dim=1)

            features_list.extend(data.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())
            probabilities.extend(prob.cpu().numpy())

            if len(features_list) >= num_examples:
                break


    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    headers = ['Пример', 'Признаки', 'Истинный класс', 'Предсказанный класс', 'Вероятность', 'Статус']

    for i in range(min(num_examples, len(features_list))):
        status = "✓" if true_labels[i] == pred_labels[i] else "✗"
        table_data.append([
            f"{i + 1}",
            f"{features_list[i][0]:.2f}, {features_list[i][1]:.2f}, ...",
            target_names[true_labels[i]],
            target_names[pred_labels[i]],
            f"{np.max(probabilities[i]):.3f}",
            status
        ])

    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    plt.title('Примеры предсказаний Iris Dataset', fontsize=16)