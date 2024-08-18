# Импортируем модули
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import time
from colorama import Fore, Style, init
from multiprocessing import freeze_support

# Инициализация colorama
init(autoreset=True)

if __name__ == '__main__':
    freeze_support()  # Добавляем это, чтобы избежать проблем на Windows

    # Проверяем, доступен ли GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available! Training on GPU...")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Training on CPU...")

    # Подготовка данных
    def prepare_data(transform):
        # Загружаем и подготавливаем обучающий и тестовый наборы данных CIFAR-10
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                                 shuffle=False, num_workers=2)
        return trainloader, testloader

    # Определение улучшенной модели
    class ImprovedCNN(nn.Module):
        def __init__(self):
            super(ImprovedCNN, self).__init__()
            # Первый сверточный блок
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Сверточный слой, 32 фильтра, размер ядра 3x3
            self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization для стабилизации
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # Второй сверточный слой, 64 фильтра
            self.bn2 = nn.BatchNorm2d(64)  # Batch Normalization

            # Второй сверточный блок
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # Третий сверточный слой, 128 фильтров
            self.bn3 = nn.BatchNorm2d(128)  # Batch Normalization

            # Полносвязные слои
            self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Полносвязный слой с 512 нейронами
            self.fc2 = nn.Linear(512, 256)  # Полносвязный слой с 256 нейронами
            self.fc3 = nn.Linear(256, 10)  # Выходной слой для 10 классов (CIFAR-10)

            # MaxPooling и Dropout для регуляризации
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.5)  # Dropout для предотвращения переобучения

        def forward(self, x):
            # Прямой проход через модель
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = x.view(-1, 128 * 4 * 4)  # Преобразуем тензор в одномерный вектор для полносвязных слоев
            x = F.relu(self.fc1(x))
            x = self.dropout(x)  # Применяем Dropout
            x = F.relu(self.fc2(x))
            x = self.dropout(x)  # Применяем Dropout
            x = self.fc3(x)
            return x

    # Функция для обучения модели
    def train_model(model, trainloader, criterion, optimizer, device, epochs):
        total_start_time = time.time()
        for epoch in range(epochs):
            running_loss = 0.0
            epoch_start_time = time.time()
            print(Fore.GREEN + f"Эпоха: {epoch + 1}" + Style.RESET_ALL)
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            print(Fore.CYAN + f"Эпоха {epoch + 1} завершена за {epoch_time:.3f} секунд" + Style.RESET_ALL)

        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        print(Fore.YELLOW + f"Обучение завершено за {total_time:.3f} секунд" + Style.RESET_ALL)

    # Функция для тестирования модели
    def test_model(model, testloader, device):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(Fore.MAGENTA + f'Точность на тестовых данных: {100 * correct / total}%' + Style.RESET_ALL)

    # Основная последовательность выполнения

    # 1. Обучаем модель без аугментации
    print(Fore.BLUE + "=== Обучение модели без аугментации ===" + Style.RESET_ALL)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainloader, testloader = prepare_data(transform)
    net = ImprovedCNN().to(device)  # Используем улучшенную модель
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # Переходим на оптимизатор Adam
    train_model(net, trainloader, criterion, optimizer, device, epochs=15)
    test_model(net, testloader, device)

    # 2. Аугментация данных и дообучение
    print(Fore.BLUE + "=== Дообучение модели с аугментацией данных (Flip, Rotation) ===" + Style.RESET_ALL)
    transform_augmented = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainloader_augmented, _ = prepare_data(transform_augmented)
    train_model(net, trainloader_augmented, criterion, optimizer, device, epochs=15)
    test_model(net, testloader, device)

    # 3. Ещё одна аугментация и дообучение
    print(Fore.BLUE + "=== Ещё одно дообучение модели с другой аугментацией данных ===" + Style.RESET_ALL)
    transform_augmented_more = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainloader_augmented_more, _ = prepare_data(transform_augmented_more)
    train_model(net, trainloader_augmented_more, criterion, optimizer, device, epochs=15)
    test_model(net, testloader, device)
