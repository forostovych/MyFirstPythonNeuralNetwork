# Импортируем модули
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import time  # Импортируем модуль time

if __name__ == '__main__':
    from multiprocessing import freeze_support
    from colorama import Fore, Style, init
    
    # Инициализация colorama
    init(autoreset=True)
    freeze_support()  # Добавляем это, чтобы избежать проблем на Windows

    # Проверяем, доступен ли GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available! Training on GPU...")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Training on CPU...")

    # Подготовка данных
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Определение модели
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, 3)
            self.fc1 = nn.Linear(32 * 6 * 6, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 32 * 6 * 6)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Обучение модели
    net = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    total_start_time = time.time()  # Начало общего времени обучения

    for epoch in range(4):  # количество эпох
        running_loss = 0.0
        epoch_start_time = time.time()  # Запоминаем время начала эпохи
        print(Fore.GREEN + "Эпоха: " + str(epoch) + Style.RESET_ALL)  # Вывод текста зелёным цветом
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # каждые 2000 мини-батчей
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        epoch_end_time = time.time()  # Запоминаем время окончания эпохи
        epoch_time = epoch_end_time - epoch_start_time  # Вычисляем время, затраченное на эпоху, в секундах
        print(Fore.CYAN + f"Эпоха {epoch + 1} завершена за {epoch_time:.3f} секунд" + Style.RESET_ALL)

    total_end_time = time.time()  # Окончание общего времени обучения
    total_time = total_end_time - total_start_time  # Общее время обучения в секундах
    print(Fore.YELLOW + f"Обучение завершено за {total_time:.3f} секунд" + Style.RESET_ALL)

    # Тестирование модели
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Точность на тестовых данных: {100 * correct / total}%')

    # Подготавливаем данные - аугментируем данные
    transform_augmented = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset_augmented = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                      download=True, transform=transform_augmented)
    trainloader_augmented = torch.utils.data.DataLoader(trainset_augmented, batch_size=32,
                                                        shuffle=True, num_workers=2)

    # Дообучаем модель на аугментированных данных
    for epoch in range(4):  # количество эпох для дообучения
        running_loss = 0.0
        epoch_start_time = time.time()
        print(Fore.GREEN + "Эпоха (дообучение): " + str(epoch-1) + Style.RESET_ALL)
        for i, data in enumerate(trainloader_augmented, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[Epoch (дообучение) {epoch + 1}, Batch {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print(Fore.CYAN + f"Эпоха (дообучение) {epoch + 1} завершена за {epoch_time:.3f} секунд" + Style.RESET_ALL)

    # Тестируем модель после дообучения
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(Fore.MAGENTA + f'Точность на тестовых данных после дообучения: {100 * correct / total}%' + Style.RESET_ALL)
