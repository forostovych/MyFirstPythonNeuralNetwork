import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from model import DeeperImprovedCNN
from train import train_and_evaluate_model_cutmix
import torchvision.models as models
from multiprocessing import freeze_support
from colorama import Fore, Style

if __name__ == '__main__':
    # Поддержка freeze_support для Windows при использовании многопроцессорности
    freeze_support()

    # Проверяем, доступен ли GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available! Training on GPU...")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Training on CPU...")

    # Функция для подготовки данных CIFAR-10
    def prepare_data(transform):
        # Загрузка и преобразование обучающего набора данных
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

        # Загрузка и преобразование тестового набора данных
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
        return trainloader, testloader

    # Обучение модели без использования аугментаций
    print(Fore.BLUE + "=== Обучение модели без аугментации ===" + Style.RESET_ALL)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Преобразование изображений в тензоры
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормализация
    ])
    trainloader, testloader = prepare_data(transform)
    net = DeeperImprovedCNN().to(device)  # Создание модели и отправка ее на устройство (GPU или CPU)
    criterion = torch.nn.CrossEntropyLoss()  # Функция потерь
    optimizer = optim.AdamW(net.parameters(), lr=0.0005, weight_decay=1e-4)  # Оптимизатор с весовой регуляризацией
    train_and_evaluate_model_cutmix(net, trainloader, testloader, criterion, optimizer, device, epochs=1)

    # Обучение модели с аугментацией данных
    print(Fore.BLUE + "=== Дообучение модели с аугментацией данных (Flip, Rotation) ===" + Style.RESET_ALL)
    transform_augmented = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Случайный горизонтальный переворот
        transforms.RandomRotation(10),  # Случайный поворот
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Изменение яркости, контрастности и насыщенности
        transforms.ToTensor(),  # Преобразование изображений в тензоры
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # Случайное удаление участков изображения
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормализация
    ])
    trainloader_augmented, _ = prepare_data(transform_augmented)
    train_and_evaluate_model_cutmix(net, trainloader_augmented, testloader, criterion, optimizer, device, epochs=1)

    # Дополнительное обучение модели с другой аугментацией данных
    print(Fore.BLUE + "=== Ещё одно дообучение модели с другой аугментацией данных ===" + Style.RESET_ALL)
    transform_augmented_more = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # Случайное изменение размера и обрезка
        transforms.RandomHorizontalFlip(),  # Случайный горизонтальный переворот
        transforms.RandomRotation(20),  # Случайный поворот
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Изменение яркости, контрастности и насыщенности
        transforms.ToTensor(),  # Преобразование изображений в тензоры
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # Случайное удаление участков изображения
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормализация
    ])
    trainloader_augmented_more, _ = prepare_data(transform_augmented_more)
    train_and_evaluate_model_cutmix(net, trainloader_augmented_more, testloader, criterion, optimizer, device, epochs=1)

    # Обучение модели ResNet с предобученными весами
    from torchvision.models import ResNet18_Weights
    print(Fore.BLUE + "=== Обучение модели ResNet ===" + Style.RESET_ALL)
    net_resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)  # Загрузка модели ResNet18 с предобученными весами
    net_resnet.fc = torch.nn.Linear(net_resnet.fc.in_features, 10)  # Замена последнего слоя на слой для 10 классов
    net_resnet = net_resnet.to(device)
    optimizer_resnet = optim.AdamW(net_resnet.parameters(), lr=0.0005, weight_decay=1e-4)  # Оптимизатор с весовой регуляризацией
    train_and_evaluate_model_cutmix(net_resnet, trainloader, testloader, criterion, optimizer_resnet, device, epochs=1)
