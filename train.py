import time
import torch
from colorama import Fore, Style
from augmentations import cutmix

def train_and_evaluate_model_cutmix(model, trainloader, testloader, criterion, optimizer, device, epochs):
    model.train()
    total_start_time = time.time()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    for epoch in range(epochs):
        running_loss = 0.0
        epoch_start_time = time.time()
        print(Fore.GREEN + f"Эпоха: {epoch + 1}" + Style.RESET_ALL)
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            inputs, targets_a, targets_b, lam = cutmix(inputs, labels)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        scheduler.step(running_loss)

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print(Fore.CYAN + f"Эпоха {epoch + 1} завершена за {epoch_time:.3f} секунд" + Style.RESET_ALL)

        accuracy = test_model(model, testloader, device)
        print(Fore.MAGENTA + f'Точность после {epoch + 1}-й эпохи: {accuracy:.2f}%' + Style.RESET_ALL)

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(Fore.YELLOW + f"Обучение завершено за {total_time:.3f} секунд" + Style.RESET_ALL)

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

    accuracy = 100 * correct / total
    return accuracy
