from torchvision import transforms, models, datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
import json

def early_stopping(validation_loss, best_loss, patience, counter, model, verbose=False):
    best_model_wts = None
    if best_loss is None or validation_loss < best_loss:
        if verbose:
            if best_loss is None:
                print(f"Validação inicial registrada: {validation_loss:.4f}")
            else:
                print(f"Validação melhorada: {validation_loss:.4f} (anterior: {best_loss:.4f})")
        best_loss = validation_loss
        counter = 0
        best_model_wts = model.state_dict()
    else:
        counter += 1
        if verbose:
            print(f"EarlyStopping: {counter}/{patience} épocas sem melhoria.")
        if counter >= patience:
            if verbose:
                print("Critério de Early Stopping atingido. Parando o treinamento.")
            return True, best_loss, counter, best_model_wts

    return False, best_loss, counter, best_model_wts


def carregar_dataset(dataset_dir, img_size):
    data_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=data_transforms)
    num_classes = len(full_dataset.classes)

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=True),
        'test': DataLoader(test_dataset, batch_size=32, shuffle=True),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return dataloaders, device, num_classes


def configurar_modelo(model_name, num_classes, pretrained=True):
    if model_name == 'alexnet':
        model = models.alexnet(weights='DEFAULT' if pretrained else None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'resnet':
        model = models.resnet50(weights='DEFAULT' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg':
        model = models.vgg16(weights='DEFAULT' if pretrained else None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'inception':
        model = models.inception_v3(weights='DEFAULT' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Modelo {model_name} não é suportado.")
    return model


def treinamento(model, model_name, optimizer_name, dataloaders, criterion, optimizer, num_epochs, device, patience=5,
                verbose=False):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_loss = None
    counter = 0
    best_model_wts = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            total = 0

            correct = torch.tensor(0, dtype=torch.int32, device=device)

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if isinstance(model, models.Inception3):
                        outputs = model(inputs)
                        loss = criterion(outputs.logits, labels)
                        preds = torch.max(outputs.logits, 1)[1]
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        preds = torch.max(outputs, 1)[1]

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                correct += torch.sum((preds == labels.data).to(torch.int))

                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = correct.double() / total

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

                stop, best_loss, counter, current_best_model_wts = early_stopping(
                    validation_loss=epoch_loss,
                    best_loss=best_loss,
                    patience=patience,
                    counter=counter,
                    model=model,
                    verbose=verbose
                )
                if current_best_model_wts:
                    best_model_wts = current_best_model_wts

                if stop:
                    print("Critério de Early Stopping atingido. Parando o treinamento.")
                    model.load_state_dict(best_model_wts)
                    torch.save(model.state_dict(), f"{model_name}_{optimizer_name}_best_model.pth")

                    # Salvar histórico
                    with open(f"{model_name}_{optimizer_name}_history.json", 'w') as f:
                        json.dump(history, f)
                    print(f"Histórico salvo como: {model_name}_{optimizer_name}_history.json")

                    return model, history

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    if best_model_wts:
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), f"{model_name}_{optimizer_name}_best_model.pth")
        print(f"Melhor modelo salvo como: {model_name}_{optimizer_name}_best_model.pth")

        # Salvar histórico
        with open(f"{model_name}_{optimizer_name}_history.json", 'w') as f:
            json.dump(history, f)
        print(f"Histórico salvo como: {model_name}_{optimizer_name}_history.json")

    return model, history


def otimizador(optimizer_name):
    if optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=0.001, betas=(0.8, 0.999))
    else:
        raise ValueError("Otimizador inválido. Escolha 'sgd' ou 'adam'.")

def gerar_grafico_historico(model_name, optimizer_name):
    filename = f"{model_name}_{optimizer_name}_history.json"
    with open(filename, 'r') as f:
        history = json.load(f)

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f"{model_name.upper()} - Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f"{model_name.upper()} - Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, 'dataset')

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"A pasta 'dataset' não foi encontrada em {dataset_dir}.")

    modelos = [
        {'model_name': 'alexnet', 'img_size': 224, 'optimizer': 'sgd'},
        {'model_name': 'alexnet', 'img_size': 224, 'optimizer': 'adam'},
        {'model_name': 'resnet', 'img_size': 224, 'optimizer': 'sgd'},
        {'model_name': 'resnet', 'img_size': 224, 'optimizer': 'adam'},
        {'model_name': 'vgg', 'img_size': 224, 'optimizer': 'sgd'},
        {'model_name': 'vgg', 'img_size': 224, 'optimizer': 'adam'},
        {'model_name': 'inception', 'img_size': 299, 'optimizer': 'sgd'},
        {'model_name': 'inception', 'img_size': 299, 'optimizer': 'adam'},
    ]

    num_epochs = 100

    for modelo in modelos:
        model_name = modelo['model_name']
        img_size = modelo['img_size']
        optimizer_name = modelo['optimizer']

        print(f"Iniciando treinamento para o modelo {model_name} com {optimizer_name}...")

        dataloaders, device, num_classes = carregar_dataset(dataset_dir, img_size)
        model = configurar_modelo(model_name, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()

        model, history = treinamento(
            model=model,
            model_name=model_name,
            optimizer_name=optimizer_name,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=otimizador(optimizer_name),
            num_epochs=num_epochs,
            device=device,
            patience=10,  # Early stopping
            verbose=True
        )

        print(f"Treinamento concluído para o modelo {model_name} com {optimizer_name}.")
