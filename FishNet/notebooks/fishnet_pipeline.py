"""
FishNet: Sistema de Deteccion y Clasificacion de Peces en Imagenes
       mediante Redes Neuronales Convolucionales (PyTorch)

Pipeline completo:
  1. Carga y preprocesamiento del dataset
  2. CNN Personalizada (Baseline)
  3. MobileNetV2 con Transfer Learning
  4. ResNet50 con Transfer Learning
  5. Evaluacion comparativa
  6. Grad-CAM visualizations
"""

import os, sys, gc, warnings, random, zipfile, copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

from PIL import Image

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')

# ─── Configuracion ───────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = BASE_DIR / 'data'
RAW_DIR         = DATA_DIR / 'raw'
PROCESSED_DIR   = DATA_DIR / 'processed'
MODELS_DIR      = BASE_DIR / 'models'
NOTEBOOKS_DIR   = BASE_DIR / 'notebooks'
CACHE_DIR       = BASE_DIR / 'cache'

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, NOTEBOOKS_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS_P1  = 30
EPOCHS_P2  = 20
LR_P1      = 1e-3
LR_P2      = 1e-5
N_CLASSES  = 9
SEED       = 42
VALID_SPLIT = 0.15
TEST_SPLIT  = 0.15
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

CLASS_NAMES = [
    'Dorada (Sparus aurata)',
    'Lubina (Dicentrarchus labrax)',
    'Trucha artica (Salvelinus alpinus)',
    'Pargo (Pagrus pagrus)',
    'Salmonete (Mullus barbatus)',
    'Caballa (Scomber scombrus)',
    'Anchoa (Engraulis encrasicolus)',
    'Jurel (Trachurus trachurus)',
    'Bacaladilla (Micromesistius poutassou)',
]

print(f"=== FishNet v1.0 (PyTorch) ===")
print(f"Dispositivo: {DEVICE}")
print(f"PyTorch {torch.__version__}  |  Python {sys.version.split()[0]}")
print(f"Dataset: {N_CLASSES} clases | Imagenes {IMG_SIZE}")
print(f"Directorios creados en {BASE_DIR}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECCION 1: CARGA Y PREPROCESAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

def explore_dataset(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"Ruta {data_path} no encontrada")
        return

    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    print(f"\n{'='*60}")
    print(f"Explorando dataset en: {data_path}")
    print(f"Total de directorios de clase: {len(class_dirs)}")
    print(f"{'='*60}")

    stats = []
    for cd in class_dirs:
        imgs = list(cd.glob('*.*'))
        ext_counts = {}
        for img in imgs:
            ext = img.suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        stats.append({
            'clase': cd.name,
            'total': len(imgs),
            'extensiones': ext_counts
        })
        print(f"  {cd.name}: {len(imgs)} imagenes {ext_counts}")

    df_stats = pd.DataFrame(stats)
    print(f"\nTotal imagenes: {df_stats['total'].sum()}")
    print(f"Promedio por clase: {df_stats['total'].mean():.1f} +/- {df_stats['total'].std():.1f}")
    print(f"Min: {df_stats['total'].min()}, Max: {df_stats['total'].max()}")

    if len(class_dirs) > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(df_stats['clase'], df_stats['total'],
                      color=sns.color_palette('viridis', len(df_stats)))
        ax.set_xlabel('Clase')
        ax.set_ylabel('Numero de imagenes')
        ax.set_title('Distribucion del dataset por clase')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        for bar, val in zip(bars, df_stats['total']):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(val), ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(BASE_DIR / 'distribucion_dataset.png', dpi=150)
        plt.show()

    return df_stats


def prepare_datasets(data_path, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Carga el dataset usando ImageFolder, divide en train/val/test,
    aplica transforms con data augmentation para train.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado en {data_path}")

    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15),
                                scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset_full = ImageFolder(str(data_path), transform=test_transform)
    class_names = dataset_full.classes
    n_classes = len(class_names)
    n_total = len(dataset_full)
    indices = list(range(n_total))
    labels = [dataset_full.targets[i] for i in indices]

    X_train_idx, X_temp_idx, _, _ = train_test_split(
        indices, labels, test_size=VALID_SPLIT + TEST_SPLIT,
        stratify=labels, random_state=SEED
    )
    val_indices, test_indices, _, _ = train_test_split(
        X_temp_idx, [labels[i] for i in X_temp_idx],
        test_size=TEST_SPLIT / (VALID_SPLIT + TEST_SPLIT),
        stratify=[labels[i] for i in X_temp_idx], random_state=SEED
    )

    # Dataset train con augmentacion
    dataset_train = ImageFolder(str(data_path), transform=train_transform)

    train_dataset = Subset(dataset_train, X_train_idx)
    val_dataset = Subset(dataset_full, val_indices)
    test_dataset = Subset(dataset_full, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    print(f"\nSplit: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print(f"Clases ({n_classes}): {class_names}")

    return train_loader, val_loader, test_loader, class_names, dataset_full


# ═══════════════════════════════════════════════════════════════════════════════
# SECCION 2: ARQUITECTURAS
# ═══════════════════════════════════════════════════════════════════════════════

class CNNBaseline(nn.Module):
    """
    CNN Personalizada (Baseline)
    Input(224x224x3)
      -> Conv2D(32, 3x3) + ReLU + MaxPooling
      -> Conv2D(64, 3x3) + ReLU + MaxPooling
      -> Conv2D(128, 3x3) + ReLU + MaxPooling
      -> Conv2D(256, 3x3) + ReLU + MaxPooling
      -> GlobalAvgPooling + Dense(256) + Dropout(0.5)
      -> Dense(N_clases)
    """
    def __init__(self, n_classes=N_CLASSES, dropout_rate=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_mobilenetv2_model(n_classes=N_CLASSES, dropout_rate=0.3):
    """
    MobileNetV2 con Transfer Learning.
    Fase 1: backbone congelado.
    Fase 2: fine-tuning ultimas capas.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, n_classes),
    )
    return model


def build_resnet50_model(n_classes=N_CLASSES, dropout_rate=0.3):
    """
    ResNet50 con Transfer Learning.
    Fase 1: backbone congelado.
    Fase 2: fine-tuning desde layer4.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, n_classes),
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# SECCION 3: ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_epoch(model, loader, criterion, optimizer, device=DEVICE):
    model.train()
    running_loss, running_acc = 0.0, 0.0
    n_samples = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_acc += torch.sum(preds == labels.data).item()
        n_samples += inputs.size(0)
    return running_loss / n_samples, running_acc / n_samples


def val_epoch(model, loader, criterion, device=DEVICE):
    model.eval()
    running_loss, running_acc = 0.0, 0.0
    n_samples = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_acc += torch.sum(preds == labels.data).item()
            n_samples += inputs.size(0)
    return running_loss / n_samples, running_acc / n_samples


def train_model(model, train_loader, val_loader, model_name,
                epochs=EPOCHS_P1, learning_rate=LR_P1, phase=1):
    print(f"\n{'='*60}")
    print(f"FASE {phase} - {model_name}: LR={learning_rate}, Epocas={epochs}")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=5, min_lr=1e-7, verbose=True)
    early_stopping = EarlyStopping(patience=10)

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_val_acc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = val_epoch(model, val_loader, criterion)

        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODELS_DIR / f'{model_name}_phase{phase}_best.pth')

        scheduler.step(val_loss)
        early_stopping(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if early_stopping.early_stop:
            print(f"  Early stopping en epoch {epoch+1}")
            break

    return history


def unfreeze_mobilenetv2(model, unfreeze_last_n=30):
    """Descongela las ultimas N capas de MobileNetV2 para fine-tuning."""
    for param in model.parameters():
        param.requires_grad = False
    layers = list(model.children())[0].features if hasattr(model, 'features') else []
    if not layers:
        layers = list(model.features)
    for layer in layers[-unfreeze_last_n:]:
        for param in layer.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    print(f"  Descongeladas ultimas {unfreeze_last_n} capas + clasificador")


def unfreeze_resnet50(model, unfreeze_from='layer4'):
    """Descongela desde una capa especifica de ResNet50."""
    for param in model.parameters():
        param.requires_grad = False
    unfreeze = False
    for name, child in model.named_children():
        if name == unfreeze_from:
            unfreeze = True
        if unfreeze or name == 'fc':
            for param in child.parameters():
                param.requires_grad = True
    print(f"  Descongelado desde {unfreeze_from}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECCION 4: EVALUACION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_model(model, loader, device=DEVICE):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def evaluate_model(model, test_loader, model_name, class_names=CLASS_NAMES):
    print(f"\n{'='*60}")
    print(f"EVALUACION: {model_name}")
    print(f"{'='*60}")

    y_true, y_pred, y_probs = predict_model(model, test_loader)

    acc = np.mean(y_true == y_pred)
    print(f"  Accuracy:  {acc:.4f}")

    print(f"\nClassification Report ({model_name}):")
    print(classification_report(y_true, y_pred, target_names=class_names[:len(np.unique(y_true))],
                                digits=4))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names[:cm.shape[0]],
                yticklabels=class_names[:cm.shape[0]], ax=ax)
    ax.set_xlabel('Prediccion')
    ax.set_ylabel('Verdad')
    ax.set_title(f'Matriz de Confusion - {model_name}')
    plt.tight_layout()
    plt.savefig(MODELS_DIR / f'cm_{model_name}.png', dpi=150)
    plt.show()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    metrics_df = pd.DataFrame({
        'Clase': class_names[:len(precision)],
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
    })
    print(f"\nMetricas por clase:")
    print(metrics_df.to_string(index=False))

    n_classes = y_probs.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Curvas ROC (One-vs-Rest) - {model_name}')
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / f'roc_{model_name}.png', dpi=150)
    plt.show()

    return {
        'model_name': model_name,
        'accuracy': acc,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_probs': y_probs,
        'cm': cm,
        'metrics_df': metrics_df,
    }


def plot_training_history(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['loss'], label='Train Loss', lw=2)
    axes[0].plot(history['val_loss'], label='Val Loss', lw=2)
    axes[0].set_xlabel('Epocas')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Curva de Loss - {model_name}')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history['accuracy'], label='Train Acc', lw=2)
    axes[1].plot(history['val_accuracy'], label='Val Acc', lw=2)
    axes[1].set_xlabel('Epocas')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'Curva de Accuracy - {model_name}')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(MODELS_DIR / f'history_{model_name}.png', dpi=150)
    plt.show()


def compare_models(results_list):
    df = pd.DataFrame(results_list)
    df = df.drop(columns=['y_true', 'y_pred', 'y_pred_probs', 'cm', 'metrics_df'],
                 errors='ignore')

    print(f"\n{'='*60}")
    print(f"COMPARACION DE MODELOS")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    metrics = ['accuracy', 'precision', 'recall']
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.25
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width,
               [r.get(metric, 0) or 0 for _, r in df.iterrows()],
               width, label=metric.capitalize())

    ax.set_xlabel('Modelo')
    ax.set_ylabel('Score')
    ax.set_title('Comparacion de metricas entre modelos')
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(df['model_name'])
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'comparacion_modelos.png', dpi=150)
    plt.show()

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECCION 5: GRAD-CAM
# ═══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Grad-CAM usando hooks de PyTorch.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        pooled_grads = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (self.activations * pooled_grads).sum(dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        return heatmap.squeeze().cpu().numpy(), class_idx


def find_last_conv(model):
    """Encuentra la ultima capa Conv2d del modelo."""
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def display_gradcam(image_path, model, class_names, img_size=IMG_SIZE, alpha=0.4):
    from PIL import Image

    original = Image.open(image_path).convert('RGB').resize(img_size)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(original).unsqueeze(0).to(DEVICE)

    last_conv = find_last_conv(model)
    if last_conv is None:
        print("  No se encontro capa Conv2d")
        return None, None, None

    gradcam = GradCAM(model, last_conv)
    heatmap, pred_class = gradcam.generate(input_tensor)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence = probs[0, pred_class].item()

    heatmap_resized = np.array(Image.fromarray(heatmap).resize(img_size, Image.BILINEAR))
    img_display = np.array(original) / 255.0
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
    superimposed = (1 - alpha) * img_display + alpha * heatmap_colored
    superimposed = np.clip(superimposed, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_display)
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    axes[2].imshow(superimposed)
    axes[2].set_title(f'Superposicion\nPred: {class_names[pred_class]} ({confidence:.2%})')
    axes[2].axis('off')
    plt.tight_layout()
    plt.show()

    return pred_class, confidence, heatmap


# ═══════════════════════════════════════════════════════════════════════════════
# SECCION 6: PIPELINE COMPLETO
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(data_path=None):
    all_results = []

    if data_path is None:
        candidates = [
            RAW_DIR / 'fish_dataset',
            RAW_DIR / 'Fish_Dataset',
            RAW_DIR / 'FishDataset',
            DATA_DIR / 'Fish_Dataset',
        ]
        for c in candidates:
            if c.exists() and any(c.iterdir()):
                data_path = c
                break
        if data_path is None:
            print("Dataset no encontrado.")
            print(f"Coloque las imagenes en: {RAW_DIR / 'fish_dataset'}/")
            return

    print(f"\nUsando dataset en: {data_path}")

    explore_dataset(data_path)

    try:
        train_loader, val_loader, test_loader, class_names, dataset_full = \
            prepare_datasets(data_path)
        n_classes = len(class_names)
    except Exception as e:
        print(f"Error preparando dataset: {e}")
        return

    # 1. CNN Baseline
    print(f"\n{'#'*60}")
    print(f"# MODELO 1: CNN Personalizada (Baseline)")
    print(f"{'#'*60}")
    cnn_model = CNNBaseline(n_classes=n_classes, dropout_rate=0.5).to(DEVICE)
    h1 = train_model(cnn_model, train_loader, val_loader, 'CNN_Baseline',
                     epochs=EPOCHS_P1, learning_rate=LR_P1, phase=1)
    plot_training_history(h1, 'CNN_Baseline')

    # Segunda fase: LR reducido
    h1b = train_model(cnn_model, train_loader, val_loader, 'CNN_Baseline',
                      epochs=10, learning_rate=LR_P2, phase=2)
    for k in h1:
        h1[k] = h1[k] + h1b[k]
    plot_training_history(h1, 'CNN_Baseline_Final')

    r1 = evaluate_model(cnn_model, test_loader, 'CNN_Baseline', class_names)
    all_results.append(r1)
    torch.save(cnn_model.state_dict(), MODELS_DIR / 'cnn_baseline_final.pth')

    # 2. MobileNetV2
    print(f"\n{'#'*60}")
    print(f"# MODELO 2: MobileNetV2 + Transfer Learning")
    print(f"{'#'*60}")
    mobilenet_model = build_mobilenetv2_model(n_classes=n_classes).to(DEVICE)
    print(mobilenet_model)
    h2 = train_model(mobilenet_model, train_loader, val_loader, 'MobileNetV2',
                     epochs=EPOCHS_P1, learning_rate=LR_P1, phase=1)
    plot_training_history(h2, 'MobileNetV2_P1')

    unfreeze_mobilenetv2(mobilenet_model, unfreeze_last_n=30)
    h2b = train_model(mobilenet_model, train_loader, val_loader, 'MobileNetV2',
                      epochs=EPOCHS_P2, learning_rate=LR_P2, phase=2)
    plot_training_history(h2b, 'MobileNetV2_P2')

    r2 = evaluate_model(mobilenet_model, test_loader, 'MobileNetV2', class_names)
    all_results.append(r2)
    torch.save(mobilenet_model.state_dict(), MODELS_DIR / 'mobilenetv2_final.pth')

    # 3. ResNet50
    print(f"\n{'#'*60}")
    print(f"# MODELO 3: ResNet50 + Transfer Learning")
    print(f"{'#'*60}")
    resnet_model = build_resnet50_model(n_classes=n_classes).to(DEVICE)
    print(resnet_model)
    h3 = train_model(resnet_model, train_loader, val_loader, 'ResNet50',
                     epochs=EPOCHS_P1, learning_rate=LR_P1, phase=1)
    plot_training_history(h3, 'ResNet50_P1')

    unfreeze_resnet50(resnet_model, unfreeze_from='layer4')
    h3b = train_model(resnet_model, train_loader, val_loader, 'ResNet50',
                      epochs=EPOCHS_P2, learning_rate=LR_P2, phase=2)
    plot_training_history(h3b, 'ResNet50_P2')

    r3 = evaluate_model(resnet_model, test_loader, 'ResNet50', class_names)
    all_results.append(r3)
    torch.save(resnet_model.state_dict(), MODELS_DIR / 'resnet50_final.pth')

    # 4. Comparacion
    compare_models(all_results)

    # 5. Grad-CAM
    print(f"\n{'#'*60}")
    print(f"# GRAD-CAM: Visualizaciones de Activacion")
    print(f"{'#'*60}")

    test_dataset = test_loader.dataset
    sample_indices = min(6, len(test_dataset))
    for i in range(sample_indices):
        img_path, _ = dataset_full.samples[test_dataset.indices[i]]
        print(f"\nEjemplo {i+1}: {Path(img_path).name}")
        try:
            display_gradcam(img_path, resnet_model, class_names)
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETADO")
    print(f"{'='*60}")
    print(f"Modelos guardados en: {MODELS_DIR}")

    return all_results


if __name__ == '__main__':
    run_pipeline()
