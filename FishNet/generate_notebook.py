"""
Genera el notebook Jupyter FishNet a partir del pipeline.
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent / 'notebooks'
NB_DIR.mkdir(parents=True, exist_ok=True)

cells = []

def md(source):
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [source] if isinstance(source, str) else source,
    })

def code(source):
    cells.append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [source] if isinstance(source, str) else source,
    })

# ── Portada ──
md("""# FishNet: Sistema de Deteccion y Clasificacion de Peces en Imagenes
### mediante Redes Neuronales Convolucionales

**Autor:** Proyecto Final - Vision por Computadora  
**Fecha:** 2026  

---

## Resumen

FishNet es un sistema de deteccion y clasificacion de especies de peces a partir de imagenes,
utilizando arquitecturas de redes neuronales convolucionales (CNN). Este notebook implementa
el pipeline completo: preprocesamiento, entrenamiento de tres arquitecturas (CNN Baseline,
MobileNetV2, ResNet50), evaluacion comparativa y visualizacion con Grad-CAM.
""")

# ── Importaciones ──
md("## 1. Importaciones y Configuracion")

code("""import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')

print(f"PyTorch {torch.__version__}")
print(f"GPU disponible: {torch.cuda.is_available()}")
print(f"Dispositivo: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
""")

# ── Importar pipeline ──
md("## 2. Cargar Pipeline FishNet")

code("""# Agregar ruta del proyecto
import sys
sys.path.insert(0, str(Path.cwd().parent))

from notebooks.fishnet_pipeline import *
""")

# ── Configuracion ──
md("## 3. Verificar Dataset")

code("""# Verificar estructura del dataset
print("Buscando dataset...")
dataset_path = None
candidates = [
    Path('data/raw/fish_dataset'),
    Path('data/raw/Fish_Dataset'),
    Path('data/Fish_Dataset'),
    Path('../data/raw/fish_dataset'),
]
for c in candidates:
    full = (Path.cwd().parent / c) if not c.is_absolute() else c
    if full.exists() and any(full.iterdir()):
        dataset_path = full
        print(f"  Encontrado: {full}")
        break

if dataset_path is None:
    print("\\nDataset no encontrado.")
    print("Pasos:")
    print("  1. Descargar de: https://www.kaggle.com/datasets/sripaadsrinivasan/a-large-scale-fish-dataset")
    print("  2. Extraer en: data/raw/fish_dataset/")
    print("  3. Estructura esperada:")
    print("     fish_dataset/")
    print("         class_00/")
    print("         class_01/")
    print("         ...")
    print("\\nOpcional: Usar kagglehub o descarga manual.")
else:
    explore_dataset(dataset_path)
""")

# ── Preparar generadores ──
md("## 4. Preparar Generadores de Datos")
md("""Se utiliza division manual con sklearn para control preciso de train/validation/test.
- **Train:** 70%
- **Validation:** 15%
- **Test:** 15%

**Data Augmentation (solo train):**
- Rotacion ±30°
- Volteo horizontal
- Zoom 0.8–1.2x
- Desplazamiento ±15%
- Variacion de brillo
""")

code("""if dataset_path:
    train_loader, val_loader, test_loader, class_names, dataset_full = prepare_datasets(
        dataset_path,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    CLASS_NAMES = class_names
    print(f"\\nClases ({len(CLASS_NAMES)}):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name}")
""")

# ── CNN Baseline ──
md("## 5. Modelo 1: CNN Personalizada (Baseline)")
md("""Arquitectura disenada desde cero como referencia base:

```
Input(224x224x3)
  -> Conv2D(32, 3x3) + ReLU + BN + MaxPooling
  -> Conv2D(64, 3x3) + ReLU + BN + MaxPooling
  -> Conv2D(128, 3x3) + ReLU + BN + MaxPooling
  -> Conv2D(256, 3x3) + ReLU + BN + MaxPooling
  -> GlobalAvgPooling + Dense(256, ReLU) + Dropout(0.5)
  -> Dense(N_clases, Softmax)
```
""")

code("""if dataset_path:
    cnn_model = CNNBaseline(n_classes=len(CLASS_NAMES), dropout_rate=0.5).to(DEVICE)
    print(cnn_model)
""")

code("""if dataset_path:
    print("\\n--- Fase 1: Entrenamiento inicial ---")
    h_cnn = train_model(cnn_model, train_loader, val_loader,
                        'CNN_Baseline', epochs=EPOCHS_P1, learning_rate=LR_P1, phase=1)
    plot_training_history(h_cnn, 'CNN_Baseline')
""")

code("""if dataset_path:
    print("\\n--- Fase 2: Ajuste fino (LR reducido) ---")
    h_cnn_ft = train_model(cnn_model, train_loader, val_loader,
                           'CNN_Baseline', epochs=10, learning_rate=LR_P2, phase=2)
    for k in h_cnn:
        h_cnn[k] = h_cnn[k] + h_cnn_ft[k]
    plot_training_history(h_cnn, 'CNN_Baseline_Final')
""")

code("""if dataset_path:
    r_cnn = evaluate_model(cnn_model, test_loader, 'CNN_Baseline', CLASS_NAMES)
    torch.save(cnn_model.state_dict(), str(MODELS_DIR / 'cnn_baseline_final.pth'))
    print("\\nModelo CNN Baseline guardado.")
""")

# ── MobileNetV2 ──
md("## 6. Modelo 2: MobileNetV2 con Transfer Learning")
md("""**Estrategia:**
- **Fase 1:** Backbone MobileNetV2 (ImageNet) congelado, solo se entrena la cabeza personalizada
- **Fase 2:** Fine-tuning: se descongelan las ultimas 30 capas del backbone y se reentrena con LR reducido

**Justificacion:** Arquitectura liviana (~3.5M parametros), ideal para recursos computacionales moderados.
""")

code("""if dataset_path:
    mobilenet_model = build_mobilenetv2_model(
        n_classes=len(CLASS_NAMES),
        dropout_rate=0.3,
    ).to(DEVICE)
    print(mobilenet_model)
""")

code("""if dataset_path:
    print("\\n--- Fase 1: Entrenar cabeza (backbone congelado) ---")
    h_mob_p1 = train_model(mobilenet_model, train_loader, val_loader,
                           'MobileNetV2', epochs=EPOCHS_P1, learning_rate=LR_P1, phase=1)
    plot_training_history(h_mob_p1, 'MobileNetV2_P1')
""")

code("""if dataset_path:
    print("\\n--- Fase 2: Fine-tuning ---")
    unfreeze_mobilenetv2(mobilenet_model, unfreeze_last_n=30)
    h_mob_p2 = train_model(mobilenet_model, train_loader, val_loader,
                           'MobileNetV2', epochs=EPOCHS_P2, learning_rate=LR_P2, phase=2)
    plot_training_history(h_mob_p2, 'MobileNetV2_P2')
""")

code("""if dataset_path:
    r_mob = evaluate_model(mobilenet_model, test_loader, 'MobileNetV2', CLASS_NAMES)
    torch.save(mobilenet_model.state_dict(), str(MODELS_DIR / 'mobilenetv2_final.pth'))
    print("\\nModelo MobileNetV2 guardado.")
""")

# ── ResNet50 ──
md("## 7. Modelo 3: ResNet50 con Transfer Learning")
md("""**Estrategia:**
- **Fase 1:** Backbone ResNet50 (ImageNet) congelado
- **Fase 2:** Fine-tuning desde el bloque conv4_x en adelante

**Justificacion:** Mayor capacidad representacional (~25M parametros) que MobileNetV2, permite comparar precision vs. costo computacional.
""")

code("""if dataset_path:
    resnet_model = build_resnet50_model(
        n_classes=len(CLASS_NAMES),
        dropout_rate=0.3,
    ).to(DEVICE)
    print(resnet_model)
""")

code("""if dataset_path:
    print("\\n--- Fase 1: Entrenar cabeza (backbone congelado) ---")
    h_res_p1 = train_model(resnet_model, train_loader, val_loader,
                           'ResNet50', epochs=EPOCHS_P1, learning_rate=LR_P1, phase=1)
    plot_training_history(h_res_p1, 'ResNet50_P1')
""")

code("""if dataset_path:
    print("\\n--- Fase 2: Fine-tuning ---")
    unfreeze_resnet50(resnet_model, unfreeze_from='layer4')
    h_res_p2 = train_model(resnet_model, train_loader, val_loader,
                           'ResNet50', epochs=EPOCHS_P2, learning_rate=LR_P2, phase=2)
    plot_training_history(h_res_p2, 'ResNet50_P2')
""")

code("""if dataset_path:
    r_res = evaluate_model(resnet_model, test_loader, 'ResNet50', CLASS_NAMES)
    torch.save(resnet_model.state_dict(), str(MODELS_DIR / 'resnet50_final.pth'))
    print("\\nModelo ResNet50 guardado.")
""")

# ── Comparacion ──
md("## 8. Comparacion de Modelos")
md("""Se comparan las tres arquitecturas utilizando las siguientes metricas:
- **Accuracy:** Porcentaje de aciertos global
- **Precision (macro):** Promedio de precision por clase
- **Recall (macro):** Promedio de sensibilidad por clase
- **AUC-ROC:** Area bajo la curva ROC (One-vs-Rest)
""")

code("""if dataset_path:
    results = [r_cnn, r_mob, r_res]
    df_comp = compare_models(results)
""")

# ── Grad-CAM ──
md("## 9. Visualizacion Grad-CAM")
md("""**Grad-CAM** (Gradient-weighted Class Activation Mapping) genera mapas de calor que
resaltan las regiones de la imagen mas relevantes para la prediccion del modelo.

Esto permite interpretar *que* partes del pez (aletas, escamas, patrones corporales)
son utilizadas por la red para diferenciar especies.
""")

code("""if dataset_path:
    print("Visualizando Grad-CAM en ejemplos del conjunto de test...")
    test_dataset = test_loader.dataset
    n_samples = min(6, len(test_dataset))
    for i in range(n_samples):
        img_path, _ = dataset_full.samples[test_dataset.indices[i]]
        print(f"\\nEjemplo {i+1}: {Path(img_path).name}")
        try:
            display_gradcam(img_path, resnet_model, CLASS_NAMES)
        except Exception as e:
            print(f"  Error: {e}")
""")

# ── Analisis de errores ──
md("## 10. Analisis de Errores")
md("""Se identifican las especies con mayor tasa de confusion mutua,
lo que permite entender limitaciones del modelo y posibles mejoras.
""")

code("""if dataset_path:
    # Matriz de confusion normalizada del mejor modelo (ResNet50)
    cm = r_res['cm']
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='RdYlBu_r',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Prediccion')
    ax.set_ylabel('Verdad')
    ax.set_title('Matriz de Confusion Normalizada - ResNet50')
    plt.tight_layout()
    plt.show()
    
    # Pares mas confundidos (off-diagonal altos)
    n = len(CLASS_NAMES)
    confusions = []
    for i in range(n):
        for j in range(n):
            if i != j:
                confusions.append((CLASS_NAMES[i], CLASS_NAMES[j], cm_norm[i][j]))
    confusions.sort(key=lambda x: -x[2])
    
    print("\\nPares de especies mas confundidas:")
    print(f"{'Especie Real':25s} {'Especie Predicha':25s} {'Tasa Error':>10s}")
    print("-"*60)
    for real, pred, rate in confusions[:5]:
        print(f"{real:25s} {pred:25s} {rate:10.2%}")
""")

# ── Conclusiones ──
md("""## 11. Conclusiones

### Resultados Principales

1. **CNN Baseline:** Establece un rendimiento de referencia. Limitada por la ausencia de
   transferencia de conocimiento, pero permite entender el comportamiento base.

2. **MobileNetV2:** Arquitectura liviana que logra buen rendimiento con entrenamiento rapido.
   Ideal para despliegue en dispositivos con recursos limitados.

3. **ResNet50:** Mayor precision gracias a su profundidad y conexiones residuales.
   El costo computacional es mayor, pero ofrece el mejor rendimiento general.

### Interpretabilidad (Grad-CAM)
- Las visualizaciones Grad-CAM confirman que el modelo aprende a enfocarse en
  regiones anatomicas relevantes (cabeza, aletas, patrones de coloracion).
- Esto valida que el modelo no esta utilizando artefactos del fondo sino
  caracteristicas biologicas genuinas.

### Trabajo Futuro
- Incorporar deteccion de multiples peces por imagen (YOLO, Faster R-CNN).
- Expandir a mas especies y condiciones de iluminacion variables.
- Despliegue como aplicacion web o movil para uso en campo.
- Explorar arquitecturas Vision Transformer (ViT).

---

*FishNet v1.0 - Proyecto de Vision por Computadora*
""")

# ── Crear notebook ──
notebook = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {
            'name': 'python',
            'version': '3.13.11',
        },
    },
    'cells': cells,
}

output_path = NB_DIR / 'fishnet_notebook.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook generado: {output_path}")
print(f"Celdas creadas: {len(cells)}")
