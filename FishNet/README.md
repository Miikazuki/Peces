# FishNet: Sistema de Deteccion y Clasificacion de Peces en Imagenes

**Mediante Redes Neuronales Convolucionales**

---

## 1. Resumen

FishNet es un sistema de vision por computadora para la identificacion automatica de
especies de peces a partir de imagenes estaticas. Utiliza tres arquitecturas de redes
neuronales convolucionales (CNN) —una personalizada desde cero, MobileNetV2 y ResNet50
con transfer learning— para clasificar 9 especies de peces del Mar Egeo.

El proyecto compara el rendimiento de estas arquitecturas usando metricas cuantitativas
estandar e incorpora visualizaciones Grad-CAM para interpretabilidad.

---

## 2. Problema y Motivacion

La identificacion manual de especies de peces es:
- **Lenta:** Requiere inspeccion visual experta de cada muestra.
- **Subjetiva:** Depende de la experiencia del observador.
- **Caro:** Necesita biólogos marinos especializados.

Un sistema automatizado permite:
- Monitoreo a gran escala de ecosistemas acuaticos.
- Apoyo a la pesca sostenible mediante identificacion rapida de capturas.
- Investigacion ecologica con datos consistentes y reproducibles.

---

## 3. Dataset

**Fuente:** [A Large Scale Fish Dataset - Kaggle](https://www.kaggle.com/datasets/sripaadsrinivasan/a-large-scale-fish-dataset)

| Caracteristica | Descripcion |
|---|---|
| **Especies** | 9 especies del Mar Egeo |
| **Imagenes** | ~9000 (1000 aumentadas por clase) |
| **Formato** | JPEG, resolucion variable |
| **Clases** | Dorada, Lubina, Trucha artica, Pargo, Salmonete, Caballa, Anchoa, Jurel, Bacaladilla |

### Preprocesamiento

- Redimension a 224x224 px
- Normalizacion [0, 1]
- Data augmentation (train): rotacion ±30°, volteo horizontal, zoom 0.8-1.2x, desplazamiento, brillo
- Division: 70% train / 15% val / 15% test

---

## 4. Arquitecturas Implementadas

### 4.1 CNN Personalizada (Baseline)

```
Input(224x224x3)
  -> Conv2D(32, 3x3) + ReLU + BN + MaxPooling
  -> Conv2D(64, 3x3) + ReLU + BN + MaxPooling
  -> Conv2D(128, 3x3) + ReLU + BN + MaxPooling
  -> Conv2D(256, 3x3) + ReLU + BN + MaxPooling
  -> GlobalAvgPooling + Dense(256) + Dropout(0.5)
  -> Dense(N_clases) + Softmax
```

### 4.2 MobileNetV2 + Transfer Learning

- Backbone: MobileNetV2 preentrenada en ImageNet
- Fase 1: backbone congelado, cabeza personalizada (GAP + Dense 128 + Dropout 0.3)
- Fase 2: fine-tuning ultimas 30 capas, LR 1e-5

### 4.3 ResNet50 + Transfer Learning

- Backbone: ResNet50 preentrenada en ImageNet
- Fase 1: backbone congelado, cabeza personalizada
- Fase 2: fine-tuning desde conv4_x, LR 1e-5

---

## 5. Metricas de Evaluacion

| Metrica | Descripcion |
|---|---|
| **Accuracy** | Porcentaje de predicciones correctas |
| **Loss** | Categorical Cross-Entropy |
| **Precision (macro)** | Promedio de precision por clase |
| **Recall (macro)** | Promedio de sensibilidad por clase |
| **F1-Score (macro)** | Media armonica de precision y recall |
| **AUC-ROC (OvR)** | Capacidad discriminativa por clase |
| **Matriz de Confusion** | Analisis visual de errores |

---

## 6. Ajuste de Hiperparametros

**Fase 1 - Busqueda manual (baseline):**
- Learning rate: {1e-3, 1e-4, 1e-5}
- Batch size: {16, 32}
- Early Stopping (patience=10)

**Fase 2 - Busqueda sistematica (opcional con KerasTuner):**
- Optimizador: Adam vs SGD + momentum
- Dropout: {0.2, 0.3, 0.5}
- Neuronas densa: {64, 128, 256}

**Regularizacion:**
- Early Stopping (val_loss)
- ReduceLROnPlateau (factor 0.5, patience 5)
- L2 Weight Decay

---

## 7. Estructura del Proyecto

```
FishNet/
├── README.md                   # Informe tecnico
├── requirements.txt            # Dependencias
├── generate_notebook.py        # Generador del notebook
├── data/                       # Datos del dataset
│   ├── raw/                    # Dataset original
│   └── processed/              # Datos preprocesados
├── notebooks/
│   ├── fishnet_notebook.ipynb  # Notebook Jupyter principal
│   └── fishnet_pipeline.py     # Pipeline completo (modulo Python)
├── models/                     # Modelos entrenados (.pth)
└── cache/                      # Cache de KerasTuner
```

---

## 8. Instrucciones de Uso

### Instalacion

```bash
pip install -r requirements.txt
```

### Descargar Dataset

```bash
# Opcion 1: Kaggle API
kaggle datasets download sripaadsrinivasan/a-large-scale-fish-dataset
# Extraer en data/raw/fish_dataset/

# Opcion 2: Manual
# Descargar de Kaggle y extraer en la ruta indicada
```

### Ejecutar Pipeline

```bash
# Opcion A: Notebook (recomendado)
jupyter notebook notebooks/fishnet_notebook.ipynb

# Opcion B: Script directo
python notebooks/fishnet_pipeline.py
```

### Resultados

Los modelos entrenados se guardan en `models/`:
- `cnn_baseline_final.pth`
- `mobilenetv2_final.pth`
- `resnet50_final.pth`

Figuras comparativas y curvas de aprendizaje se generan automaticamente.

---

## 9. Interpretabilidad (Grad-CAM)

Grad-CAM genera mapas de calor que resaltan las regiones de la imagen mas relevantes
para la prediccion. Esto permite:

- Verificar que el modelo se enfoca en caracteristicas biologicas (aletas, patrones corporales)
- Identificar posibles sesgos (fondo, iluminacion)
- Generar confianza en las predicciones (XAI - Explainable AI)

---

## 10. Resultados Esperados

| Modelo | Accuracy (esperado) | Parametros | Framework |
|---|---|---|---|
| CNN Baseline | ~75-85% | ~1.5M | PyTorch |
| MobileNetV2 | ~88-94% | ~3.5M | PyTorch |
| ResNet50 | ~92-97% | ~25M | PyTorch |

*Resultados preliminares; dependen del dataset y configuracion especifica.*

---

## 11. Trabajo Futuro

- Deteccion de multiples peces por imagen (YOLOv8, Faster R-CNN)
- Clasificacion en tiempo real desde video
- Aplicacion web/movil para uso en campo
- Arquitecturas Vision Transformer (ViT)
- Dataset con mas especies y condiciones variadas

---

## 12. Referencias

1. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks.
2. Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks.
3. He, K., et al. (2016). Deep Residual Learning for Image Recognition.
4. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
5. A Large Scale Fish Dataset - Kaggle: https://www.kaggle.com/datasets/sripaadsrinivasan/a-large-scale-fish-dataset

---

*Proyecto de Vision por Computadora - 2026*
