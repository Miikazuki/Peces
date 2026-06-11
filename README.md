# 🐟 FishNet: Clasificación de Especies Ícticas mediante CNN y Grad-CAM

[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0.0-orange)](https://pytorch.org/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.67%25-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

Sistema de visión por computador que clasifica automáticamente imágenes de peces en diez especies usando una red neuronal convolucional (SimpleCNN), con explicaciones visuales mediante mapas de activación **Grad-CAM**. Incluye una interfaz gráfica en Tkinter para usuarios sin formación técnica.

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Dataset](#dataset)
- [Arquitectura del Pipeline](#arquitectura-del-pipeline)
- [Modelo](#modelo)
- [Resultados](#resultados)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Instalación y Uso](#instalación-y-uso)
- [Interfaz Gráfica](#interfaz-gráfica)
- [Grad-CAM: Explicabilidad Visual](#grad-cam-explicabilidad-visual)
- [Dependencias](#dependencias)
- [Autores](#autores)

---

## Descripción del Proyecto

La identificación manual de especies de peces por parte de ictiólogos es un proceso lento, costoso y difícil de escalar. **FishNet** automatiza esta tarea mediante un pipeline modular de visión por computador que:

1. Carga y preprocesa imágenes de peces.
2. Clasifica la especie usando una CNN ligera entrenada desde cero en PyTorch.
3. Genera mapas de activación Grad-CAM para explicar visualmente las predicciones, señalando las regiones morfológicas (cabeza, aletas, contorno) que más influyeron en la decisión del modelo.

El sistema fue desarrollado como proyecto final de la asignatura **Redes Neuronales y Aprendizaje Profundo** de la Universidad Nacional de Colombia Sede La Paz, y se presentó en el congreso de ictiología celebrado en la sede el 28 de mayo de 2026.

---

## Dataset

El dataset proviene del repositorio [FishNet](https://github.com/vc-2026-i/proyecto-2x1/tree/main), utilizando la subdivisión `mini_dataset` generada a partir del *Fish Dataset* original.

| Parámetro | Valor |
|---|---|
| Total de imágenes | 1 500 |
| Clases | 10 especies |
| Imágenes por clase | 150 |
| Formato | PNG / JPEG |
| Condiciones de captura | Laboratorio controlado |

**Especies incluidas:**

| # | Especie |
|---|---|
| 1 | Astrolebpus |
| 2 | Black Sea Sprat |
| 3 | Gilt-Head Bream |
| 4 | Hourse Mackerel |
| 5 | Red Mullet |
| 6 | Red Sea Bream |
| 7 | Sea Bass |
| 8 | Shrimp |
| 9 | Striped Red Mullet |
| 10 | Trout |

**Partición de datos** (muestreo estratificado):

| Conjunto | Imágenes | Porcentaje |
|---|---|---|
| Entrenamiento | 1 275 | 85 % |
| Validación | 150 | 10 % |
| Prueba | 75 | 5 % |

Los datos se ubican en `data/mini_dataset/` organizados en subdirectorios por clase. Ver [instrucciones de descarga](#instalación-y-uso).

---

## Arquitectura del Pipeline

El sistema se estructura en **cinco etapas modulares**, cada una implementada como un módulo Python independiente en `pipeline/`:

```
Imagen de entrada
      │
      ▼
┌─────────────┐
│ 1. Adquisición     │  pipeline/adquisicion.py
│ (archivo o random) │
└──────┬──────┘
       │  PIL.Image RGB
       ▼
┌─────────────┐
│ 2. Preprocesamiento│  pipeline/preprocesamiento.py
│ Resize 224×224     │
│ Normalize ImageNet │
└──────┬──────┘
       │  Tensor (1,3,224,224)
       ▼
┌─────────────────────┐
│ 3. Extracción Grad-CAM │  pipeline/extraccion_caracteristicas.py
│ Forward + gradientes    │
│ Heatmap 224×224         │
└──────┬──────────────┘
       │
       ▼
┌─────────────┐
│ 4. Clasificación   │  pipeline/clasificacion.py
│ SimpleCNN + Softmax│
└──────┬──────┘
       │  clase, confianza, probabilidades
       ▼
┌────────────────────┐
│ 5. Postprocesamiento│  pipeline/postprocesamiento.py
│ Overlay Grad-CAM    │
│ Visualización       │
└────────────────────┘
```

### Preprocesamiento

- Redimensionamiento a **224 × 224** píxeles.
- Normalización por canal con parámetros ImageNet: `µ = [0.485, 0.456, 0.406]`, `σ = [0.229, 0.224, 0.225]`.
- Augmentación en entrenamiento: `RandomHorizontalFlip` y `RandomVerticalFlip` (p = 0.5).

### Grad-CAM (Etapa 3 + 5)

El mapa de activación se calcula sobre la última capa `Conv2d`:

```
L^c = ReLU( Σ_k α^c_k · A^k )
```

Donde `α^c_k` es el promedio espacial del gradiente de la clase predicha respecto a los mapas de activación `A^k`. El heatmap se colorea con `jet` y se superpone a la imagen original con opacidad `α = 0.4`.

---

## Modelo

### SimpleCNN

CNN diseñada a medida, con ~200 K parámetros, optimizada para el tamaño del dataset:

| Capa | Detalle |
|---|---|
| Conv1 | 3 → 16 canales, kernel 5×5, padding 2, BatchNorm, LeakyReLU(0.1) |
| Conv2 | 16 → 32 canales, kernel 5×5, padding 2, BatchNorm, LeakyReLU(0.1), MaxPool 2×2 |
| FC | 32 × 112 × 112 → 10, Dropout(0.2), BatchNorm, LeakyReLU(0.01) |

**Configuración de entrenamiento:**

| Hiperparámetro | Valor |
|---|---|
| Optimizador | Adam (lr = 0.001) |
| Función de pérdida | CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau (factor=0.1, patience=2) |
| Épocas máximas | 30 |
| Early stopping | Paciencia = 5 |
| Batch size | 64 |

**¿Por qué SimpleCNN y no ResNet/EfficientNet?**

| Modelo | Parámetros | Accuracy | Inferencia (CPU) |
|---|---|---|---|
| **SimpleCNN** | ~200 K | **98.67 %** | ~50 ms |
| ResNet-18 | ~11 M | 99.2 %* | ~200 ms |
| EfficientNet-B0 | ~5.3 M | 99.0 %* | ~180 ms |

*Valores estimados con transfer learning. Con solo 1 350 imágenes de entrenamiento, una red profunda sería propensa al sobreajuste. SimpleCNN ofrece el mejor balance velocidad/precisión y es compatible nativamente con Grad-CAM sobre su última capa Conv2d (mapas 32 × 112 × 112).

---

## Resultados

### Métricas globales (75 imágenes de prueba)

| Métrica | Valor |
|---|---|
| **Exactitud (Accuracy)** | **98.67 %** |
| Precisión (macro) | 0.99 |
| Recall (macro) | 0.97 |
| F1-score (macro) | 0.98 |

### Reporte por especie

| Clase | Precisión | Recall | F1 | N |
|---|---|---|---|---|
| Astrolebpus | 1.00 | 1.00 | 1.00 | 10 |
| Black Sea Sprat | 1.00 | 1.00 | 1.00 | 6 |
| Gilt-Head Bream | 0.86 | 1.00 | 0.92 | 6 |
| Hourse Mackerel | 1.00 | 1.00 | 1.00 | 5 |
| Red Mullet | 1.00 | 1.00 | 1.00 | 12 |
| Red Sea Bream | 1.00 | 1.00 | 1.00 | 7 |
| Sea Bass | 1.00 | 0.75 | 0.86 | 4 |
| Shrimp | 1.00 | 1.00 | 1.00 | 5 |
| Striped Red Mullet | 1.00 | 1.00 | 1.00 | 11 |
| Trout | 1.00 | 1.00 | 1.00 | 9 |

### Análisis de errores

Los errores se concentran en **Gilt-Head Bream** (precisión 0.86) y **Sea Bass** (recall 0.75). Ambas especies comparten cuerpo ovalado y comprimido lateralmente. Posibles mejoras: aumento de datos específico para este par, o añadir características morfológicas más finas.

Los mapas Grad-CAM confirman que el modelo se enfoca en regiones biológicamente relevantes: **contorno del cuerpo, cabeza y aletas**, lo que acerca el comportamiento de la red al razonamiento de un especialista humano.

---

## Estructura del Repositorio

```
FishNet/
├── data/
│   └── mini_dataset/          # Dataset por clase (ver instrucciones de descarga)
│       ├── Astrolebpus/
│       ├── Black Sea Sprat/
│       └── ...
├── pipeline/
│   ├── adquisicion.py         # Etapa 1: carga de imagen
│   ├── preprocesamiento.py    # Etapa 2: transformaciones
│   ├── extraccion_caracteristicas.py  # Etapa 3: Grad-CAM
│   ├── clasificacion.py       # Etapa 4: inferencia SimpleCNN
│   └── postprocesamiento.py   # Etapa 5: visualización overlay
├── gui/
│   └── app.py                 # Interfaz gráfica Tkinter
├── models/
│   └── fish_classification_model.pt  # Pesos entrenados
├── test1.ipynb                # Notebook completo reproducible
├── informe.pdf                # Documento resumen del proyecto
├── requirements.txt           # Dependencias
├── .gitignore
└── README.md
```

---

## Instalación y Uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/vc-2026-i/proyecto-2x1.git
cd proyecto-2x1
```

### 2. Crear el entorno y instalar dependencias

```bash
conda create -n vision python=3.9
conda activate vision
pip install -r requirements.txt
```

### 3. Descargar el dataset

El dataset `mini_dataset` está disponible en el repositorio. Si no está incluido por tamaño, descárgalo desde:

```
https://github.com/vc-2026-i/proyecto-2x1/tree/main
```

Colócalo en `data/mini_dataset/` con subdirectorios por clase.

### 4. Ejecutar el notebook (entrenamiento completo)

```bash
jupyter notebook test1.ipynb
```

El notebook cubre en orden: carga de datos → preprocesamiento → definición del modelo → entrenamiento con early stopping → evaluación (accuracy, classification report, matriz de confusión, curvas ROC) → visualización Grad-CAM.

### 5. Ejecutar la interfaz gráfica

```bash
python gui/app.py
```

---

## Interfaz Gráfica

La GUI en **Tkinter** permite a usuarios sin formación técnica operar el sistema completo:

| Componente | Función |
|---|---|
| Panel superior | Cargar imagen desde archivo, seleccionar imagen aleatoria del dataset, ejecutar pipeline |
| Panel central | Vista previa de la imagen cargada |
| Panel de resultados | Clase predicha y confianza (verde > 50 %, rojo ≤ 50 %) |
| Botón Grad-CAM | Abre ventana con tres paneles: imagen original, heatmap, superposición |
| Botón Probabilidades | Tabla completa de probabilidades por clase |

**Flujo de uso típico:**
1. Cargar imagen → 2. Ejecutar pipeline → 3. Ver resultado → 4. Inspeccionar Grad-CAM

---

## Grad-CAM: Explicabilidad Visual

Grad-CAM (*Gradient-weighted Class Activation Mapping*, Selvaraju et al., ICCV 2017) genera un mapa de calor que indica qué píxeles de la imagen fueron más relevantes para la predicción. En FishNet, el modelo consistentemente destaca:

- **Contorno del cuerpo** — forma general de la especie
- **Cabeza** — rasgos faciales y dentición
- **Aletas** — morfología característica por especie

Esto valida que el modelo aprende representaciones biológicamente significativas en lugar de artefactos del fondo.

---

## Dependencias

| Biblioteca | Versión mínima |
|---|---|
| PyTorch | ≥ 2.0.0 |
| TorchVision | ≥ 0.15.0 |
| NumPy | ≥ 1.24.0 |
| Matplotlib | ≥ 3.7.0 |
| scikit-learn | ≥ 1.2.0 |
| Pillow | ≥ 9.0.0 |
| pandas | ≥ 1.5.0 |
| seaborn | ≥ 0.12.0 |
| albumentations | ≥ 1.3.0 |

Ver `requirements.txt` para la lista completa con versiones exactas.

**Entorno de desarrollo:**
- SO: Windows 11 / Ubuntu 22.04
- Python 3.9
- GPU: NVIDIA RTX 3060 Ti, CUDA 13.1 (la inferencia también corre en CPU)

---

## Referencias

1. FishNet, "Fish classification dataset and baseline models," 2026. [GitHub](https://github.com/vc-2026-i/proyecto-2x1/tree/main)
2. R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," *Proc. IEEE ICCV*, 2017, pp. 618–626.
3. I. Goodfellow, Y. Bengio y A. Courville, *Deep Learning*. MIT Press, 2016.

---

## Autores

**Luis Daniel Reyes Rodríguez**  
Ingeniería Mecatrónica — Universidad Nacional de Colombia Sede La Paz  
lureyesr@unal.edu.co

**Jean Carlos Mejia Jimenez**  
Ingeniería Mecatrónica — Universidad Nacional de Colombia Sede La Paz  
jemejiaj@unal.edu.co
---

*Proyecto final — Asignatura: Redes Neuronales y Aprendizaje Profundo*  
*Universidad Nacional de Colombia Sede La Paz, 2026*
