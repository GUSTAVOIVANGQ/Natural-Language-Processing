# Clickbait Detection - Práctica IV

Detección automática de clickbait en tweets mediante técnicas de Procesamiento de Lenguaje Natural. El proyecto se divide en dos partes: una con modelos de Machine Learning tradicionales y otra con el modelo de lenguaje BERT.

---

## Tabla de Contenidos

1. [Descripción](#descripción)
2. [Pipeline](#pipeline)
3. [Herramientas Usadas](#herramientas-usadas)
4. [Estructura del Proyecto](#estructura-del-proyecto)
5. [Instrucciones de Uso](#instrucciones-de-uso)
6. [Resultados](#resultados)
7. [Archivos de Salida](#archivos-de-salida)
8. [Requisitos](#requisitos)
9. [Notas Adicionales](#notas-adicionales)

---

## Descripción

El objetivo de esta práctica es construir un clasificador binario que identifique si un tweet es clickbait (`1`) o no (`0`). Se utilizan los datos del corpus `TA1C_dataset_detection_train.csv` para entrenamiento y `TA1C_dataset_detection_dev.csv` para evaluación.

---

## Pipeline

### Parte 1 – Modelos de ML Tradicionales

```
Datos raw (CSV)
      │
      ▼
Normalización de texto (normalizacionTexto.py)
  ├── Tokenización
  ├── Eliminación de Stopwords
  └── Lematización (spaCy, es_core_news_sm)
      │
      ▼
Vectorización del texto
  ├── Frecuencia (CountVectorizer)
  ├── Binaria   (CountVectorizer binary=True)
  └── TF-IDF    (TfidfVectorizer)
      │
      ▼
Entrenamiento y evaluación (clickbaitModeloVFinal.py)
  ├── Naive Bayes
  ├── Logistic Regression
  ├── SVM (SVC lineal)
  └── Random Forest
      │
      ▼
Selección del mejor modelo (F1-Macro + Validación Cruzada 5-fold)
      │
      ▼
Predicciones sobre conjunto de desarrollo → detection.csv
```

### Parte 2 – BERT

```
Datos raw (CSV)
      │
      ▼
Tokenización con BertTokenizer (bert-base-multilingual-cased)
      │
      ▼
Fine-tuning del modelo BERT (Google Colab / GPU)
  └── Múltiples configuraciones de hiperparámetros
      │
      ▼
Evaluación y selección del mejor checkpoint
      │
      ▼
Predicciones sobre conjunto de desarrollo → detection.csv
```

---

## Herramientas Usadas

| Herramienta / Librería | Versión recomendada | Uso |
|---|---|---|
| Python | 3.8+ | Lenguaje principal |
| spaCy (`es_core_news_sm`) | 3.x | Tokenización, lematización y eliminación de stopwords |
| scikit-learn | latest | Vectorización, modelos ML, métricas y validación cruzada |
| imbalanced-learn | latest | Balanceo de clases (`RandomOverSampler`) |
| pandas | latest | Manejo de datos en CSV |
| numpy | latest | Operaciones numéricas |
| PyTorch | 1.11.0 | Backend para entrenamiento de BERT |
| Transformers (HuggingFace) | 4.18.0 | Modelo y tokenizador BERT |
| matplotlib / seaborn | latest | Visualización de resultados |
| Google Colab | — | Entorno con GPU para la Parte 2 |

---

## Estructura del Proyecto

```
Practice IV - Clickbait detection/
├── corpus/                             # Datos originales
│   ├── TA1C_dataset_detection_train.csv
│   └── TA1C_dataset_detection_dev.csv
├── corpus_tokenizado/                  # Datos preprocesados
│   ├── train/                          # Variantes normalizadas del conjunto de entrenamiento
│   └── dev/                            # Variantes normalizadas del conjunto de desarrollo
├── resultados_configuraciones/         # CSVs con métricas de cada experimento
├── resultados_predicciones/            # Archivos de predicciones generados
├── tabla_evidencias/                   # Tablas resumen de evidencias (Parte 1)
├── tabla_evidencias_tipo_normalizacion/# Comparativo de resultados por tipo de normalización
├── pkls_modelos/                       # Modelos entrenados serializados (.pkl)
├── normalizacionTexto.py               # Módulo de preprocesamiento de texto (Parte 1)
├── clickbaitModeloVFinal.py            # Detector de clickbait con ML tradicional (Parte 1)
├── Clickbait_BERT.ipynb                # Notebook de detección con BERT (Parte 2)
├── comparativo_normalizaciones.csv     # Resumen comparativo de tipos de normalización
├── detection.csv                       # Predicciones finales (archivo de entrega)
└── requerimientosPractica/             # Especificaciones de la práctica
```

---

## Instrucciones de Uso

### Parte 1 – Modelos de ML Tradicionales

#### 1. Preprocesar el texto

```bash
python normalizacionTexto.py
```

Esto genera las versiones normalizadas del corpus en `corpus_tokenizado/`. Las variantes disponibles son:

| Modo | Descripción |
|---|---|
| `Completo` | Tokenización + Stopwords + Lematización |
| `Tokenizacion` | Solo tokenización |
| `Stopwords` | Solo eliminación de stopwords |
| `Lematizacion` | Solo lematización |
| `Tokenizacion_Stopwords` | Tokenización + eliminación de stopwords |
| `Tokenizacion_Lematizacion` | Tokenización + lematización |
| `Stopwords_Lematizacion` | Eliminación de stopwords + lematización |

#### 2. Ejecutar experimentos

```bash
# Ejecutar para un tipo de normalización específico
python clickbaitModeloVFinal.py completo Tokenizacion

# Ejecutar para todos los tipos de normalización
python clickbaitModeloVFinal.py completo todos
```

#### 3. Generar predicciones

```bash
python clickbaitModeloVFinal.py prediccion Tokenizacion
```

### Parte 2 – BERT (Google Colab)

1. Sube el archivo `Clickbait_BERT.ipynb` a [Google Colab](https://colab.research.google.com/)
2. Activa el entorno de ejecución con GPU (`Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU`)
3. Ejecuta todas las celdas en orden
4. Cuando se solicite, sube los archivos de datos:
   - `TA1C_dataset_detection_train.csv`
   - `TA1C_dataset_detection_dev.csv`
5. El notebook realiza automáticamente:
   - Instalación de dependencias
   - Carga y preprocesamiento de datos
   - Fine-tuning de BERT con distintas configuraciones
   - Evaluación y selección del mejor modelo
   - Generación de `detection.csv` y tabla comparativa de experimentos

#### Problemas comunes

**Error `evaluation_strategy`:**
```
TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'
```
Solución: instala la versión correcta de `transformers`:
```python
!pip install transformers==4.18.0
```

**Problemas de memoria en GPU:**
Reduce los siguientes parámetros en el notebook:
- `per_device_train_batch_size` → 4 u 8
- `max_length` en la tokenización → 64 o 128

---

## Resultados

### Parte 1 – Comparativo por tipo de normalización (mejor modelo por variante)

Los experimentos evaluaron combinaciones de 4 modelos × 3 vectorizaciones × 3 rangos de n-gramas por cada una de las 7 variantes de normalización.

| Normalización | Mejor Modelo | Vectorización | N-gramas | F1-Macro | CV Media |
|---|---|---|---|---|---|
| Tokenización | Logistic Regression | Binaria | (1,1) | **0.7743** | 0.7425 |
| Lematización | Random Forest | Frecuencia | (1,1) | 0.7620 | 0.6898 |
| Tokenización + Lematización | SVC | TF-IDF | (1,1) | 0.7518 | 0.7147 |
| Stopwords | Random Forest | TF-IDF | (1,1) | 0.7068 | 0.6452 |
| Tokenización + Stopwords | Logistic Regression | TF-IDF | (1,1) | 0.6975 | 0.5680 |
| Completo | Naive Bayes | Binaria | (1,1) | 0.6647 | 0.5797 |
| Stopwords + Lematización | Naive Bayes | Binaria | (1,1) | 0.6647 | 0.5797 |

> **Mejor configuración global (Parte 1):** Logistic Regression con vectorización binaria, unigramas y solo tokenización → **F1-Macro: 0.7743**

### Parte 2 – BERT

El notebook `Clickbait_BERT.ipynb` realiza fine-tuning de `bert-base-multilingual-cased` y genera métricas en el conjunto de desarrollo. Los resultados detallados se almacenan en `resumen_experimentos_clickbait.csv` tras la ejecución.

---

## Archivos de Salida

| Archivo | Descripción |
|---|---|
| `detection.csv` | Predicciones finales (columnas: `Tweet ID`, `Tag Value`) — archivo de entrega |
| `comparativo_normalizaciones.csv` | Resumen del mejor modelo por cada tipo de normalización |
| `resultados_configuraciones/` | CSVs con métricas de todos los experimentos de la Parte 1 |
| `tabla_evidencias/` | Tablas de evidencia en formato requerido por la práctica |
| `modelo_<Normalizacion>.pkl` | Modelos serializados del mejor experimento por normalización |
| `resumen_experimentos_clickbait.csv` | Tabla comparativa generada por el notebook BERT |

---

## Requisitos

```
Python >= 3.8
spacy >= 3.0           # + modelo: python -m spacy download es_core_news_sm
scikit-learn
imbalanced-learn
pandas
numpy
# Para la Parte 2 (BERT):
torch == 1.11.0
transformers == 4.18.0
matplotlib
seaborn
```

Instalar dependencias:
```bash
pip install scikit-learn imbalanced-learn pandas numpy spacy
python -m spacy download es_core_news_sm
# Para BERT (en Colab se instalan automáticamente):
pip install torch==1.11.0 transformers==4.18.0 matplotlib seaborn
```

---

## Notas Adicionales

- El entrenamiento de BERT puede tardar varias horas; se recomienda usar GPU en Google Colab.
- Los modelos `.pkl` de la Parte 1 se guardan en el directorio raíz de la práctica y en `pkls_modelos/`.
- Los n-gramas de orden superior a (1,1) generalmente reducen el rendimiento en este corpus.
- El método de balanceo `RandomOverSampler` está implementado pero no fue el que produjo los mejores resultados; los experimentos finales usaron el corpus sin balanceo.
