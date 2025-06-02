# Clickbait Detection - Práctica IV

Este repositorio contiene el código para la Práctica IV sobre detección de clickbait utilizando aprendizaje profundo.

## Estructura del Proyecto

- `Clickbait_BERT.ipynb`: Notebook principal para la segunda parte (detección de clickbait con BERT)
- `clickbaitModeloVFinal.py`: Implementación de la primera parte con modelos de ML tradicionales
- `normalizacionTexto.py`: Funciones de normalización de texto para la primera parte
- `corpus/`: Directorio con los archivos de datos originales
  - `TA1C_dataset_detection_train.csv`: Conjunto de entrenamiento
  - `TA1C_dataset_detection_dev.csv`: Conjunto de desarrollo para pruebas

## Instrucciones para la Segunda Parte (BERT)

### Ejecutar en Google Colab

1. Sube el archivo `Clickbait_BERT.ipynb` a Google Colab
2. Ejecuta todas las celdas en orden
3. Cuando se te solicite, sube los archivos de datos:
   - `TA1C_dataset_detection_train.csv`
   - `TA1C_dataset_detection_dev.csv`
4. El notebook se encargará de:
   - Instalar las dependencias necesarias
   - Cargar y procesar los datos
   - Entrenar varios modelos BERT con diferentes configuraciones
   - Evaluar los modelos
   - Generar el archivo `detection.csv` con las predicciones finales
   - Generar una tabla comparativa para el informe

### Problemas Comunes y Soluciones

#### Error con evaluation_strategy

Si ves un error como:
```
TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'
```

Es debido a la versión de la biblioteca `transformers`. El notebook ya incluye código para instalar la versión correcta (4.18.0), pero si persiste el problema, ejecuta manualmente:

```python
!pip install transformers==4.18.0
```

#### Problemas de Memoria en GPU

Si experimentas problemas de memoria en la GPU, puedes reducir los siguientes parámetros:
- `per_device_train_batch_size`: Reducir a 4 u 8
- `max_length` en la tokenización: Reducir a 64 o 128

## Archivos de Salida

El notebook genera dos archivos principales:

1. `detection.csv`: Contiene las predicciones del mejor modelo en el conjunto de desarrollo
   - Formato: columnas "Tweet ID" y "Tag Value", separadas por coma
   - Este archivo es el requerido para la entrega

2. `resumen_experimentos_clickbait.csv`: Tabla con el resumen de todos los experimentos realizados
   - Útil para incluir en el informe final

## Requisitos

- Python 3.6+
- PyTorch 1.11.0
- Transformers 4.18.0
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Notas Adicionales

- El entrenamiento completo puede tardar varias horas dependiendo del hardware disponible
- Se recomienda usar una GPU para acelerar el entrenamiento
- Los modelos se guardan temporalmente en el directorio `./results/` durante la ejecución
