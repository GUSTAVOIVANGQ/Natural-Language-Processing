import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import logging
import os
import pickle
import warnings
import csv
from datetime import datetime

from torch.utils.data import Dataset
from transformers import (
    BertTokenizer, RobertaTokenizer, ElectraTokenizerFast,
    RobertaForSequenceClassification, BertForSequenceClassification, ElectraForSequenceClassification,
    TrainingArguments, Trainer, pipeline
)
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log',
    filemode='a'
)

logger = logging.getLogger("ClickBait_Models")

class DetectorClickBait:
    
    def __init__(self, tipoModelo="bert"):
        self.modelosDisponibles = {
            "bert": {
                "nombre": "dccuchile/bert-base-spanish-wwm-cased",
                "descripcion": "Modelo BERT Base en español"
            },
            "roberta": {
                "nombre": "PlanTL-GOB-ES/roberta-base-bne",
                "descripcion": "Modelo RoBERTa Base en español"
            },
            "electra": {
                "nombre": "google/electra-base-discriminator",
                "descripcion": "Modelo ELECTRA Base en español"
            }
        }
        self.tipoModelo = tipoModelo
        self.tokenizador = None
        self.modelo = None
        self.codificadorEtiquetas = None
        self.modeloEntrenado = False
        self.rutaModeloGuardado = f"modelo_guardado_{tipoModelo}"

        if tipoModelo in self.modelosDisponibles:
            self.configuracionModelo = self.modelosDisponibles[tipoModelo]
            self.nombreModelo = self.configuracionModelo["nombre"]
            logger.info(f"Modelo seleccionado: {self.configuracionModelo['descripcion']}")
            logger.info(f"Ruta del modelo: {self.nombreModelo}")
        else:
            logger.warning(f"Modelo {tipoModelo} no disponible o no reconocido.")
            for clave, configuracion in self.modelosDisponibles.items():
                logger.info(f"{clave}: {configuracion['descripcion']}")
            raise ValueError(f"Modelo {tipoModelo} no disponible.")
        
    def cargarYProcesarDatos(self, rutaArchivo, columnaTeaser, columnaTarget):
        logger.info(f"Cargando datos desde {rutaArchivo}...")
        try:
            df_data = pd.read_csv(rutaArchivo, sep=",", engine="python", usecols=[columnaTeaser])
            df_target = pd.read_csv(rutaArchivo, sep=",", engine="python", usecols=[columnaTarget])
        except Exception as ErrorCargarArchivo:
            logger.error(f"Error al cargar el archivo: {ErrorCargarArchivo}")
            raise

        X = df_data.iloc[:, 0].tolist()
        y = df_target.iloc[:, 0].tolist()
        etiquetasTarget = np.unique(y)
        logger.info(f"Dataset cargado con {len(X)} muestras, {len(etiquetasTarget)} clases.")
        logger.info(f"Clases: {etiquetasTarget}")
        return X, y, etiquetasTarget
        
    def cargarDosArchivos(self, rutaEntrenamiento, rutaEvaluacion, columnaTeaser, columnaTarget):
        logger.info(f"Cargando archivo de entrenamiento: {rutaEntrenamiento}")
        try:
            df_train_data = pd.read_csv(rutaEntrenamiento, sep=",", engine="python")
            df_eval_data = pd.read_csv(rutaEvaluacion, sep=",", engine="python")
        except Exception as error:
            logger.error(f"Error al cargar archivos: {error}")
            raise

        X_train = df_train_data.iloc[:, columnaTeaser].tolist()
        y_train = df_train_data.iloc[:, columnaTarget].tolist()
        
        X_eval = df_eval_data.iloc[:, columnaTeaser].tolist()
        y_eval = df_eval_data.iloc[:, columnaTarget].tolist()
        
        logger.info(f"Entrenamiento: {len(X_train)} muestras")
        logger.info(f"Evaluación: {len(X_eval)} muestras")
        
        return X_train, y_train, X_eval, y_eval, df_eval_data
    
    def entrenamientoEsquema1(self, rutaEntrenamiento, rutaTest, columnaTeaser=4, columnaTarget=5):
        logger.info("Esquema 1: 75% train / 25% eval del archivo train + predicciones en dev_gold")
        
        X, y, etiquetasTarget = self.cargarYProcesarDatos(rutaEntrenamiento, columnaTeaser, columnaTarget)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
        
        try:
            df_test = pd.read_csv(rutaTest, sep=",", engine="python")
            X_test = df_test.iloc[:, columnaTeaser].tolist()
            y_test = df_test.iloc[:, columnaTarget].tolist()
        except Exception as error:
            logger.error(f"Error al cargar archivo de test: {error}")
            raise
        
        self.codificadorEtiquetas = LabelEncoder()
        y_train_enc = self.codificadorEtiquetas.fit_transform(y_train)
        y_val_enc = self.codificadorEtiquetas.transform(y_val)
        y_test_enc = self.codificadorEtiquetas.transform(y_test)
        
        if self.tokenizador is None:
            self.configurarTokenizador()
        
        train_encodings = self.tokenizador(X_train, truncation=True, padding=True)
        val_encodings = self.tokenizador(X_val, truncation=True, padding=True)
        test_encodings = self.tokenizador(X_test, truncation=True, padding=True)
        
        train_dataset = DatasetPersonalizado(train_encodings, y_train_enc)
        val_dataset = DatasetPersonalizado(val_encodings, y_val_enc)
        test_dataset = DatasetPersonalizado(test_encodings, y_test_enc)
        
        trainer = self.entrenarModelo(train_dataset, val_dataset)
        
        if trainer:
            logger.info("Evaluando en conjunto de test (dev_gold)")
            self.evaluarModelo(trainer, test_dataset, self.codificadorEtiquetas.classes_)
            self.guardarModelo(trainer)
            return trainer, df_test
        return None, None
    
    def entrenamientoEsquema2(self, rutaEntrenamiento, rutaEvaluacion, columnaTeaser=4, columnaTarget=5):
        logger.info("Esquema 2: 100% train + 100% dev_gold para evaluación")
        
        X_train, y_train, X_eval, y_eval, df_eval = self.cargarDosArchivos(
            rutaEntrenamiento, rutaEvaluacion, columnaTeaser, columnaTarget)
        
        self.codificadorEtiquetas = LabelEncoder()
        y_train_enc = self.codificadorEtiquetas.fit_transform(y_train)
        y_eval_enc = self.codificadorEtiquetas.transform(y_eval)
        
        if self.tokenizador is None:
            self.configurarTokenizador()
        
        train_encodings = self.tokenizador(X_train, truncation=True, padding=True)
        eval_encodings = self.tokenizador(X_eval, truncation=True, padding=True)
        
        train_dataset = DatasetPersonalizado(train_encodings, y_train_enc)
        eval_dataset = DatasetPersonalizado(eval_encodings, y_eval_enc)
        
        trainer = self.entrenarModelo(train_dataset, eval_dataset)
        
        if trainer:
            logger.info("Evaluando en dev_gold completo")
            self.evaluarModelo(trainer, eval_dataset, self.codificadorEtiquetas.classes_)
            self.guardarModelo(trainer)
            return trainer, df_eval
        return None, None
    
    def entrenamientoEsquema3(self, rutaEntrenamiento, rutaEvaluacion, porcentajeEval=0.5, columnaTeaser=4, columnaTarget=5):
        logger.info(f"Esquema 3: 100% train + {porcentajeEval*100}% dev_gold eval + {(1-porcentajeEval)*100}% dev_gold test")
        
        X_train, y_train, X_dev_total, y_dev_total, df_dev = self.cargarDosArchivos(
            rutaEntrenamiento, rutaEvaluacion, columnaTeaser, columnaTarget)
        
        indices_dev = list(range(len(X_dev_total)))
        X_eval, X_test, y_eval, y_test, indices_eval, indices_test = train_test_split(
            X_dev_total, y_dev_total, indices_dev, test_size=(1-porcentajeEval), random_state=0, stratify=y_dev_total)
        
        df_eval_split = df_dev.iloc[indices_eval]
        df_test_split = df_dev.iloc[indices_test]
        
        self.codificadorEtiquetas = LabelEncoder()
        y_train_enc = self.codificadorEtiquetas.fit_transform(y_train)
        y_eval_enc = self.codificadorEtiquetas.transform(y_eval)
        y_test_enc = self.codificadorEtiquetas.transform(y_test)
        
        if self.tokenizador is None:
            self.configurarTokenizador()
        
        train_encodings = self.tokenizador(X_train, truncation=True, padding=True)
        eval_encodings = self.tokenizador(X_eval, truncation=True, padding=True)
        test_encodings = self.tokenizador(X_test, truncation=True, padding=True)
        
        train_dataset = DatasetPersonalizado(train_encodings, y_train_enc)
        eval_dataset = DatasetPersonalizado(eval_encodings, y_eval_enc)
        test_dataset = DatasetPersonalizado(test_encodings, y_test_enc)
        
        trainer = self.entrenarModelo(train_dataset, eval_dataset)
        
        if trainer:
            logger.info("Evaluando en porción de test de dev_gold")
            self.evaluarModelo(trainer, test_dataset, self.codificadorEtiquetas.classes_)
            self.guardarModelo(trainer)
            return trainer, df_test_split
        return None, None
        """Carga un CSV solo con textos para hacer predicciones"""
        logger.info(f"Cargando datos para predicción desde {rutaArchivo}...")
        try:
            if isinstance(columnaTeaser, int):
                # Si es un índice de columna
                df = pd.read_csv(rutaArchivo, sep=",", engine="python")
                textos = df.iloc[:, columnaTeaser].tolist()
            else:
                # Si es el nombre de la columna
                df = pd.read_csv(rutaArchivo, sep=",", engine="python")
                textos = df[columnaTeaser].tolist()
            
            logger.info(f"Cargados {len(textos)} textos para predicción.")
            return textos, df
        except Exception as error:
            logger.error(f"Error al cargar el archivo: {error}")
            raise
    
    def dividirDataset(self, X, y, tamanioTest=0.2, tamanioValidacion=0.25):
        logger.info(f"Dividiendo el dataset para el conjunto de entrenamiento...")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=tamanioTest,
            random_state=0,
            stratify=y)
        
        logger.info(f"Dividiendo el dataset para el conjunto de validacion...")
        X_train_val, X_val, y_train_val, y_val = train_test_split(
            X_train,
            y_train,
            test_size=tamanioValidacion,
            random_state=0,
            stratify=y_train)
        
        logger.info(f"Tamanio de los conjuntos:")
        logger.info(f"Conjunto de entrenamiento: {len(X_train_val)}")
        logger.info(f"Conjunto de validacion: {len(X_val)}")
        logger.info(f"Conjunto de prueba: {len(X_test)}")

        return X_train_val, X_val, X_test, y_train_val, y_val, y_test
    
    def codificarEtiquetas(self, y_train, y_val, y_test):
        logger.info(f"Codificando las etiquetas...")
        self.codificadorEtiquetas = LabelEncoder()
        
        y_train_encoded = self.codificadorEtiquetas.fit_transform(y_train)
        y_val_encoded = self.codificadorEtiquetas.transform(y_val)
        y_test_encoded = self.codificadorEtiquetas.transform(y_test)
        logger.info(f"Mapeo de etiquetas: {self.codificadorEtiquetas.classes_}")

        return y_train_encoded, y_val_encoded, y_test_encoded
    
    def configurarTokenizador(self):
        """Configura el tokenizador según el tipo de modelo"""
        if self.tipoModelo == "bert":
            logger.info(f"Configurando BertTokenizer...")
            self.tokenizador = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
        elif self.tipoModelo == "roberta":
            logger.info(f"Configurando RoBERTaTokenizer...")
            self.tokenizador = RobertaTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
        elif self.tipoModelo == "electra":
            logger.info(f"Configurando ElectraTokenizer...")
            self.tokenizador = ElectraTokenizerFast.from_pretrained("google/electra-base-discriminator")
        else:
            logger.error(f"Tipo de modelo no reconocido: {self.tipoModelo}")
            raise ValueError(f"Tipo de modelo no reconocido: {self.tipoModelo}")
    
    def prepararDataset(self, X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded):
        logger.info(f"Tokenizando los textos...")
        
        if self.tokenizador is None:
            self.configurarTokenizador()

        logger.info(f"Tokenizando el conjunto de entrenamiento...")
        train_encodings = self.tokenizador(X_train, truncation=True, padding=True)

        logger.info("Tokenizando conjunto de validación...")
        validation_encodings = self.tokenizador(X_val, truncation=True, padding=True)

        logger.info("Tokenizando conjunto de prueba...")
        test_encodings = self.tokenizador(X_test, truncation=True, padding=True)

        trainDataset = DatasetPersonalizado(train_encodings, y_train_encoded)
        valDataset = DatasetPersonalizado(validation_encodings, y_val_encoded)
        testDataset = DatasetPersonalizado(test_encodings, y_test_encoded)

        return trainDataset, valDataset, testDataset
    
    def calcularMetricas(self, pred):
        etiquetas = pred.label_ids
        predicciones = pred.predictions.argmax(-1)
        f1 = f1_score(etiquetas, predicciones, average="macro")
        accuracy = accuracy_score(etiquetas, predicciones)
        return {"accuracy": accuracy, "f1": f1}
    
    def entrenarModelo(self, trainDataset, valDataset, numeroEpocas=3):
        logger.info(f"Definiendo los parámetros para el entrenamiento del modelo...")
        numeroEtiquetas = len(self.codificadorEtiquetas.classes_)
        
        # Crear directorio de salida con nombre del modelo
        directorioSalida = f"{self.tipoModelo}_output"
        
        try:
            if self.tipoModelo == "bert":
                modelo = BertForSequenceClassification.from_pretrained(
                    'dccuchile/bert-base-spanish-wwm-cased', 
                    num_labels=numeroEtiquetas
                )
            elif self.tipoModelo == "roberta":
                modelo = RobertaForSequenceClassification.from_pretrained(
                    'PlanTL-GOB-ES/roberta-base-bne', 
                    num_labels=numeroEtiquetas
                )
            elif self.tipoModelo == "electra":
                modelo = ElectraForSequenceClassification.from_pretrained(
                    'google/electra-base-discriminator', 
                    num_labels=numeroEtiquetas
                )
            else:
                logger.error(f"No se puede generar el modelo para el tipo {self.tipoModelo}")
                raise ValueError(f"Tipo de modelo no válido: {self.tipoModelo}")
            
            # Información para el usuario sobre el proceso de fine-tuning
            logger.info("Modelo cargado para fine-tuning (es normal que algunos pesos se inicialicen nuevamente)")
                
        except Exception as error:
            logger.error(f"Error al generar el modelo: {error}")
            raise ValueError(f"Error al generar el modelo: {error}")
        
        logger.info(f"Configurando el entrenamiento del modelo...")
        
        trainingArgs = TrainingArguments(
            output_dir=directorioSalida,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            num_train_epochs=numeroEpocas,
            seed=0,
            load_best_model_at_end=True,
            fp16=True,
            logging_steps=50
        )
        
        trainer = Trainer(
            model=modelo,
            args=trainingArgs,
            train_dataset=trainDataset,
            eval_dataset=valDataset,
            compute_metrics=self.calcularMetricas
        )
        
        try:
            logger.info(f"Comenzando el entrenamiento del modelo...")
            trainer.train()
            self.modeloEntrenado = True
            self.modelo = trainer.model
            logger.info(f"Entrenamiento completado exitosamente.")
        except Exception as error:
            logger.error(f"Error al entrenar el modelo: {error}")
            raise ValueError(f"Error al entrenar el modelo: {error}")
        
        return trainer
    
    def guardarModelo(self, trainer=None):
        """Guarda el modelo entrenado y el codificador de etiquetas"""
        if not self.modeloEntrenado:
            logger.warning("No hay modelo entrenado para guardar.")
            return False
        
        try:
            # Crear directorio si no existe
            os.makedirs(self.rutaModeloGuardado, exist_ok=True)
            
            # Guardar el modelo y tokenizador
            if trainer:
                trainer.save_model(self.rutaModeloGuardado)
            else:
                self.modelo.save_pretrained(self.rutaModeloGuardado)
            
            self.tokenizador.save_pretrained(self.rutaModeloGuardado)
            
            # Guardar el codificador de etiquetas
            with open(f"{self.rutaModeloGuardado}/label_encoder.pkl", "wb") as f:
                pickle.dump(self.codificadorEtiquetas, f)
            
            # Guardar metadatos del modelo
            metadatos = {
                "tipo_modelo": self.tipoModelo,
                "fecha_entrenamiento": datetime.now().isoformat(),
                "clases": self.codificadorEtiquetas.classes_.tolist()
            }
            
            with open(f"{self.rutaModeloGuardado}/metadatos.pkl", "wb") as f:
                pickle.dump(metadatos, f)
            
            logger.info(f"Modelo guardado exitosamente en: {self.rutaModeloGuardado}")
            return True
            
        except Exception as error:
            logger.error(f"Error al guardar el modelo: {error}")
            return False
    
    def cargarModelo(self, rutaModelo=None):
        """Carga un modelo previamente entrenado"""
        if rutaModelo is None:
            rutaModelo = self.rutaModeloGuardado
        
        try:
            # Verificar que el directorio existe
            if not os.path.exists(rutaModelo):
                logger.error(f"No se encontró el modelo en: {rutaModelo}")
                return False
            
            # Cargar metadatos
            with open(f"{rutaModelo}/metadatos.pkl", "rb") as f:
                metadatos = pickle.load(f)
            
            logger.info(f"Cargando modelo entrenado el: {metadatos['fecha_entrenamiento']}")
            
            # Cargar el codificador de etiquetas
            with open(f"{rutaModelo}/label_encoder.pkl", "rb") as f:
                self.codificadorEtiquetas = pickle.load(f)
            
            # Configurar tokenizador
            self.configurarTokenizador()
            
            # Cargar el modelo
            if self.tipoModelo == "bert":
                self.modelo = BertForSequenceClassification.from_pretrained(rutaModelo)
            elif self.tipoModelo == "roberta":
                self.modelo = RobertaForSequenceClassification.from_pretrained(rutaModelo)
            elif self.tipoModelo == "electra":
                self.modelo = ElectraForSequenceClassification.from_pretrained(rutaModelo)
            
            self.modeloEntrenado = True
            logger.info(f"Modelo cargado exitosamente desde: {rutaModelo}")
            return True
            
        except Exception as error:
            logger.error(f"Error al cargar el modelo: {error}")
            return False
    
    def predecir(self, textos):
        """Hace predicciones sobre una lista de textos"""
        if not self.modeloEntrenado:
            logger.error("No hay modelo entrenado disponible.")
            return None
        
        try:
            # Tokenizar los textos
            encodings = self.tokenizador(textos, truncation=True, padding=True, return_tensors="pt")
            
            # Hacer predicciones
            self.modelo.eval()
            with torch.no_grad():
                outputs = self.modelo(**encodings)
                predicciones = torch.nn.functional.softmax(outputs.logits, dim=-1)
                clases_predichas = torch.argmax(predicciones, dim=-1)
            
            # Convertir a etiquetas originales
            etiquetas_predichas = self.codificadorEtiquetas.inverse_transform(clases_predichas.numpy())
            probabilidades = predicciones.numpy()
            
            resultados = []
            for i, texto in enumerate(textos):
                resultado = {
                    "texto": texto,
                    "prediccion": etiquetas_predichas[i],
                    "confianza": float(np.max(probabilidades[i])),
                    "probabilidades": {
                        clase: float(prob) 
                        for clase, prob in zip(self.codificadorEtiquetas.classes_, probabilidades[i])
                    }
                }
                resultados.append(resultado)
            
            return resultados
            
        except Exception as error:
            logger.error(f"Error al hacer predicciones: {error}")
            return None
    
    def predecirDesdeCSV(self, rutaArchivo, columnaTexto=0, guardarResultados=True):
        """Hace predicciones sobre un archivo CSV y opcionalmente guarda los resultados"""
        try:
            textos, df_original = self.cargarDatosParaPrediccion(rutaArchivo, columnaTexto)
            resultados = self.predecir(textos)
            
            if resultados is None:
                return None
            
            # Agregar resultados al DataFrame original
            df_original['prediccion_clickbait'] = [r['prediccion'] for r in resultados]
            df_original['confianza'] = [r['confianza'] for r in resultados]
            
            if guardarResultados:
                nombre_archivo = f"predicciones_{self.tipoModelo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df_original.to_csv(nombre_archivo, index=False)
                logger.info(f"Resultados guardados en: {nombre_archivo}")
            
            return resultados, df_original
            
        except Exception as error:
            logger.error(f"Error al procesar CSV: {error}")
            return None
        
    def predecirYGuardar(self, datos_prueba, archivo_salida="detection.csv", columnaID=0, columnaTexto=4):
        if not self.modeloEntrenado:
            raise ValueError("El modelo no ha sido entrenado. Entrena o carga un modelo primero.")
        
        try:
            if isinstance(datos_prueba, str):
                df = pd.read_csv(datos_prueba, sep=",", engine="python")
            else:
                df = datos_prueba
            
            if isinstance(columnaTexto, int):
                textos = df.iloc[:, columnaTexto].tolist()
            else:
                textos = df[columnaTexto].tolist()
            
            if isinstance(columnaID, int):
                tweet_ids = df.iloc[:, columnaID].tolist()
            else:
                tweet_ids = df[columnaID].tolist()
            
            resultados = self.predecir(textos)
            
            if resultados is None:
                raise ValueError("Error al hacer predicciones")
            
            archivo_base = archivo_salida.replace('.csv', '')
            archivo_simple = f"{archivo_base}.csv"
            archivo_completo = f"{archivo_base}_completo.csv"
            
            with open(archivo_simple, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Tweet ID", "Tag Value"])
                for tweet_id, resultado in zip(tweet_ids, resultados):
                    writer.writerow([tweet_id, resultado['prediccion']])
            
            with open(archivo_completo, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                header = ["Tweet ID", "Texto", "Prediccion", "Confianza"]
                for clase in self.codificadorEtiquetas.classes_:
                    header.append(f"Prob_{clase}")
                writer.writerow(header)
                
                for tweet_id, resultado in zip(tweet_ids, resultados):
                    row = [
                        tweet_id,
                        resultado['texto'][:100] + "..." if len(resultado['texto']) > 100 else resultado['texto'],
                        resultado['prediccion'],
                        f"{resultado['confianza']:.4f}"
                    ]
                    for clase in self.codificadorEtiquetas.classes_:
                        row.append(f"{resultado['probabilidades'][clase]:.4f}")
                    writer.writerow(row)
            
            logger.info(f"Predicciones guardadas en {archivo_simple}")
            logger.info(f"Predicciones completas guardadas en {archivo_completo}")
            
            conteo = {}
            for resultado in resultados:
                pred = resultado['prediccion']
                conteo[pred] = conteo.get(pred, 0) + 1
            
            logger.info("Resumen de predicciones:")
            for clase, cantidad in conteo.items():
                porcentaje = (cantidad / len(resultados)) * 100
                logger.info(f"   {clase}: {cantidad} ({porcentaje:.1f}%)")
            
            return resultados
            
        except Exception as e:
            logger.error(f"Error al predecir: {e}")
            raise ValueError(f"Error al predecir: {e}")
    
    def evaluarModelo(self, trainer, testDataset, y_test_encoded):
        logger.info(f"Evaluando el modelo...")
        
        predicciones = trainer.predict(testDataset)
        prediccionesClases = np.argmax(predicciones.predictions, axis=-1)
        etiquetasVerdaderas = predicciones.label_ids

        reporteClasificacion = classification_report(
            etiquetasVerdaderas,
            prediccionesClases,
            target_names=self.codificadorEtiquetas.classes_,
        )
        logger.info(f"Reporte de clasificación del modelo {self.tipoModelo}:\n{reporteClasificacion}")
        self.crearMatrizConfusion(prediccionesClases, etiquetasVerdaderas, self.codificadorEtiquetas.classes_)

        return reporteClasificacion
    
    def crearMatrizConfusion(self, prediccionesClases, etiquetasVerdaderas, etiquetasTarget):
        logger.info(f"Creando la matriz de confusión...")
        matrizConfusion = confusion_matrix(etiquetasVerdaderas, prediccionesClases, normalize="true")

        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=matrizConfusion, display_labels=etiquetasTarget)
        disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
        plt.title("Matriz de confusión normalizada")
        plt.tight_layout()
        plt.savefig(f"matriz_confusion_{self.tipoModelo}.png")
        plt.close()


class DatasetPersonalizado(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def entrenamientoCompleto(rutaDataset, tipoModelo, numeroEpocas=3):
    logger.info(f"Iniciando el entrenamiento para el modelo {tipoModelo}...")
    detector = DetectorClickBait(tipoModelo)

    X, y, etiquetasTarget = detector.cargarYProcesarDatos(rutaDataset, 4, 5)
    X_train_val, X_val, X_test, y_train_val, y_val, y_test = detector.dividirDataset(X, y)
    
    y_train_enc, y_val_enc, y_test_enc = detector.codificarEtiquetas(y_train_val, y_val, y_test)
    
    train_dataset, val_dataset, test_dataset = detector.prepararDataset(
        X_train_val, X_val, X_test, y_train_enc, y_val_enc, y_test_enc
    )

    trainer = detector.entrenarModelo(train_dataset, val_dataset, numeroEpocas)    
    if trainer is None:
        return None
    
    detector.evaluarModelo(trainer, test_dataset, y_test_enc)
    
    detector.guardarModelo(trainer)
    
    logger.info(f"Entrenamiento completado exitosamente.")
    return detector


def modoStandby():
    print("Detector de ClickBait - Modo Standby")
    print("=" * 50)
    
    detector = None
    
    while True:
        print("\nOpciones disponibles:")
        print("1. Entrenar modelo básico (división interna)")
        print("2. Cargar modelo existente")
        print("3. Hacer predicción individual")
        print("4. Procesar archivo CSV")
        print("5. Esquema 1: 75% train / 25% eval + predicciones en dev_gold")
        print("6. Esquema 2: 100% train + 100% dev_gold evaluación")
        print("7. Esquema 3: 100% train + 50% dev_gold eval + 50% test")
        print("8. Predicciones con formato específico (Tweet ID)")
        print("9. Mostrar información del modelo actual")
        print("10. Salir")
        
        opcion = input("\nSelecciona una opción (1-10): ").strip()
        
        if opcion == "1":
            print("\nEntrenando nuevo modelo...")
            ruta_dataset = input("Ruta del dataset de entrenamiento: ").strip()
            tipo_modelo = input("Tipo de modelo (bert/roberta/electra): ").strip().lower()
            epocas = input("Número de épocas (por defecto 3): ").strip()
            
            if not epocas.isdigit():
                epocas = 3
            else:
                epocas = int(epocas)
            
            if tipo_modelo in ["bert", "roberta", "electra"]:
                try:
                    print(f"Iniciando entrenamiento con {epocas} épocas...")
                    detector = entrenamientoCompleto(ruta_dataset, tipo_modelo, epocas)
                    if detector:
                        print("Modelo entrenado y guardado exitosamente!")
                        print(f"Guardado en: {detector.rutaModeloGuardado}")
                except Exception as e:
                    print(f"Error al entrenar: {e}")
            else:
                print("Tipo de modelo no válido.")
        
        elif opcion == "2":
            print("\nCargando modelo existente...")
            tipo_modelo = input("Tipo de modelo a cargar (bert/roberta/electra): ").strip().lower()
            ruta_modelo = input("Ruta del modelo (opcional, presiona Enter para usar ruta por defecto): ").strip()
            
            if tipo_modelo in ["bert", "roberta", "electra"]:
                detector = DetectorClickBait(tipo_modelo)
                if detector.cargarModelo(ruta_modelo if ruta_modelo else None):
                    print("Modelo cargado exitosamente!")
                else:
                    print("Error al cargar el modelo.")
                    detector = None
            else:
                print("Tipo de modelo no válido.")
        
        elif opcion == "3":
            if detector is None or not detector.modeloEntrenado:
                print("Primero debes entrenar o cargar un modelo.")
                continue
            
            print("\nHacer predicción individual...")
            texto = input("Ingresa el texto a analizar: ").strip()
            
            if texto:
                resultados = detector.predecir([texto])
                if resultados:
                    resultado = resultados[0]
                    print(f"\nResultado:")
                    print(f"   Predicción: {resultado['prediccion']}")
                    print(f"   Confianza: {resultado['confianza']:.2%}")
                    print(f"   Probabilidades:")
                    for clase, prob in resultado['probabilidades'].items():
                        print(f"      {clase}: {prob:.2%}")
        
        elif opcion == "4":
            if detector is None or not detector.modeloEntrenado:
                print("Primero debes entrenar o cargar un modelo.")
                continue
            
            print("\nProcesar archivo CSV...")
            ruta_csv = input("Ruta del archivo CSV: ").strip()
            columna = input("Nombre o número de columna con los textos (por defecto 0): ").strip()
            
            try:
                columna = int(columna) if columna.isdigit() else (columna if columna else 0)
                resultados, df = detector.predecirDesdeCSV(ruta_csv, columna)
                
                if resultados:
                    print(f"Procesados {len(resultados)} textos exitosamente!")
                    print("Resumen de predicciones:")
                    
                    conteo = {}
                    for r in resultados:
                        pred = r['prediccion']
                        conteo[pred] = conteo.get(pred, 0) + 1
                    
                    for clase, cantidad in conteo.items():
                        porcentaje = (cantidad / len(resultados)) * 100
                        print(f"   {clase}: {cantidad} ({porcentaje:.1f}%)")
                        
            except Exception as e:
                print(f"Error al procesar el archivo: {e}")
        
        elif opcion == "5":
            print("\nEsquema 1: 75% train / 25% eval + predicciones en dev_gold")
            ruta_train = input("Ruta archivo entrenamiento: ").strip()
            ruta_test = input("Ruta archivo dev_gold: ").strip()
            tipo_modelo = input("Tipo de modelo (bert/roberta/electra): ").strip().lower()
            
            if tipo_modelo in ["bert", "roberta", "electra"]:
                try:
                    detector = DetectorClickBait(tipo_modelo)
                    trainer, df_test = detector.entrenamientoEsquema1(ruta_train, ruta_test)
                    if trainer:
                        print("Entrenamiento completado. Evaluación en dev_gold realizada.")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Tipo de modelo no válido.")
        
        elif opcion == "6":
            print("\nEsquema 2: 100% train + 100% dev_gold evaluación")
            ruta_train = input("Ruta archivo entrenamiento: ").strip()
            ruta_eval = input("Ruta archivo dev_gold: ").strip()
            tipo_modelo = input("Tipo de modelo (bert/roberta/electra): ").strip().lower()
            
            if tipo_modelo in ["bert", "roberta", "electra"]:
                try:
                    detector = DetectorClickBait(tipo_modelo)
                    trainer, df_eval = detector.entrenamientoEsquema2(ruta_train, ruta_eval)
                    if trainer:
                        print("Entrenamiento completado. Evaluación en dev_gold completo realizada.")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Tipo de modelo no válido.")
        
        elif opcion == "7":
            print("\nEsquema 3: 100% train + 50% dev_gold eval + 50% test")
            ruta_train = input("Ruta archivo entrenamiento: ").strip()
            ruta_dev = input("Ruta archivo dev_gold: ").strip()
            porcentaje = input("Porcentaje para evaluación (por defecto 50): ").strip()
            tipo_modelo = input("Tipo de modelo (bert/roberta/electra): ").strip().lower()
            
            if not porcentaje:
                porcentaje = 0.5
            else:
                porcentaje = float(porcentaje) / 100 if float(porcentaje) > 1 else float(porcentaje)
            
            if tipo_modelo in ["bert", "roberta", "electra"]:
                try:
                    detector = DetectorClickBait(tipo_modelo)
                    trainer, df_test = detector.entrenamientoEsquema3(ruta_train, ruta_dev, porcentaje)
                    if trainer:
                        print("Entrenamiento completado. Evaluación en porción de dev_gold realizada.")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Tipo de modelo no válido.")
        
        elif opcion == "8":
            if detector is None or not detector.modeloEntrenado:
                print("Primero debes entrenar o cargar un modelo.")
                continue
            
            print("\nPredicciones con formato específico...")
            ruta_csv = input("Ruta del archivo CSV: ").strip()
            archivo_salida = input("Nombre archivo salida (por defecto 'detection.csv'): ").strip()
            if not archivo_salida:
                archivo_salida = "detection.csv"
            
            columna_id = input("Columna Tweet ID (por defecto 0): ").strip()
            columna_texto = input("Columna texto (por defecto 4): ").strip()
            
            columna_id = int(columna_id) if columna_id.isdigit() else 0
            columna_texto = int(columna_texto) if columna_texto.isdigit() else 4
            
            try:
                resultados = detector.predecirYGuardar(ruta_csv, archivo_salida, columna_id, columna_texto)
                print("Predicciones guardadas exitosamente!")
            except Exception as e:
                print(f"Error: {e}")
        
        elif opcion == "9":
            if detector is None:
                print("No hay modelo cargado.")
            else:
                print(f"\nInformación del modelo:")
                print(f"   Tipo: {detector.tipoModelo}")
                print(f"   Entrenado: {'Sí' if detector.modeloEntrenado else 'No'}")
                if detector.codificadorEtiquetas:
                    print(f"   Clases: {list(detector.codificadorEtiquetas.classes_)}")
                print(f"   Ruta guardado: {detector.rutaModeloGuardado}")
        
        elif opcion == "10":
            print("Hasta luego!")
            break
        
        else:
            print("Opción no válida. Por favor selecciona 1-10.")


def ejemploEsquemas():
    """Ejemplos de uso de los diferentes esquemas"""
    
    # Esquema 1: 75% train / 25% eval + predicciones en dev_gold
    print("Ejecutando Esquema 1...")
    detector1 = DetectorClickBait("bert")
    trainer1, df_test1 = detector1.entrenamientoEsquema1(
        "TA1C_dataset_detection_train.csv", 
        "TA1C_dataset_detection_dev_gold.csv"
    )
    
    # Esquema 2: 100% train + 100% dev_gold evaluación
    print("Ejecutando Esquema 2...")
    detector2 = DetectorClickBait("bert")
    trainer2, df_eval2 = detector2.entrenamientoEsquema2(
        "TA1C_dataset_detection_train.csv", 
        "TA1C_dataset_detection_dev_gold.csv"
    )
    
    # Esquema 3: 100% train + 50% dev_gold eval + 50% test
    print("Ejecutando Esquema 3...")
    detector3 = DetectorClickBait("bert")
    trainer3, df_test3 = detector3.entrenamientoEsquema3(
        "TA1C_dataset_detection_train.csv", 
        "TA1C_dataset_detection_dev_gold.csv",
        porcentajeEval=0.5
    )
    
    # Ejemplo de predicciones con formato específico
    if detector1 and detector1.modeloEntrenado:
        detector1.predecirYGuardar(
            "TA1C_dataset_detection_dev_gold.csv",
            "predicciones_esquema1.csv",
            columnaID=0,
            columnaTexto=4
        )


if __name__ == "__main__":
    usar_standby = input("¿Usar modo standby? (s/n): ").strip().lower()
    
    if usar_standby in ['s', 'si', 'yes', 'y']:
        modoStandby()
    else:
        # Entrenamiento directo original
        rutaDataset = "/home/servidor/Documentos/GitHub/Procesamiento-Lenguaje-Natural-7CM2/Practica_5/corpus/TA1C_dataset_detection_train.csv"
        detector = entrenamientoCompleto(rutaDataset, "bert")