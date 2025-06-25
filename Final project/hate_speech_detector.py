import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import logging
import os
import pickle
import warnings
import csv
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    BertTokenizer, BertForSequenceClassification,
    TrainingArguments, Trainer, pipeline
)
from peft import (
    LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
)
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, 
    confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
import seaborn as sns

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hate_speech_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HateSpeech_Detector")

class DetectorHateSpeech:
    """
    Detector de Hate Speech enfocado a grupos específicos usando LLMs
    Soporta clasificación binaria y multi-etiqueta
    """
    
    def __init__(self, tipoModelo="roberta", usar_lora=True, multi_etiqueta=False):
        """
        Inicializa el detector de hate speech
        
        Args:
            tipoModelo: Tipo de modelo (roberta, bert, beto)
            usar_lora: Si usar LoRA para fine-tuning eficiente
            multi_etiqueta: Si el problema es multi-etiqueta
        """
        self.modelosDisponibles = {
            "roberta": {
                "nombre": "PlanTL-GOB-ES/roberta-base-bne",
                "descripcion": "RoBERTa Base en español (Plan TL)"
            },
            "roberta_large": {
                "nombre": "PlanTL-GOB-ES/roberta-large-bne",
                "descripcion": "RoBERTa Large en español (Plan TL)"
            },
            "bert": {
                "nombre": "dccuchile/bert-base-spanish-wwm-cased",
                "descripcion": "BERT Base en español (UC Chile)"
            },
            "beto": {
                "nombre": "dccuchile/bert-base-spanish-wwm-uncased",
                "descripcion": "BETO - BERT español uncased"
            },
            "distilbert": {
                "nombre": "distilbert-base-multilingual-cased",
                "descripcion": "DistilBERT multilingüe"
            }
        }
        
        self.tipoModelo = tipoModelo
        self.usar_lora = usar_lora
        self.multi_etiqueta = multi_etiqueta
        self.tokenizador = None
        self.modelo = None
        self.peft_model = None
        self.codificadorEtiquetas = None
        self.modeloEntrenado = False
        self.rutaModeloGuardado = f"modelo_hate_speech_{tipoModelo}{'_lora' if usar_lora else ''}"
        
        # Configuración LoRA
        self.lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=16,  # Rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "value", "key", "dense"]  # Para RoBERTa/BERT
        )
        
        if tipoModelo in self.modelosDisponibles:
            self.configuracionModelo = self.modelosDisponibles[tipoModelo]
            self.nombreModelo = self.configuracionModelo["nombre"]
            logger.info(f"Modelo seleccionado: {self.configuracionModelo['descripcion']}")
            logger.info(f"LoRA activado: {usar_lora}")
            logger.info(f"Multi-etiqueta: {multi_etiqueta}")
        else:
            logger.error(f"Modelo {tipoModelo} no disponible")
            raise ValueError(f"Modelo {tipoModelo} no disponible")
    
    def cargarYProcesarDatos(self, rutaArchivo: str, columnaTexto: str = "text", 
                           columnaEtiqueta: str = "label", columnaVariacion: str = "variation"):
        """
        Carga y procesa el dataset de hate speech
        
        Args:
            rutaArchivo: Ruta del archivo CSV
            columnaTexto: Nombre de la columna con los textos
            columnaEtiqueta: Nombre de la columna con las etiquetas
            columnaVariacion: Nombre de la columna con la variación/grupo
        """
        logger.info(f"Cargando datos desde {rutaArchivo}...")
        
        try:
            df = pd.read_csv(rutaArchivo)
            logger.info(f"Dataset cargado: {len(df)} muestras")
            
            # Información del dataset
            logger.info(f"Columnas disponibles: {list(df.columns)}")
            logger.info(f"Distribución de etiquetas:")
            logger.info(df[columnaEtiqueta].value_counts())
            
            if columnaVariacion in df.columns:
                logger.info(f"Distribución por grupo/variación:")
                logger.info(df[columnaVariacion].value_counts())
            
            X = df[columnaTexto].tolist()
            y = df[columnaEtiqueta].tolist()
            
            # Si hay columna de variación, la incluimos
            variaciones = df[columnaVariacion].tolist() if columnaVariacion in df.columns else None
            
            etiquetas_unicas = np.unique(y)
            logger.info(f"Clases encontradas: {etiquetas_unicas}")
            
            return X, y, variaciones, df, etiquetas_unicas
            
        except Exception as error:
            logger.error(f"Error al cargar el archivo: {error}")
            raise
    
    def preprocesarTextos(self, textos: List[str]) -> List[str]:
        """
        Preprocesa los textos para limpiarlos
        """
        textos_limpios = []
        for texto in textos:
            # Remover URLs
            import re
            texto_limpio = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
            # Remover menciones excesivas de @
            texto_limpio = re.sub(r'@\w+', '@USER', texto_limpio)
            # Remover espacios múltiples
            texto_limpio = re.sub(r'\s+', ' ', texto_limpio).strip()
            textos_limpios.append(texto_limpio)
        
        return textos_limpios
    
    def configurarTokenizador(self):
        """Configura el tokenizador según el tipo de modelo"""
        logger.info(f"Configurando tokenizador para {self.tipoModelo}...")
        
        if self.tipoModelo in ["roberta", "roberta_large"]:
            self.tokenizador = AutoTokenizer.from_pretrained(self.nombreModelo)
        elif self.tipoModelo in ["bert", "beto"]:
            self.tokenizador = AutoTokenizer.from_pretrained(self.nombreModelo)
        elif self.tipoModelo == "distilbert":
            self.tokenizador = AutoTokenizer.from_pretrained(self.nombreModelo)
        else:
            raise ValueError(f"Tipo de modelo no reconocido: {self.tipoModelo}")
    
    def configurarModelo(self, num_labels: int):
        """
        Configura el modelo base y aplica LoRA si está habilitado
        """
        logger.info(f"Configurando modelo {self.tipoModelo} con {num_labels} etiquetas...")
        
        # Cargar modelo base
        if self.multi_etiqueta:
            # Para multi-etiqueta, configuramos problema como tal
            self.modelo = AutoModelForSequenceClassification.from_pretrained(
                self.nombreModelo,
                num_labels=num_labels,
                problem_type="multi_label_classification"
            )
        else:
            self.modelo = AutoModelForSequenceClassification.from_pretrained(
                self.nombreModelo,
                num_labels=num_labels
            )
        
        # Aplicar LoRA si está habilitado
        if self.usar_lora:
            logger.info("Aplicando configuración LoRA...")
            self.peft_model = get_peft_model(self.modelo, self.lora_config)
            self.peft_model.print_trainable_parameters()
            return self.peft_model
        
        return self.modelo
    
    def entrenamientoCompleto(self, rutaArchivo: str, esquema: int = 1, 
                            parametros_entrenamiento: dict = None):
        """
        Entrenamiento completo con diferentes esquemas
        
        Args:
            rutaArchivo: Ruta del dataset
            esquema: 1, 2 o 3 según las especificaciones
            parametros_entrenamiento: Parámetros personalizados de entrenamiento
        """
        # Cargar datos
        X, y, variaciones, df, etiquetas_unicas = self.cargarYProcesarDatos(rutaArchivo)
        
        # Preprocesar textos
        X = self.preprocesarTextos(X)
        
        # Configurar codificador de etiquetas
        if self.multi_etiqueta:
            # Para multi-etiqueta, usar MultiLabelBinarizer
            self.codificadorEtiquetas = MultiLabelBinarizer()
            # Asumiendo que las etiquetas están separadas por comas o son listas
            if isinstance(y[0], str):
                y = [label.split(',') if ',' in label else [label] for label in y]
            y_encoded = self.codificadorEtiquetas.fit_transform(y)
        else:
            self.codificadorEtiquetas = LabelEncoder()
            y_encoded = self.codificadorEtiquetas.fit_transform(y)
        
        # Configurar tokenizador y modelo
        self.configurarTokenizador()
        num_labels = len(self.codificadorEtiquetas.classes_) if not self.multi_etiqueta else y_encoded.shape[1]
        modelo = self.configurarModelo(num_labels)
        
        # Ejecutar esquema de entrenamiento
        if esquema == 1:
            return self._esquema1(X, y_encoded, parametros_entrenamiento)
        elif esquema == 2:
            return self._esquema2(X, y_encoded, parametros_entrenamiento)
        elif esquema == 3:
            return self._esquema3(X, y_encoded, parametros_entrenamiento)
        else:
            raise ValueError("Esquema debe ser 1, 2 o 3")
    
    def _esquema1(self, X, y, parametros_entrenamiento=None):
        """Esquema 1: 75% train / 25% eval"""
        logger.info("Ejecutando Esquema 1: 75% train / 25% eval")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=0, stratify=y if not self.multi_etiqueta else None
        )
        
        return self._entrenar(X_train, X_val, y_train, y_val, parametros_entrenamiento)
    
    def _esquema2(self, X, y, parametros_entrenamiento=None):
        """Esquema 2: Cross-validation"""
        logger.info("Ejecutando Esquema 2: Cross-validation")
        
        # Para cross-validation, usamos una porción para entrenamiento directo
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=y if not self.multi_etiqueta else None
        )
        
        return self._entrenar(X_train, X_val, y_train, y_val, parametros_entrenamiento)
    
    def _esquema3(self, X, y, parametros_entrenamiento=None):
        """Esquema 3: Con balanceo de clases"""
        logger.info("Ejecutando Esquema 3: Con balanceo de clases")
        
        # Dividir primero
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=0, stratify=y if not self.multi_etiqueta else None
        )
        
        # Aplicar balanceo solo si no es multi-etiqueta
        if not self.multi_etiqueta:
            # Para balanceo de clases, aplicamos técnicas de sampling
            logger.info("Aplicando balanceo de clases...")
            
            # Obtener conteos de clases
            unique, counts = np.unique(y_train, return_counts=True)
            logger.info(f"Distribución original: {dict(zip(unique, counts))}")
            
            # Aplicar oversampling simple para clases minoritarias
            max_count = max(counts)
            X_train_balanced = []
            y_train_balanced = []
            
            for clase in unique:
                indices_clase = [i for i, label in enumerate(y_train) if label == clase]
                X_clase = [X_train[i] for i in indices_clase]
                y_clase = [y_train[i] for i in indices_clase]
                
                # Duplicar muestras para llegar al balance
                factor = max_count // len(indices_clase)
                resto = max_count % len(indices_clase)
                
                X_train_balanced.extend(X_clase * factor)
                y_train_balanced.extend(y_clase * factor)
                
                if resto > 0:
                    X_train_balanced.extend(X_clase[:resto])
                    y_train_balanced.extend(y_clase[:resto])
            
            logger.info(f"Tamaño después del balanceo: {len(X_train_balanced)}")
            X_train = X_train_balanced
            y_train = y_train_balanced
        
        return self._entrenar(X_train, X_val, y_train, y_val, parametros_entrenamiento)
    
    def _entrenar(self, X_train, X_val, y_train, y_val, parametros_entrenamiento=None):
        """Método de entrenamiento común"""
        
        # Tokenizar
        train_encodings = self.tokenizador(X_train, truncation=True, padding=True, max_length=512)
        val_encodings = self.tokenizador(X_val, truncation=True, padding=True, max_length=512)
        
        # Crear datasets
        train_dataset = DatasetHateSpeech(train_encodings, y_train, self.multi_etiqueta)
        val_dataset = DatasetHateSpeech(val_encodings, y_val, self.multi_etiqueta)
        
        # Configurar parámetros de entrenamiento
        params_default = {
            "output_dir": f"./resultados_{self.tipoModelo}",
            "eval_strategy": "steps",
            "eval_steps": 100,
            "save_strategy": "steps",
            "save_steps": 500,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "logging_dir": f"./logs_{self.tipoModelo}",
            "logging_steps": 50,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_f1_macro",
            "greater_is_better": True,
            "seed": 0,
            "fp16": True,
            "gradient_accumulation_steps": 2,
            "learning_rate": 2e-5 if not self.usar_lora else 5e-4,  # LoRA usa LR más alto
        }
        
        if parametros_entrenamiento:
            params_default.update(parametros_entrenamiento)
        
        training_args = TrainingArguments(**params_default)
        
        # Configurar trainer
        modelo_final = self.peft_model if self.usar_lora else self.modelo
        
        trainer = Trainer(
            model=modelo_final,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.calcular_metricas,
        )
        
        # Entrenar
        logger.info("Iniciando entrenamiento...")
        trainer.train()
        
        # Evaluar
        logger.info("Evaluando modelo...")
        eval_results = trainer.evaluate()
        logger.info(f"Resultados de evaluación: {eval_results}")
        
        # Guardar modelo
        self.modeloEntrenado = True
        self.guardarModelo(trainer)
        
        return trainer, eval_results
    
    def calcular_metricas(self, eval_pred):
        """Calcula métricas de evaluación"""
        predictions, labels = eval_pred
        
        if self.multi_etiqueta:
            # Para multi-etiqueta
            predictions = torch.sigmoid(torch.tensor(predictions))
            predictions = (predictions > 0.5).int().numpy()
            
            # Calcular métricas micro y macro
            precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
                labels, predictions, average='micro'
            )
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                labels, predictions, average='macro'
            )
            
            return {
                'accuracy': accuracy_score(labels, predictions),
                'f1_micro': f1_micro,
                'f1_macro': f1_macro,
                'precision_micro': precision_micro,
                'precision_macro': precision_macro,
                'recall_micro': recall_micro,
                'recall_macro': recall_macro,
            }
        else:
            # Para clasificación binaria/multiclase
            predictions = np.argmax(predictions, axis=1)
            
            return {
                'accuracy': accuracy_score(labels, predictions),
                'f1_macro': f1_score(labels, predictions, average='macro'),
                'f1_weighted': f1_score(labels, predictions, average='weighted'),
                'precision_macro': precision_recall_fscore_support(labels, predictions, average='macro')[0],
                'recall_macro': precision_recall_fscore_support(labels, predictions, average='macro')[1],
            }
    
    def predecir(self, textos: List[str], batch_size: int = 8):
        """
        Hace predicciones sobre una lista de textos
        """
        if not self.modeloEntrenado:
            logger.error("No hay modelo entrenado disponible.")
            return None
        
        try:
            # Preprocesar textos
            textos = self.preprocesarTextos(textos)
            
            # Tokenizar
            encodings = self.tokenizador(
                textos, 
                truncation=True, 
                padding=True, 
                max_length=512,
                return_tensors="pt"
            )
            
            # Hacer predicciones
            modelo_final = self.peft_model if self.usar_lora else self.modelo
            modelo_final.eval()
            
            resultados = []
            
            # Procesar en batches
            for i in range(0, len(textos), batch_size):
                batch_texts = textos[i:i+batch_size]
                batch_encodings = self.tokenizador(
                    batch_texts, 
                    truncation=True, 
                    padding=True, 
                    max_length=512,
                    return_tensors="pt"
                )
                
                with torch.no_grad():
                    outputs = modelo_final(**batch_encodings)
                    
                    if self.multi_etiqueta:
                        # Multi-etiqueta
                        probabilidades = torch.sigmoid(outputs.logits)
                        predicciones = (probabilidades > 0.5).int()
                        
                        for j, texto in enumerate(batch_texts):
                            pred_labels = []
                            prob_dict = {}
                            
                            for k, (pred, prob) in enumerate(zip(predicciones[j], probabilidades[j])):
                                label = self.codificadorEtiquetas.classes_[k]
                                prob_dict[label] = float(prob)
                                if pred == 1:
                                    pred_labels.append(label)
                            
                            resultados.append({
                                "texto": texto,
                                "predicciones": pred_labels,
                                "probabilidades": prob_dict,
                                "confianza_max": float(torch.max(probabilidades[j]))
                            })
                    else:
                        # Clasificación normal
                        probabilidades = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        clases_predichas = torch.argmax(probabilidades, dim=-1)
                        
                        for j, texto in enumerate(batch_texts):
                            etiqueta_predicha = self.codificadorEtiquetas.inverse_transform([clases_predichas[j]])[0]
                            
                            resultados.append({
                                "texto": texto,
                                "prediccion": etiqueta_predicha,
                                "confianza": float(torch.max(probabilidades[j])),
                                "probabilidades": {
                                    clase: float(prob) 
                                    for clase, prob in zip(self.codificadorEtiquetas.classes_, probabilidades[j])
                                }
                            })
            
            return resultados
            
        except Exception as error:
            logger.error(f"Error al hacer predicciones: {error}")
            return None
    
    def guardarModelo(self, trainer=None):
        """Guarda el modelo entrenado"""
        if not self.modeloEntrenado:
            logger.warning("No hay modelo entrenado para guardar.")
            return False
        
        try:
            os.makedirs(self.rutaModeloGuardado, exist_ok=True)
            
            # Guardar modelo
            if self.usar_lora and self.peft_model:
                self.peft_model.save_pretrained(self.rutaModeloGuardado)
            elif trainer:
                trainer.save_model(self.rutaModeloGuardado)
            else:
                self.modelo.save_pretrained(self.rutaModeloGuardado)
            
            # Guardar tokenizador
            self.tokenizador.save_pretrained(self.rutaModeloGuardado)
            
            # Guardar codificador de etiquetas y metadatos
            with open(f"{self.rutaModeloGuardado}/label_encoder.pkl", "wb") as f:
                pickle.dump(self.codificadorEtiquetas, f)
            
            metadatos = {
                "tipo_modelo": self.tipoModelo,
                "usar_lora": self.usar_lora,
                "multi_etiqueta": self.multi_etiqueta,
                "fecha_entrenamiento": datetime.now().isoformat(),
                "clases": (self.codificadorEtiquetas.classes_.tolist() 
                          if hasattr(self.codificadorEtiquetas, 'classes_') 
                          else list(self.codificadorEtiquetas.classes_))
            }
            
            with open(f"{self.rutaModeloGuardado}/metadatos.pkl", "wb") as f:
                pickle.dump(metadatos, f)
            
            logger.info(f"Modelo guardado en: {self.rutaModeloGuardado}")
            return True
            
        except Exception as error:
            logger.error(f"Error al guardar modelo: {error}")
            return False
    
    def cargarModelo(self, rutaModelo=None):
        """Carga un modelo previamente entrenado"""
        if rutaModelo is None:
            rutaModelo = self.rutaModeloGuardado
        
        try:
            # Cargar metadatos
            with open(f"{rutaModelo}/metadatos.pkl", "rb") as f:
                metadatos = pickle.load(f)
            
            logger.info(f"Cargando modelo: {metadatos}")
            
            # Cargar codificador de etiquetas
            with open(f"{rutaModelo}/label_encoder.pkl", "rb") as f:
                self.codificadorEtiquetas = pickle.load(f)
            
            # Configurar tokenizador
            self.configurarTokenizador()
            
            # Cargar modelo
            if metadatos.get("usar_lora", False):
                # Cargar modelo base primero
                modelo_base = AutoModelForSequenceClassification.from_pretrained(self.nombreModelo)
                # Cargar adaptador LoRA
                self.peft_model = PeftModel.from_pretrained(modelo_base, rutaModelo)
            else:
                self.modelo = AutoModelForSequenceClassification.from_pretrained(rutaModelo)
            
            self.modeloEntrenado = True
            logger.info("Modelo cargado exitosamente")
            return True
            
        except Exception as error:
            logger.error(f"Error al cargar modelo: {error}")
            return False
    
    def generar_reporte(self, resultados_evaluacion: dict, X_test, y_test, y_pred):
        """
        Genera un reporte completo de evaluación
        """
        try:
            # Crear directorio para reportes
            os.makedirs("reportes", exist_ok=True)
            
            # Reporte de clasificación
            if not self.multi_etiqueta:
                reporte_clf = classification_report(
                    y_test, y_pred, 
                    target_names=self.codificadorEtiquetas.classes_,
                    output_dict=True
                )
                
                # Matriz de confusión
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    xticklabels=self.codificadorEtiquetas.classes_,
                    yticklabels=self.codificadorEtiquetas.classes_
                )
                plt.title(f'Matriz de Confusión - {self.tipoModelo}')
                plt.ylabel('Etiqueta Real')
                plt.xlabel('Etiqueta Predicha')
                plt.tight_layout()
                plt.savefig(f'reportes/matriz_confusion_{self.tipoModelo}.png')
                plt.close()
            
            # Guardar métricas en CSV
            df_metricas = pd.DataFrame([resultados_evaluacion])
            df_metricas.to_csv(f'reportes/metricas_{self.tipoModelo}.csv', index=False)
            
            logger.info("Reporte generado exitosamente")
            
        except Exception as error:
            logger.error(f"Error al generar reporte: {error}")


class DatasetHateSpeech(Dataset):
    """Dataset personalizado para hate speech"""
    
    def __init__(self, encodings, labels, multi_etiqueta=False):
        self.encodings = encodings
        self.labels = labels
        self.multi_etiqueta = multi_etiqueta

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        
        if self.multi_etiqueta:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

    def __len__(self):
        return len(self.labels)


# Funciones de utilidad para Colab
def ejecutar_experimento_completo(ruta_dataset: str, configuraciones: List[Dict]):
    """
    Ejecuta múltiples experimentos con diferentes configuraciones
    """
    resultados = []
    
    for i, config in enumerate(configuraciones):
        logger.info(f"Ejecutando experimento {i+1}/{len(configuraciones)}")
        logger.info(f"Configuración: {config}")
        
        try:
            detector = DetectorHateSpeech(**config)
            trainer, eval_results = detector.entrenamientoCompleto(ruta_dataset)
            
            resultado = {
                "experimento": i+1,
                "configuracion": config,
                "resultados": eval_results,
                "modelo_guardado": detector.rutaModeloGuardado
            }
            
            resultados.append(resultado)
            
        except Exception as error:
            logger.error(f"Error en experimento {i+1}: {error}")
            resultado = {
                "experimento": i+1,
                "configuracion": config,
                "error": str(error)
            }
            resultados.append(resultado)
    
    # Guardar resultados
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("resultados_experimentos.csv", index=False)
    
    return resultados


def modo_interactivo():
    """
    Modo interactivo para usar en Colab
    """
    print("=== Detector de Hate Speech - Proyecto Final ===")
    print("Selecciona una opción:")
    print("1. Entrenar modelo individual")
    print("2. Ejecutar múltiples experimentos")
    print("3. Cargar modelo y hacer predicciones")
    print("4. Modo de prueba rápida")
    
    opcion = input("Opción (1-4): ").strip()
    
    if opcion == "1":
        # Configurar parámetros
        tipo_modelo = input("Tipo de modelo (roberta/bert/beto): ").strip()
        usar_lora = input("¿Usar LoRA? (s/n): ").strip().lower() == 's'
        multi_etiqueta = input("¿Multi-etiqueta? (s/n): ").strip().lower() == 's'
        esquema = int(input("Esquema de entrenamiento (1-3): ").strip())
        ruta_dataset = input("Ruta del dataset: ").strip()
        
        # Crear detector
        detector = DetectorHateSpeech(
            tipoModelo=tipo_modelo,
            usar_lora=usar_lora,
            multi_etiqueta=multi_etiqueta
        )
        
        # Entrenar
        try:
            trainer, resultados = detector.entrenamientoCompleto(ruta_dataset, esquema)
            print("Entrenamiento completado exitosamente!")
            print(f"Resultados: {resultados}")
            
            # Preguntar si quiere hacer predicciones
            if input("¿Hacer predicciones de prueba? (s/n): ").strip().lower() == 's':
                texto_prueba = input("Ingresa texto para probar: ")
                prediccion = detector.predecir([texto_prueba])
                print(f"Predicción: {prediccion}")
                
        except Exception as e:
            print(f"Error durante el entrenamiento: {e}")
    
    elif opcion == "2":
        ruta_dataset = input("Ruta del dataset: ").strip()
        
        # Configuraciones predefinidas
        configuraciones = [
            {"tipoModelo": "roberta", "usar_lora": True, "multi_etiqueta": False},
            {"tipoModelo": "bert", "usar_lora": True, "multi_etiqueta": False},
            {"tipoModelo": "roberta", "usar_lora": False, "multi_etiqueta": False},
        ]
        
        print(f"Ejecutando {len(configuraciones)} experimentos...")
        resultados = ejecutar_experimento_completo(ruta_dataset, configuraciones)
        
        print("Experimentos completados!")
        for i, resultado in enumerate(resultados):
            if "error" not in resultado:
                print(f"Experimento {i+1}: F1-macro = {resultado['resultados'].get('eval_f1_macro', 'N/A')}")
            else:
                print(f"Experimento {i+1}: Error - {resultado['error']}")
    
    elif opcion == "3":
        ruta_modelo = input("Ruta del modelo guardado: ").strip()
        tipo_modelo = input("Tipo de modelo (roberta/bert/beto): ").strip()
        
        detector = DetectorHateSpeech(tipoModelo=tipo_modelo)
        
        if detector.cargarModelo(ruta_modelo):
            print("Modelo cargado exitosamente!")
            
            while True:
                texto = input("Ingresa texto (o 'salir' para terminar): ").strip()
                if texto.lower() == 'salir':
                    break
                
                if texto:
                    prediccion = detector.predecir([texto])
                    print(f"Predicción: {prediccion}")
        else:
            print("Error al cargar el modelo")
    
    elif opcion == "4":
        print("Modo de prueba rápida con dataset por defecto...")
        
        # Usar configuración por defecto
        detector = DetectorHateSpeech(tipoModelo="roberta", usar_lora=True)
        
        # Intentar usar el dataset disponible
        try:
            trainer, resultados = detector.entrenamientoCompleto("hascosva_2022.csv", esquema=1)
            print("Prueba rápida completada!")
            print(f"Resultados: {resultados}")
            
            # Algunas predicciones de ejemplo
            ejemplos = [
                "Este es un texto normal",
                "Odio a ese grupo de personas",
                "Me gusta la diversidad cultural"
            ]
            
            for ejemplo in ejemplos:
                prediccion = detector.predecir([ejemplo])
                print(f"'{ejemplo}' -> {prediccion}")
                
        except Exception as e:
            print(f"Error en prueba rápida: {e}")
    
    else:
        print("Opción no válida")


def configurar_entorno_colab():
    """
    Configura el entorno de Google Colab con las dependencias necesarias
    """
    print("Configurando entorno para Google Colab...")
    
    # Instalar dependencias si no están disponibles
    try:
        import transformers
        print("✓ Transformers disponible")
    except ImportError:
        print("Instalando transformers...")
        os.system("pip install transformers")
    
    try:
        import peft
        print("✓ PEFT disponible")
    except ImportError:
        print("Instalando PEFT...")
        os.system("pip install peft")
    
    try:
        import torch
        print("✓ PyTorch disponible")
        print(f"CUDA disponible: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch no encontrado - debería estar preinstalado en Colab")
    
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo seleccionado: {device}")
    
    return device


def ejemplo_uso_basico():
    """
    Función de ejemplo que muestra cómo usar el detector
    """
    print("=== Ejemplo de Uso Básico ===")
    
    # 1. Crear detector
    detector = DetectorHateSpeech(
        tipoModelo="roberta",
        usar_lora=True,
        multi_etiqueta=False
    )
    
    # 2. Entrenar (usando dataset por defecto)
    try:
        print("Entrenando modelo...")
        trainer, resultados = detector.entrenamientoCompleto(
            "hascosva_2022.csv", 
            esquema=1
        )
        
        print("Entrenamiento completado!")
        print(f"Métricas de evaluación: {resultados}")
        
        # 3. Hacer predicciones
        textos_prueba = [
            "Los refugiados merecen respeto y dignidad",
            "Ese grupo de personas no debería estar aquí",
            "La diversidad cultural enriquece nuestra sociedad"
        ]
        
        print("\n=== Predicciones de Prueba ===")
        for texto in textos_prueba:
            prediccion = detector.predecir([texto])
            print(f"Texto: '{texto}'")
            print(f"Predicción: {prediccion[0]}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error en ejemplo: {e}")


# Función principal para ejecutar en Colab
def main():
    """
    Función principal para ejecutar en Google Colab
    """
    print("=" * 60)
    print("DETECTOR DE HATE SPEECH - PROYECTO FINAL NLP")
    print("=" * 60)
    
    # Configurar entorno
    device = configurar_entorno_colab()
    
    print("\nOpciones disponibles:")
    print("1. Ejecutar ejemplo básico")
    print("2. Modo interactivo")
    print("3. Ejecutar experimentos predefinidos")
    
    try:
        opcion = input("\nSelecciona una opción (1-3): ").strip()
        
        if opcion == "1":
            ejemplo_uso_basico()
        elif opcion == "2":
            modo_interactivo()
        elif opcion == "3":
            # Experimentos predefinidos
            configuraciones = [
                {"tipoModelo": "roberta", "usar_lora": True, "multi_etiqueta": False},
                {"tipoModelo": "bert", "usar_lora": True, "multi_etiqueta": False},
                {"tipoModelo": "roberta", "usar_lora": False, "multi_etiqueta": False},
            ]
            
            resultados = ejecutar_experimento_completo("hascosva_2022.csv", configuraciones)
            
            print("\n=== Resumen de Experimentos ===")
            for resultado in resultados:
                if "error" not in resultado:
                    config = resultado["configuracion"]
                    metrics = resultado["resultados"]
                    print(f"Modelo: {config['tipoModelo']}, LoRA: {config['usar_lora']}")
                    print(f"  F1-macro: {metrics.get('eval_f1_macro', 'N/A'):.4f}")
                    print(f"  Accuracy: {metrics.get('eval_accuracy', 'N/A'):.4f}")
                else:
                    print(f"Error: {resultado['error']}")
        else:
            print("Opción no válida")
            
    except KeyboardInterrupt:
        print("\nEjecución interrumpida por el usuario")
    except Exception as e:
        print(f"Error inesperado: {e}")


if __name__ == "__main__":
    main()