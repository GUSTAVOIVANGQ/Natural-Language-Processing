"""
DETECTOR DE HATE SPEECH - PROYECTO FINAL NLP 2025
Código optimizado para Google Colab

Este archivo contiene toda la implementación del detector de hate speech
usando LLMs (Large Language Models) con técnicas de fine-tuning y LoRA.
"""

# ============================================================================
# INSTALACIONES NECESARIAS PARA COLAB
# ============================================================================

# Ejecutar esta celda primero en Colab:
"""
!pip install transformers==4.36.0
!pip install peft==0.6.0
!pip install datasets==2.14.0
!pip install accelerate==0.24.0
!pip install scikit-learn==1.3.0
!pip install seaborn==0.12.0
!pip install matplotlib==3.7.0
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
"""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import logging
import os
import pickle
import warnings
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

# Transformers y PEFT
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, 
    EarlyStoppingCallback
)
from peft import (
    LoraConfig, get_peft_model, TaskType, PeftModel
)

# Scikit-learn
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, 
    confusion_matrix, ConfusionMatrixDisplay, 
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# PyTorch
from torch.utils.data import Dataset
import torch.nn.functional as F

# Configurar warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HateSpeechDetector")

# ============================================================================
# CLASE PRINCIPAL DEL DETECTOR
# ============================================================================

class DetectorHateSpeechColab:
    """
    Detector de Hate Speech optimizado para Google Colab
    Incluye soporte para LoRA, diferentes modelos LLM y métricas completas
    """
    
    def __init__(self, 
                 tipo_modelo: str = "roberta", 
                 usar_lora: bool = True,
                 dispositivo: str = None):
        """
        Inicializa el detector
        
        Args:
            tipo_modelo: Tipo de modelo LLM ('roberta', 'bert', 'beto', 'distilbert')
            usar_lora: Si usar LoRA para fine-tuning eficiente
            dispositivo: Dispositivo ('cuda' o 'cpu')
        """
        
        # Configurar dispositivo
        if dispositivo is None:
            self.dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dispositivo = torch.device(dispositivo)
        
        logger.info(f"Dispositivo seleccionado: {self.dispositivo}")
        
        # Modelos disponibles
        self.modelos_disponibles = {
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
        
        # Configuración
        self.tipo_modelo = tipo_modelo
        self.usar_lora = usar_lora
        self.tokenizador = None
        self.modelo = None
        self.peft_model = None
        self.codificador_etiquetas = None
        self.modelo_entrenado = False
        
        # Configuración LoRA
        self.lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "value", "key", "dense"]
        )
        
        # Validar modelo
        if tipo_modelo not in self.modelos_disponibles:
            raise ValueError(f"Modelo {tipo_modelo} no disponible. Opciones: {list(self.modelos_disponibles.keys())}")
        
        self.config_modelo = self.modelos_disponibles[tipo_modelo]
        self.nombre_modelo = self.config_modelo["nombre"]
        
        logger.info(f"Modelo seleccionado: {self.config_modelo['descripcion']}")
        logger.info(f"LoRA activado: {usar_lora}")
    
    def cargar_datos(self, 
                     archivo_csv: str,
                     columna_texto: str = "text",
                     columna_etiqueta: str = "label",
                     columna_grupo: str = "variation") -> Tuple[List[str], List[int], pd.DataFrame]:
        """
        Carga y procesa el dataset
        
        Args:
            archivo_csv: Ruta del archivo CSV
            columna_texto: Nombre de la columna con textos
            columna_etiqueta: Nombre de la columna con etiquetas
            columna_grupo: Nombre de la columna con grupos/variaciones
            
        Returns:
            Tuple con textos, etiquetas y DataFrame original
        """
        logger.info(f"Cargando datos desde: {archivo_csv}")
        
        try:
            df = pd.read_csv(archivo_csv)
            logger.info(f"Dataset cargado: {len(df)} muestras")
            
            # Información del dataset
            logger.info(f"Columnas: {list(df.columns)}")
            logger.info(f"Distribución de etiquetas:")
            print(df[columna_etiqueta].value_counts())
            
            if columna_grupo in df.columns:
                logger.info(f"Distribución por grupo:")
                print(df[columna_grupo].value_counts())
            
            # Extraer datos
            textos = df[columna_texto].fillna("").astype(str).tolist()
            etiquetas = df[columna_etiqueta].tolist()
            
            logger.info(f"Clases únicas: {np.unique(etiquetas)}")
            
            return textos, etiquetas, df
            
        except Exception as e:
            logger.error(f"Error cargando datos: {e}")
            raise
    
    def preprocesar_texto(self, texto: str) -> str:
        """
        Preprocesa un texto individual
        
        Args:
            texto: Texto a preprocesar
            
        Returns:
            Texto preprocesado
        """
        # Convertir a string si no lo es
        texto = str(texto)
        
        # Remover URLs
        texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
        
        # Normalizar menciones
        texto = re.sub(r'@\w+', '@USER', texto)
        
        # Remover hashtags pero mantener el texto
        texto = re.sub(r'#(\w+)', r'\1', texto)
        
        # Limpiar espacios múltiples
        texto = re.sub(r'\s+', ' ', texto).strip()
        
        return texto
    
    def preprocesar_textos(self, textos: List[str]) -> List[str]:
        """
        Preprocesa una lista de textos
        
        Args:
            textos: Lista de textos
            
        Returns:
            Lista de textos preprocesados
        """
        return [self.preprocesar_texto(texto) for texto in textos]
    
    def configurar_tokenizador(self):
        """Configura el tokenizador"""
        logger.info(f"Configurando tokenizador para {self.tipo_modelo}")
        
        self.tokenizador = AutoTokenizer.from_pretrained(
            self.nombre_modelo,
            use_fast=True
        )
        
        # Añadir token de padding si no existe
        if self.tokenizador.pad_token is None:
            self.tokenizador.pad_token = self.tokenizador.eos_token
    
    def configurar_modelo(self, num_labels: int):
        """
        Configura el modelo
        
        Args:
            num_labels: Número de etiquetas/clases
        """
        logger.info(f"Configurando modelo con {num_labels} etiquetas")
        
        # Cargar modelo base
        self.modelo = AutoModelForSequenceClassification.from_pretrained(
            self.nombre_modelo,
            num_labels=num_labels,
            torch_dtype=torch.float16 if self.dispositivo.type == "cuda" else torch.float32
        )
        
        # Mover modelo al dispositivo
        self.modelo.to(self.dispositivo)
        
        # Aplicar LoRA si está habilitado
        if self.usar_lora:
            logger.info("Aplicando LoRA...")
            self.peft_model = get_peft_model(self.modelo, self.lora_config)
            self.peft_model.print_trainable_parameters()
            return self.peft_model
        
        return self.modelo
    
    def entrenar(self,
                 textos: List[str],
                 etiquetas: List[int],
                 test_size: float = 0.2,
                 epochs: int = 3,
                 batch_size: int = 8,
                 learning_rate: float = None,
                 usar_early_stopping: bool = True) -> Dict:
        """
        Entrena el modelo
        
        Args:
            textos: Lista de textos
            etiquetas: Lista de etiquetas
            test_size: Proporción para validación
            epochs: Número de épocas
            batch_size: Tamaño del batch
            learning_rate: Tasa de aprendizaje
            usar_early_stopping: Si usar early stopping
            
        Returns:
            Diccionario con métricas de evaluación
        """
        logger.info("Iniciando entrenamiento...")
        
        # Preprocesar textos
        textos = self.preprocesar_textos(textos)
        
        # Configurar codificador de etiquetas
        self.codificador_etiquetas = LabelEncoder()
        etiquetas_encoded = self.codificador_etiquetas.fit_transform(etiquetas)
        
        # Dividir datos
        X_train, X_val, y_train, y_val = train_test_split(
            textos, etiquetas_encoded,
            test_size=test_size,
            random_state=42,
            stratify=etiquetas_encoded
        )
        
        logger.info(f"Datos de entrenamiento: {len(X_train)}")
        logger.info(f"Datos de validación: {len(X_val)}")
        
        # Configurar tokenizador y modelo
        self.configurar_tokenizador()
        num_labels = len(self.codificador_etiquetas.classes_)
        modelo_final = self.configurar_modelo(num_labels)
        
        # Tokenizar datos
        train_encodings = self.tokenizador(
            X_train,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        val_encodings = self.tokenizador(
            X_val,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Crear datasets
        train_dataset = HateSpeechDataset(train_encodings, y_train)
        val_dataset = HateSpeechDataset(val_encodings, y_val)
        
        # Configurar parámetros de entrenamiento
        if learning_rate is None:
            learning_rate = 5e-4 if self.usar_lora else 2e-5
        
        training_args = TrainingArguments(
            output_dir="./resultados",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None,
            learning_rate=learning_rate,
            fp16=True if self.dispositivo.type == "cuda" else False,
            gradient_accumulation_steps=2,
            seed=42,
        )
        
        # Configurar callbacks
        callbacks = []
        if usar_early_stopping:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        # Crear trainer
        trainer = Trainer(
            model=modelo_final,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.calcular_metricas,
            callbacks=callbacks
        )
        
        # Entrenar
        trainer.train()
        
        # Evaluar
        eval_results = trainer.evaluate()
        
        # Marcar como entrenado
        self.modelo_entrenado = True
        
        # Guardar modelo
        self.guardar_modelo(trainer)
        
        logger.info("Entrenamiento completado!")
        return eval_results
    
    def calcular_metricas(self, eval_pred):
        """Calcula métricas de evaluación"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calcular métricas
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        
        # Precision y recall
        precision, recall, _, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision,
            'recall_macro': recall,
        }
    
    def predecir(self, textos: List[str]) -> List[Dict]:
        """
        Hace predicciones sobre textos
        
        Args:
            textos: Lista de textos
            
        Returns:
            Lista de diccionarios con predicciones
        """
        if not self.modelo_entrenado:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Preprocesar
        textos = self.preprocesar_textos(textos)
        
        # Tokenizar
        encodings = self.tokenizador(
            textos,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Mover al dispositivo
        encodings = {k: v.to(self.dispositivo) for k, v in encodings.items()}
        
        # Predicción
        modelo_final = self.peft_model if self.usar_lora else self.modelo
        modelo_final.eval()
        
        resultados = []
        
        with torch.no_grad():
            outputs = modelo_final(**encodings)
            probabilidades = F.softmax(outputs.logits, dim=-1)
            predicciones = torch.argmax(probabilidades, dim=-1)
        
        # Procesar resultados
        for i, texto in enumerate(textos):
            pred_idx = predicciones[i].item()
            pred_label = self.codificador_etiquetas.inverse_transform([pred_idx])[0]
            confianza = probabilidades[i][pred_idx].item()
            
            # Probabilidades por clase
            probs_dict = {}
            for j, clase in enumerate(self.codificador_etiquetas.classes_):
                probs_dict[str(clase)] = probabilidades[i][j].item()
            
            resultados.append({
                'texto': texto,
                'prediccion': pred_label,
                'confianza': confianza,
                'probabilidades': probs_dict
            })
        
        return resultados
    
    def guardar_modelo(self, trainer=None, ruta: str = None):
        """Guarda el modelo entrenado"""
        if ruta is None:
            ruta = f"modelo_hate_speech_{self.tipo_modelo}"
        
        try:
            os.makedirs(ruta, exist_ok=True)
            
            # Guardar modelo
            if trainer:
                trainer.save_model(ruta)
            elif self.usar_lora and self.peft_model:
                self.peft_model.save_pretrained(ruta)
            else:
                self.modelo.save_pretrained(ruta)
            
            # Guardar tokenizador
            self.tokenizador.save_pretrained(ruta)
            
            # Guardar metadatos
            metadatos = {
                'tipo_modelo': self.tipo_modelo,
                'usar_lora': self.usar_lora,
                'clases': self.codificador_etiquetas.classes_.tolist(),
                'fecha_entrenamiento': datetime.now().isoformat()
            }
            
            with open(f"{ruta}/metadatos.json", "w") as f:
                import json
                json.dump(metadatos, f, indent=2)
            
            # Guardar codificador
            with open(f"{ruta}/label_encoder.pkl", "wb") as f:
                pickle.dump(self.codificador_etiquetas, f)
            
            logger.info(f"Modelo guardado en: {ruta}")
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
    
    def generar_reporte(self, 
                       textos_test: List[str], 
                       etiquetas_test: List[int],
                       mostrar_graficos: bool = True) -> Dict:
        """
        Genera un reporte completo de evaluación
        
        Args:
            textos_test: Textos de prueba
            etiquetas_test: Etiquetas verdaderas
            mostrar_graficos: Si mostrar gráficos
            
        Returns:
            Diccionario con métricas
        """
        logger.info("Generando reporte de evaluación...")
        
        # Hacer predicciones
        predicciones = self.predecir(textos_test)
        pred_labels = [pred['prediccion'] for pred in predicciones]
        
        # Convertir etiquetas a mismo formato
        etiquetas_str = [str(label) for label in etiquetas_test]
        
        # Calcular métricas
        accuracy = accuracy_score(etiquetas_str, pred_labels)
        f1_macro = f1_score(etiquetas_str, pred_labels, average='macro')
        f1_weighted = f1_score(etiquetas_str, pred_labels, average='weighted')
        
        # Reporte de clasificación
        reporte_clf = classification_report(
            etiquetas_str, pred_labels, 
            output_dict=True,
            zero_division=0
        )
        
        # Matriz de confusión
        cm = confusion_matrix(etiquetas_str, pred_labels)
        
        if mostrar_graficos:
            # Gráfico de matriz de confusión
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', 
                       xticklabels=np.unique(etiquetas_str),
                       yticklabels=np.unique(etiquetas_str))
            plt.title('Matriz de Confusión')
            plt.ylabel('Etiqueta Real')
            plt.xlabel('Etiqueta Predicha')
            plt.tight_layout()
            plt.show()
            
            # Gráfico de métricas por clase
            clases = list(reporte_clf.keys())[:-3]  # Excluir 'accuracy', 'macro avg', 'weighted avg'
            f1_scores = [reporte_clf[clase]['f1-score'] for clase in clases]
            
            plt.figure(figsize=(10, 6))
            plt.bar(clases, f1_scores)
            plt.title('F1-Score por Clase')
            plt.ylabel('F1-Score')
            plt.xlabel('Clase')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        
        # Compilar resultados
        resultado = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'matriz_confusion': cm.tolist(),
            'reporte_clasificacion': reporte_clf,
            'num_muestras': len(textos_test)
        }
        
        # Mostrar resumen
        print("\n" + "="*50)
        print("RESUMEN DE EVALUACIÓN")
        print("="*50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (macro): {f1_macro:.4f}")
        print(f"F1-Score (weighted): {f1_weighted:.4f}")
        print(f"Número de muestras: {len(textos_test)}")
        print("="*50)
        
        return resultado


# ============================================================================
# DATASET PERSONALIZADO
# ============================================================================

class HateSpeechDataset(Dataset):
    """Dataset personalizado para hate speech"""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
    def __len__(self):
        return len(self.labels)


# ============================================================================
# FUNCIONES DE UTILIDAD PARA COLAB
# ============================================================================

def configurar_entorno():
    """Configura el entorno de Google Colab"""
    print("🔧 Configurando entorno...")
    
    # Verificar GPU
    if torch.cuda.is_available():
        print(f"✅ GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  GPU no disponible, usando CPU")
    
    # Configurar matplotlib para Colab
    plt.style.use('default')
    
    print("✅ Entorno configurado correctamente")


def cargar_dataset_ejemplo():
    """Carga el dataset de ejemplo"""
    # Intentar cargar el dataset por defecto
    try:
        detector = DetectorHateSpeechColab()
        textos, etiquetas, df = detector.cargar_datos("hascosva_2022.csv")
        print(f"✅ Dataset cargado: {len(textos)} muestras")
        return textos, etiquetas, df
    except:
        print("⚠️  No se pudo cargar el dataset por defecto")
        return None, None, None


def ejecutar_experimento_completo(ruta_dataset: str = "hascosva_2022.csv"):
    """
    Ejecuta un experimento completo de entrenamiento y evaluación
    
    Args:
        ruta_dataset: Ruta del dataset
    """
    configurar_entorno()
    
    print("\n🚀 Iniciando experimento completo...")
    print("="*60)
    
    try:
        # Crear detector
        detector = DetectorHateSpeechColab(
            tipo_modelo="roberta",
            usar_lora=True
        )
        
        # Cargar datos
        textos, etiquetas, df = detector.cargar_datos(ruta_dataset)
        
        # Dividir datos para evaluación final
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            textos, etiquetas, 
            test_size=0.2, 
            random_state=42, 
            stratify=etiquetas
        )
        
        # Entrenar modelo
        print("\n📚 Entrenando modelo...")
        metricas_entrenamiento = detector.entrenar(
            X_train_val, y_train_val,
            epochs=3,
            batch_size=8,
            usar_early_stopping=True
        )
        
        print("\n📊 Métricas de entrenamiento:")
        for key, value in metricas_entrenamiento.items():
            if key.startswith('eval_'):
                print(f"  {key}: {value:.4f}")
        
        # Evaluación final
        print("\n🔍 Evaluación final en conjunto de prueba...")
        reporte_final = detector.generar_reporte(
            X_test, y_test, 
            mostrar_graficos=True
        )
        
        # Ejemplos de predicción
        print("\n🎯 Ejemplos de predicción:")
        ejemplos = [
            "Los refugiados merecen respeto y apoyo",
            "No quiero a esos extranjeros en mi país",
            "La diversidad cultural es enriquecedora",
            "Todos los inmigrantes son delincuentes"
        ]
        
        predicciones_ejemplo = detector.predecir(ejemplos)
        
        for pred in predicciones_ejemplo:
            print(f"\nTexto: '{pred['texto']}'")
            print(f"Predicción: {pred['prediccion']} (confianza: {pred['confianza']:.3f})")
        
        print("\n✅ Experimento completado exitosamente!")
        return detector, reporte_final
        
    except Exception as e:
        print(f"❌ Error en el experimento: {e}")
        return None, None


def comparar_modelos(ruta_dataset: str = "hascosva_2022.csv"):
    """
    Compara diferentes modelos y configuraciones
    
    Args:
        ruta_dataset: Ruta del dataset
    """
    configurar_entorno()
    
    print("\n🔬 Comparando modelos...")
    print("="*60)
    
    # Configuraciones a probar
    configuraciones = [
        {"tipo_modelo": "roberta", "usar_lora": True, "nombre": "RoBERTa + LoRA"},
        {"tipo_modelo": "bert", "usar_lora": True, "nombre": "BERT + LoRA"},
        {"tipo_modelo": "roberta", "usar_lora": False, "nombre": "RoBERTa (completo)"},
    ]
    
    resultados = []
    
    for i, config in enumerate(configuraciones):
        print(f"\n🔄 Evaluando configuración {i+1}/{len(configuraciones)}: {config['nombre']}")
        
        try:
            # Crear detector
            detector = DetectorHateSpeechColab(
                tipo_modelo=config["tipo_modelo"],
                usar_lora=config["usar_lora"]
            )
            
            # Cargar datos
            textos, etiquetas, _ = detector.cargar_datos(ruta_dataset)
            
            # Entrenar
            metricas = detector.entrenar(
                textos, etiquetas,
                epochs=2,  # Menos épocas para comparación rápida
                batch_size=8
            )
            
            # Guardar resultados
            resultado = {
                "configuracion": config["nombre"],
                "f1_macro": metricas.get("eval_f1_macro", 0),
                "accuracy": metricas.get("eval_accuracy", 0),
                "tipo_modelo": config["tipo_modelo"],
                "usar_lora": config["usar_lora"]
            }
            
            resultados.append(resultado)
            
            print(f"   ✅ F1-macro: {resultado['f1_macro']:.4f}")
            print(f"   ✅ Accuracy: {resultado['accuracy']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            resultado = {
                "configuracion": config["nombre"],
                "error": str(e)
            }
            resultados.append(resultado)
    
    # Mostrar comparación final
    print("\n📊 COMPARACIÓN FINAL")
    print("="*60)
    
    df_resultados = pd.DataFrame([r for r in resultados if "error" not in r])
    
    if not df_resultados.empty:
        # Gráfico de comparación
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(df_resultados['configuracion'], df_resultados['f1_macro'])
        plt.title('F1-Score Macro por Configuración')
        plt.ylabel('F1-Score')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        plt.bar(df_resultados['configuracion'], df_resultados['accuracy'])
        plt.title('Accuracy por Configuración')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Mostrar mejor configuración
        mejor_config = df_resultados.loc[df_resultados['f1_macro'].idxmax()]
        print(f"\n🏆 Mejor configuración: {mejor_config['configuracion']}")
        print(f"   F1-macro: {mejor_config['f1_macro']:.4f}")
        print(f"   Accuracy: {mejor_config['accuracy']:.4f}")
    
    return resultados


def demo_interactivo():
    """Demostración interactiva para Colab"""
    configurar_entorno()
    
    print("\n🎮 DEMOSTRACIÓN INTERACTIVA")
    print("="*60)
    
    # Cargar modelo preentrenado o entrenar uno nuevo
    print("Selecciona una opción:")
    print("1. Entrenar modelo nuevo")
    print("2. Usar modelo de demostración")
    
    # Para Colab, siempre entrenar modelo nuevo
    print("\n🚀 Entrenando modelo de demostración...")
    
    try:
        detector = DetectorHateSpeechColab(tipo_modelo="roberta", usar_lora=True)
        textos, etiquetas, _ = detector.cargar_datos("hascosva_2022.csv")
        
        # Entrenar con subset pequeño para demo rápida
        subset_size = min(1000, len(textos))
        indices = np.random.choice(len(textos), subset_size, replace=False)
        textos_demo = [textos[i] for i in indices]
        etiquetas_demo = [etiquetas[i] for i in indices]
        
        detector.entrenar(textos_demo, etiquetas_demo, epochs=2, batch_size=8)
        
        print("\n✅ Modelo entrenado! Ahora puedes probar predicciones:")
        
        # Ejemplos predefinidos
        ejemplos = [
            "Los refugiados merecen ayuda y comprensión",
            "Odio a todos los inmigrantes",
            "La diversidad hace más rica nuestra cultura",
            "Deberían deportar a todos los extranjeros"
        ]
        
        print("\n🎯 Predicciones en ejemplos:")
        predicciones = detector.predecir(ejemplos)
        
        for pred in predicciones:
            emoji = "😡" if pred['prediccion'] == 1 else "😊"
            print(f"{emoji} '{pred['texto']}'")
            print(f"   Predicción: {'Hate Speech' if pred['prediccion'] == 1 else 'No Hate Speech'}")
            print(f"   Confianza: {pred['confianza']:.3f}\n")
        
        return detector
        
    except Exception as e:
        print(f"❌ Error en la demostración: {e}")
        return None


# ============================================================================
# FUNCIÓN PRINCIPAL PARA EJECUTAR EN COLAB
# ============================================================================

def main():
    """
    Función principal para ejecutar en Google Colab
    
    Para usar en Colab, ejecuta:
    
    # Primera celda - Instalaciones
    !pip install transformers peft datasets accelerate
    
    # Segunda celda - Ejecutar código
    from hate_speech_colab import main
    main()
    """
    print("🎯 DETECTOR DE HATE SPEECH - PROYECTO FINAL NLP")
    print("="*70)
    print("Desarrollado para Google Colab")
    print("="*70)
    
    configurar_entorno()
    
    print("\n📋 Opciones disponibles:")
    print("1. 🚀 Experimento completo (recomendado)")
    print("2. 🔬 Comparar modelos")
    print("3. 🎮 Demostración interactiva")
    print("4. 📚 Entrenar modelo personalizado")
    
    # Para Colab, ejecutar experimento completo por defecto
    print("\n🚀 Ejecutando experimento completo...")
    
    try:
        detector, reporte = ejecutar_experimento_completo()
        
        if detector is not None:
            print("\n🎉 ¡Experimento completado exitosamente!")
            print("\nPuedes usar el detector entrenado para hacer predicciones:")
            print("predicciones = detector.predecir(['tu texto aquí'])")
            
            return detector
        else:
            print("\n❌ El experimento no se completó correctamente")
            return None
            
    except Exception as e:
        print(f"\n❌ Error ejecutando el experimento: {e}")
        return None


# ============================================================================
# CÓDIGO PARA EJECUTAR AUTOMÁTICAMENTE EN COLAB
# ============================================================================

if __name__ == "__main__":
    # Si se ejecuta como script, mostrar instrucciones
    print("📋 INSTRUCCIONES PARA GOOGLE COLAB:")
    print("="*50)
    print("1. Sube el archivo 'hascosva_2022.csv' a tu sesión de Colab")
    print("2. Ejecuta: from hate_speech_colab import main")
    print("3. Ejecuta: detector = main()")
    print("4. ¡Disfruta del detector de hate speech!")
    print("="*50)
