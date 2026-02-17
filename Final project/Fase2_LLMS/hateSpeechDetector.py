import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import logging
import pickle
import os

from personalizacionDataset import DatasetPersonalizado
from transformers import (
    BertTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    ElectraTokenizerFast,
    BertForSequenceClassification,
    ElectraForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="app.log",  # Omitir para solo consola
    filemode="a",  # 'w' para sobrescribir, 'a' para añadir
)

logger = logging.getLogger("hateSpeechDetector")

torch.cuda.is_available()


class DetectorHateSpeech:

    def __init__(self, tipoLLM):
        self.modelosDisponibles = {
            "bert": {
                "nombre": "dccuchile/bert-base-spanish-wwm-cased",
                "descripcion": "Modelo BERT Base en español",
            },
            "roberta": {
                "nombre": "PlanTL-GOB-ES/roberta-base-bne",
                "descripcion": "Modelo RoBERTa Base en español",
            },
            "electra": {
                "nombre": "google/electra-base-discriminator",
                "descripcion": "Modelo ELECTRA Base en español",
            },
        }
        self.tipoLLM = tipoLLM
        self.tokenizador = None
        self.modelo = None
        self.codificadorEtiquetas = None
        self.modeloEntrenado = False
        self.rutaModeloGuardado = f"modelo_guardado_{tipoLLM}"

        if tipoLLM in self.modelosDisponibles:
            self.configuracionModelo = self.modelosDisponibles[tipoLLM]
            self.nombreModelo = self.configuracionModelo["nombre"]
            logger.info(
                f"Modelo seleccionado: \t {self.configuracionModelo['descripcion']}"
            )
            logger.info(f"Ruta del modelo: \t {self.nombreModelo}")

        else:
            logger.warning(
                f"Modelo {tipoLLM} no disponible o no reconocido.\n Modelos disponibles:\n"
            )
            for clave, configuracion in self.modelosDisponibles.items():
                logger.info(f"{clave}: {configuracion['descripcion']}")
            raise ValueError(f"Modelo {tipoLLM} no disponible.")

    def cargarProcesarDatos(self, rutaArchivo, columnaTexto=0, columnaTarget=1):
        logger.info(f"Cargando datos desde {rutaArchivo}...")
        try:
            df_data = pd.read_csv(
                rutaArchivo, sep=",", engine="python", usecols=[columnaTexto]
            )
            df_target = pd.read_csv(
                rutaArchivo, sep=",", engine="python", usecols=[columnaTarget]
            )
        except Exception as ErrorCargaDatos:
            logger.error(f"Error al cargar los datos: {ErrorCargaDatos}")
            raise ValueError(f"Error al cargar los datos: {ErrorCargaDatos}")

        X = df_data.iloc[:, 0].tolist()
        y = df_target.iloc[:, 0].tolist()
        etiquetasTarget = np.unique(y)
        logger.info(
            f"Dataset cargado con {len(X)} muestras, {len(etiquetasTarget)} clases."
        )
        logger.info(f"Clases: {etiquetasTarget}")
        return X, y, etiquetasTarget

    def dividirDataset(self, X, y, tamanioTest=0.25):
        logger.info(f"Dividiendo el dataset para el conjunto de entrenamiento...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=tamanioTest, random_state=0, stratify=y
        )
        logger.info(
            f"Dataset dividido en {len(X_train)} muestras para entrenamiento y {len(X_test)} para prueba."
        )
        return X_train, X_test, y_train, y_test

    def codificarEtiquetas(self, y_train, y_test):
        logger.info("Codificando etiquetas...")
        self.codificadorEtiquetas = LabelEncoder()
        y_train_codificado = self.codificadorEtiquetas.fit_transform(y_train)
        y_test_codificado = self.codificadorEtiquetas.transform(y_test)
        logger.info(f"Mapeo de etiquetas: {self.codificadorEtiquetas.classes_}")
        logger.info(
            f"Etiquetas codificadas: {len(np.unique(y_train_codificado))} clases en entrenamiento, {len(np.unique(y_test_codificado))} clases en prueba."
        )

        return y_train_codificado, y_test_codificado

    def configurarTokenizador(self):
        logger.info("Configurando el tokenizador...")
        if self.tipoLLM == "bert":
            logger.info(f"Configurando BertTokenizer...")
            self.tokenizador = BertTokenizer.from_pretrained(
                "dccuchile/bert-base-spanish-wwm-cased"
            )
        elif self.tipoLLM == "roberta":
            logger.info(f"Configurando RobertaTokenizer...")
            self.tokenizador = RobertaTokenizer.from_pretrained(
                "PlanTL-GOB-ES/roberta-base-bne"
            )
        elif self.tipoLLM == "electra":
            logger.info(f"Configurando ElectraTokenizer...")
            self.tokenizador = ElectraTokenizerFast.from_pretrained(
                "google/electra-base-discriminator"
            )
        else:
            logger.error(
                f"Tipo de modelo no reconocido: {self.tipoLLM}. No se puede configurar un tokenizador."
            )

        logger.info(f"Tokenizador configurado - {self.tokenizador}")

    def prepararDataset(self, X_train, X_test, y_train_codificado, y_test_codificado):
        logger.info("Preparando el tokenizador...")
        if self.tokenizador is None:
            self.configurarTokenizador()

        logger.info(f"Tokenizador configurado - {self.tokenizador}")

        logger.info("Tokenizando el conjunto de entrenamiento...")
        trainEncoding = self.tokenizador(X_train, padding=True, truncation=True)

        logger.info("Tokenizando el conjunto de prueba...")
        testEncoding = self.tokenizador(X_test, padding=True, truncation=True)

        logger.info("Creando datasets personalizados...")
        trainDataset = DatasetPersonalizado(trainEncoding, y_train_codificado)
        testDataset = DatasetPersonalizado(testEncoding, y_test_codificado)

        logger.info(
            f"Dataset de entrenamiento preparado con {len(trainDataset)} muestras."
        )
        logger.info(f"Dataset de prueba preparado con {len(testDataset)} muestras.")
        return trainDataset, testDataset

    def calcularMetricas(self, pred):
        etiquetas = pred.label_ids
        predicciones = pred.predictions.argmax(-1)
        f1 = f1_score(etiquetas, predicciones, average="macro")
        accuracy = accuracy_score(etiquetas, predicciones)
        return {"accuracy": accuracy, "f1": f1}

    def entrenarModelo(self, trainDataset, testDataset, numeroEpocas=3):
        logger.info(f"Definiendo el modelo...")
        numeroEtiquetas = len(self.codificadorEtiquetas.classes_)

        directorioSalidaModelo = (
            f"{self.tipoLLM}_output"  # Directorio de salida para archivos del modelo
        )

        try:
            if self.tipoLLM == "bert":
                modelo = BertForSequenceClassification.from_pretrained(
                    "dccuchile/bert-base-spanish-wwm-cased", num_labels=numeroEtiquetas
                )
            elif self.tipoLLM == "roberta":
                modelo = RobertaForSequenceClassification.from_pretrained(
                    "PlanTL-GOB-ES/roberta-base-bne", num_labels=numeroEtiquetas
                )
            elif self.tipoLLM == "electra":
                modelo = ElectraForSequenceClassification.from_pretrained(
                    "google/electra-base-discriminator", num_labels=numeroEtiquetas
                )
        except Exception as ErrorCargaModelo:
            logger.error(f"Error al cargar el modelo: {ErrorCargaModelo}")
            raise ValueError(f"Modelo {self.tipoLLM} no disponible o no reconocido.")

        logger.info(
            f"Configurando los argumentos de entrenamiento del {self.tipoLLM}..."
        )

        argsEntrenamiento = TrainingArguments(
            output_dir=directorioSalidaModelo,
            learning_rate=2e-5,  # Mantener bajo
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_total_limit=3,
            seed=0,
            load_best_model_at_end=True,
            fp16=True,
            logging_steps=50,
            num_train_epochs=numeroEpocas,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            # dataloader_num_workers=4
            # per_device_train_batch_size=8,  # Batch más pequeño
            # per_device_eval_batch_size=16,
            # weight_decay=0.01,  # Agregar regularización
            # warmup_ratio=0.1,  # Agregar warmup
        )

        entrenador = Trainer(
            model=modelo,
            args=argsEntrenamiento,
            train_dataset=trainDataset,
            eval_dataset=testDataset,
            compute_metrics=self.calcularMetricas,
        )

        try:
            logger.info(f"Comenzando el entrenamiento del modelo...")
            entrenador.train()
            self.modeloEntrenado = True
            self.modelo = entrenador.model
            logger.info(f"Entrenamiento completado exitosamente.")
        except Exception as error:
            logger.error(f"Error al entrenar el modelo: {error}")
            raise ValueError(f"Error al entrenar el modelo: {error}")

        return entrenador

    def guardarModelo(self, entrenador=None):
        if not self.modeloEntrenado:
            logger.warning("El modelo no ha sido entrenado. No se puede guardar.")
            return False

        try:
            os.makedirs(self.rutaModeloGuardado, exist_ok=True)
            if entrenador:
                entrenador.save_model(self.rutaModeloGuardado)
            else:
                self.modelo.save_pretrained(self.rutaModeloGuardado)
            self.tokenizador.save_pretrained(self.rutaModeloGuardado)

            logger.info(f"Modelo guardado en {self.rutaModeloGuardado}")
            with open(
                f"{self.rutaModeloGuardado}/codificador_etiquetas.pkl", "wb"
            ) as codificadorEtiquetas:
                pickle.dump(self.codificadorEtiquetas, codificadorEtiquetas)

            metadatos = {
                "tipo_llm": self.tipoLLM,
                "fecha_entrenamiento": datetime.datetime.now().isoformat(),
                "clases": self.codificadorEtiquetas.classes_.tolist(),
            }
            with open(
                f"{self.rutaModeloGuardado}/metadatos.pkl", "wb"
            ) as archivoMetadatos:
                pickle.dump(metadatos, archivoMetadatos)
            logger.info(
                f"Metadatos guardados en {self.rutaModeloGuardado}/metadatos.pkl"
            )
            return True
        except Exception as errorGuardarModelo:
            logger.error(f"Error al guardar el modelo: {errorGuardarModelo}")
            return False

    def cargarModelo(self, rutaModelo=None):
        if rutaModelo is None:
            rutaModelo = self.rutaModeloGuardado

        try:
            if not os.path.exists(rutaModelo):
                logger.error(f"La ruta del modelo {rutaModelo} no existe.")
                return False

            with open(f"{rutaModelo}/metadatos.pkl", "rb") as f:
                metadatos = pickle.load(f)
            logger.info(
                f"Cargando el modelo entrenado el {metadatos['fecha_entrenamiento']}..."
            )

            with open(f"{rutaModelo}/codificador_etiquetas.pkl", "rb") as f:
                self.codificadorEtiquetas = pickle.load(f)

            self.configurarTokenizador()

            if self.tipoLLM == "bert":
                self.modelo = BertForSequenceClassification.from_pretrained(rutaModelo)
            elif self.tipoLLM == "roberta":
                self.modelo = RobertaForSequenceClassification.from_pretrained(
                    rutaModelo
                )
            elif self.tipoLLM == "electra":
                self.modelo = ElectraForSequenceClassification.from_pretrained(
                    rutaModelo
                )

            self.modeloEntrenado = True
            logger.info(f"Modelo cargado exitosamente desde {rutaModelo}.")
            return True
        except Exception as errorCargarModelo:
            logger.error(f"Error al cargar el modelo: {errorCargarModelo}")
            return False

    def realizaInferencia(self, textos):
        if not self.modeloEntrenado:
            logger.error(
                "El modelo no ha sido entrenado o cargado. No se puede realizar inferencia."
            )
            return None

        try:
            disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            entrada = self.tokenizador(
                textos, truncation=True, padding=True, return_tensors="pt"
            )  # Pienso que aquí es el error

            entrada = {
                k: v.to(disp) for k, v in entrada.items()
            }  # Mover tensores a GPU si está disponible
            self.modelo.to(disp)  # Mover el modelo a GPU si está disponible
            logger.info(f"Realizando inferencia para {len(textos)} ...")
            self.modelo.eval()

            with torch.no_grad():
                output = self.modelo(**entrada)
                predicciones = torch.nn.functional.softmax(output.logits, dim=-1)
                clases_predichas = torch.argmax(predicciones, dim=-1)

            # Convertir a etiquetas originales
            etiquetas_predichas = self.codificadorEtiquetas.inverse_transform(
                clases_predichas.cpu().numpy()
            )
            probabilidades = predicciones.cpu().numpy()

            resultados = []
            for i, texto in enumerate(textos):

                resultado = {
                    "texto": texto,
                    "prediccion": etiquetas_predichas[i],
                    "confianza": float(np.max(probabilidades[i])),
                    "probabilidades": {
                        clase: float(prob)
                        for clase, prob in zip(
                            self.codificadorEtiquetas.classes_, probabilidades[i]
                        )
                    },
                }

                resultados.append(resultado)

            logger.info(f"Inferencia realizada con éxito para {len(textos)} textos.")
            return resultados
        except Exception as errorInferencia:
            logger.error(f"Error al realizar la inferencia: {errorInferencia}")
            return None

    def evaluarModelo(self, entrenador, testDataset):
        """
        Evalúa el modelo usando el conjunto de prueba.
        Ya no necesita recibir las clases como parámetro porque usa self.codificadorEtiquetas
        """
        if not self.modeloEntrenado:
            logger.error(
                "El modelo no ha sido entrenado o cargado. No se puede evaluar."
            )
            return None

        predicciones = entrenador.predict(testDataset)
        prediccionesClases = np.argmax(predicciones.predictions, axis=-1)
        etiquetasVerdaderas = predicciones.label_ids

        reporteClasificacion = classification_report(
            etiquetasVerdaderas,
            prediccionesClases,
            target_names=self.codificadorEtiquetas.classes_,
        )
        logger.info(
            f"Reporte de clasificación del modelo {self.tipoLLM}:\n{reporteClasificacion}"
        )
        self.crearMatrizConfusion(
            prediccionesClases, etiquetasVerdaderas, self.codificadorEtiquetas.classes_
        )

        return reporteClasificacion

    def crearMatrizConfusion(
        self, prediccionesClases, etiquetasVerdaderas, etiquetasTarget
    ):
        logger.info(f"Creando la matriz de confusión...")
        matrizConfusion = confusion_matrix(
            etiquetasVerdaderas, prediccionesClases, normalize="true"
        )

        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=matrizConfusion, display_labels=etiquetasTarget
        )
        disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
        plt.title("Matriz de confusión normalizada")
        plt.tight_layout()
        plt.savefig(f"matriz_confusion_{self.tipoLLM}.png")
        plt.close()

    @classmethod
    def entrenarNuevoModelo(cls, rutaDataset, tipoLLM, numeroEpocas=4):
        """
        Método de clase para entrenar un nuevo modelo desde cero.
        Retorna una instancia del detector con el modelo entrenado.
        """
        logger.info(f"Iniciando el entrenamiento del modelo {tipoLLM}...")

        # Crear nueva instancia del detector
        detector = cls(tipoLLM)

        # Cargar y procesar datos
        X, y, etiquetasTarget = detector.cargarProcesarDatos(rutaDataset, 0, 1)
        X_train, X_test, y_train, y_test = detector.dividirDataset(X, y)

        # Codificar etiquetas
        y_train_enc, y_test_enc = detector.codificarEtiquetas(y_train, y_test)

        # Preparar datasets
        trainDataset, testDataset = detector.prepararDataset(
            X_train, X_test, y_train_enc, y_test_enc
        )

        # Entrenar modelo
        entrenador = detector.entrenarModelo(trainDataset, testDataset, numeroEpocas)
        if entrenador is None:
            return None

        # Evaluar modelo - ahora sin el tercer parámetro
        detector.evaluarModelo(entrenador, testDataset)

        # Guardar modelo
        detector.guardarModelo(entrenador)
        logger.info(
            f"Entrenamiento del modelo {tipoLLM} completado y guardado exitosamente."
        )
        return detector


def menu():
    print("=" * 60)
    print("Bienvenido al detector de discurso de odio")
    print("=" * 60)
    detector = None
    while True:
        print("\n1. Entrenar modelo")
        print("2. Cargar modelo existente")
        print("3. Realizar inferencia")
        print("4. Mostrar información del modelo actual")
        print("5. Salir")

        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            print("Entrenando modelo...")
            ruta_dataset = input("Ruta del dataset de entrenamiento: ").strip()
            tipo_modelo = (
                input("Tipo de modelo (bert/roberta/electra): ").strip().lower()
            )
            epocas = input("Número de épocas (por defecto 3): ").strip()
            if not epocas.isdigit():
                epocas = 3
            else:
                epocas = int(epocas)

            if tipo_modelo in ["bert", "roberta", "electra"]:
                try:
                    print(
                        f"Iniciando entrenamiento con {epocas} épocas y el modelo {tipo_modelo}..."
                    )
                    # Usar el nuevo método de clase
                    detector = DetectorHateSpeech.entrenarNuevoModelo(
                        ruta_dataset, tipo_modelo, epocas
                    )
                    logger.info(f"Modelo {tipo_modelo} entrenado")

                    if detector:
                        print(
                            f"Modelo {tipo_modelo} entrenado y guardado exitosamente."
                        )
                        print(f"Modelo guardado en : {detector.rutaModeloGuardado}")
                except Exception as e:
                    print(f"Error durante el entrenamiento: {e}")
                    logger.error(f"Error durante el entrenamiento: {e}")
            else:
                print(
                    "Tipo de modelo no reconocido. Por favor, elija entre 'bert', 'roberta' o 'electra'."
                )
                logger.error(
                    "Tipo de modelo no reconocido. Por favor, elija entre 'bert', 'roberta' o 'electra'."
                )

        elif opcion == "2":
            print("\nCargando modelo existente...")
            tipo_modelo = (
                input("Tipo de modelo a cargar (bert/roberta/electra): ")
                .strip()
                .lower()
            )
            ruta_modelo = input(
                "Ruta del modelo (opcional, presiona Enter para usar ruta por defecto): "
            ).strip()

            if tipo_modelo in ["bert", "roberta", "electra"]:
                detector = DetectorHateSpeech(tipo_modelo)
                if detector.cargarModelo(ruta_modelo if ruta_modelo else None):
                    print("Modelo cargado exitosamente!")
                else:
                    print("Error al cargar el modelo.")
                    detector = None
            else:
                print("Tipo de modelo no válido.")

        elif opcion == "3":
            if detector is None:
                print("Por favor, entrena o carga un modelo primero.")
                continue

            print("\nRealizando inferencia...")
            texto = input("Ingrese el texto a evaluar: ").strip()
            logger.info(f"El texto ingresado es : {texto}")

            if texto:
                resultados = detector.realizaInferencia([texto])
                if resultados:
                    resultado = resultados[0]
                    print(f"\nTexto: {resultado['texto']}")
                    print(f"resultados :\n")

                    print(f"Etiqueta predicha: {resultado['prediccion']}")
                    print(f"Confianza: {resultado['confianza']:.2%}")
                    print("Probabilidades por clase:")
                    for clase, prob in resultado["probabilidades"].items():
                        print(f"  {clase}: {prob:.2%}")

        elif opcion == "4":
            if detector is None:
                print("No hay modelo cargado.")
            else:
                print(f"\nInformación del modelo:")
                print(f"   Tipo: {detector.tipoLLM}")
                print(f"   Entrenado: {'Sí' if detector.modeloEntrenado else 'No'}")
                if detector.codificadorEtiquetas:
                    clases = list(detector.codificadorEtiquetas.classes_)
                    # Mostrar las clases con su significado
                    clases_descriptivas = []
                    for clase in clases:
                        if str(clase) == "0":
                            clases_descriptivas.append(f"{clase} (No odio)")
                        elif str(clase) == "1":
                            clases_descriptivas.append(f"{clase} (Discurso de odio)")
                        else:
                            clases_descriptivas.append(str(clase))
                    print(f"   Clases: {clases_descriptivas}")
                print(f"   Ruta guardado: {detector.rutaModeloGuardado}")

        elif opcion == "5":
            print("Hasta luego!")
            break

        else:
            print("Opción no válida. Por favor selecciona 1-5.")


if __name__ == "__main__":
    menu()
    logger.info("Programa finalizado por el usuario.")
    print("Programa finalizado. Revisa el archivo app.log para más detalles.")
