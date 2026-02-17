import spacy
import csv
import logging
import re

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log',  # Omitir para solo consola
    filemode='a'         # 'w' para sobrescribir, 'a' para añadir
)

logger = logging.getLogger("normalizacionTexto")

class NormalizadorDiscursoOdio:

    def __init__(self, archivoDataset):
        self.archivoDataset = archivoDataset
        self.pln = self.cargarModelo()
        self.categoriasExcluir = ["DET", "ADP", "CCONJ", "SCONJ", "PRON"]
    
    def cargarModelo(self):
        """Carga el modelo de spaCy para español"""
        try:
            return spacy.load("es_core_news_sm")
        except:
            logger.error("Error al cargar el modelo de spaCy")
            print("Ejecuta: python -m spacy download es_core_news_sm")
            exit()

    def tokenizar(self, texto):
        """Realiza solo la tokenización del texto"""
        contenido = self.pln(texto)
        tokens = [(token.text, token.pos_, token.tag_, token.lemma_) for token in contenido if not token.is_space]
        return tokens
        
    def eliminarPalabrasVacias(self, tokens):
        """
        Elimina las palabras vacías basándose en las categorías gramaticales.
        Recibe una lista de tuplas (token, pos, tag, lemma) y retorna solo los tokens cuyo POS no está en categoriasExcluir.
        """
        tokensFiltrados = [
            token[0] for token in tokens 
            if token[1] not in self.categoriasExcluir and token[0].isalpha()]
        return tokensFiltrados

    def lematizar(self, tokens):
        """
        Devuelve solo los lemas de los tokens, omitiendo aquellos que no sean alfabéticos.
        Recibe una lista de tuplas (token, pos, tag, lemma) y retorna una lista de lemas.
        """
        tokensFiltrados = [
            token[3] for token in tokens if token[3].isalpha()]
        
        return tokensFiltrados

    def lematizarSinStopwords(self, tokens):
        """
        Lematiza y elimina palabras vacías según la categoría gramatical.
        Recibe una lista de tuplas (token, pos, tag, lemma) y retorna una lista de lemas.
        """
        return [token[3] for token in tokens if token[1] not in self.categoriasExcluir and token[3].isalpha()]
        
    def procesamientoTipoNormalizacion(self, texto, modo = "completo"):

        if modo == "tokenizacion":
            tokensLimpios = self.tokenizar(texto)
            return [token[0] for token in tokensLimpios]
        elif modo == "stopwords":
            tokensLimpios = self.tokenizar(texto)
            return self.eliminarPalabrasVacias(tokensLimpios)
        elif modo == "lematizacion":
            tokensLimpios = self.tokenizar(texto)
            return self.lematizar(tokensLimpios)
        elif modo == "stopwords_lematizacion":
            tokensLimpios = self.tokenizar(texto)
            return self.lematizarSinStopwords(tokensLimpios)

        elif modo == "tokenizacion_stopwords":
            tokensLimpios = self.tokenizar(texto)
            return self.eliminarPalabrasVacias(tokensLimpios)
        elif modo == "tokenizacion_lematizacion":
            tokensLimpios = self.tokenizar(texto)
            return self.lematizar(tokensLimpios)
        elif modo == "completo":
            tokensLimpios = self.tokenizar(texto)
            tokensFiltrados = [
                token[3] for token in tokensLimpios 
                if token[1] not in self.categoriasExcluir and token[3].isalpha()
            ]
            return tokensFiltrados

    def procesarArchivoTexto(self, nombreArchivoSalida, modoLimpieza):
        nombreArchivoEntrada = self.archivoDataset
        tokensPreProcesados = []
        with open(nombreArchivoEntrada, newline='', encoding="utf-8") as archivoSinTokenizar:
            leerArchivo = csv.DictReader(archivoSinTokenizar, delimiter=",")

            for fila in leerArchivo:
                textoHateSpeech = fila["text"]

                resultadoLimpieza = self.procesamientoTipoNormalizacion(textoHateSpeech, modoLimpieza)
                resultadoTextoLimpio = " ".join(resultadoLimpieza) if isinstance(resultadoLimpieza, list) else resultadoLimpieza

                registro = {
                    "text": resultadoTextoLimpio,
                    "hate_speech": fila["label"]
                }
                tokensPreProcesados.append(registro)
        campos = ["text", "hate_speech"]
        with open(nombreArchivoSalida, 'w', newline='', encoding="utf-8") as archivoTokenizado:
            escritorCSV = csv.DictWriter(archivoTokenizado, fieldnames=campos, delimiter=",")
            escritorCSV.writeheader()
            escritorCSV.writerows(tokensPreProcesados)
        print(f"Archivo procesado y guardado en: {nombreArchivoSalida}")
        return tokensPreProcesados
    
normalizadorDiscurso = NormalizadorDiscursoOdio("./Data/hascosva_2022.csv")

# 1. Solo Tokenizacion
normalizadorDiscurso.procesarArchivoTexto("./Data/normalizado/HateSpeech_Tokenizado.csv", modoLimpieza="tokenizacion")

# 2. Solo eliminacion de stopwords
normalizadorDiscurso.procesarArchivoTexto("./Data/normalizado/HateSpeech_Stopwords.csv", modoLimpieza="stopwords")

# 3. Solo lematizacion
normalizadorDiscurso.procesarArchivoTexto("./Data/normalizado/HateSpeech_Lematizado.csv", modoLimpieza="lematizacion")

# 4. Lematizacion y stopwords
normalizadorDiscurso.procesarArchivoTexto("./Data/normalizado/HateSpeech_Lematizado_Stopwords.csv", modoLimpieza="stopwords_lematizacion")

# 5. Tokenizacion y stopwords
normalizadorDiscurso.procesarArchivoTexto("./Data/normalizado/HateSpeech_Tokenizado_Stopwords.csv", modoLimpieza="tokenizacion_stopwords")

# 6. Tokenizacion y lematizacion
normalizadorDiscurso.procesarArchivoTexto("./Data/normalizado/HateSpeech_Tokenizado_Lematizado.csv", modoLimpieza="tokenizacion_lematizacion")

# 7. Tokenizacion, stopwords y lematizacion
normalizadorDiscurso.procesarArchivoTexto("./Data/normalizado/HateSpeech_Tokenizado_Stopwords_Lematizado.csv", modoLimpieza="completo")

