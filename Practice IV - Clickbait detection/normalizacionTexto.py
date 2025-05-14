import spacy
import csv

class NormalizadorTexto():

    def __init__(self, archivoCSVTweets):
        self.archivoCSVTweets = archivoCSVTweets

    def cargarModelo(self):
        try:
            return spacy.load("es_core_news_sm")
        except:
            print("Modelo 'es_core_news_sm' no encontrado. Ejecuta:")
            print("python -m spacy download es_core_news_sm")
            exit()

    def tokenizadorTexto(self, pln, texto):
        contenido = pln(texto)
        tokens = [(token.text, token.pos_, token.tag_, token.lemma_) for token in contenido if not token.is_space]
        tokens = self.eliminarPalabrasVaciasLemmatizacion(tokens)
        return tokens
    
    def eliminarPalabrasVaciasLemmatizacion(self, tokens):
        categoriasExcluir = ["DET", "ADP", "CCONJ", "SCONJ", "PRON"]
        tokensFiltrados = [
            token[3] for token in tokens 
            if token[1] not in categoriasExcluir and token[3].isalpha()]
        return tokensFiltrados
    
    def procesarArchivo(self, nombreArchivoSalida):
        pln = self.cargarModelo()
        nombreArchivoEntrada = self.archivoCSVTweets

        tokensPreProcesados = []
        with open(nombreArchivoEntrada, newline='', encoding="utf-8") as archivoSinTokenizar:
            leerArchivo = csv.DictReader(archivoSinTokenizar, delimiter=",")

            for fila in leerArchivo:
                teaserTweet = fila["Teaser Text"]
                teaserTokens = self.tokenizadorTexto(pln, teaserTweet)
                teaserTokensLimpios = " ".join(teaserTokens)
                registro = {
                    "Tweet_ID": fila["Tweet ID"],
                    "Tweet_Date": fila["Tweet Date"],
                    "Media_Name": fila["Media Name"],
                    "Media_Origin": fila["Media Origin"],
                    "Teaser_Tokens": teaserTokensLimpios}
                if "Tag Value" in fila:
                    registro["Tag_Value"] = fila["Tag Value"]
                tokensPreProcesados.append(registro)

                if "Tag Value" in fila:
                    campos = ["Tweet_ID", "Tweet_Date", "Media_Name", "Media_Origin", "Teaser_Tokens", "Tag_Value"]
                else:
                    campos = ["Tweet_ID", "Tweet_Date", "Media_Name", "Media_Origin", "Teaser_Tokens"]

        with open(nombreArchivoSalida, mode='w', newline='', encoding="utf-8") as archivoTokenizado:
            escritorCSV = csv.DictWriter(archivoTokenizado, fieldnames=campos)
            escritorCSV.writeheader()
            escritorCSV.writerows(tokensPreProcesados)
        print(f"Archivo procesado y guardado como {nombreArchivoSalida}")
        return tokensPreProcesados
    
normalizarTexto = NormalizadorTexto("./corpus/TA1C_dataset_detection_train.csv")
normalizarTexto.procesarArchivo("TA1C_dataset_Normalize_detection_train.csv")

normalizarTextodev = NormalizadorTexto("./corpus/TA1C_dataset_detection_dev.csv")
normalizarTextodev.procesarArchivo("TA1C_dataset_Normalize_detection_dev.csv")