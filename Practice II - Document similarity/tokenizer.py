from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import csv
import spacy

def cargar_modelo():
    try:
        return spacy.load("en_core_web_sm")
    except:
        print("Modelo 'en_core_web_sm' no encontrado. Ejecuta:")
        print("python -m spacy download en_core_web_sm")
        exit()

def tokenizar_texto(nlp, texto):
    doc = nlp(texto)
    tokens = [(token.text, token.pos_, token.tag_, token.lemma_) for token in doc if not token.is_punct and not token.is_space]
    return tokens

def palabrasVacias(tokens):
    # Se eliminan palabras vacias por categoria gramatial artículos, preposiciones, conjunciones y pronombres.
    categoriasExcluir = ["DT", "IN", "CC", "PRP", "PRP$"]
    tokensFiltrados = [token[3] for token in tokens if token[2] not in categoriasExcluir]   # Se toma el lemma de la palabra
    return tokensFiltrados

def procesar_archivo(nombre_entrada, nombre_salida, es_arxiv=True):
    nlp = cargar_modelo()
    
    filas_procesadas = []
    tokensPreRepresentacionVectorial = []

    with open(nombre_entrada, newline='', encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo, delimiter=",")
        

        for fila in lector:
            titulo = fila["Titulo"]
            abstract = fila["Abstract"]

            titulo_tokens = tokenizar_texto(nlp, titulo)
            abstract_tokens = tokenizar_texto(nlp, abstract)

            tituloTokensLimpios = palabrasVacias(titulo_tokens)
            abstractTokensLimpios = palabrasVacias(abstract_tokens)

            tituloTokensLimpios = " ".join(tituloTokensLimpios)
            abstractTokensLimpios = " ".join(abstractTokensLimpios)

            nueva_fila = {
                "DOI": fila["DOI"],
                "Titulo_Tokens": " ".join(tituloTokensLimpios),
                "Abstract_Tokens": " ".join(abstractTokensLimpios),
                "Autores": fila["Autores"],
                "Fecha": fila["Fecha"]
            }

            if es_arxiv:
                nueva_fila["Seccion"] = fila["Seccion"]
            else:
                nueva_fila["Journal"] = fila["Journal"]

            filas_procesadas.append(nueva_fila)

    campos = list(filas_procesadas[0].keys())
    with open(nombre_salida, "w", newline='', encoding="utf-8") as archivo_salida:
        escritor = csv.DictWriter(archivo_salida, fieldnames=campos, delimiter=",")
        escritor.writeheader()
        for fila in filas_procesadas:
            escritor.writerow(fila)

    print(f"\nArchivo generado exitosamente: {nombre_salida}")

    return tokensPreRepresentacionVectorial

def vectorizarCorpus(corpus, tipoVectorizacion, extraeCaracteristicas):

    corpus = [texto["Tokens_Titulo"] + texto["Tokens_Abstract"] for texto in corpus]

    if tipoVectorizacion == 1:
        vectorizadorTitulo = CountVectorizer()
        vectorizadorAbstract = CountVectorizer()
    elif tipoVectorizacion == 2:
        vectorizadorTitulo = CountVectorizer(binary=True)
        vectorizadorAbstract = CountVectorizer(binary=True)
    elif tipoVectorizacion == 3:
        vectorizadorTitulo = TfidfVectorizer()
        vectorizadorAbstract = TfidfVectorizer()

    X_Titulo = []
    X_Abstract = []

    return print(X.toarray())
    

def menu():
    print("\nSeleccione el corpus que desea tokenizar:")
    print("1 - ArXiv")
    print("2 - PubMed")
    opcionToken = input("Ingrese el número de su opción: ")

    if opcionToken == "1":
        print(procesar_archivo("./archivos/ArticulosArXiv.csv", "arxiv_normalized_corpus.csv", es_arxiv=True))
        
    elif opcionToken == "2":
        procesar_archivo("pubmed_raw_corpus.csv", "pubmed_normalized_corpus.csv", es_arxiv=False)
    else:
        print("Opción inválida. Intente de nuevo.")
        menu()

    #print("\n\nSeleccione la representación vectorial que desea utilizar en el corpus:")
    #print("1 - Por frecuencia")
    #print("2 - Binarizada")
    #print("3 - TF-IDF")
    #opcionVector = input("Ingrese el número de su opción: ")
    #if opcionVector == "1":
    #    print("Vectorización por frecuencia")
    #elif opcionVector == "2":
    #    print("Vectorización binarizada")
    #elif opcionVector == "3":
    #    print("Vectorización TF-IDF")
    #else:
    #    print("Opción inválida. Intente de nuevo.")
    #    menu()

if __name__ == "__main__":
    menu()
