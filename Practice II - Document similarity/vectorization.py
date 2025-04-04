import os
import pickle
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd
import sys

def cargar_corpus(archivo_csv, num_articulos=None):
    """
    Carga el corpus desde un archivo CSV normalizado.
    
    Args:
        archivo_csv: Ruta del archivo CSV
        num_articulos: Número de artículos a procesar (None = todos)
    
    Returns:
        Diccionario con DOIs, títulos y abstracts tokenizados
    """
    print(f"Cargando corpus desde {archivo_csv}...")
    
    corpus = {
        "DOI": [],
        "Titulo_Tokens": [],
        "Abstract_Tokens": []
    }
    
    try:
        with open(archivo_csv, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            contador = 0
            
            for row in reader:
                corpus["DOI"].append(row["DOI"])
                corpus["Titulo_Tokens"].append(row["Titulo_Tokens"])
                corpus["Abstract_Tokens"].append(row["Abstract_Tokens"])
                
                contador += 1
                if num_articulos is not None and contador >= num_articulos:
                    break
                    
        print(f"Se cargaron {len(corpus['DOI'])} artículos correctamente.")
        return corpus
    except Exception as e:
        print(f"Error al cargar el corpus: {str(e)}")
        return None

def guardar_vectorizacion(vectorizador, matriz, ruta_archivo):
    """
    Guarda un vectorizador y su matriz en un archivo pickle.
    
    Args:
        vectorizador: Instancia del vectorizador
        matriz: Matriz de características
        ruta_archivo: Ruta donde guardar el archivo
    """
    try:
        with open(ruta_archivo, 'wb') as archivo:
            pickle.dump((vectorizador, matriz), archivo)
        print(f"Vectorización guardada en: {ruta_archivo}")
        return True
    except Exception as e:
        print(f"Error al guardar la vectorización: {str(e)}")
        return False

def vectorizar_texto(textos, tipo_vectorizacion, ngram_range):
    """
    Vectoriza una lista de textos según el tipo de vectorización especificado.
    
    Args:
        textos: Lista de textos a vectorizar
        tipo_vectorizacion: 'frequency', 'binary' o 'tfidf'
        ngram_range: Rango de n-gramas a extraer
    
    Returns:
        Tupla (vectorizador, matriz)
    """
    if tipo_vectorizacion == 'frequency':
        vectorizador = CountVectorizer(ngram_range=ngram_range, token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?')
    elif tipo_vectorizacion == 'binary':
        vectorizador = CountVectorizer(binary=True, ngram_range=ngram_range, token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?')
    elif tipo_vectorizacion == 'tfidf':
        vectorizador = TfidfVectorizer(ngram_range=ngram_range, token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?')
    else:
        raise ValueError("Tipo de vectorización no válido. Debe ser 'frequency', 'binary' o 'tfidf'.")
    
    matriz = vectorizador.fit_transform(textos)
    return vectorizador, matriz

def imprimir_estadisticas_vectorizacion(vectorizador, matriz, tipo, campo, max_elementos=100):
    """
    Imprime estadísticas de la vectorización realizada.
    
    Args:
        vectorizador: Instancia del vectorizador
        matriz: Matriz de características
        tipo: Tipo de vectorización
        campo: Campo vectorizado ('Titulo' o 'Abstract')
        max_elementos: Número máximo de elementos a mostrar
    """
    print(f"\n--- Estadísticas de vectorización {tipo} para {campo} ---")
    print(f"Dimensiones de la matriz: {matriz.shape}")
    print(f"Número de características: {len(vectorizador.get_feature_names_out())}")
    print(f"Número de valores no cero: {matriz.nnz}")
    print(f"Densidad de la matriz: {matriz.nnz / (matriz.shape[0] * matriz.shape[1]):.6f}")
    
    print(f"\nPrimeras {min(max_elementos, len(vectorizador.get_feature_names_out()))} características:")
    for i, feature in enumerate(vectorizador.get_feature_names_out()[:max_elementos]):
        print(f"  {i+1}. {feature}")
    
    print(f"\nPrimeras {min(max_elementos, matriz.shape[0])} filas (documentos) de la matriz:")
    matriz_densa = matriz.toarray()
    for i in range(min(max_elementos, matriz.shape[0])):
        # Mostrar solo los primeros elementos por fila para no saturar la pantalla
        valores_fila = matriz_densa[i][:10]
        print(f"  Doc {i+1}: {valores_fila}")

def procesar_y_guardar(corpus, campo, tipo_vectorizacion, dir_salida, ngram_range):
    """
    Procesa y guarda la vectorización de un campo específico.
    
    Args:
        corpus: Diccionario con los datos del corpus
        campo: Campo a vectorizar ('Titulo_Tokens' o 'Abstract_Tokens')
        tipo_vectorizacion: Tipo de vectorización a realizar
        dir_salida: Directorio de salida
        ngram_range: Rango de n-gramas
    
    Returns:
        Tupla (vectorizador, matriz)
    """
    print(f"\nVectorizando {campo} usando {tipo_vectorizacion}...")
    vectorizador, matriz = vectorizar_texto(corpus[campo], tipo_vectorizacion, ngram_range)
    
    # Guardar la vectorización
    nombre_archivo = f"{campo.lower()}_{tipo_vectorizacion}_{ngram_range[0]}_{ngram_range[1]}.pkl"
    ruta_completa = os.path.join(dir_salida, nombre_archivo)
    guardar_vectorizacion(vectorizador, matriz, ruta_completa)
    
    # Imprimir estadísticas
    imprimir_estadisticas_vectorizacion(vectorizador, matriz, tipo_vectorizacion, campo)
    
    return vectorizador, matriz, matriz.toarray()

def representacion_vectorial_completa(corpus_type="arxiv"):
    """
    Ejecuta todas las representaciones vectoriales y guarda los
    resultados en arreglos separados para unigramas y bigramas.
    
    Args:
        corpus_type: Tipo de corpus a procesar ("arxiv", "pubmed", o "both")
    """
    print("\n===== REPRESENTACIÓN VECTORIAL COMPLETA DE TEXTOS =====")
    
    # Crear directorio para almacenar los archivos de vectorización
    dir_salida = os.path.join(os.getcwd(), "vectorizaciones")
    if not os.path.exists(dir_salida):
        os.makedirs(dir_salida)
    
    # Determinar qué corpus procesar
    corpus_files = []
    if corpus_type == "arxiv" or corpus_type == "both":
        corpus_files.append(("arxiv", "arxiv_normalized_corpus.csv"))
    if corpus_type == "pubmed" or corpus_type == "both":
        corpus_files.append(("pubmed", "pubmed_normalized_corpus.csv"))
    
    # Configuraciones para procesar
    tipos_vectorizacion = ["frequency", "binary", "tfidf"]
    campos = ["Titulo_Tokens", "Abstract_Tokens"]
    ngram_ranges = [(1, 1), (2, 2)]  # Unigramas, Bigramas
    
    # Procesar cada corpus seleccionado
    for corpus_prefix, archivo_normalizado in corpus_files:
        print(f"\nProcesando corpus: {archivo_normalizado}")
        corpus = cargar_corpus(archivo_normalizado, None)  # None = todos los artículos
        
        if corpus is None:
            print(f"No se pudo cargar el corpus {archivo_normalizado}.")
            continue
        
        print(f"\nProcesando todos los artículos del corpus ({len(corpus['DOI'])} artículos)...")
        
        # Diccionarios para almacenar los resultados
        resultados = {}
        
        # Procesar todas las combinaciones
        for campo in campos:
            resultados[campo] = {}
            
            for tipo in tipos_vectorizacion:
                resultados[campo][tipo] = {}
                
                for ngram_range in ngram_ranges:
                    print(f"\n{'='*50}")
                    print(f"Procesando {campo} con {tipo} y n-gramas {ngram_range} para {corpus_prefix}")
                    print(f"{'='*50}")
                    
                    # Nombre del archivo con prefijo del corpus
                    nombre_archivo = f"{corpus_prefix}_{campo.lower()}_{tipo}_{ngram_range[0]}_{ngram_range[1]}.pkl"
                    ruta_completa = os.path.join(dir_salida, nombre_archivo)
                    
                    # Vectorizar
                    vectorizador, matriz = vectorizar_texto(corpus[campo], tipo, ngram_range)
                    
                    # Guardar la vectorización
                    guardar_vectorizacion(vectorizador, matriz, ruta_completa)
                    
                    # Imprimir estadísticas
                    imprimir_estadisticas_vectorizacion(vectorizador, matriz, tipo, campo)
                    
                    # Guardar el resultado en el diccionario
                    ngram_key = f"ngram_{ngram_range[0]}_{ngram_range[1]}"
                    resultados[campo][tipo][ngram_key] = {
                        "vectorizador": vectorizador,
                        "matriz": matriz,
                        "matriz_densa": matriz.toarray()
                    }
                    
                    print(f"\nMatriz {tipo} para {campo} con n-gramas {ngram_range} en {corpus_prefix}:")
                    print(f"Forma: {matriz.shape}")
                    print("Primeras 3 filas (truncadas) de la matriz:")
                    matriz_densa = matriz.toarray()
                    for i in range(min(3, matriz_densa.shape[0])):
                        print(f"  Doc {i+1}: {matriz_densa[i][:5]}...")
    
    print("\n¡Proceso de vectorización completa finalizado con éxito!")
    print(f"Todos los archivos se han guardado en el directorio: {dir_salida}")
    
    return resultados

if __name__ == "__main__":
    # Determinar qué corpus procesar a partir de argumentos de línea de comandos
    if len(sys.argv) > 1 and sys.argv[1] in ["arxiv", "pubmed", "both"]:
        corpus_type = sys.argv[1]
    else:
        corpus_type = "both"  # Por defecto, procesar ambos corpus
    
    print(f"Procesando corpus de tipo: {corpus_type}")
    representacion_vectorial_completa(corpus_type)

# En el archivo vectorization.py, los archivos `.pkl` se guardan utilizando la función `guardar_vectorizacion`. 
# Aquí está el proceso explicado y comentado:
#   - Opción 1: Solo unigramas - `ngram_range = (1, 1)`
#   - Opción 2: Solo bigramas - `ngram_range = (2, 2)`
#   - Opción 3: Unigramas y bigramas (por defecto) - `ngram_range = (1, 2)`
### Proceso de Guardado de Archivos `.pkl`

### Resumen del Proceso
# 1. Se vectorizan los textos del corpus.
# 2. Se guarda el vectorizador y la matriz en un archivo `.pkl` en el directorio vectorizaciones.
# 3. El nombre del archivo incluye información sobre el campo, tipo de vectorización, y rango de n-gramas.
# 4. Se imprimen estadísticas para verificar la calidad de la vectorización.