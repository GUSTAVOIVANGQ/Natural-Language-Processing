from asyncio import sleep
import os
from webScraping import WebScraping
from tokenizer import cargar_modelo, tokenizar_texto, procesar_archivo, menu

def mostrarMenu():

    print("\n\t\tRecoleccion de datos de articulos en repositorios....")
    print("\t1 - ArXiv")
    print("\t2 - PubMed")
    print("\t3 - Salir")

    return int(input("Selecciona el repositorio: "))

def numeroArticulos():
    
    print("\nIndica el numero de articulos a obtener...")

    return int(input("Numero de articulos: \t"))

def obtenerArticulosArXiv():

    print("\nQuieres obtener articulos de Vision&Patrones, Computacion&Lenguaje o ambos?")
    print("\t1 - Vision&Patrones")
    print("\t2 - Computacion&Lenguaje")
    print("\t3 - Ambos")
    opcion = int(input("Selecciona una opcion: "))

    numArticulos = numeroArticulos()
    instanciaArticulosArXiv = WebScraping("arxiv", numArticulos)
    articulos = []

    if opcion == 1:
        print("Vision&Patrones")
        articulos = instanciaArticulosArXiv.obtenerArticulos("Vision&Patrones")

    elif opcion == 2:
        print("Computacion&Lenguaje")
        articulos = instanciaArticulosArXiv.obtenerArticulos("Computacion&Lenguaje")

    elif opcion == 3:
        print("\nObteniendo articulos de Vision&Patrones y Computacion&Lenguaje...")
        articulos = instanciaArticulosArXiv.obtenerArticulos("Vision&Patrones")
        articulos.extend(instanciaArticulosArXiv.obtenerArticulos("Computacion&Lenguaje"))

    else:
        print("Opcion no valida")
        return None, None

    return instanciaArticulosArXiv, articulos

def obtenerArticulosPubMed():
    
    print("Articulos de PubMed")
    numArticulos = numeroArticulos()
    obtenerArticulosPubMed = WebScraping("pubmed", numArticulos)
    articulos = obtenerArticulosPubMed.obtenerArticulos()

    return obtenerArticulosPubMed, articulos

def generarArchivos(instanciaWebScraping, articulos):

    if not articulos:
        print("No hay articulos para generar un archivo")
        return 
    
    print("\nQuieres generar el archivo de los articulos obtenidos?")
    print("\t1 - Si")
    print("\t2 - No")
    opcion = int(input("Selecciona una opcion: "))

    if opcion == 1:
        print("Generando archivos...\n\n")
        instanciaWebScraping.generaArchivo(articulos,"ArticulosTestEscuela.csv")

    elif opcion == 2:
        print("Saliendo...")

    else:
        print("Opcion no valida")

def mainWebScraping():
    while True:
        opcion = mostrarMenu()
        
        try:
            if opcion == 1:
                instancia, articulos = obtenerArticulosArXiv()
            elif opcion == 2:
                instancia, articulos = obtenerArticulosPubMed()
            elif opcion == 3:
                print("Saliendo...")
                break
            
            if instancia and articulos:
                generarArchivos(instancia, articulos)

            os.system("clear")

        except ValueError:
            print("Error: Debe ingresar un numero valido")

def normalizacionTexto():
    menu()

def similitudDocumentos():
    from document_similarity import buscar_documentos_similares
    buscar_documentos_similares()

def mostrarFasePractica():
    while True:
        print("\n\t\tPractica similitud en documentos")
        print("\t1 - Web Scraping")
        print("\t2 - Normalizacion de texto")
        print("\t3 - Similitud de documentos")
        print("\t4 - Salir")

        opcionPrincipal = int(input("Selecciona una opcion: "))
        try:
            if opcionPrincipal == 1:
                mainWebScraping()
            elif opcionPrincipal == 2:
                normalizacionTexto()
            elif opcionPrincipal == 3:
                similitudDocumentos()
            else:
                print("Opcion no valida")
                break
            
            sleep(10)
            os.system("clear")
        except ValueError:
            print("Error: Debe ingresar un numero valido")
    

mostrarFasePractica()