from vectorization import representacion_vectorial_completa
import sys

if __name__ == "__main__":
    print("Generando archivos de vectorización para búsqueda de documentos similares...")
    
    # Verificar si se proporcionó un argumento para el tipo de corpus
    corpus_type = "both"  # Valor predeterminado
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ["arxiv", "pubmed", "both"]:
            corpus_type = sys.argv[1]
        else:
            print(f"Tipo de corpus no válido: {sys.argv[1]}")
            print("Opciones válidas: arxiv, pubmed, both")
            sys.exit(1)
    
    print(f"Procesando corpus de tipo: {corpus_type}")
    representacion_vectorial_completa(corpus_type)
    print("\nTerminado. Ahora puede ejecutar search_similar_documents.py")
