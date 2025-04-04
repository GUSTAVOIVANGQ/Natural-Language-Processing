import os
import pickle
import numpy as np
import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import re
import sys

# Add imports from existing modules
from tokenizer import cargar_modelo, tokenizar_texto, palabrasVacias

def cargar_vectorizacion(tipo_vectorizacion, campo, ngram_range, corpus_type):
    """
    Carga una vectorización previamente guardada.
    
    Args:
        tipo_vectorizacion: Tipo de vectorización ('frequency', 'binary', 'tfidf')
        campo: Campo a usar ('Titulo_Tokens' o 'Abstract_Tokens')
        ngram_range: Rango de n-gramas, ejemplo: (1, 1) para unigramas
        corpus_type: Tipo de corpus ('arxiv' o 'pubmed')
    
    Returns:
        vectorizador, matriz, corpus_metadata
    """
    dir_vectorizaciones = os.path.join(os.getcwd(), "vectorizaciones")
    
    # Crear nombre del archivo con prefijo del corpus
    nombre_archivo = f"{corpus_type}_{campo.lower()}_{tipo_vectorizacion}_{ngram_range[0]}_{ngram_range[1]}.pkl"
    ruta_completa = os.path.join(dir_vectorizaciones, nombre_archivo)
    
    # Verificar que existe el archivo
    if not os.path.exists(ruta_completa):
        # Depuración: listar archivos disponibles en el directorio de vectorizaciones
        print(f"No se encontró {nombre_archivo}. Archivos disponibles en {dir_vectorizaciones}:")
        if os.path.exists(dir_vectorizaciones):
            archivos = [f for f in os.listdir(dir_vectorizaciones) if f.endswith('.pkl')]
            for archivo in archivos:
                print(f"  - {archivo}")
        else:
            print(f"  El directorio {dir_vectorizaciones} no existe.")
            
        # Intentar con el formato alternativo (sin prefijo de corpus)
        nombre_archivo_alt = f"{campo.lower()}_{tipo_vectorizacion}_{ngram_range[0]}_{ngram_range[1]}.pkl"
        ruta_completa_alt = os.path.join(dir_vectorizaciones, nombre_archivo_alt)
        
        if os.path.exists(ruta_completa_alt):
            print(f"Usando archivo alternativo: {nombre_archivo_alt}")
            ruta_completa = ruta_completa_alt
        else:
            print(f"Error: No se encontró el archivo de vectorización {nombre_archivo}")
            print(f"Tampoco se encontró formato alternativo: {nombre_archivo_alt}")
            print(f"Asegúrese de haber ejecutado vectorization.py primero con 'python vectorization.py both' para generar los archivos necesarios.")
            return None, None, None
    
    try:
        # Cargar el archivo
        with open(ruta_completa, 'rb') as f:
            datos = pickle.load(f)
            
            # Manejar formatos tanto de diccionario como de tupla
            if isinstance(datos, tuple) and len(datos) == 2:
                # Si es una tupla (formato de vectorization.py)
                vectorizador, matriz = datos
            elif isinstance(datos, dict) and 'vectorizador' in datos and 'matriz' in datos:
                # Si es un diccionario
                vectorizador = datos['vectorizador']
                matriz = datos['matriz']
            else:
                print(f"Formato de archivo de vectorización no reconocido en: {ruta_completa}")
                return None, None, None
        
        # Cargar el corpus correcto basado en el tipo de corpus
        archivo_corpus = f"{corpus_type}_normalized_corpus.csv"
        if not os.path.exists(archivo_corpus):
            print(f"Advertencia: No se encontró el archivo de corpus normalizado: {archivo_corpus}")
            # Listar archivos CSV disponibles para ayudar al diagnóstico
            csv_files = [f for f in os.listdir() if f.endswith('.csv')]
            print(f"Archivos CSV disponibles: {csv_files}")
            
            # Buscar otros archivos de corpus específicos para el tipo seleccionado
            archivos_corpus = [f for f in os.listdir() if f.startswith(f"{corpus_type}_") and 
                              (f.endswith('_normalized_corpus.csv') or f.endswith('_corpus.csv'))]
            
            # Si no hay archivos específicos, buscar cualquier corpus
            if not archivos_corpus:
                archivos_corpus = [f for f in os.listdir() if 
                                 f.endswith('_normalized_corpus.csv') or f.endswith('_corpus.csv')]
            
            if archivos_corpus:
                archivo_corpus = archivos_corpus[0]
                print(f"Usando corpus alternativo: {archivo_corpus}")
            else:
                print("No se encontró ningún archivo de corpus. Verifique que el archivo existe.")
                return vectorizador, matriz, None
        
        corpus = pd.read_csv(archivo_corpus)
        
        # Depuración: mostrar columnas disponibles
        print(f"Columnas en el corpus {archivo_corpus}: {corpus.columns.tolist()}")
        
        # Map between different naming conventions
        if "TokensTitulo" in corpus.columns and campo == "Titulo_Tokens":
            print("Renombrando TokensTitulo a Titulo_Tokens para compatibilidad")
            corpus = corpus.rename(columns={"TokensTitulo": "Titulo_Tokens"})
            
        if "TokensAbstract" in corpus.columns and campo == "Abstract_Tokens":
            print("Renombrando TokensAbstract a Abstract_Tokens para compatibilidad")
            corpus = corpus.rename(columns={"TokensAbstract": "Abstract_Tokens"})
        
        print(f"Vectorización cargada desde {os.path.basename(ruta_completa)} para corpus {corpus_type}")
        return vectorizador, matriz, corpus
    except Exception as e:
        print(f"Error al cargar vectorización: {str(e)}")
        return None, None, None

def extraer_contenido_bibTeX(texto):
    """
    Extrae el título y abstract de un archivo BibTeX.
    """
    titulo = ""
    abstract = ""
    
    # Buscar título
    titulo_match = re.search(r'title\s*=\s*[{"](.*?)[}"],?', texto, re.DOTALL)
    if titulo_match:
        titulo = titulo_match.group(1).strip()
    
    # Buscar abstract
    abstract_match = re.search(r'abstract\s*=\s*[{"](.*?)[}"],?', texto, re.DOTALL)
    if abstract_match:
        abstract = abstract_match.group(1).strip()
    
    return titulo, abstract

def extraer_contenido_ris(texto):
    """
    Extrae el título y abstract de un archivo RIS.
    """
    titulo = ""
    abstract = ""
    
    # Buscar título (TI o T1)
    titulo_match = re.search(r'(TI|T1)\s*-\s*(.*?)(?:\r?\n)', texto, re.DOTALL)
    if titulo_match:
        titulo = titulo_match.group(2).strip()
    
    # Buscar abstract (AB)
    abstract_match = re.search(r'AB\s*-\s*(.*?)(?:\r?\n\w\w|\Z)', texto, re.DOTALL)
    if abstract_match:
        abstract = abstract_match.group(1).strip()
        # Limpiar múltiples espacios y saltos de línea en el abstract
        abstract = re.sub(r'\s+', ' ', abstract)
    
    return titulo, abstract

def cargar_archivo_consulta():
    """
    Abre un diálogo para seleccionar un archivo BibTeX o RIS.
    """
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal
    
    archivo = filedialog.askopenfilename(
        title="Seleccionar archivo BibTeX o RIS",
        filetypes=[
            ("Archivos BibTeX", "*.bib"),
            ("Archivos RIS", "*.ris"),
            ("Todos los archivos", "*.*")
        ]
    )
    
    if not archivo:
        return None, None, None
    
    # Determinar el tipo de archivo y parsearlo
    extension = os.path.splitext(archivo)[1].lower()
    
    try:
        with open(archivo, 'r', encoding='utf-8') as f:
            contenido = f.read()
        
        if extension == '.bib':
            titulo, abstract = extraer_contenido_bibTeX(contenido)
            tipo_archivo = "BibTeX"
        elif extension == '.ris':
            titulo, abstract = extraer_contenido_ris(contenido)
            tipo_archivo = "RIS"
        else:
            # Intentar detectar el formato basado en el contenido
            if '@article' in contenido or '@inproceedings' in contenido:
                titulo, abstract = extraer_contenido_bibTeX(contenido)
                tipo_archivo = "BibTeX"
            elif 'TY  -' in contenido:
                titulo, abstract = extraer_contenido_ris(contenido)
                tipo_archivo = "RIS"
            else:
                messagebox.showerror("Error", "Formato de archivo no reconocido. Use BibTeX (.bib) o RIS (.ris)")
                return None, None, None
        
        if not titulo and not abstract:
            messagebox.showwarning("Advertencia", "No se pudo extraer título ni abstract del archivo seleccionado.")
            
        return titulo, abstract, tipo_archivo
        
    except Exception as e:
        messagebox.showerror("Error", f"Error al leer el archivo: {str(e)}")
        return None, None, None

def normalizar_texto(texto, nlp):
    """
    Aplica la normalización al texto (tokenización y filtrado de stopwords).
    """
    if not texto:
        return ""
    
    tokens = tokenizar_texto(nlp, texto)
    tokens_filtrados = palabrasVacias(tokens)
    return " ".join(tokens_filtrados)

class SimilarityApp:
    def __init__(self, master=None):
        if master is None:
            self.root = tk.Tk()
        else:
            self.root = master
            
        self.root.title("Búsqueda de Documentos Similares")
        self.root.geometry("650x580")
        
        # Initialize model
        self.nlp = cargar_modelo()
        
        # Initialize document properties
        self.titulo = ""
        self.abstract = ""
        self.tipo_archivo = ""
        
        # Initialize variables with proper master reference
        self.campo_var = tk.StringVar(self.root)
        self.campo_var.set("Titulo_Tokens")
        
        self.feature_var = tk.StringVar(self.root)
        self.feature_var.set("Unigrama")
        
        self.vectorizacion_var = tk.StringVar(self.root)
        self.vectorizacion_var.set("tfidf")
        
        self.corpus_var = tk.StringVar(self.root)
        self.corpus_var.set("both")
        
        # Create UI elements
        self.create_widgets()
        
        # Store search results
        self.resultados = []
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        ttk.Label(main_frame, text="Búsqueda de Documentos Similares", font=('Arial', 14, 'bold')).grid(
            row=0, column=0, columnspan=4, pady=(0, 20))
        
        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="Cargar documento")
        file_frame.grid(row=1, column=0, columnspan=4, sticky='ew', pady=(0, 15))
        
        self.file_label = ttk.Label(file_frame, text="Ningún archivo seleccionado")
        self.file_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(file_frame, text="Seleccionar archivo", command=self.load_file).pack(
            side=tk.RIGHT, padx=5, pady=5)
        
        # Options frame
        options_frame = ttk.LabelFrame(main_frame, text="Opciones de búsqueda")
        options_frame.grid(row=2, column=0, columnspan=4, sticky='ew', pady=(0, 15))
        
        # Campo selection
        ttk.Label(options_frame, text="Contenido a comparar:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        campo_frame = ttk.Frame(options_frame)
        campo_frame.grid(row=0, column=1, sticky='w')
        
        # Create radiobuttons for explicit value assignment to ensure they always work
        ttk.Radiobutton(campo_frame, text="Título", variable=self.campo_var, value="Titulo_Tokens", 
                        command=lambda: self.update_selection_explicit("campo", "Titulo_Tokens")).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(campo_frame, text="Abstract", variable=self.campo_var, value="Abstract_Tokens", 
                        command=lambda: self.update_selection_explicit("campo", "Abstract_Tokens")).pack(side=tk.LEFT, padx=5)
        
        # Feature selection
        ttk.Label(options_frame, text="Características:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        feature_frame = ttk.Frame(options_frame)
        feature_frame.grid(row=1, column=1, sticky='w')
        
        ttk.Radiobutton(feature_frame, text="Unigrama", variable=self.feature_var, value="Unigrama", 
                        command=lambda: self.update_selection_explicit("feature", "Unigrama")).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(feature_frame, text="Bigrama", variable=self.feature_var, value="Bigrama", 
                        command=lambda: self.update_selection_explicit("feature", "Bigrama")).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(feature_frame, text="Trigrama", variable=self.feature_var, value="Trigrama", 
                        command=lambda: self.update_selection_explicit("feature", "Trigrama")).pack(side=tk.LEFT, padx=5)
        
        # Vectorization selection
        ttk.Label(options_frame, text="Vectorización:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        vectorizacion_frame = ttk.Frame(options_frame)
        vectorizacion_frame.grid(row=2, column=1, sticky='w')
        
        ttk.Radiobutton(vectorizacion_frame, text="TF-IDF", variable=self.vectorizacion_var, value="tfidf", 
                        command=lambda: self.update_selection_explicit("vectorizacion", "tfidf")).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(vectorizacion_frame, text="Frecuencia", variable=self.vectorizacion_var, value="frequency", 
                        command=lambda: self.update_selection_explicit("vectorizacion", "frequency")).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(vectorizacion_frame, text="Binaria", variable=self.vectorizacion_var, value="binary", 
                        command=lambda: self.update_selection_explicit("vectorizacion", "binary")).pack(side=tk.LEFT, padx=5)
        
        # Corpus selection
        ttk.Label(options_frame, text="Corpus:").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        corpus_frame = ttk.Frame(options_frame)
        corpus_frame.grid(row=3, column=1, sticky='w')
        
        ttk.Radiobutton(corpus_frame, text="ArXiv", variable=self.corpus_var, value="arxiv", 
                        command=lambda: self.update_selection_explicit("corpus", "arxiv")).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(corpus_frame, text="PubMed", variable=self.corpus_var, value="pubmed", 
                        command=lambda: self.update_selection_explicit("corpus", "pubmed")).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(corpus_frame, text="Ambos", variable=self.corpus_var, value="both", 
                        command=lambda: self.update_selection_explicit("corpus", "both")).pack(side=tk.LEFT, padx=5)
        
        # Current selection display
        self.selection_frame = ttk.LabelFrame(options_frame, text="Selección actual")
        self.selection_frame.grid(row=4, column=0, columnspan=2, sticky='ew', pady=(10, 5), padx=5)
        
        self.selection_label = ttk.Label(self.selection_frame, text=self.get_current_selection_text())
        self.selection_label.pack(padx=5, pady=5)
        
        # Document preview
        preview_frame = ttk.LabelFrame(main_frame, text="Vista previa del documento")
        preview_frame.grid(row=3, column=0, columnspan=4, sticky='nsew', pady=(0, 15))
        
        # Make this frame expand to use available space
        main_frame.rowconfigure(3, weight=1)
        
        # Title display
        ttk.Label(preview_frame, text="Título:").grid(row=0, column=0, sticky='nw', padx=5, pady=5)
        self.titulo_display = tk.Text(preview_frame, height=2, width=50, wrap=tk.WORD)
        self.titulo_display.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        self.titulo_display.config(state=tk.DISABLED)
        
        # Abstract display
        ttk.Label(preview_frame, text="Abstract:").grid(row=1, column=0, sticky='nw', padx=5, pady=5)
        
        abstract_container = ttk.Frame(preview_frame)
        abstract_container.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)
        preview_frame.rowconfigure(1, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        
        self.abstract_display = tk.Text(abstract_container, height=6, width=50, wrap=tk.WORD)
        self.abstract_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        abstract_scrollbar = ttk.Scrollbar(abstract_container, command=self.abstract_display.yview)
        abstract_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.abstract_display.config(yscrollcommand=abstract_scrollbar.set, state=tk.DISABLED)
        
        # Search button
        search_button = ttk.Button(main_frame, text="Buscar Documentos Similares", 
                                  command=self.search_documents)
        search_button.grid(row=4, column=0, columnspan=4, pady=15)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="")
        self.status_label.grid(row=5, column=0, columnspan=4)
    
    def update_selection_explicit(self, var_type, value):
        """Update the selection variables explicitly and refresh display"""
        if var_type == "campo":
            self.campo_var.set(value)
        elif var_type == "feature":
            self.feature_var.set(value)
        elif var_type == "vectorizacion":
            self.vectorizacion_var.set(value)
        elif var_type == "corpus":
            self.corpus_var.set(value)
        
        # Update the selection display
        self.selection_label.config(text=self.get_current_selection_text())
        
        # Print current state for debugging
        print(f"Updated {var_type} to: {value}")
        print(f"Current selections: Campo={self.campo_var.get()}, Feature={self.feature_var.get()}, "
              f"Vectorización={self.vectorizacion_var.get()}, Corpus={self.corpus_var.get()}")
    
    def get_current_selection_text(self):
        """Get a formatted string showing current selections"""
        campo_map = {"Titulo_Tokens": "Título", "Abstract_Tokens": "Abstract"}
        vectorizacion_map = {"tfidf": "TF-IDF", "frequency": "Frecuencia", "binary": "Binaria"}
        corpus_map = {"arxiv": "ArXiv", "pubmed": "PubMed", "both": "Ambos"}
        
        return (f"Campo: {campo_map.get(self.campo_var.get(), self.campo_var.get())} | "
                f"Caract.: {self.feature_var.get()} | "
                f"Vector.: {vectorizacion_map.get(self.vectorizacion_var.get(), self.vectorizacion_var.get())} | "
                f"Corpus: {corpus_map.get(self.corpus_var.get(), self.corpus_var.get())}")

    def update_selection(self):
        """Legacy method - now just updates the selection display"""
        self.selection_label.config(text=self.get_current_selection_text())
        
        # Print selections for debugging
        print(f"Campo: {self.campo_var.get()}")
        print(f"Feature: {self.feature_var.get()}")
        print(f"Vectorización: {self.vectorizacion_var.get()}")
        print(f"Corpus: {self.corpus_var.get()}")
    
    def load_file(self):
        """Load a BibTeX or RIS file"""
        self.titulo, self.abstract, self.tipo_archivo = cargar_archivo_consulta()
        
        if not self.titulo and not self.abstract:
            self.file_label.config(text="No se pudo cargar el archivo o está vacío")
            return
        
        # Update file label
        self.file_label.config(text=f"Archivo {self.tipo_archivo} cargado correctamente")
        
        # Update preview
        self.update_preview()
    
    def update_preview(self):
        """Update the document preview with loaded content"""
        # Update title preview
        self.titulo_display.config(state=tk.NORMAL)
        self.titulo_display.delete(1.0, tk.END)
        self.titulo_display.insert(tk.END, self.titulo if self.titulo else "No disponible")
        self.titulo_display.config(state=tk.DISABLED)
        
        # Update abstract preview
        self.abstract_display.config(state=tk.NORMAL)
        self.abstract_display.delete(1.0, tk.END)
        self.abstract_display.insert(tk.END, self.abstract if self.abstract else "No disponible")
        self.abstract_display.config(state=tk.DISABLED)
    
    def get_ngram_range(self):
        """Convert feature selection to appropriate ngram range"""
        feature = self.feature_var.get()
        if feature == "Unigrama":
            return (1, 1)
        elif feature == "Bigrama":
            return (2, 2)
        elif feature == "Trigrama":
            return (3, 3)
        else:
            return (1, 1)  # Default to unigram
    
    def search_documents(self):
        """Execute search for similar documents using current settings"""
        if not self.titulo and not self.abstract:
            messagebox.showwarning("Advertencia", "Primero debe cargar un archivo.")
            return
        
        # Get selected options - read directly from the StringVars to ensure they're current
        campo = self.campo_var.get()
        vectorizacion = self.vectorizacion_var.get()
        corpus_type = self.corpus_var.get()
        ngram_range = self.get_ngram_range()
        
        # Log the current selections for debugging
        print(f"Searching with: Campo={campo}, Feature={self.feature_var.get()}, "
              f"Vectorización={vectorizacion}, Corpus={corpus_type}, NGram={ngram_range}")
        
        # Show current selections in status label
        self.status_label.config(text=f"Buscando con: {self.get_current_selection_text()}...")
        self.root.update()
        
        # Check if the required field has content
        if campo == "Titulo_Tokens" and not self.titulo:
            messagebox.showwarning("Advertencia", "El título está vacío pero ha seleccionado comparar por título.")
            return
        elif campo == "Abstract_Tokens" and not self.abstract:
            messagebox.showwarning("Advertencia", "El abstract está vacío pero ha seleccionado comparar por abstract.")
            return
        
        # Show status
        self.status_label.config(text="Normalizando texto...")
        self.root.update()
        
        # Get and normalize text based on selected field
        texto_consulta = self.titulo if campo == "Titulo_Tokens" else self.abstract
        texto_normalizado = normalizar_texto(texto_consulta, self.nlp)
        
        if not texto_normalizado:
            messagebox.showwarning("Advertencia", f"Después de la normalización, el texto quedó vacío.")
            self.status_label.config(text="")
            return
        
        # Determine which corpus to process
        if corpus_type == "both":
            corpus_types = ["arxiv", "pubmed"]
        else:
            corpus_types = [corpus_type]
        
        # Clear previous results
        self.resultados = []
        
        # Process each selected corpus
        for current_corpus in corpus_types:
            self.status_label.config(text=f"Procesando corpus {current_corpus}...")
            self.root.update()
            
            # Load vectorization data
            vectorizador, matriz, corpus = cargar_vectorizacion(
                vectorizacion, campo, ngram_range, current_corpus
            )
            
            # Fix the way we check if vectorization was loaded successfully
            if vectorizador is None or matriz is None or corpus is None:
                messagebox.showwarning(
                    "Error de vectorización", 
                    f"No se pudo cargar la vectorización para {current_corpus}.\n"
                    "Verifique que ha ejecutado vectorization.py primero."
                )
                continue
            
            # Transform query into vector space
            query_vector = vectorizador.transform([texto_normalizado])
            
            # Calculate similarities
            similitudes = cosine_similarity(query_vector, matriz)[0]
            
            # Get indices of top results
            num_results = 10 if len(corpus_types) == 1 else 5
            indices_top = similitudes.argsort()[-num_results:][::-1]
            
            # Extract results
            for i, indice in enumerate(indices_top):
                similitud = similitudes[indice]
                if similitud > 0:  # Only include if there's some similarity
                    # Handle different column naming conventions
                    doi = corpus.iloc[indice].get("DOI", "N/A")
                    titulo_doc = corpus.iloc[indice].get("Titulo_Tokens", 
                                corpus.iloc[indice].get("TokensTitulo", "Título no disponible"))
                    abstract_doc = corpus.iloc[indice].get("Abstract_Tokens",
                                  corpus.iloc[indice].get("TokensAbstract", "Abstract no disponible"))
                    
                    self.resultados.append({
                        "posicion": i+1,
                        "doi": doi,
                        "titulo": titulo_doc,
                        "abstract": abstract_doc,
                        "similitud": similitud,
                        "corpus": current_corpus
                    })
        
        self.status_label.config(text="")
        
        # If no results found
        if not self.resultados:
            messagebox.showinfo("Información", "No se encontraron documentos similares.")
            return
        
        # Sort combined results by similarity
        self.resultados.sort(key=lambda x: x["similitud"], reverse=True)
        
        # Update positions after sorting
        for i, doc in enumerate(self.resultados):
            doc["posicion"] = i + 1
        
        # Display results in a new window
        self.display_results()
    
    def display_results(self):
        """Display search results in a new window"""
        results_window = tk.Toplevel(self.root)
        results_window.title("Resultados de búsqueda")
        results_window.geometry("900x600")
        
        # Main frame
        main_frame = ttk.Frame(results_window, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="Documentos similares encontrados", 
                font=('Arial', 14, 'bold')).pack(pady=(0, 20))
        
        # Count results by corpus
        corpus_counts = {"arxiv": 0, "pubmed": 0}
        for doc in self.resultados:
            if doc["corpus"] in corpus_counts:
                corpus_counts[doc["corpus"]] += 1
        
        # Show a summary if we have results from both corpus types
        if corpus_counts["arxiv"] > 0 and corpus_counts["pubmed"] > 0:
            summary = f"Resultados combinados: {corpus_counts['arxiv']} de ArXiv y {corpus_counts['pubmed']} de PubMed"
            ttk.Label(main_frame, text=summary, font=('Arial', 10, 'italic')).pack(pady=(0, 10))
        
        # Create the treeview to display results
        columns = ("pos", "corpus", "doi", "title", "similarity")
        tree = ttk.Treeview(main_frame, columns=columns, show="headings", height=15)
        
        # Configure columns
        tree.heading("pos", text="#")
        tree.heading("corpus", text="Corpus")
        tree.heading("doi", text="DOI")
        tree.heading("title", text="Título")
        tree.heading("similarity", text="Similitud")
        
        tree.column("pos", width=40, anchor=tk.CENTER)
        tree.column("corpus", width=80, anchor=tk.CENTER)
        tree.column("doi", width=150)
        tree.column("title", width=450)
        tree.column("similarity", width=80, anchor=tk.CENTER)
        
        # Insert results
        for doc in self.resultados:
            corpus_name = "ArXiv" if doc["corpus"] == "arxiv" else "PubMed"
            tag = f"corpus_{doc['corpus']}"
            
            tree.insert("", tk.END, values=(
                doc["posicion"],
                corpus_name,
                doc["doi"],
                doc["titulo"][:70] + "..." if len(doc["titulo"]) > 70 else doc["titulo"],
                f"{doc['similitud']:.4f}"
            ), tags=(tag,))
        
        # Apply visual styling to rows based on corpus
        tree.tag_configure("corpus_arxiv", background="#E6F3FF")  # Light blue for ArXiv
        tree.tag_configure("corpus_pubmed", background="#FFF0F5")  # Light pink for PubMed
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack the treeview and scrollbar
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Double-click to view details
        def show_details(event):
            selection = tree.selection()
            if selection:
                item = selection[0]
                pos = int(tree.item(item, "values")[0])
                doc = next((d for d in self.resultados if d["posicion"] == pos), None)
                if doc:
                    self.show_document_details(doc)
        
        tree.bind("<Double-1>", show_details)
    
    def show_document_details(self, doc):
        """Show detailed information about a document"""
        details_window = tk.Toplevel(self.root)
        details_window.title(f"Detalles del documento - {doc['corpus'].upper()}")
        details_window.geometry("700x500")
        
        # Main frame
        details_frame = ttk.Frame(details_window, padding=20)
        details_frame.pack(fill=tk.BOTH, expand=True)
        
        # Document information
        ttk.Label(details_frame, text="Corpus:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=5)
        corpus_name = "ArXiv" if doc["corpus"] == "arxiv" else "PubMed"
        corpus_label = ttk.Label(details_frame, text=corpus_name)
        corpus_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(details_frame, text="DOI:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Label(details_frame, text=doc["doi"], wraplength=600).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(details_frame, text="Título:", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Label(details_frame, text=doc["titulo"], wraplength=600).grid(row=2, column=1, sticky=tk.W)
        
        ttk.Label(details_frame, text="Abstract:", font=('Arial', 10, 'bold')).grid(row=3, column=0, sticky=tk.NW, pady=5)
        
        # Text widget for abstract with scrollbar
        abstract_text = tk.Text(details_frame, wrap=tk.WORD, height=15, width=70)
        abstract_text.grid(row=3, column=1, sticky=tk.W)
        abstract_text.insert(tk.END, doc["abstract"])
        abstract_text.config(state=tk.DISABLED)
        
        abstract_scroll = ttk.Scrollbar(details_frame, command=abstract_text.yview)
        abstract_scroll.grid(row=3, column=2, sticky=tk.NS)
        abstract_text.config(yscrollcommand=abstract_scroll.set)
        
        ttk.Label(details_frame, text="Similitud:", font=('Arial', 10, 'bold')).grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Label(details_frame, text=f"{doc['similitud']:.6f}").grid(row=4, column=1, sticky=tk.W)
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

# Update the main function to use our new class
def buscar_documentos_similares():
    """
    Función principal para búsqueda de documentos similares.
    """
    app = SimilarityApp()
    app.run()

# Ensure the entry point remains consistent
if __name__ == "__main__":
    buscar_documentos_similares()
