# Hate Speech Detection - Copilot Instructions

## Project Overview
Comparative study of hate speech detection using HASCOSVA 2022 Spanish dataset. Two-phase approach: traditional ML with systematic text normalization vs. fine-tuned transformer models (BERT/RoBERTa/ELECTRA).

## Architecture Patterns

### Two-Phase Design Pattern
- **Phase 1** (`Fase1_MLTradicional/`): Systematic experimentation with 7 text normalization strategies × 4 ML algorithms × 3 vectorization methods × 3 n-gram configurations = 252 experiments
- **Phase 2** (`Fase2_LLMS/`): Transformer fine-tuning with Spanish pre-trained models

### Text Processing Pipeline
All normalization uses spaCy Spanish model (`es_core_news_sm`):
```python
self.categoriasExcluir = ["DET", "ADP", "CCONJ", "SCONJ", "PRON"]  # POS-based stopword filtering
```
Seven normalization strategies available: `Completo`, `Tokenizacion`, `Stopwords`, `Lematizacion`, `Tokenizacion_Stopwords`, `Tokenizacion_Lematizacion`, `Stopwords_Lematizacion`

## Development Workflows

### Traditional ML Experimentation
```bash
# Run specific normalization type
python hateSpeechDetector.py completo Tokenizacion

# Run all normalization types
python hateSpeechDetector.py completo todos
```

### LLM Fine-tuning
Interactive menu system in `Fase2_LLMS/hateSpeechDetector.py`:
1. Train new model (saves to `modelo_guardado_{tipoLLM}/`)
2. Load existing model 
3. Perform inference
4. Model information

### Key File Patterns
- **Model Configuration**: `archivoNormalizacion` dictionary maps normalization types to CSV files
- **Experiment Definition**: `experimentos` list defines all ML algorithm/vectorization/n-gram combinations
- **Result Generation**: Automatic CSV generation (`resultados_experimentos_{tipo}.csv`, `Evidencia_{tipo}.csv`)

## Data Flow Conventions

### Dataset Structure
Binary classification with class imbalance:
- Class 0: "No Odio" (86.2% - 3,448 samples) 
- Class 1: "Discurso Odio" (13.8% - 552 samples)

### File Organization
```
Data/
├── hascosva_2022.csv                    # Original dataset
└── normalizado/                         # Preprocessed versions
    ├── HateSpeech_Completo.csv
    ├── HateSpeech_Tokenizacion.csv
    └── ... (one per normalization type)
```

### Performance Tracking
Always use F1-macro as primary metric due to class imbalance. Results include:
- Cross-validation with StratifiedKFold (n_splits=5)
- Comprehensive classification reports
- Confusion matrices for LLM models

## Model Management

### Traditional ML Models
- Pickle serialization: `modelo_{tipoNormalizacion}.pkl`
- Pipeline structure: `[('text_representation', vectorizer), ('classifier', model)]`
- Best performing: SVM + Frequency Vectorization + Unigrams (F1-macro: 0.7206)

### Transformer Models
- Save pattern: `modelo_guardado_{tipoLLM}/` contains model, tokenizer, label encoder, and metadata
- Spanish models: `dccuchile/bert-base-spanish-wwm-cased`, `PlanTL-GOB-ES/roberta-base-bne`, `google/electra-base-discriminator`
- Custom Dataset class in `personalizacionDataset.py`

## Critical Dependencies

### Required Models
```bash
python -m spacy download es_core_news_sm  # Spanish NLP model
```

### Hardware Considerations
- LLM training: CUDA support recommended (`torch.cuda.is_available()`)
- Traditional ML: CPU sufficient for 252 experiments

## Debugging Patterns

### Logging Configuration
Both phases use structured logging to `app.log`:
```python
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='app.log', filemode='a')
```

### Common Issues
- **spaCy model missing**: Run download command above
- **CUDA memory**: Reduce batch size in TrainingArguments
- **File encoding**: All CSV operations use `encoding='utf-8'`
- **Cross-validation failures**: Check stratification with imbalanced data

## Performance Benchmarks

### Phase 1 Optimal Configuration
- Normalization: Tokenization only
- Algorithm: SVM (linear kernel, C=1.0)  
- Vectorization: Frequency (CountVectorizer)
- N-grams: Unigrams (1,1)
- **Result**: F1-macro 0.7206, Accuracy 0.8840

### Phase 2 Best Model
- Model: RoBERTa (`PlanTL-GOB-ES/roberta-base-bne`)
- **Result**: F1-macro 0.76, Accuracy 0.89
- Training time: ~4 minutes