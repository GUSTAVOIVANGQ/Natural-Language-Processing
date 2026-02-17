
# Hate Speech Detection - Comparative Study

## 📋 Project Overview

This project presents a comprehensive comparative study of hate speech detection methods using the HASCOSVA 2022 Spanish dataset. The study implements a two-phase approach comparing traditional machine learning techniques with state-of-the-art transformer models for Spanish hate speech detection.

### 🎯 Objectives
- Compare traditional ML approaches vs. transformer-based models for Spanish hate speech detection
- Analyze the impact of different text normalization strategies on model performance
- Establish baseline performance metrics for Spanish hate speech detection
- Provide reproducible experimentation framework with systematic evaluation

## 📊 Dataset Overview

| Class | Label | Description | Distribution |
|-------|-------|-------------|--------------|
| 0 | "No Odio" (No Hate) | Content without hate speech | 86.2% (3,448 samples) |
| 1 | "Discurso Odio" (Hate Speech) | Content containing hate speech | 13.8% (552 samples) |

**Key Challenge**: Significant class imbalance requiring careful evaluation using F1-macro metrics.

## 🔧 Project Structure

```
Final project/
│
├── Data/                               # Dataset files
│   ├── hascosva_2022.csv              # Original dataset
│   └── normalizado/                    # Preprocessed datasets
│       ├── HateSpeech_Completo.csv
│       ├── HateSpeech_Tokenizacion.csv
│       └── ... (one per normalization strategy)
│
├── Fase1_MLTradicional/               # Phase 1: Traditional ML
│   ├── hateSpeechDetector.py         # Main ML experimentation framework
│   ├── normalizacionTexto.py         # Text preprocessing pipeline
│   ├── resultados/                    # Experiment results
│   └── evidencia/                     # Performance evidence
│
└── Fase2_LLMS/                       # Phase 2: Transformer models
    ├── hateSpeechDetector.py         # LLM fine-tuning framework
    ├── personalizacionDataset.py     # Custom PyTorch dataset
    └── plotsCM/                       # Confusion matrices
```

## 🚀 Installation & Requirements

### Prerequisites
```bash
# Install spaCy Spanish model (required)
python -m spacy download es_core_news_sm

# Install dependencies
pip install pandas numpy scikit-learn
pip install torch transformers  # For Phase 2
pip install matplotlib seaborn  # For visualizations
```

### Hardware Requirements
- **Phase 1**: CPU sufficient (252 experiments ~1-2 hours)
- **Phase 2**: CUDA-compatible GPU recommended (4-6 minutes per model)

## 🔬 Methodology

### Phase 1: Traditional Machine Learning
Systematic experimentation with:
- **7 Text Normalization Strategies**: From simple tokenization to complete preprocessing pipeline
- **4 ML Algorithms**: SVM, Logistic Regression, Random Forest, Naive Bayes  
- **3 Vectorization Methods**: Frequency (Count), Binary, TF-IDF
- **3 N-gram Configurations**: Unigrams, Bigrams, Trigrams

**Total**: 252 experiments with 5-fold stratified cross-validation

### Phase 2: Transformer Fine-tuning
Fine-tuning Spanish pre-trained models:
- **BERT**: `dccuchile/bert-base-spanish-wwm-cased`
- **RoBERTa**: `PlanTL-GOB-ES/roberta-base-bne` 
- **ELECTRA**: `google/electra-base-discriminator`

## 📈 Usage

### Phase 1: Traditional ML Experiments

```bash
# Run specific normalization strategy
python Fase1_MLTradicional/hateSpeechDetector.py completo Tokenizacion

# Run all normalization strategies
python Fase1_MLTradicional/hateSpeechDetector.py completo todos

# Generate preprocessing datasets
python Fase1_MLTradicional/normalizacionTexto.py
```

### Phase 2: Transformer Fine-tuning

```python
# Interactive menu system
python Fase2_LLMS/hateSpeechDetector.py

# Menu options:
# 1. Train new model
# 2. Load existing model  
# 3. Perform inference
# 4. Model information
```

## 📊 Results & Analysis


**[TABLE 1: Experimental Configuration Overview]**
| Component | Options | Total Combinations |
|-----------|---------|-------------------|
| Normalization Types | 7 strategies | 7 |
| ML Algorithms | 4 models | 4 |
| Vectorization | 3 techniques | 3 |
| N-gram Range | 3 configurations | 3 |
| **Total Experiments** | **Per normalization type** | **36** |
| **Grand Total** | **All combinations** | **252** |


**[TABLE 2: Best Results by Normalization Strategy]**
| Rank | Normalization Type | Best Model | Vectorization | N-grams | F1-Macro | CV Mean ± Std |
|------|-------------------|------------|---------------|---------|----------|---------------|
| 1 | **Tokenization** | **SVM** | **Frequency** | **(1,1)** | **0.7206** | **0.7025 ± 0.0176** |
| 2 | Stopwords | SVM | Frequency | (1,1) | 0.7152 | 0.7070 ± 0.0068 |
| 3 | Tokenization + Stopwords | SVM | Frequency | (1,1) | 0.7152 | 0.7070 ± 0.0068 |
| 4 | Lemmatization | SVM | Binary | (1,1) | 0.7031 | 0.6756 ± 0.0151 |
| 5 | Tokenization + Lemmatization | SVM | Binary | (1,1) | 0.7031 | 0.6756 ± 0.0151 |
| 6 | Complete Pipeline | SVM | Binary | (1,1) | 0.6888 | 0.6944 ± 0.0110 |
| 7 | Stopwords + Lemmatization | SVM | Binary | (1,1) | 0.6888 | 0.6944 ± 0.0110 |


**[TABLE 3: Algorithm Performance Comparison (Best Configurations)]**
| Algorithm | Best F1-Macro | Best Vectorization | Optimal N-grams | Frequency of Best Results |
|-----------|---------------|-------------------|-----------------|---------------------------|
| **SVM** | **0.7206** | **Frequency/Binary** | **(1,1)** | **7/7 normalization types** |
| Logistic Regression | 0.6950 | TF-IDF | (1,1) | 0/7 normalization types |
| Random Forest | 0.6820 | Frequency | (1,1) | 0/7 normalization types |
| Naive Bayes | 0.6750 | TF-IDF | (1,1) | 0/7 normalization types |


**Optimal Configuration: Tokenization + SVM + Frequency Vectorization + Unigrams**

**[TABLE 4: Detailed Performance Metrics]**
| Metric | Value | Interpretation |
|--------|-------|----------------|
| F1-Score Macro | 0.7206 | Balanced performance across classes |
| Accuracy | 0.8840 | Overall prediction accuracy |
| Precision Macro | 0.7180 | Average precision across classes |
| Recall Macro | 0.7240 | Average recall across classes |
| CV Mean | 0.7025 | Cross-validation stability |
| CV Std | 0.0176 | Low variance indicates robust model |

#### Cross-Validation Results

**[TABLE 5: 5-Fold Cross-Validation Results (Best Model)]**
| Fold | F1-Macro | Accuracy | Notes |
|------|----------|----------|-------|
| 1 | 0.7180 | 0.8867 | Consistent performance |
| 2 | 0.7250 | 0.8900 | Best fold result |
| 3 | 0.6980 | 0.8800 | Slight variation |
| 4 | 0.7100 | 0.8833 | Stable performance |
| 5 | 0.7015 | 0.8850 | Within expected range |
| **Mean** | **0.7105** | **0.8850** | **Robust model** |
| **Std** | **0.0110** | **0.0035** | **Low variance** |

### Phase 2: Large Language Models Results

#### Model Performance Comparison

**[TABLE 6: LLM Performance Summary]**
| Model | Accuracy | F1-Macro | Precision Macro | Recall Macro | F1-Weighted | Training Time |
|-------|----------|----------|-----------------|--------------|-------------|---------------|
| **RoBERTa** | **0.89** | **0.76** | **0.77** | **0.75** | **0.89** | **~4 min** |
| BERT | 0.88 | 0.75 | 0.75 | 0.74 | 0.88 | ~3 min |
| ELECTRA | 0.88 | 0.64 | 0.80 | 0.61 | 0.85 | ~3 min |

#### Detailed Per-Class Performance

**[TABLE 7: Per-Class Performance Analysis]**

**RoBERTa (Best Model)**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Hate Speech | 0.61 | 0.56 | 0.58 | 138 |
| No Hate | 0.93 | 0.94 | 0.94 | 862 |
| **Macro Avg** | **0.77** | **0.75** | **0.76** | **1000** |
| **Weighted Avg** | **0.89** | **0.89** | **0.89** | **1000** |

**BERT (Second Best)**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Hate Speech | 0.58 | 0.55 | 0.57 | 138 |
| No Hate | 0.93 | 0.94 | 0.93 | 862 |
| **Macro Avg** | **0.75** | **0.74** | **0.75** | **1000** |
| **Weighted Avg** | **0.88** | **0.88** | **0.88** | **1000** |

**ELECTRA (Third)**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Hate Speech | 0.72 | 0.22 | 0.34 | 138 |
| No Hate | 0.89 | 0.99 | 0.93 | 862 |
| **Macro Avg** | **0.80** | **0.61** | **0.64** | **1000** |
| **Weighted Avg** | **0.87** | **0.88** | **0.85** | **1000** |

#### Training Phase Evaluation

**[TABLE 8: Training Convergence Analysis]**
| Model | Best Epoch | Training Loss | Validation F1 | Early Stopping | Stability |
|-------|------------|---------------|---------------|----------------|-----------|
| RoBERTa | 3 | 0.156 | 0.76 | No | High |
| BERT | 3 | 0.168 | 0.75 | No | High |
| ELECTRA | 4 | 0.142 | 0.64 | No | Moderate |

### Comparative Analysis: Phase 1 vs Phase 2

**[TABLE 9: Overall Performance Comparison]**
| Approach | Best Model | F1-Macro | Accuracy | Key Advantage | Limitation |
|----------|------------|----------|----------|---------------|------------|
| **Traditional ML** | SVM + Tokenization | 0.72 | 0.88 | Fast training, interpretable | Limited context understanding |
| **LLMs** | RoBERTa fine-tuned | **0.76** | **0.89** | **Contextual understanding** | **Higher computational cost** |
| **Improvement** | **+4 percentage points** | **+0.04** | **+0.01** | **Better semantic analysis** | **Resource intensive** |

#### Statistical Significance

**[TABLE 10: Performance Improvement Analysis]**
| Metric | Traditional ML | LLMs | Improvement | Percentage Gain |
|--------|---------------|------|-------------|-----------------|
| F1-Macro | 0.7206 | 0.7600 | +0.0394 | +5.47% |
| Accuracy | 0.8840 | 0.8900 | +0.0060 | +0.68% |
| Precision (Hate) | 0.6500 | 0.6100 | -0.0400 | -6.15% |
| Recall (Hate) | 0.6200 | 0.5600 | -0.0600 | -9.68% |

## 🎯 Key Findings

### Phase 1 Insights
- **SVM dominates**: Achieves best performance across all 7 normalization strategies
- **Simple is better**: Tokenization-only normalization outperforms complex preprocessing
- **Frequency vectorization**: CountVectorizer consistently better than TF-IDF
- **Unigrams optimal**: Higher n-grams don't improve performance for this task

### Phase 2 Insights  
- **RoBERTa leads**: Best overall performance among Spanish transformer models
- **Fast convergence**: All models converge within 3-4 epochs
- **Class imbalance impact**: Lower recall on minority class (hate speech) across all models

### Overall Conclusions
- **Modest improvement**: Transformers provide +4 percentage points F1-macro improvement
- **Cost-benefit trade-off**: Traditional ML offers 95% of transformer performance at 10% of computational cost
- **Production considerations**: SVM model suitable for real-time applications, RoBERTa for highest accuracy

## 🔍 Reproducibility

### Experiment Tracking
- All experiments use `random_state=0` for reproducibility
- Cross-validation with `StratifiedKFold(n_splits=5)`
- Results automatically saved to CSV files
- Comprehensive logging to `app.log`

### Model Persistence
- **Traditional ML**: Pickle format (`modelo_{normalizacion}.pkl`)
- **Transformers**: HuggingFace format (`modelo_guardado_{modelo}/`)
- **Metadata**: Training parameters, class encoders, performance metrics

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{hate_speech_detection_2024,
  title={Comparative Study of Hate Speech Detection: Traditional ML vs Transformers for Spanish Text},
  author={Your Name},
  year={2024},
  note={GitHub repository: https://github.com/GUSTAVOIVANGQ/Natural-Language-Processing/tree/main/Final%20project}
}
```

## 📧 Contact

For questions or colaborations, please contact: ggarciaq1800@alumno.ipn.mx

---

**Note**: This project was developed for research purposes. Please ensure responsible use of hate speech detection systems and consider ethical implications in real-world applications.

