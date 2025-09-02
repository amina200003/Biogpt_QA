# Medical Question Answering with RAG-Enhanced BioBERT

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system to enhance BioBERT's ability to answer medical questions. The system combines traditional information retrieval techniques with state-of-the-art question-answering models to provide accurate, contextually relevant medical information.

## 🚀 Key Features

- **RAG Architecture**: Combines retrieval and generation for improved medical QA
- **BioBERT Integration**: Uses the `dmis-lab/biobert-base-cased-v1.1-squad` model fine-tuned on medical data
- **TF-IDF Retrieval**: Fast semantic search through medical knowledge base
- **Intelligent Chunking**: Smart text segmentation for optimal context processing
- **Comprehensive Medical Coverage**: Handles 200+ diverse medical questions
- **Performance Optimization**: GPU acceleration with fallback to CPU

## 🏗️ Architecture

```
User Question → TF-IDF Retrieval → Context Selection → BioBERT QA → Answer Generation
                    ↓                    ↓              ↓
              Medical Database    Top-K Contexts   Best Answer + Score
```

### Components

1. **Retrieval Engine**: TF-IDF vectorization with n-gram features (1-2 grams)
2. **Context Processor**: Intelligent text chunking with overlap for optimal BERT input
3. **QA Model**: BioBERT fine-tuned on SQuAD for extractive question answering
4. **Scoring System**: Confidence scoring for answer quality assessment

## 📋 Prerequisites

- Python 3.7+
- CUDA-compatible GPU (optional, for acceleration)
- Medical QA dataset in CSV format

### Required Packages

```bash
pip install transformers torch pandas scikit-learn tqdm
```

## 🗄️ Data Format

The system expects a CSV file with the following structure:

```csv
question,answer,source,focus_area
"What are the symptoms of breast cancer?", "Breast cancer symptoms include...", "Medical Journal", "Oncology"
```

**Required columns:**
- `answer`: Medical knowledge content (used as retrieval context)
- `source`: Source of the medical information
- `focus_area`: Medical specialty or category

## ⚙️ Configuration

### Key Parameters

```python
# Retrieval settings
TOP_K_CONTEXTS = 30        # Number of contexts to retrieve per question
CHUNK_WORDS = 220          # Maximum words per text chunk
CHUNK_OVERLAP = 60         # Overlap between chunks

# Model settings
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1-squad"
DEVICE = 0 if torch.cuda.is_available() else -1
```

### File Paths

```python
train_file = "path/to/your/medical_QA_dataset.csv"
OUT_JSON = "path/to/output/predictions.json"
OUT_CSV = "path/to/output/predictions.csv"
```

## 🚀 Usage

### Basic Usage

1. **Prepare your medical dataset** in the required CSV format
2. **Update file paths** in the configuration section
3. **Run the notebook** or script

### Custom Questions

Add your medical questions to the `query` list:

```python
query = [
    "What are the symptoms of diabetes?",
    "How to treat hypertension?",
    "What causes migraines?",
    # Add more questions...
]
```

## 📊 Output Format

### JSON Output

```json
{
  "question": "What are the symptoms of breast cancer?",
  "pred_answer": "Common symptoms include...",
  "pred_score": 0.85,
  "best_context": "Retrieved medical text...",
  "best_context_row": 42,
  "source": "Medical Journal",
  "focus_area": "Oncology"
}
```

### CSV Output

Flat table format with all predictions and metadata for easy analysis.

## 🔧 Technical Details

### Retrieval Process

1. **TF-IDF Vectorization**: Creates document-term matrices for fast similarity search
2. **Cosine Similarity**: Ranks contexts by relevance to the question
3. **Top-K Selection**: Retrieves most relevant contexts for detailed processing

### Context Processing

- **Smart Chunking**: Splits long texts into BERT-compatible chunks
- **Overlap Strategy**: Maintains context continuity between chunks
- **Token Optimization**: Ensures chunks fit within BERT's input limits

### QA Pipeline

- **Question-Context Pairing**: Combines question with retrieved context
- **BioBERT Processing**: Extracts answers using medical domain knowledge
- **Score Ranking**: Selects best answer based on confidence scores

## 📈 Performance Considerations

### Optimization Tips

- **GPU Acceleration**: Use CUDA for faster processing
- **Batch Processing**: Process multiple questions efficiently
- **Memory Management**: Adjust chunk sizes based on available RAM
- **Context Limits**: Balance between retrieval breadth and processing speed

### Scalability

- **Dataset Size**: Handles large medical knowledge bases
- **Question Volume**: Processes hundreds of questions efficiently
- **Parallel Processing**: GPU acceleration for multiple QA operations

## 🏥 Medical Applications

### Use Cases

- **Clinical Decision Support**: Assist healthcare providers with medical queries
- **Patient Education**: Provide accurate medical information to patients
- **Medical Research**: Support literature review and knowledge discovery
- **Healthcare Training**: Educational tool for medical students and professionals

### Medical Domains Covered

- **General Medicine**: Common symptoms and treatments
- **Specialized Care**: Oncology, cardiology, neurology, etc.
- **Emergency Medicine**: Acute conditions and urgent care
- **Preventive Care**: Screening, vaccination, lifestyle advice

## 🔬 Research Contributions

This project demonstrates:

1. **RAG Effectiveness**: How retrieval augmentation improves medical QA accuracy
2. **Domain Adaptation**: BioBERT optimization for medical applications
3. **Context Optimization**: Intelligent text chunking for medical knowledge
4. **Scalable Architecture**: Framework for large-scale medical QA systems

## 📚 References

- **BioBERT**: [dmis-lab/biobert-base-cased-v1.1-squad](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1-squad)
- **Transformers**: Hugging Face Transformers library
- **RAG Methodology**: Retrieval-Augmented Generation for knowledge-intensive tasks

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- **Additional Medical Models**: Integration with other medical AI models
- **Enhanced Retrieval**: Advanced semantic search algorithms
- **Evaluation Metrics**: Medical QA-specific performance measures
- **Multi-language Support**: International medical knowledge bases

## 📄 License

This project is for research and educational purposes. Please ensure compliance with medical data privacy regulations (HIPAA, GDPR, etc.) when using with real patient data.

## ⚠️ Disclaimer

**This system is for research and educational purposes only. It should not be used for clinical decision-making or medical advice. Always consult qualified healthcare professionals for medical concerns.**

---

**Built with ❤️ for advancing medical AI research**
