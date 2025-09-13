# PyTorch-Based K-Means Clustering Project

## 📋 Project Overview

This project implements **comparative clustering analysis** using PyTorch-based neural network approaches for **unsupervised news article classification**. The project focuses on non-deterministic clustering methods and stochastic embeddings, fulfilling course requirements for advanced neural network implementations.

### Key Features
- ✅ **Custom PyTorch K-Means Implementation** - Built from scratch using PyTorch tensors and CUDA acceleration
- ✅ **Stochastic Embedding Networks** - Non-deterministic clustering with uncertainty quantification (z = f(x) + ε)
- ✅ **Comparative Analysis** - Deterministic vs. non-deterministic clustering approaches
- ✅ **Comprehensive Evaluation** - Multiple clustering metrics (Silhouette, ARI, NMI, etc.)
- ✅ **Uncertainty Quantification** - Reparameterization trick and input-dependent noise modeling
- ✅ **Data Preprocessing Pipeline** - Complete TF-IDF feature extraction and text preprocessing
- ✅ **Rich Visualizations** - PCA/t-SNE projections and performance comparisons

---

## 🛠️ Installation & Requirements

### System Requirements
- **Operating System**: Linux (Ubuntu/Debian preferred), macOS, or Windows
- **Python Version**: Python 3.8 or higher (3.9-3.11 recommended)
- **Memory**: Minimum 8GB RAM (16GB recommended for large datasets)
- **Storage**: At least 2GB free space for dependencies and data

### Required Libraries and Versions

#### Core Dependencies
```bash
# Data Science Stack
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Machine Learning
scikit-learn>=1.0.0
scipy>=1.7.0

# PyTorch Ecosystem (CUDA support recommended)
torch>=1.12.0
torchvision>=0.13.0  # Optional, for additional utilities

# Natural Language Processing
nltk>=3.6.0
datasets>=2.0.0      # Hugging Face datasets for BBC News

# Feature Extraction
wordcloud>=1.8.0

# Visualization
plotly>=5.0.0        # Interactive plots

# Jupyter Environment
jupyter>=1.0.0
ipykernel>=6.0.0
notebook>=6.4.0

# Additional Utilities  
tqdm>=4.62.0         # Progress bars (if used)
```

### Installation Instructions

#### Option 1: Using pip (Recommended)

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd Cse425-Project-Pytorch-based-basic-kmeans
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # Using venv
   python3 -m venv pytorch_clustering_env
   source pytorch_clustering_env/bin/activate  # Linux/macOS
   # pytorch_clustering_env\Scripts\activate   # Windows

   # OR using conda
   conda create -n pytorch_clustering python=3.9
   conda activate pytorch_clustering
   ```

3. **Install core dependencies**:
   ```bash
   # Basic data science stack
   pip install pandas numpy matplotlib seaborn scipy scikit-learn
   
   # PyTorch (CPU version)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   
   # For CUDA 11.8 (if you have compatible GPU)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # NLP and datasets
   pip install nltk datasets wordcloud
   
   # Visualization
   pip install plotly
   
   # Jupyter
   pip install jupyter ipykernel notebook
   ```

4. **Verify PyTorch installation**:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

#### Option 2: Using conda (Alternative)

```bash
# Create environment with conda-forge packages
conda create -n pytorch_clustering python=3.9
conda activate pytorch_clustering

# Install packages from conda-forge
conda install -c conda-forge pandas numpy matplotlib seaborn scikit-learn scipy nltk wordcloud plotly jupyter

# Install PyTorch from conda
conda install pytorch torchvision cpuonly -c pytorch  # CPU version
# OR for CUDA:
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install datasets via pip (not available in conda)
pip install datasets
```

### Quick Installation Script

Create a file called `install_requirements.sh`:

```bash
#!/bin/bash
echo "Setting up PyTorch K-Means Clustering Project..."

# Create virtual environment
python3 -m venv pytorch_clustering_env
source pytorch_clustering_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install pandas>=1.3.0 numpy>=1.21.0 matplotlib>=3.4.0 seaborn>=0.11.0
pip install scikit-learn>=1.0.0 scipy>=1.7.0
pip install torch>=1.12.0 torchvision>=0.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install nltk>=3.6.0 datasets>=2.0.0 wordcloud>=1.8.0
pip install plotly>=5.0.0 jupyter>=1.0.0 ipykernel>=6.0.0 notebook>=6.4.0

echo "Installation complete! Activate environment with: source pytorch_clustering_env/bin/activate"
```

Run it with:
```bash
chmod +x install_requirements.sh
./install_requirements.sh
```

---

## 🚀 How to Run the Project

### Step-by-Step Execution

#### 1. **Environment Setup**
```bash
# Activate your environment
source pytorch_clustering_env/bin/activate  # or conda activate pytorch_clustering

# Navigate to project directory
cd /path/to/Cse425-Project-Pytorch-based-basic-kmeans
```

#### 2. **Start Jupyter Notebook**
```bash
jupyter notebook
# OR for JupyterLab
jupyter lab
```

#### 3. **Execute Notebooks in Order**

**IMPORTANT**: Run notebooks in this exact sequence:

##### A. Data Preprocessing (REQUIRED FIRST)
```
📓 dataset_preprocessing.ipynb
```
- **Purpose**: Downloads and preprocesses BBC News dataset
- **Runtime**: ~5-10 minutes
- **Output Files**:
  - `processed_news_data.csv`
  - `tfidf_features.pkl`
  - `preprocessing_summary.pkl`
- **What it does**:
  - Downloads BBC News dataset from Hugging Face
  - Performs text cleaning and preprocessing
  - Extracts TF-IDF features with multiple configurations
  - Creates vocabulary analysis and word clouds
  - Saves processed data for model training

##### B. Model Training and Evaluation (RUN SECOND)
```
📓 training_and_comparison.ipynb
```
- **Purpose**: Trains PyTorch models and performs comparative analysis
- **Runtime**: ~10-20 minutes (depends on GPU availability)
- **Prerequisites**: Must complete `dataset_preprocessing.ipynb` first
- **Output Files**:
  - `model_results.pkl`
  - `model_evaluation_results.csv`
  - `pytorch_model_comparison.png`
  - `pytorch_clustering_visualization.png`
- **What it does**:
  - Implements custom PyTorch K-Means from scratch
  - Trains Stochastic Embedding Networks
  - Performs uncertainty quantification
  - Evaluates models with multiple metrics
  - Creates comparative visualizations

### Alternative: Command Line Execution

If you prefer running notebooks from command line:

```bash
# Install nbconvert if not already installed
pip install nbconvert

# Execute notebooks in sequence
jupyter nbconvert --to notebook --execute dataset_preprocessing.ipynb
jupyter nbconvert --to notebook --execute training_and_comparison.ipynb
```

---

## 📊 Expected Results and Outputs

### Generated Files

After successful execution, you'll have these files:

#### Data Files
- **`processed_news_data.csv`** - Cleaned and preprocessed news articles with TF-IDF features
- **`tfidf_features.pkl`** - Serialized TF-IDF vectorizers and matrices
- **`preprocessing_summary.pkl`** - Summary statistics from preprocessing

#### Model Files
- **`model_results.pkl`** - Trained PyTorch models and complete results
- **`model_evaluation_results.csv`** - Quantitative evaluation metrics in CSV format

#### Visualization Files
- **`pytorch_model_comparison.png`** - Performance comparison charts
- **`pytorch_clustering_visualization.png`** - Side-by-side cluster visualizations
- **`pytorch_methods_comparison.png`** - Method comparison plots (if generated)

### Performance Benchmarks

**Typical results on BBC News dataset (~2,200 articles)**:

| Model | Silhouette Score | Training Time | Clusters Found |
|-------|------------------|---------------|----------------|
| PyTorch K-Means | ~0.025-0.035 | 0.05-0.1s | 5-8 |
| Stochastic Embedding | ~0.001-0.010 | 3-5s | 2-6 |

**Note**: Results may vary based on hardware and random initialization.

---

## 🏗️ Project Architecture

### Core Components

#### 1. **PyTorchKMeans Class**
```python
class PyTorchKMeans(nn.Module):
    """Custom K-Means implementation using PyTorch tensors"""
```
- **Features**: K-means++ initialization, CUDA acceleration, gradient-friendly operations
- **Methods**: `fit()`, `predict()`, `fit_predict()`
- **Location**: `training_and_comparison.ipynb`

#### 2. **StochasticEmbeddingNetwork Class**
```python
class StochasticEmbeddingNetwork(nn.Module):
    """Non-deterministic clustering with z = f(x) + ε"""
```
- **Features**: Reparameterization trick, uncertainty quantification, temperature scaling
- **Formula**: z = f(x) + ε where ε ~ N(0, σ²(x))
- **Location**: `training_and_comparison.ipynb`

#### 3. **PyTorchModelTrainer Class**
```python
class PyTorchModelTrainer:
    """Unified trainer for all PyTorch models"""
```
- **Features**: Model comparison, evaluation metrics, uncertainty analysis
- **Location**: `training_and_comparison.ipynb`

### Data Pipeline

```
BBC News Dataset (Hugging Face)
    ↓
Text Preprocessing & Cleaning
    ↓
TF-IDF Feature Extraction (3 configurations)
    ↓
PyTorch Tensor Conversion
    ↓
Model Training (K-Means + Stochastic)
    ↓
Evaluation & Comparison
    ↓
Visualization & Results
```

---

## ⚙️ Configuration Options

### Model Hyperparameters

You can modify these in the notebooks:

#### PyTorch K-Means
```python
# In training_and_comparison.ipynb
pytorch_kmeans = PyTorchKMeans(
    n_clusters=8,          # Number of clusters (auto-tuned in range 2-12)
    max_iter=300,          # Maximum iterations
    tol=1e-4,             # Convergence tolerance
    init='k-means++'      # Initialization method
)
```

#### Stochastic Embedding Network
```python
# In training_and_comparison.ipynb
model = StochasticEmbeddingNetwork(
    input_dim=1500,        # Input feature dimension (auto-detected)
    n_clusters=5,          # Number of clusters (auto-tuned in range 2-8)
    hidden_dim=256,        # Hidden layer size
    embedding_dim=128,     # Embedding dimension
    noise_scale=0.1        # Initial noise scale
)
```

### TF-IDF Configurations

Available in `dataset_preprocessing.ipynb`:

```python
tfidf_configs = {
    'basic': {
        'max_features': 1000,
        'min_df': 2,
        'max_df': 0.8,
        'ngram_range': (1, 1)  # Unigrams only
    },
    'bigrams': {
        'max_features': 1500,
        'min_df': 2, 
        'max_df': 0.8,
        'ngram_range': (1, 2)  # Unigrams + Bigrams (DEFAULT)
    },
    'trigrams': {
        'max_features': 2000,
        'min_df': 2,
        'max_df': 0.8,
        'ngram_range': (1, 3)  # Unigrams + Bigrams + Trigrams
    }
}
```

### Hardware Optimization

#### For GPU Users
The code automatically detects and uses CUDA if available:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

#### For CPU-Only Systems
- Project works fully on CPU
- Expect longer training times for stochastic models (~5-10x slower)
- Consider reducing `hidden_dim` and `epochs` for faster execution

---

## 🧪 Advanced Usage

### Custom Dataset Integration

To use your own text dataset:

1. **Replace data loading** in `dataset_preprocessing.ipynb`:
   ```python
   # Instead of BBC News dataset
   # dataset = load_dataset("SetFit/bbc-news")
   
   # Load your CSV with 'text' and optionally 'label' columns
   news_df = pd.read_csv('your_dataset.csv')
   # Ensure columns: 'text', 'label_text' (optional)
   ```

2. **Adjust preprocessing parameters** based on your text characteristics

### Model Customization

#### Adding New Clustering Methods

Extend the `PyTorchModelTrainer` class:

```python
def train_custom_model(self, **params):
    """Add your custom clustering method here"""
    # Implementation
    pass
```

#### Modifying Network Architecture

In `StochasticEmbeddingNetwork`:
- Adjust encoder layers
- Change activation functions
- Add regularization techniques

### Batch Processing

For large datasets, modify batch processing:

```python
# In _train_stochastic_model method
batch_size = 512  # Increase for more memory, decrease for less
```

---

## 📈 Evaluation Metrics Explained

### Unsupervised Clustering Metrics

1. **Silhouette Score** (-1 to 1, higher is better)
   - Measures how similar points are to their own cluster vs. other clusters
   - Values > 0.5 indicate good clustering

2. **Calinski-Harabasz Index** (higher is better)
   - Ratio of between-cluster dispersion to within-cluster dispersion
   - No fixed range, compare relative values

3. **Davies-Bouldin Index** (lower is better)
   - Average similarity ratio of each cluster with its most similar cluster
   - Values closer to 0 indicate better clustering

### Supervised Metrics (when ground truth available)

1. **Adjusted Rand Index (ARI)** (-1 to 1, higher is better)
   - Measures similarity between true and predicted clusters
   - Adjusted for chance, 0 = random clustering

2. **Normalized Mutual Information (NMI)** (0 to 1, higher is better)
   - Information theoretic measure of clustering agreement
   - 1 = perfect clustering, 0 = no agreement

3. **Homogeneity Score** (0 to 1, higher is better)
   - Whether each cluster contains only members of a single class
   - 1 = perfectly homogeneous clusters

---

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. **PyTorch Installation Issues**
```bash
# Error: "No module named 'torch'"
pip install torch torchvision

# CUDA compatibility issues
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 2. **Memory Issues**
```python
# In notebooks, reduce batch size
batch_size = 64  # Instead of 256

# Reduce model complexity
hidden_dim = 64   # Instead of 256
embedding_dim = 32  # Instead of 128
```

#### 3. **NLTK Download Errors**
```python
# Run in notebook cell
import nltk
nltk.download('all')  # Downloads all NLTK data (may take time)
```

#### 4. **Dataset Loading Failures**
```bash
# Check internet connection for Hugging Face datasets
pip install --upgrade datasets

# Alternative: Use local dataset
# Place your CSV file in project directory and modify data loading code
```

#### 5. **Jupyter Kernel Issues**
```bash
# Install ipykernel in your environment
pip install ipykernel
python -m ipykernel install --user --name pytorch_clustering --display-name "PyTorch Clustering"
```

#### 6. **Visualization Issues**
```bash
# For matplotlib backend issues
pip install --upgrade matplotlib

# For plotly display issues in Jupyter
pip install --upgrade plotly nbformat

# If wordcloud fails to install
conda install -c conda-forge wordcloud
# OR
pip install wordcloud --no-cache-dir
```

### Performance Optimization Tips

1. **Enable CUDA** if you have a compatible GPU:
   ```python
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
   ```

2. **Adjust batch sizes** based on available memory:
   - GPU with 8GB+: batch_size = 512
   - GPU with 4GB: batch_size = 256
   - CPU only: batch_size = 64-128

3. **Reduce dataset size** for testing:
   ```python
   # In dataset_preprocessing.ipynb
   news_df = news_df.sample(n=1000, random_state=42)  # Use smaller subset
   ```

---

## 🎯 Assignment Compliance

This project fulfills the following academic requirements:

### Core Requirements ✅
- **Non-deterministic Models**: Stochastic Embedding Network with z = f(x) + ε
- **Reparameterization Trick**: Implemented for gradient-based training  
- **Uncertainty Quantification**: Input-dependent noise modeling
- **Comparative Analysis**: Multiple clustering approaches compared
- **Comprehensive Evaluation**: Multiple metrics (Silhouette, ARI, NMI, etc.)

### Technical Implementation ✅
- **PyTorch Framework**: All models built using PyTorch tensors and nn.Module
- **CUDA Support**: Automatic GPU acceleration when available
- **From-Scratch Implementation**: Custom K-Means implementation in PyTorch
- **Neural Network Architecture**: Deep embedding networks for clustering
- **Gradient-Based Optimization**: Adam optimizer with custom loss functions

### Deliverables ✅
- **Complete Code**: Two comprehensive Jupyter notebooks
- **Documentation**: This comprehensive README
- **Results**: Quantitative evaluation metrics and visualizations  
- **Reproducibility**: Detailed installation and execution instructions

---

## 📚 References and Resources

### Academic Papers
- "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013) - Reparameterization trick
- "Deep Clustering: A Comprehensive Survey" (Min et al., 2018)
- "Unsupervised Deep Embedding for Clustering Analysis" (Xie et al., 2016)

### Technical Documentation
- [PyTorch Official Documentation](https://pytorch.org/docs/)
- [scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)

### Datasets
- **BBC News Dataset**: Available via Hugging Face `datasets` library
- **Alternative**: Any text classification dataset with 'text' and 'label' columns

---

## 👥 Contributing and Support

### Project Structure
```
Cse425-Project-Pytorch-based-basic-kmeans/
├── README.md                              # This comprehensive guide
├── dataset_preprocessing.ipynb            # Data preprocessing and TF-IDF extraction
├── training_and_comparison.ipynb         # Model training and evaluation
├── processed_news_data.csv              # Preprocessed dataset (generated)
├── model_results.pkl                     # Trained models (generated)
├── model_evaluation_results.csv         # Evaluation metrics (generated)
├── pytorch_model_comparison.png         # Performance visualizations (generated)
├── pytorch_clustering_visualization.png # Cluster visualizations (generated)
├── preprocessing_summary.pkl            # Preprocessing statistics (generated)
├── tfidf_features.pkl                  # TF-IDF features (generated)
└── research_paper.pdf                   # Academic paper (if available)
```

### Getting Help

1. **Check this README** for installation and usage instructions
2. **Review error messages** carefully - most issues are dependency-related
3. **Verify environment setup** - ensure all packages are installed correctly
4. **Check hardware compatibility** - GPU support is optional but recommended
5. **Test with smaller datasets** if encountering memory issues

### Known Limitations

- **Stochastic models** may show high variance in results due to randomness
- **Large datasets** (>10k articles) may require significant memory and time  
- **GPU memory** limitations may require batch size adjustments
- **NLTK data** requires internet connection for initial download

---

## 📋 Project Summary

This project provides a **comprehensive implementation** of PyTorch-based clustering methods with emphasis on:

🔬 **Research Focus**: Non-deterministic clustering with uncertainty quantification  
⚙️ **Technical Depth**: Custom PyTorch implementations built from scratch  
📊 **Evaluation Rigor**: Multiple clustering metrics and comparative analysis  
🎨 **Visualization**: Rich plots and interactive comparisons  
📖 **Documentation**: Complete setup and usage instructions  
🔧 **Reproducibility**: Detailed environment specifications and execution steps  

**Target Audience**: Students, researchers, and practitioners interested in advanced clustering methods and PyTorch implementation techniques.

**Academic Level**: Graduate-level neural networks and machine learning courses.

---

**Last Updated**: September 2025  
**Python Version**: 3.8+  
**PyTorch Version**: 1.12.0+  
**Platform**: Cross-platform (Linux, macOS, Windows)
