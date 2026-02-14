# Neural Storyteller – Image Captioning with CNN-LSTM Seq2Seq Architecture

A deep learning project that generates natural language descriptions for images using a Sequence-to-Sequence (Seq2Seq) architecture with PyTorch.

## 🎯 Project Overview

This project implements an end-to-end image captioning system that:
- Extracts visual features from images using pretrained ResNet50 (CNN)
- Generates natural language captions using an LSTM-based Seq2Seq decoder
- Supports both Greedy Search (fast) and Beam Search (high-quality) inference
- Provides comprehensive evaluation metrics (BLEU-4, Precision, Recall, F1)
- Includes a production-ready Streamlit web application with modern UI

## 📁 Project Structure

```
GenAss/
├── app.py                              # Streamlit web application
├── GenAi-01_neural_storyteller_notebook.ipynb  # Training notebook (Kaggle)
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── image_captioning_model.pth         # Trained model weights
├── vocab.pkl                          # Vocabulary dictionary
├── inv_vocab.pkl                      # Inverse vocabulary mapping
└── flickr30k_features.pkl             # Cached ResNet50 features
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/muneeb-codehub/Gen-Ai-Assignment-Neural-Storyteller-Image-Captioning-using-CNN-LSTM-Seq2Seq-Architecture.git
cd Gen-Ai-Assignment-Neural-Storyteller-Image-Captioning-using-CNN-LSTM-Seq2Seq-Architecture
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch torchvision streamlit pillow pandas numpy matplotlib scikit-learn nltk tqdm
```

### 3. Download Model Files

Download the pre-trained model files and place them in the project root:
- `image_captioning_model.pth` - Trained LSTM decoder weights
- `vocab.pkl` - Vocabulary mapping
- `inv_vocab.pkl` - Inverse vocabulary
- `flickr30k_features.pkl` - Cached ResNet50 features (optional, only for training)

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

The web app will open at `http://localhost:8501`

## 🎓 Training from Scratch (Kaggle)

### Prerequisites
1. Create a Kaggle account
2. Enable GPU acceleration (T4 x2 recommended)
3. Add dataset: `adityajn105/flickr30k`

### Steps
1. Upload `GenAi-01_neural_storyteller_notebook.ipynb` to Kaggle
2. Enable GPU T4 x2 accelerator in notebook settings
3. Run all cells sequentially
4. Download generated files:
   - `image_captioning_model.pth`
   - `vocab.pkl`
   - `inv_vocab.pkl`
   - `flickr30k_features.pkl`

The training takes approximately 2-3 hours on dual T4 GPUs.

## 🏗️ Architecture

### Encoder
- Input: 2048-dim ResNet50 features
- Output: 512-dim hidden state
- Architecture: Single linear layer

### Decoder
- Embedding: vocab_size → 512
- LSTM: 512 hidden units
- Output: Linear layer → vocab_size
- Uses teacher forcing during training

### Why LSTM over GRU?
LSTM has separate cell state and hidden state, providing better long-term memory for longer captions compared to GRU.

## 📊 Evaluation Metrics

### BLEU-4 Score
Measures n-gram overlap between generated and reference captions.

### Token-level Metrics
- **Precision**: Accuracy of predicted tokens
- **Recall**: Coverage of ground truth tokens
- **F1-Score**: Harmonic mean of precision and recall

### Important Note on Metrics
Image captioning is subjective. Multiple correct captions can describe the same image with different wording:
- Ground Truth: "a man riding a bike"
- Generated: "a person cycling on the road"

Both are semantically correct, but metrics penalize lexical variations. **Semantic correctness is more meaningful than exact lexical matching.**

## 🎨 Streamlit App Features

- **Gradient UI** with glassmorphism effects
- **Dual inference methods**: Greedy Search (faster) and Beam Search (better quality)
- **Real-time caption generation**
- **Responsive design** with smooth animations
- **Professional typography** using Google Fonts

## 📝 Deliverables

1. ✅ **Caption Examples**: 5 random test images with ground truth and generated captions
2. ✅ **Loss Curve**: Training and validation loss visualization
3. ✅ **Quantitative Evaluation**: BLEU-4, Precision, Recall, F1-score
4. ✅ **App Deployment**: Streamlit app with beautiful UI

## 🔧 Technical Details

- **Platform**: Kaggle Notebook
- **Accelerator**: GPU T4 x2 (Dual GPU)
- **Dataset**: Flickr30k (31,000+ images)
- **Framework**: PyTorch
- **Epochs**: 15
- **Optimizer**: Adam (lr=1e-3)
- **Loss Function**: CrossEntropyLoss (ignore padding)
- **Batch Size**: 64

## 📚 Key Concepts

### Feature Caching
Instead of training CNN alongside RNN (computationally expensive), we:
1. Extract features once using pretrained ResNet50
2. Cache features to disk
3. Use cached features during caption training

This approach is:
- Much faster
- Requires less GPU memory
- Industry standard practice

### Inference Methods

**Greedy Search**
- Selects highest probability word at each step
- Fast but may miss better overall sequences
- O(n) complexity

**Beam Search**
- Maintains top-k candidates at each step
- Better quality captions
- O(k×n) complexity

## 🎓 Academic Context

This project demonstrates:
- Multimodal deep learning (vision + language)
- Sequence-to-sequence architectures
- Transfer learning with pretrained CNNs
- Proper evaluation of generative models
- Production deployment of ML models

## 📖 References

- ResNet: "Deep Residual Learning for Image Recognition" (He et al., 2015)
- Seq2Seq: "Sequence to Sequence Learning with Neural Networks" (Sutskever et al., 2014)
- Image Captioning: "Show and Tell: A Neural Image Caption Generator" (Vinyals et al., 2015)
- Dataset: Flickr30k (31,783 images with 158,915 captions)

## 📧 Contact

**Muneeb Arif**  
📧 Email: muneebarif226@gmail.com  
🔗 GitHub: [@muneeb-codehub](https://github.com/muneeb-codehub)  


*Generative AI Assignment - Image Captioning with CNN-LSTM Seq2Seq Architecture*
