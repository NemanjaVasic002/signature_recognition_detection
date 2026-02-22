# Comparative Analysis of Deep Learning Architectures for Signature Verification

This project implements a complete computer vision pipeline for automated signature detection and authenticity verification from scanned documents. The research focuses on applying Deep Metric Learning to distinguish genuine signatures from skilled forgeries.

## Key Features
* **End-to-End Pipeline:** From raw scanned documents to final authenticity verification.
* **Advanced Segmentation:** Utilizes Retinex filtering and the Suzuki-Abe algorithm for precise signature extraction from complex backgrounds.
* **Deep Metric Learning:** Implementation of Siamese and Triplet networks focused on similarity learning.
* **Comparative Study:** Detailed evaluation and benchmarking of CNN, Siamese, and Triplet architectures.

##Technologies
* **Language:** Python
* **Deep Learning:** TensorFlow, Keras, PyTorch
* **Computer Vision:** OpenCV (Retinex, CCL, Suzuki-Abe)
* **Data Analysis:** NumPy, Pandas, Matplotlib, Scikit-learn

##System Architecture

### 1. Detection and Extraction (Preprocessing)
Before verification, the system localizes signatures using:
* **Retinex Filter:** For illumination normalization.
* **Connected Component Labeling (CCL):** Combined with histogram analysis for text isolation.
* **Suzuki-Abe Algorithm:** For precise contour detection and signature extraction.

### 2. Verification Models
Three main architectures were implemented and tested:
* **Standard CNN:** Used as a baseline for classification tasks.
* **Siamese Networks:** Utilizing pair-wise inputs and Contrastive Loss.
* **Triplet Networks:** Utilizing Triplet Loss (Anchor, Positive, Negative) to create a robust embedding space.

## 📊 Evaluation Metrics
Models were evaluated using metrics optimized for imbalanced datasets:
* **AUC-ROC Curve**
* **Precision-Recall Curve**
* **Cost-Sensitive Accuracy**

The Triplet Network architecture demonstrated the highest robustness in detecting skilled forgeries by optimizing the distance between signature embeddings.

Installation and Usage
```bash
# Clone the repository
git clone [https://github.com/your-username/signature-verification.git](https://github.com/your-username/signature-verification.git)

# Install dependencies
pip install -r requirements.txt

# Run the main extraction and verification script
python main.py --input sample_document.pdf
