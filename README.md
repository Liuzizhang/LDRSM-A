📖 Overview
Railway Point Machines are critical for train routing and safety. Traditional fault diagnosis methods are often computationally heavy and impractical for real-time use. LDRSM-a addresses this by integrating:

Feature Extraction Module (FEM): Multi-scale convolutional layers with depthwise separable convolutions.

Temporal Modeling Module (TMM): BiGRU with temporal attention mechanism.

Classification Module (CM): Fully connected layers for fault classification.

The model is optimized for low latency and minimal memory footprint, enabling on-site diagnostics via handheld devices.

🚀 Features
✅ Lightweight: Only 0.066M parameters, 22.009 MFLOPs

✅ High Accuracy: 98.36% on ZDJ9-RPM dataset

✅ Real-Time Capable: Optimized for edge deployment

✅ End-to-End: Raw audio input to fault classification

✅ Interpretable: Temporal attention highlights critical audio segments

📦 Installation
Prerequisites
Python 3.8+

PyTorch 1.9+

torchaudio

scikit-learn

matplotlib

tqdm
