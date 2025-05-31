# Cassava Disease Classifier 🌿

A simple web application that classifies cassava leaf images into five categories using a fine-tuned **ResNet-50** model. The app is built using **PyTorch** and **Gradio**.


## 📈 Class Labels

The model can classify the following cassava leaf conditions:

1. Cassava Bacterial Blight (CBB)
2. Cassava Brown Streak Disease (CBSD)
3. Cassava Green Mite Damage (CGM)
4. Cassava Mosaic Disease (CMD)
5. Healthy

## 📊 Model Details

* **Architecture**: ResNet-50 (pretrained on ImageNet)
* **Framework**: PyTorch
* **Input Size**: 224 x 224
* **Normalization**:

  * mean: `[0.4326, 0.4953, 0.3120]`
  * std: `[0.2178, 0.2214, 0.2091]`
* **Output Classes**: 5

## 🛦 Project Structure

```
cassava-disease-classifier/
├── app.py                  # Main Gradio app
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── examples/               # Sample cassava leaf images
    ├── healthy.jpg
    ├── cbb.jpg
    ├── cbsd.jpg
    ├── cgm.jpg
    └── cmd.jpg
```

## 🔧 Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/cassava-disease-classifier.git
cd cassava-disease-classifier
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app**

```bash
python app.py
```

## 🔄 Example Usage

Upload an image of a cassava leaf or click one of the example images. The model will return the predicted disease class or "Healthy".
