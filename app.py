import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# ----- ConvertToRGB class -----
class ConvertToRGB(object):
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

# ----- Step 1: Class labels -----
class_labels = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Healthy",
    4: "Cassava Mosaic Disease (CMD)"
}

# ----- Step 2: Updated transform with training-time normalization -----
transform = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4326, 0.4953, 0.3120],
                         std=[0.2178, 0.2214, 0.2091])
])

# ----- Step 3: Load model -----
def load_model():
   model = torch.load("model/model_trained.pth", weights_only=False,map_location=torch.device('cpu'))
   model.eval()

   return model


model = load_model()

# ----- Step 4: Prediction function -----
def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        results = {class_labels[i]: float(probs[i]) for i in range(5)}
    return results

# ----- Step 5: Gradio interface -----


examples = [
    ["examples/Healthy.jpg"],
    ["examples/Cassava Brown Streak Disease (CBSD).jpg"],
    ["examples/Cassava Mosaic Disease (CMD).jpg"],
    ["examples/Cassava Green Mottle (CGM).jpg"],
]

view = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="Cassava Leaf Disease Classifier",
    description="Upload a cassava leaf image to detect whether it is healthy or affected by disease.",
    examples=examples,
    theme = gr.themes.Soft(),
)

# ----- Step 6: Run app -----
if __name__ == "__main__":
    view.launch()
