import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import streamlit as st
from staffClassifier.components.models import ImageClassifier, get_custom_model
from staffClassifier.config_manager.configuration import ConfigurationManager


# Define transformations for the input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(model,image):
    # Perform inference
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image_tensor)

    # Get the predicted class label
    predicted_class = torch.argmax(output).item()

    # Load the ImageNet class labels
    classes = ['Non-Staff','Staff']
    #
    # Print the predicted class label
    print("Predicted class:", classes[predicted_class])
    return  classes[predicted_class]


if __name__ == '__main__':
    # Import model configs
    model_config = ConfigurationManager().get_model_config()
    ckpt_path = r'artifacts/checkpoints/epoch=2-step=5019.ckpt'
    checkpoint = torch.load(ckpt_path,map_location=torch.device('cpu'))
    model = ImageClassifier(model_config,)
    model.load_state_dict(checkpoint["state_dict"])
    
    

    # Load and preprocess the input image
    st.title("Image Uploader")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        pred = predict(model,image)
        st.write("## Predicted class :",pred)