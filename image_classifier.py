import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

# Function to load a pre-trained model
def load_model(model_name):
    if model_name == "ResNet50":
        return ResNet50(weights='imagenet')
    elif model_name == "VGG16":
        return VGG16(weights='imagenet')
    else:
        raise ValueError("Invalid model name")

# Function to classify an image
def classify_image(model, image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=3)[0]

    return decoded_preds

# Function to visualize the image with classification results
def visualize_results(image_path, predictions):
    img = image.load_img(image_path, target_size=(224, 224))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    for _, label, prob in predictions:
        print(f"{label}: {prob*100:.2f}%")

# Main program
if __name__ == "__main__":
    # Allow user to choose a model
    model_name = input("Enter the name of the model (ResNet50/VGG16): ")
    model = load_model(model_name)

    # Allow user to choose an image
    image_path = input("Enter the path to the image: ")

    # Classify the image
    predictions = classify_image(model, image_path)

    # Visualize results and show top predictions
    visualize_results(image_path, predictions)
