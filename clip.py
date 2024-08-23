import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image_url):
    """Fetch image from URL and return its CLIP embedding."""
    #Here I could and probably should add processing which eliminates the background of the
    #picture so that only the item of clothing appears
    response = requests.get(image_url)
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features

def get_text_embedding(description):
    """Return the CLIP embedding for a text description."""
    inputs = processor(text=description, return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features

def cosine_similarity(a, b):
    """Compute the cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_best_match(image_urls, description_embedding):
    """Find the best matching image for a given text embedding."""
    best_match = None
    highest_similarity = -1

    for image_url in image_urls:
        image_embedding = get_image_embedding(image_url).numpy().flatten()
        similarity = cosine_similarity(image_embedding, description_embedding)

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = image_url

    return best_match

def generate_outfit(shirts, pants, shoes, description):
    """Generate an outfit based on description and lists of clothing item URLs."""
    description_embedding = get_text_embedding(description).numpy().flatten()
    print('shirts:', shirts)

    best_shirt = find_best_match(shirts, description_embedding)
    best_pant = find_best_match(pants, description_embedding)
    best_shoe = find_best_match(shoes, description_embedding)

    return {
        "shirt": best_shirt,
        "pant": best_pant,
        "shoe": best_shoe
    }

# EXAMPLE USAGE for testing
# shirts = [
#     "https://firebasestorage.googleapis.com/v0/b/fits4u-519aa.appspot.com/o/Shirts%2F1722623522752whtshirt.png?alt=media&token=9d2387cf-f10e-4ec3-8abc-ea9156197291"
#     # Add more shirt URLs
# ]

# pants = [
#     "https://firebasestorage.googleapis.com/v0/b/fits4u-519aa.appspot.com/o/Pants%2F1720903866459pp.png?alt=media&token=743b3783-501f-4691-b13c-167c93756f26",
#     "https://firebasestorage.googleapis.com/v0/b/fits4u-519aa.appspot.com/o/Pants%2F1721070385359gp.jpg?alt=media&token=5cf16f3a-4f5e-49e1-a6b5-c03326bd1aad",
#     "https://firebasestorage.googleapis.com/v0/b/fits4u-519aa.appspot.com/o/Pants%2F1722624803543blckpant.png?alt=media&token=9ceccbda-06f7-464c-bbb4-9e26e87af832"
#     # Add more pant URLs
# ]

# shoes = [
#     "https://firebasestorage.googleapis.com/v0/b/fits4u-519aa.appspot.com/o/Shoes%2F1722623528803whtshoe.png?alt=media&token=a13e6b80-4f9f-429b-861b-b976ecb96904",
#     # Add more shoe URLs
# ]

# #SUGGESTION: could preprocess the description to filter out the irrelevant words that could skew the cosine similarity
# description = "A stylish outfit with dark pants for a professional occasion."

# outfit = generate_outfit(shirts, pants, shoes, description)
# print("Generated Outfit:", outfit)
