import streamlit as st
from PIL import Image
from model import get_ingredient

# Streamlit application
st.title("Ingredient Extractor")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image on the left side
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image to a temporary location
    image_path = f"./temp_image.{uploaded_file.type.split('/')[-1]}"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Get ingredients from the image
    ingredients = get_ingredient(image_path)
    print(ingredients)
    # Display ingredients in table or labels on the left side
    st.subheader("Ingredients")
    for key, value in ingredients.items():
        st.write(f"**{key}:** {value}")
