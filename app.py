import streamlit as st
import  cv2
import numpy as np
# Charger le modèle de détection de visage
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Fonction pour détecter les visages et enregistrer l'image
def detecter_et_enregistrer_visages(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    st.image(image, channels="BGR")
    if st.button("Enregistrer l'image"):
        cv2.imwrite("image_avec_visages.jpg", image)
        st.success("Image enregistrée avec succès!")

# Interface Streamlit
st.title("Détection de visage et enregistrement")
uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
couleur_rectangle = st.color_picker("Choisissez la couleur des rectangles", "#FF0000")
couleur_rectangle = tuple(int(couleur_rectangle[i:i+2], 16) for i in (1, 3, 5))
min_neighbors = st.slider("MinNeighbors", min_value=1, max_value=10, value=4)
scale_factor = st.slider("ScaleFactor", min_value=1.01, max_value=2.0, value=1.1, step=0.01)


if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    detecter_et_enregistrer_visages(image)

