import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>

.stApp{
background: linear-gradient(to right,#0f2027,#203a43,#2c5364);
color:white;
}

.main-title{
text-align:center;
font-size:42px;
font-weight:bold;
color:white;
}

.sub-title{
text-align:center;
font-size:18px;
color:#d6d6d6;
}

.card{
background:rgba(255,255,255,0.08);
padding:20px;
border-radius:18px;
backdrop-filter:blur(15px);
box-shadow:0px 5px 20px rgba(0,0,0,0.3);
}

.result-success{
background:#00C853;
padding:18px;
border-radius:12px;
font-size:22px;
font-weight:bold;
color:white;
text-align:center;
}

.result-danger{
background:#D50000;
padding:18px;
border-radius:12px;
font-size:22px;
font-weight:bold;
color:white;
text-align:center;
}

.stButton>button{
width:100%;
background:#00b4db;
background:linear-gradient(to right,#0083B0,#00B4DB);
color:white;
border-radius:10px;
height:55px;
font-size:18px;
font-weight:bold;
border:none;
}

.stButton>button:hover{
background:linear-gradient(to right,#00B4DB,#0083B0);
color:white;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        r"C:\Users\ASUS\Downloads\face_mask_model.keras"
    )

model = load_model()

mask_labels = ["with_mask", "without_mask"]

# ==========================================
# HEADER
# ==========================================
st.markdown(
    "<div class='main-title'> Face Mask Detection System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Deep Learning CNN | Image Upload + Live Webcam</div>",
    unsafe_allow_html=True
)

st.write("")

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("Navigation")

option = st.sidebar.radio(
    "Choose Detection Mode",
    ["Image Detection", "Live Webcam"]
)

st.sidebar.markdown("---")
st.sidebar.info(
"""
### Model Details

✔ CNN Model

✔ Classes :

- With Mask

- Without Mask
"""
)

# ==========================================
# IMAGE DETECTION
# ==========================================
if option == "Image Detection":

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg","jpeg","png","webp"]
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_image is not None:

        image = Image.open(uploaded_image).convert("RGB")

        col1,col2 = st.columns([1,1])

        with col1:

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with col2:

            if st.button("Predict Mask"):

                img = image.resize((128,128))

                img_array = tf.keras.utils.img_to_array(img)

                img_array = img_array.astype("float32")

                img_array = np.expand_dims(img_array,axis=0)

                prediction = model.predict(img_array,verbose=0)

                predicted_class = np.argmax(prediction)

                confidence = np.max(prediction)*100

                st.subheader("Prediction")

                st.progress(int(confidence))

                if mask_labels[predicted_class] == "with_mask":

                    st.markdown(
                        f"<div class='result-success'>✅ MASK DETECTED<br>{confidence:.2f}%</div>",
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        f"<div class='result-danger'>❌ NO MASK DETECTED<br>{confidence:.2f}%</div>",
                        unsafe_allow_html=True
                    )

# ==========================================
# LIVE WEBCAM
# ==========================================
else:

    st.warning(
        "Press **Q** in the camera window to stop detection."
    )

    if st.button("Start Webcam"):

        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        cap = cv2.VideoCapture(0)

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5
            )

            for (x,y,w,h) in faces:

                face = frame[y:y+h,x:x+w]

                face = cv2.resize(
                    face,
                    (128,128)
                )

                face = face.astype("float32")

                face = np.expand_dims(
                    face,
                    axis=0
                )

                prediction = model.predict(
                    face,
                    verbose=0
                )[0]

                class_index = np.argmax(prediction)

                confidence = np.max(prediction)*100

                label = mask_labels[class_index]

                if label=="with_mask":

                    color=(0,255,0)

                    text=f"Mask {confidence:.1f}%"

                else:

                    color=(0,0,255)

                    text=f"No Mask {confidence:.1f}%"

                cv2.rectangle(
                    frame,
                    (x,y),
                    (x+w,y+h),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    text,
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

            cv2.imshow(
                "Face Mask Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()

        cv2.destroyAllWindows()

# ==========================================
# FOOTER
# ==========================================
footer = """
<div style="text-align:center;">
🤖 <b>   Face Mask Detection System | Deep Learning CNN</b><br><br>

Made with ❤️ using
<b>Streamlit</b> •
<b>TensorFlow</b> •
<b>OpenCV</b>

<br><br>

© 2026 <b>Created by Rijo Thomas</b>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)