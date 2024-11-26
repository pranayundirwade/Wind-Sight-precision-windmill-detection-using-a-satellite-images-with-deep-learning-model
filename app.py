import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pymongo 
import urllib.parse
import random  
import smtplib  


@st.cache_resource
def init_db():
    
    username = urllib.parse.quote_plus("pranayundirwade") 
    password = urllib.parse.quote_plus("Pranay@358575")  

    client = pymongo.MongoClient(f"mongodb+srv://{username}:{password}@cluster0.9tmwm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    
    db = client["windmill_db"] 
    collection = db["predictions"]
    users_collection = db["users"]  
    return collection, users_collection


@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(r'C:\Users\Pranay\Desktop\Final_Project\windmill_model.h5', compile=False)
       # st.write("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise

def preprocess_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')  
    elif image.mode == 'L':
        image = image.convert('RGB')  

    image = image.resize((512, 512))
    image_np = np.array(image) / 255.0

    if image_np.shape[-1] == 4:
        image_np = image_np[..., :3]  

    image_np = np.expand_dims(image_np, axis=0)
    image_np = image_np.astype(np.float32)

    if image_np.shape != (1, 512, 512, 3):
        raise ValueError(f"Processed image has incorrect shape: {image_np.shape}. Expected (1, 512, 512, 3).")

    return image_np

# Prediction function
@tf.function
def predict_mask(model, image):
    return model(image)

# Predict new image
def predict_new_image(model, image):
    image = preprocess_image(image)
    predicted_mask = predict_mask(model, image)
    predicted_mask_binary = (predicted_mask > 0.5).numpy().astype(np.uint8)
    return image[0], predicted_mask_binary[0]

# Visualize the prediction
def visualize_prediction(image, predicted_mask):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(predicted_mask.squeeze(), cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    st.pyplot(fig)

# Function to save results in MongoDB
def save_prediction_to_db(collection, image_name, prediction_result):
    try:
        record = {
            "image_name": image_name,
            "prediction_result": prediction_result,
        }
        collection.insert_one(record)
        st.write("Prediction saved to the database")
    except Exception as e:
        st.error(f"Error saving prediction to the database: {e}")

# Sign up function to register users
def sign_up(users_collection, username, password, email):
    if users_collection.find_one({"username": username}):
        return False, "Username already exists!"
    users_collection.insert_one({"username": username, "password": password, "email": email})
    return True, "User registered successfully!"

# Login function to authenticate users
def login(users_collection, username, password):
    user = users_collection.find_one({"username": username, "password": password})
    if user:
        return True, "Login successful!"
    return False, "Invalid username or password!"

# Forgot password function - sends OTP to user's email
def send_otp(email, otp):
    try:
        # Replace these credentials with your actual SMTP server setup
        sender_email = "youremail@example.com"
        sender_password = "yourpassword"
        smtp_server = "smtp.example.com"

        server = smtplib.SMTP(smtp_server, 587)
        server.starttls()
        server.login(sender_email, sender_password)

        message = f"Subject: Your OTP for Password Reset\n\nYour OTP is {otp}."
        server.sendmail(sender_email, email, message)
        server.quit()
        return True, "OTP sent successfully!"
    except Exception as e:
        return False, f"Failed to send OTP: {e}"

st.title("Windmill Detection - Login & Signup")

collection, users_collection = init_db()

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

option = st.sidebar.selectbox("Choose an option", ["Login", "Sign Up", "Forgot Password"])

if option == "Sign Up":
    st.subheader("Create an account")
    new_username = st.text_input("Username")
    new_password = st.text_input("Password", type="password")
    new_email = st.text_input("Email")

    if st.button("Sign Up"):
        if new_username and new_password and new_email:
            success, message = sign_up(users_collection, new_username, new_password, new_email)
            if success:
                st.success(message)
            else:
                st.error(message)
        else:
            st.error("Please fill all fields")

elif option == "Login":
    st.subheader("Login to your account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username and password:
            success, message = login(users_collection, username, password)
            if success:
                st.success(message)
                st.session_state['logged_in'] = True  

if st.session_state['logged_in']:
    st.subheader("You are logged in!")
    uploaded_file = st.file_uploader("Upload images (JPG, JPEG, PNG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            st.write(f"File uploaded: {uploaded_file.name}")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            model = load_model()

            st.write("Processing image...")
            image, predicted_mask = predict_new_image(model, image)

            st.subheader("Prediction Result")
            visualize_prediction(image, predicted_mask)

            prediction_result = "Windmill Present" if np.sum(predicted_mask) > 0 else "No Windmill Present"
            st.write(f"Prediction result: {prediction_result}")

            if np.sum(predicted_mask) > 0:
                st.success(prediction_result)
            else:
                st.warning(prediction_result)

            save_prediction_to_db(collection, uploaded_file.name, prediction_result)

        except ValueError as e:
            st.error(f"Error in prediction: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

elif option == "Forgot Password":
    st.subheader("Reset your password")
    email = st.text_input("Enter your registered email")
    if st.button("Send OTP"):
        if email:
            otp = random.randint(100000, 999999)  
            success, message = send_otp(email, otp)
            if success:
                st.success(message)
                entered_otp = st.text_input("Enter the OTP sent to your email")
                new_password = st.text_input("Enter new password", type="password")
                if st.button("Reset Password"):
                    if str(otp) == entered_otp:
                        user = users_collection.find_one({"email": email})
                        if user:
                            users_collection.update_one({"email": email}, {"$set": {"password": new_password}})
                            st.success("Password reset successful!")
                        else:
                            st.error("User not found!")
                    else:
                        st.error("Invalid OTP!")
            else:
                st.error(message)
        else:
            st.error("Please enter your email.")