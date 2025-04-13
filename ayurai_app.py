
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load model, encoder, and column structure
model = load_model("ayurai_model.h5")
label_encoder = joblib.load("label_encoder.pkl")
column_template = pd.read_csv("sample_model_input_columns.csv", header=None).values[0].tolist()

st.title("🧘‍♀️ AyurAI – Discover Your Dosha 🌿")
st.markdown("Answer a few questions and let AyurAI predict your dominant dosha and offer personalized recommendations.")

# User form inputs (25+ features)
with st.form("dosha_quiz"):
    skin = st.selectbox("Your skin texture?", ["Dry,Rough", "Soft,Sweating", "Moist,Greasy"])
    complexion = st.selectbox("Your complexion?", ["Pinkish", "Dark", "Pale", "Yellowish"])
    hair = st.selectbox("Hair type?", ["Dry", "Greasy", "Normal"])
    nails = st.selectbox("Nail color?", ["Redish", "Blackish", "Pinkish", "Bluish"])
    mood = st.selectbox("Your typical mood?", ["Anxious", "Irritated", "Sluggish"])
    sleep = st.selectbox("Sleep pattern?", ["Late", "Early", "Heavy"])
    digestion = st.selectbox("How’s your digestion?", ["Constipated", "Acidic", "Slow"])
    hunger = st.selectbox("How’s your hunger?", ["Skips Meal", "Irregular", "Sudden and Sharp"])
    thirst = st.selectbox("Your thirst level?", ["Excessive", "Normal", "Low"])
    voice = st.selectbox("Quality of your voice?", ["Hoarse", "Clear", "Rough"])
    dreams = st.selectbox("Dreams about?", ["Sky", "Fire", "Water"])
    social = st.selectbox("Your social nature?", ["Introvert", "Ambivert", "Extrovert"])
    mental = st.selectbox("Mental activity?", ["Restless", "Aggressive", "Calm"])
    memory = st.selectbox("Memory type?", ["Short", "Sharp", "Slow"])
    pulse = st.selectbox("Your pulse feels?", ["Irregular", "Moderate", "Regular"])
    body_frame = st.selectbox("Your body frame?", ["Thin and Lean", "Medium", "Well Built"])
    walk = st.selectbox("Pace of walking?", ["Fast", "Normal", "Slow"])
    nature = st.selectbox("You are mostly…", ["Forgiving,Grateful", "Jealous,Fearful", "Calm,Stable"])
    energy = st.selectbox("Body energy level?", ["Fluctuates", "High", "Low"])
    weather = st.selectbox("You dislike?", ["Cold", "Hot", "Humid"])
    bowel = st.selectbox("Bowel movement?", ["Irregular", "Burning", "Heavy"])
    odor = st.selectbox("Your body odor is?", ["Mild", "Negligible", "Strong"])
    eyes = st.selectbox("Eye type?", ["Small,Dry", "Sharp,Red", "Large,Attractive"])
    walk_style = st.selectbox("Your walking style is?", ["Fast", "Moderate", "Slow"])
    perspiration = st.selectbox("Level of perspiration?", ["Less", "Moderate", "Excessive"])
    submit = st.form_submit_button("🔍 Predict My Dosha")

if submit:
    user_input = pd.DataFrame([{
        "Skin": skin,
        "Complexion": complexion,
        "Type of Hair": hair,
        "Nails": nails,
        "Mood": mood,
        "Sleep Pattern": sleep,
        "Digestion": digestion,
        "Hunger": hunger,
        "Thirst": thirst,
        "Quality of Voice": voice,
        "Dreams": dreams,
        "Social Relations": social,
        "Mental Activity": mental,
        "Memory": memory,
        "Pulse Movement": pulse,
        "Body Frame": body_frame,
        "Pace of Performing Work": walk,
        "Nature": nature,
        "Body Energy": energy,
        "Weather Conditions": weather,
        "Bowel Movement": bowel,
        "Body Odor": odor,
        "Eyes": eyes,
        "Walking Style": walk_style,
        "Perspiration": perspiration
    }])

    # Encode and align with training columns
    user_encoded = pd.get_dummies(user_input)
    user_encoded = user_encoded.reindex(columns=column_template, fill_value=0)
    prediction = model.predict(user_encoded.astype(np.float32).values)
    dosha = label_encoder.inverse_transform([np.argmax(prediction)])

    st.success(f"🌸 You are primarily **{dosha[0]}** dosha.")

    # Recommend herbs and tips
    if dosha[0] == "Vata":
        st.markdown("**🌿 Recommendations for Vata:** Warm, moist foods, oils, ashwagandha, ginger tea, meditation.")
    elif dosha[0] == "Pitta":
        st.markdown("**🧊 Recommendations for Pitta:** Cooling foods, aloe vera, coconut water, time in nature.")
    elif dosha[0] == "Kapha":
        st.markdown("**🔥 Recommendations for Kapha:** Light, spicy foods, turmeric, early wake-up, vigorous exercise.")
