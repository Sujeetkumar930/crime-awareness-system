import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load datasets
master_dataset = pd.read_csv(os.path.join(BASE_DIR, "master_dataset.csv"))
d6 = pd.read_csv(os.path.join(BASE_DIR, "d6_cleaned.csv"))

# Load model + encoders
model = joblib.load(os.path.join(BASE_DIR, "crime_model_dr1.pkl"))
le_state = joblib.load(os.path.join(BASE_DIR, "le_state.pkl"))
le_district = joblib.load(os.path.join(BASE_DIR, "le_district.pkl"))
le_city = joblib.load(os.path.join(BASE_DIR, "le_city.pkl"))
le_crime = joblib.load(os.path.join(BASE_DIR, "le_crime.pkl"))

# -----------------------------
# Streamlit UI
st.set_page_config(page_title=" Crime Awareness System", layout="centered")
st.title("🚨  Crime Awareness System")

# Dropdowns
state = st.selectbox("Select State", sorted(master_dataset["state"].dropna().unique()))
district = st.selectbox(
    "Select District",
    sorted(
        master_dataset[master_dataset["state"] == state]["district"].dropna().unique()
    ),
)
city = st.selectbox(
    "Select City",
    sorted(
        master_dataset[master_dataset["district"] == district]["city"].dropna().unique()
    ),
)
crime_type = st.selectbox(
    "Select Crime Type", sorted(master_dataset["crime_type"].dropna().unique())
)
year = st.number_input("Enter Year", min_value=2000, max_value=2030, value=2022)

if st.button("🔮 Predict Crime Count"):
    # Encode input
    state_enc = le_state.transform([state])[0]
    district_enc = le_district.transform([district])[0]
    city_enc = le_city.transform([city])[0]
    crime_type_enc = le_crime.transform([crime_type])[0]

    X_input = pd.DataFrame(
        [
            {
                "state_enc": state_enc,
                "district_enc": district_enc,
                "city_enc": city_enc,
                "crime_type_enc": crime_type_enc,
                "year": year,
            }
        ]
    )

    # Prediction
    prediction = int(model.predict(X_input)[0])
    st.success(f"Predicted Crime Count: {prediction}")

    # Total crime count from dataset
    mask = (
        (master_dataset["state"] == state)
        & (master_dataset["district"] == district)
        & (master_dataset["city"] == city)
        & (master_dataset["year"] == year)
        & (master_dataset["crime_type"] == crime_type)
    )
    total_count = int(master_dataset.loc[mask, "crime_count"].sum())
    st.info(f"📊 Total Recorded Crime Count: {total_count}")

    # d6 info
    d6_row = d6[(d6["state"] == state) & (d6["year"] == year)]
    if not d6_row.empty:
        st.subheader("📌 State Information")
        st.write(f"🏢 Police Stations: {int(d6_row['no.of_police_station'].values[0])}")
        st.write(f"📖 Literacy Rate: {float(d6_row['literacy_rate_in_%'].values[0])}%")
        st.write(f"🌍 Area: {float(d6_row['area_in_kilometer_square'].values[0])} km²")
        st.write(f"👥 Population: {int(d6_row['population'].values[0])}")

# -----------------------------
# Simple Chatbot (fun messages)
st.sidebar.title("💬 Crime Awareness Chatbot")
user_msg = st.sidebar.text_input("Ask me something about crime & safety:")
if user_msg:
    responses = [
        "🚔 Always report crimes to the nearest police station.",
        "🔦 Stay aware of your surroundings, especially at night.",
        "👮 Join community safety programs in your city.",
        "📱 Save emergency helpline numbers on your phone.",
        "🚨 Prevention is better than cure — avoid unsafe areas.",
    ]
    import random

    bot_reply = random.choice(responses)
    st.sidebar.success(f"Bot:{bot_reply}")
