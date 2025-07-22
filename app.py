import streamlit as st
import numpy as np
import joblib

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
le_team = joblib.load('le_team.pkl')
le_opp = joblib.load('le_opp.pkl')
le_venue = joblib.load('le_venue.pkl')
le_result = joblib.load('le_result.pkl')

st.title("Prediksi Hasil Pertandingan Serie A")

team = st.selectbox("Pilih Tim", le_team.classes_)
opponent = st.selectbox("Pilih Lawan", le_opp.classes_)
venue = st.selectbox("Venue (Home/Away)", le_venue.classes_)

xG = st.slider("Expected Goals (xG)", 0.0, 5.0, step=0.1)
xGA = st.slider("Expected Goals Against (xGA)", 0.0, 5.0, step=0.1)
poss = st.slider("Penguasaan Bola (%)", 0.0, 100.0, step=1.0)
sh = st.slider("Jumlah Tembakan", 0, 30, step=1)
sot = st.slider("Tembakan Tepat Sasaran", 0, 20, step=1)
dist = st.slider("Jarak Rata-rata Tembakan", 0.0, 30.0, step=0.1)

team_enc = le_team.transform([team])[0]
opp_enc = le_opp.transform([opponent])[0]
venue_enc = le_venue.transform([venue])[0]

input_data = np.array([[team_enc, opp_enc, venue_enc, xG, xGA, poss, sh, sot, dist]])
input_scaled = scaler.transform(input_data)

if st.button("Prediksi Hasil Pertandingan"):
    prediction = model.predict(input_scaled)
    pred_label = le_result.inverse_transform(prediction)[0]
    st.success(f"Prediksi hasil pertandingan: **{pred_label}**")