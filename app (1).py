import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="House Price Prediction")
st.title("🏠 House Price Prediction using ML")

st.image(
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRvQdqIasHkDTf5733FK14z5mPQ18VPhg_R_Q&s"
)

# Load dataset
df = pd.read_csv("house_data.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Scaling
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

st.sidebar.title("Select House Features")
st.sidebar.image(
    "https://media.tenor.com/Dyg_gZa4Vl4AAAAM/for-sale.gif"
)

user_input = []

for col in X.columns:
    min_val = float(X[col].min())
    max_val = float(X[col].max())
    val = st.sidebar.slider(
        f"{col}",
        min_value=min_val,
        max_value=max_val,
        value=min_val
    )
    user_input.append(val)

st.write("### Selected Values")
st.write(user_input)
