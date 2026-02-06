import streamlit as st

st.title("AI Electronics Recommender")

product = st.selectbox("Choose a Product", df["name"].tolist())

if st.button("Recommend"):
    results = recommend(product)
    st.dataframe(results)
