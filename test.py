import streamlit as st

st.title("Aplicação Simples com Streamlit")

nome = st.text_input("Digite seu nome:")

if st.button("Saudar"):
    st.write(f"Olá, {nome}! Bem-vindo à aplicação Streamlit.")