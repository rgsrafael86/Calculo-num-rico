import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

st.title("📊 Análise de Produção x Diâmetro")

# --- Entrada de dados ---
st.subheader("Entrada de dados")

dados_iniciais = pd.DataFrame({
    "Produção (unidades)": [0, 9077, 20443, 32649, 47798, 69231],
    "Diâmetro (mm)": [190.70, 188.20, 186.35, 183.88, 181.40, 178.20]
})

tabela_editada = st.data_editor(dados_iniciais, num_rows="dynamic")

X = tabela_editada["Produção (unidades)"].to_numpy()
Y = tabela_editada["Diâmetro (mm)"].to_numpy()

# --- Regressão linear ---
result = linregress(X, Y)
slope = result.slope
intercept = result.intercept
r_value = result.rvalue
p_value = result.pvalue
std_err = result.stderr

# Classificação da correlação
if r_value > 0.7:
    classificacao = "Altamente positiva"
elif r_value > 0.3:
    classificacao = "Positiva"
elif r_value > -0.3:
    classificacao = "Sem correlação linear"
elif r_value > -0.7:
    classificacao = "Negativa"
else:
    classificacao = "Altamente negativa"

# --- Resultados formatados ---
st.subheader("Resultados da regressão e correlação")

st.write(f"**Equação da reta:** Y = {intercept:.2f} + ({slope:.6f}) * X")
st.write(f"**Correlação (r):** {r_value:.3f} | **p-valor:** {p_value:.3e} | **erro do slope:** {std_err:.6f}")
st.write(f"**Classificação da correlação:** {classificacao}")

# --- Previsão para diâmetro alvo ---
Y_target = st.number_input("Defina o diâmetro alvo (mm)", value=168.0)
X_target = (Y_target - intercept) / slope
st.write(f"📌 Produção necessária para atingir {Y_target:.1f} mm: **{X_target:.0f} unidades**")

# --- Gráfico ---
st.subheader("Gráfico dos pontos e reta ajustada")
Y_pred = intercept + slope * X

plt.figure(figsize=(8,5))
plt.scatter(X, Y, color='blue', label='Dados coletados')
plt.plot(X, Y_pred, color='red', label='Regressão linear')
plt.scatter(X_target, Y_target, color='green', marker='x', s=120, label=f"Alvo: {Y_target} mm")
plt.text(X_target, Y_target+1, f"{int(X_target)} un", color='green')

plt.xlabel("Produção (unidades)")
plt.ylabel("Diâmetro (mm)")
plt.title("Produção vs Diâmetro: regressão linear e alvo")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

st.pyplot(plt)
