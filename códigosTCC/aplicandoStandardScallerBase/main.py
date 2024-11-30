file_path = "C:\\Users\\Claudio e Victor\\PycharmProjects\\comparandoAlgoritmoPadrão\\dataset\\crops1.csv"
df = pd.read_csv(file_path)import pandas as pd
from sklearn.preprocessing import StandardScaler

# Carrega a base de dados
file_path = "C:\\Users\\Claudio e Victor\\PycharmProjects\\comparandoAlgoritmoPadrão\\dataset\\crops1.csv"
df = pd.read_csv(file_path)

# Exibe a base de dados antes da normalização
print("Base de dados sem normalização:")
print(df.head())

# Aplica a normalização com o StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Exibe a base de dados após a normalização
print("\nBase de dados com normalização (StandardScaler):")
print(df_scaled.head())
