import pandas as pd
from sklearn.preprocessing import StandardScaler

# Carregando o CSV em um DataFrame
file_path = "C:\\Users\\Claudio e Victor\\PycharmProjects\\comparandoAlgoritmoPadrão\\dataset\\crops1.csv"
df = pd.read_csv(file_path)

# Reorganizar as colunas para que 'label' esteja no final
columns = [col for col in df.columns if col != 'label'] + ['label']
df = df[columns]

# Exibir as 5 primeiras colunas da base de dados antes da normalização
print("Base de dados sem normalização (5 primeiras colunas):")
print(df.iloc[:5, :5])

# Separar as features da coluna 'label'
features = df.drop(columns=['label'])
labels = df['label']

# Normalizar as features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Criar um DataFrame com as features normalizadas e adicionar a coluna 'label' de volta
df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
df_scaled['label'] = labels.values

# Exibir as 5 primeiras colunas da base de dados após a normalização
print("\nBase de dados com normalização (StandardScaler) - 5 primeiras colunas:")
print(df_scaled.iloc[:5, :5])

print(df_scaled['K'])
