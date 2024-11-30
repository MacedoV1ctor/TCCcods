import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time  # Importando o módulo time
import warnings

# Supressão de avisos
#warnings.filterwarnings("ignore", category=UserWarning)

# Definindo a seed para reprodutibilidade
seed = 42
np.random.seed(seed)

# Carregando o CSV em um DataFrame
file_path = "C:\\Users\\Claudio e Victor\\PycharmProjects\\comparandoAlgoritmoPadrão\\dataset\\crops1.csv"
df = pd.read_csv(file_path)

# Converter a coluna 'label' para valores numéricos usando LabelEncoder
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Separar as features da coluna 'label'
features = df.drop(columns=['label'])
labels = df['label']

# Aplicar o RFE com o estimador RandomForest na base não normalizada
estimator = RandomForestClassifier(random_state=seed)
rfe = RFE(estimator, n_features_to_select=len(features.columns) - 2)  # Remove 2 features
rfe.fit(features, labels)

# Features selecionadas pelo RFE
selected_features = rfe.support_
selected_columns = features.columns[selected_features]

# Atualizar o DataFrame com as features selecionadas (não normalizadas)
features_selected = features[selected_columns]

# Verificar por valores NaN ou infinitos após a seleção de features
print("Existem valores NaN no conjunto de features selecionadas?", np.any(np.isnan(features_selected)))
print("Existem valores infinitos no conjunto de features selecionadas?", np.any(np.isinf(features_selected)))

# Substituir valores infinitos por NaN e remover NaN se existirem
# features_selected.replace([np.inf, -np.inf], np.nan, inplace=True)
# features_selected.dropna(inplace=True)

# Verificar tipos de dados
print(features_selected.dtypes)

# Certificar que todas as colunas são numéricas (inteiros ou floats)
features_selected = features_selected.apply(pd.to_numeric, errors='coerce')

# Aplicar normalização às features selecionadas
scaler = StandardScaler()
features_selected_normalized = scaler.fit_transform(features_selected)

# Definir o esquema de validação cruzada com 5 folds
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# Definir a grade de parâmetros otimizados, reduzindo as opções para acelerar
param_grid = {
    'n_estimators': [100, 200, 300],  # Número de árvores
    'max_depth': [10, 20, None],  # Profundidade máxima da árvore
    'min_samples_split': [2, 5, 10],  # Número mínimo de amostras para dividir um nó
    'min_samples_leaf': [1, 2, 4],  # Número mínimo de amostras em cada folha
    'max_features': [None, 'sqrt', 'log2'],  # Corrigido para usar None, 'sqrt', ou 'log2'
    'criterion': ['gini', 'entropy'],  # Critério de impureza para medir a qualidade da divisão artigo
}

# Definir as métricas para avaliação
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# Instanciar o RandomForestClassifier
random_forest = RandomForestClassifier(random_state=seed)

# Instanciar o GridSearchCV
grid_search = GridSearchCV(
    estimator=random_forest,
    param_grid=param_grid,
    cv=cv,
    scoring=scoring,  # Avaliando múltiplas métricas
    refit='f1_macro',  # Refitar o modelo com base na métrica f1_macro
    # n_jobs=-1,  # Usar todos os núcleos disponíveis
    verbose=1
)

start_time = time.time() #Iniciar temporizador

# Executar o GridSearchCV
grid_search.fit(features_selected_normalized, labels)

end_time = time.time() #Finalizar o temporizador

elapsed_time = end_time - start_time
print(f"Tempo total para o GridSearchCV: {elapsed_time:.2f} segundos")

# Exibir o melhor conjunto de hiperparâmetros
print(f"Melhores hiperparâmetros: {grid_search.best_params_}")

# Exibir o melhor estimador
print(f"Melhor estimador: {grid_search.best_estimator_}")

# Exibir as métricas de desempenho
print(f"Melhor F1-score: {grid_search.best_score_}")
