import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score
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

# Atualizar a grade de parâmetros para o SVC
param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Parâmetro de regularização
    'kernel': ['linear', 'rbf', 'poly'],  # Tipos de kernel
    'degree': [2, 3, 5],  # Apenas relevante para o kernel polinomial
    'gamma': ['scale', 'auto', 0.01, 0.1]  # Coeficiente para kernels 'rbf', 'poly'
}

# Criar métricas personalizadas com zero_division ajustado
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
    'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
    'f1_macro': make_scorer(f1_score, average='macro', zero_division=0)
}

# Instanciar o SVC
svc = SVC(random_state=seed, probability=True)  # Ativar probabilidade para ROC AUC

# Instanciar o GridSearchCV
grid_search = GridSearchCV(
    estimator=svc,
    param_grid=param_grid,
    cv=cv,
    scoring=scoring,  # Avaliando múltiplas métricas
    refit='f1_macro',  # Refitar o modelo com base na métrica f1_macro
    # n_jobs=-1,  # Usar todos os núcleos disponíveis
    verbose=1
)

# Iniciar o temporizador
start_time = time.time()

# Executar o GridSearchCV
grid_search.fit(features_selected_normalized, labels)

# Parar o temporizador
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Tempo total para o GridSearchCV: {elapsed_time:.2f} segundos")

# Exibir o melhor conjunto de hiperparâmetros
print(f"Melhores hiperparâmetros: {grid_search.best_params_}")

# Exibir o melhor estimador
print(f"Melhor estimador: {grid_search.best_estimator_}")

# Exibir as métricas de desempenho
print(f"Melhor F1-score: {grid_search.best_score_}")
