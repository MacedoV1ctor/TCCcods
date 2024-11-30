import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
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

# Definir o esquema de validação cruzada com 5 folds
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# Definir a grade de parâmetros para otimização do SVM
param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Parâmetro de regularização
    'kernel': ['linear', 'rbf', 'poly'],  # Tipos de kernel
    'degree': [2, 3, 5],  # Apenas relevante para o kernel polinomial
    'gamma': ['scale', 'auto', 0.01, 0.1]  # Coeficiente para kernels 'rbf', 'poly'
}

# Definir as métricas personalizadas com zero_division=1
scoring = {
    'accuracy': 'accuracy',
    'precision_macro': make_scorer(precision_score, average='macro'),
    'recall_macro': make_scorer(recall_score, average='macro'),
    'f1_macro': make_scorer(f1_score, average='macro')
}

# Instanciar o SVC (Suport Vector Classifier)
svm = SVC(random_state=seed)

# Instanciar o GridSearchCV
grid_search = GridSearchCV(
    estimator=svm,
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
grid_search.fit(features_selected, labels)

# Parar o temporizador
end_time = time.time()

# Calcular e exibir o tempo decorrido
elapsed_time = end_time - start_time
print(f"Tempo total para o GridSearchCV: {elapsed_time:.2f} segundos")

# Exibir o melhor conjunto de hiperparâmetros
print(f"Melhores hiperparâmetros: {grid_search.best_params_}")

# Exibir o melhor estimador
print(f"Melhor estimador: {grid_search.best_estimator_}")

# Exibir as métricas de desempenho
print(f"Melhor F1-score: {grid_search.best_score_}")
