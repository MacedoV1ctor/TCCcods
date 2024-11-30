import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt

import warnings

# Supressão de avisos de convergência
warnings.filterwarnings("ignore", category=UserWarning)

# Definindo a seed para reprodutibilidade
seed = 42
np.random.seed(seed)

# Carregando o CSV em um DataFrame
file_path = "C:\\Users\\Claudio e Victor\\PycharmProjects\\aplicandoPCA\\dataset\\crops1.csv"
df = pd.read_csv(file_path)

# Colocar a coluna da label no final do DataFrame
columns = list(df.columns)
columns.append(columns.pop(columns.index('label')))
df = df[columns]

# Separar as features da coluna 'label'
features = df.drop(columns=['label'])
labels = df['label']

# Normalizar as features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# Função para definir os modelos com diferentes números de features para Random Forest, SVM e MLP
def get_models():
    models = dict()
    # Random Forest
    for i in range(2, 9):
        rfe_rf = RFE(estimator=RandomForestClassifier(random_state=seed, n_jobs=-1), n_features_to_select=i)
        model_rf = RandomForestClassifier(random_state=seed)
        models[f'RF_{i}'] = Pipeline(steps=[('s', rfe_rf), ('m', model_rf)])

    # SVM
    for i in range(2, 9):
        rfe_svm = RFE(estimator=SVC(kernel='linear', random_state=seed), n_features_to_select=i)
        model_svm = SVC(random_state=seed)
        models[f'SVM_{i}'] = Pipeline(steps=[('s', rfe_svm), ('m', model_svm)])

    # MLP
    for i in range(2, 9):
        rfe_mlp = RFE(estimator=RandomForestClassifier(random_state=seed, n_jobs=-1),
                      n_features_to_select=i)  # Estimador padrão
        model_mlp = MLPClassifier(random_state=seed)
        models[f'MLP_{i}'] = Pipeline(steps=[('s', rfe_mlp), ('m', model_mlp)])

    return models


# Função para avaliar um modelo usando validação cruzada
def evaluate_model(model, X, y):
    cv = StratifiedKFold(n_splits=5)  # Reduzido para 5 folds para acelerar
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# Obtendo os modelos para avaliar
models = get_models()

# Avaliando os modelos e armazenando os resultados
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, features_scaled, labels)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))

# Plotando a performance dos modelos para comparação
plt.boxplot(results, labels=names, showmeans=True)
plt.xlabel('Modelo e Número de features selecionadas')
plt.ylabel('Acurácia')
plt.title('Comparação de Acurácia para Random Forest, SVM e MLP')
plt.xticks(rotation=90)  # Rotacionando os rótulos do eixo X para melhor visualização
plt.show()
