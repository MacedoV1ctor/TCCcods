import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, make_scorer
import warnings

# Supressão de avisos de convergência
warnings.filterwarnings("ignore", category=UserWarning)

# Definindo o caminho do arquivo
file_path = "C:\\Users\\Claudio e Victor\\PycharmProjects\\aplicandoPCA\\dataset\\crops1.csv"

# Carregando o CSV em um DataFrame
df = pd.read_csv(file_path)

# Colocar a coluna da label no final do DataFrame
columns = list(df.columns)
columns.append(columns.pop(columns.index('label')))
df = df[columns]

# Separar as features da coluna 'label'
features = df.drop(columns=['label'])
labels = df['label']

# 1. Heatmap da correlação de Pearson na base completa não normalizada

print("Colunas antes da normalização:")
print(features.columns)
print("\nDados antes da normalização:")
print(features.head())

corr_matrix = features.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap da Matriz de Correlação Original (Não Normalizada)')
plt.show()

# 2. Normalizar as features e exibir o heatmap da correlação de Pearson na base normalizada
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Converta o array normalizado de volta para um DataFrame, mantendo as colunas originais
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

# Agora você pode visualizar as colunas e os valores normalizados
print("\nColunas após a normalização:")
print(features_scaled_df.columns)

print("\nDados após a normalização:")
print(features_scaled_df.head())


normalized_corr_matrix = pd.DataFrame(features_scaled, columns=features.columns).corr()

plt.figure(figsize=(12, 10))
sns.heatmap(normalized_corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap da Matriz de Correlação (Base Normalizada)')
plt.show()

# 3. Remover features com correlação maior que 0.79 e exibir o heatmap
threshold = 0.79
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
features_reduced = features.drop(columns=to_drop)

# Exibindo o heatmap após a remoção de features altamente correlacionadas
reduced_corr_matrix = features_reduced.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(reduced_corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap da Matriz de Correlação Após Remoção de Features Altamente Correlacionadas')
plt.show()

# Normalizar as features reduzidas e exibir o heatmap
features_reduced_scaled = scaler.fit_transform(features_reduced)

normalized_reduced_corr_matrix = pd.DataFrame(features_reduced_scaled, columns=features_reduced.columns).corr()

plt.figure(figsize=(12, 10))
sns.heatmap(normalized_reduced_corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap da Matriz de Correlação das Features Normalizadas e Reduzidas')
plt.show()

# Configurando os modelos com ajustes para o MLP
svm_model = SVC(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
mlp_model = MLPClassifier(random_state=42)

# Configurando a validação cruzada com 10 folds
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Criando scorers personalizados para validação cruzada
precision_macro_scorer = make_scorer(precision_score, average='macro', zero_division=0)
recall_macro_scorer = make_scorer(recall_score, average='macro', zero_division=0)
f1_macro_scorer = make_scorer(f1_score, average='macro', zero_division=0)

# Função para realizar a validação cruzada e imprimir os resultados
def evaluate_model(model, model_name, features, labels):
    print(f"\n{'='*10} Resultados da Validação Cruzada para {model_name} {'='*10}")
    scores = cross_validate(model, features, labels, cv=cv,
                            scoring={
                                'accuracy': 'accuracy',
                                'precision_macro': precision_macro_scorer,
                                'recall_macro': recall_macro_scorer,
                                'f1_macro': f1_macro_scorer
                            },
                            return_train_score=True,
                            error_score='raise')

    # Exibindo os resultados de treino
    print(f'\n{model_name} - Desempenho no Conjunto de Treino:')
    print(f'Acurácia média: {scores["train_accuracy"].mean():.4f}')
    print(f'Precisão média: {scores["train_precision_macro"].mean():.4f}')
    print(f'Recall médio: {scores["train_recall_macro"].mean():.4f}')
    print(f'F1-score médio: {scores["train_f1_macro"].mean():.4f}')

    # Exibindo os resultados de teste
    print(f'\n{model_name} - Desempenho no Conjunto de Teste:')
    print(f'Acurácia média: {scores["test_accuracy"].mean():.4f}')
    print(f'Precisão média: {scores["test_precision_macro"].mean():.4f}')
    print(f'Recall médio: {scores["test_recall_macro"].mean():.4f}')
    print(f'F1-score médio: {scores["test_f1_macro"].mean():.4f}')

# Função para verificar overfitting comparando treino e teste
def check_overfitting(model, model_name, features, labels):
    print(f"\n{'='*10} Verificação de Overfitting para {model_name} {'='*10}")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    train_precision = precision_score(y_train, model.predict(X_train), average='macro', zero_division=0)
    test_precision = precision_score(y_test, model.predict(X_test), average='macro', zero_division=0)
    train_recall = recall_score(y_train, model.predict(X_train), average='macro', zero_division=0)
    test_recall = recall_score(y_test, model.predict(X_test), average='macro', zero_division=0)
    train_f1 = f1_score(y_train, model.predict(X_train), average='macro', zero_division=0)
    test_f1 = f1_score(y_test, model.predict(X_test), average='macro', zero_division=0)

    print(f'Acurácia no Treino: {train_acc:.4f}, Acurácia no Teste: {test_acc:.4f}, Diferença: {train_acc - test_acc:.4f}')
    print(f'Precisão no Treino: {train_precision:.4f}, Precisão no Teste: {test_precision:.4f}')
    print(f'Recall no Treino: {train_recall:.4f}, Recall no Teste: {test_recall:.4f}')
    print(f'F1-score no Treino: {train_f1:.4f}, F1-score no Teste: {test_f1:.4f}')

    # print(f"\nAnálise detalhada das previsões para {model_name}:")
    # report = classification_report(y_test, model.predict(X_test), zero_division=0)
    # print(report)
    #
    # # Identificar classes não previstas no conjunto de teste
    # predicted_classes = np.unique(model.predict(X_test))
    # missing_classes = set(np.unique(y_test)) - set(predicted_classes)
    #
    # if missing_classes:
    #     print(f"Classes não previstas no conjunto de teste: {missing_classes}")
    #     for cls in missing_classes:
    #         print(f"Classe {cls} tem {np.sum(y_test == cls)} amostras no conjunto de teste.")
    # else:
    #     print("Todas as classes foram previstas no conjunto de teste.")

# Avaliação dos modelos nas diferentes versões dos dados

# SVM
# evaluate_model(svm_model, "SVM na base original (não normalizada)", features, labels)
# evaluate_model(svm_model, "SVM na base original (normalizada)", features_scaled, labels)
# evaluate_model(svm_model, "SVM com features reduzidas", features_reduced_scaled, labels)
#
# check_overfitting(svm_model, "SVM na base original (não normalizada)", features, labels)
# check_overfitting(svm_model, "SVM na base original (normalizada)", features_scaled, labels)
# check_overfitting(svm_model, "SVM com features reduzidas", features_reduced_scaled, labels)

# # Random Forest
# evaluate_model(rf_model, "Random Forest na base original (não normalizada)", features, labels)
# evaluate_model(rf_model, "Random Forest na base original (normalizada)", features_scaled, labels)
# evaluate_model(rf_model, "Random Forest com features reduzidas", features_reduced_scaled, labels)
#
# check_overfitting(rf_model, "Random Forest na base original (não normalizada)", features, labels)
# check_overfitting(rf_model, "Random Forest na base original (normalizada)", features_scaled, labels)
# check_overfitting(rf_model, "Random Forest com features reduzidas", features_reduced_scaled, labels)

# MLP (Perceptron de Múltiplas Camadas)
evaluate_model(mlp_model, "MLP na base original (não normalizada)", features, labels)
evaluate_model(mlp_model, "MLP na base original (normalizada)", features_scaled, labels)
evaluate_model(mlp_model, "MLP com features reduzidas", features_reduced_scaled, labels)

check_overfitting(mlp_model, "MLP na base original (não normalizada)", features, labels)
check_overfitting(mlp_model, "MLP na base original (normalizada)", features_scaled, labels)
check_overfitting(mlp_model, "MLP com features reduzidas", features_reduced_scaled, labels)


# Avaliação do RF na base reduzida não normalizada após correlação de Pearson
# evaluate_model(svm_model, "SVM Pearson N.N (cruzado)", features_reduced, labels)
# check_overfitting(svm_model, "SVM Pearson N.N (padrão)", features_reduced, labels)

# evaluate_model(rf_model, "RF Pearson N.N (cruzado)", features_reduced, labels)
# check_overfitting(rf_model, "RF Pearson N.N (padrão)", features_reduced, labels)

evaluate_model(mlp_model, "MLP Pearson N.N (cruzado)", features_reduced, labels)
check_overfitting(mlp_model, "MLP Pearson N.N (padrão)", features_reduced, labels)
