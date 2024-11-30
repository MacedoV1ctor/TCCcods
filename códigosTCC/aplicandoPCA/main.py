import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, make_scorer
from sklearn.utils.multiclass import unique_labels
import warnings

# Supressão de avisos de convergência
warnings.filterwarnings("ignore", category=UserWarning)

# Definindo o caminho do arquivo
file_path = "C:\\Users\\Claudio e Victor\\PycharmProjects\\aplicandoPCA\\dataset\\crops1.csv"

# Carregando o CSV em um DataFrame
df = pd.read_csv(file_path)

print(df.head())

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

# Aplicar o PCA para reter cerca de 90% da variância dos dados
pca_90 = PCA(n_components=0.90, random_state=42)
features_pca_90 = pca_90.fit_transform(features_scaled)

# Criar um DataFrame com os componentes principais
pca_columns = [f'PC{i+1}' for i in range(features_pca_90.shape[1])]
df_pca_90 = pd.DataFrame(features_pca_90, columns=pca_columns)

print(df_pca_90);

# Base de dados não normalizada
naoNormalizadaPCA = pca_90.fit_transform(features)

# Configurando os modelos com ajustes para o MLP
svm_model = SVC(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
mlp_model = MLPClassifier(random_state=42)  # Aumentando max_iter para 1000

# Configurando a validação cruzada com 10 folds
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Criando scorers personalizados para validação cruzada, ajustando para evitar divisão por zero no SVM e MLP
precision_macro_scorer = make_scorer(precision_score, average='macro', zero_division=0)
recall_macro_scorer = make_scorer(recall_score, average='macro', zero_division=0)
f1_macro_scorer = make_scorer(f1_score, average='macro', zero_division=0)


# Função para realizar a validação cruzada e imprimir os resultados
def evaluate_model(model, model_name, features, labels):
    print(f"\n{'='*10} Resultados da Validação Cruzada para {model_name} {'='*10}")
    # Ajustando para usar zero_division=0 em 'precision_macro' e 'recall_macro' para evitar o problema de divisão por zero
    scores = cross_validate(model, features, labels, cv=cv,
                            scoring={
                                'accuracy': 'accuracy',
                                'precision_macro': precision_macro_scorer,
                                'recall_macro': recall_macro_scorer,
                                'f1_macro': f1_macro_scorer
                            },
                            return_train_score=True,  # Incluir os resultados do treino
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
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42,
                                                        stratify=labels)
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

    # # Análise detalhada das classes que não estão sendo previstas
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

# #SVM
# evaluate_model(svm_model, "SVM na base original (não normalizada)", features, labels)
# evaluate_model(svm_model, "SVM na base original (normalizada)", features_scaled, labels)
# evaluate_model(svm_model, "SVM na base PCA", features_pca_90, labels)
#
# check_overfitting(svm_model, "SVM na base original (não normalizada)", features, labels)
# check_overfitting(svm_model, "SVM na base original (normalizada)", features_scaled, labels)
# check_overfitting(svm_model, "SVM na base PCA", features_pca_90, labels)
#
# Random Forest
# evaluate_model(rf_model, "Random Forest na base original (não normalizada)", features, labels)
# evaluate_model(rf_model, "Random Forest na base original (normalizada)", features_scaled, labels)
# evaluate_model(rf_model, "Random Forest na base PCA", features_pca_90, labels)
#
# check_overfitting(rf_model, "Random Forest na base original (não normalizada)", features, labels)
# check_overfitting(rf_model, "Random Forest na base original (normalizada)", features_scaled, labels)
# check_overfitting(rf_model, "Random Forest na base PCA", features_pca_90, labels)

#MLP (Perceptron de Múltiplas Camadas)
evaluate_model(mlp_model, "MLP na base original (não normalizada)", features, labels)
evaluate_model(mlp_model, "MLP na base original (normalizada)", features_scaled, labels)
evaluate_model(mlp_model, "MLP na base PCA", features_pca_90, labels)

check_overfitting(mlp_model, "MLP na base original (não normalizada)", features, labels)
check_overfitting(mlp_model, "MLP na base original (normalizada)", features_scaled, labels)
check_overfitting(mlp_model, "MLP na base PCA", features_pca_90, labels)
#
#SVM PCA não normalizado
# evaluate_model(svm_model, "SVM na base PCA não normalizado (cruzado)", naoNormalizadaPCA, labels)
# check_overfitting(svm_model, "SVM na base PCA não normalizado (padrão)", naoNormalizadaPCA, labels)
#
#
# #RF PCA não normalizado
# evaluate_model(rf_model, "RF na base PCA não normalizado (cruzado)", naoNormalizadaPCA, labels)
# check_overfitting(rf_model, "RF na base PCA não normalizado (padrão)", naoNormalizadaPCA, labels)
# #
# # PCA NA BASE NÃO NORMALIZADA
# print("PCA NA BASE NORMALIZADA")
evaluate_model(mlp_model, "MLP na base PCA não normalizada (cruzada)", naoNormalizadaPCA, labels)
check_overfitting(mlp_model, "MLP na base PCA não normalizada (padrão)", naoNormalizadaPCA, labels)
