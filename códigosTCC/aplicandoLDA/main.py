
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, make_scorer
import matplotlib.pyplot as plt
import warnings

# Supressão de avisos de convergência
warnings.filterwarnings("ignore", category=UserWarning)

# Carregar o dataset
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

# Aplicar o PCA para reter cerca de 90% da variância dos dados (para a avaliação)
pca_90 = PCA(n_components=0.90)
features_pca_90 = pca_90.fit_transform(features_scaled)

# Aplicar o LDA completo para avaliação
lda_full = LDA()
features_lda_full = lda_full.fit_transform(features_scaled, labels)
LDAnaoNomalizada = lda_full.fit_transform(features, labels)

# Identificar as colunas removidas após LDA
removed_columns = set(features.columns) - set(features.columns[:features_lda_full.shape[1]])
print(f"Colunas removidas após aplicação do LDA: {removed_columns}")

print("Colunas base de dados LDA aplicado: ", features_lda_full.shape[1])
print("Colunas base de dados PCA aplicado: ", features_pca_90.shape[1])

# Mostrar a base de dados antes e depois da aplicação do LDA
print("\nBase de dados original (primeiras 5 linhas):")
print(df.head())

# Criar um DataFrame para a base de dados transformada pelo LDA
df_lda = pd.DataFrame(features_lda_full)
df_lda['label'] = labels.values

print("\nBase de dados após aplicação do LDA (primeiras 5 linhas):")
print(df_lda.head())

# Aplicar o PCA e LDA para visualização em 2D
pca_2d = PCA(n_components=2)
features_pca_2d = pca_2d.fit_transform(features_scaled)

lda_2d = LDA(n_components=2)
features_lda_2d = lda_2d.fit_transform(features_scaled, labels)

# Configurando os modelos com random_state
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
    print("="*40)

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



# print("\n### SVM ###")
# evaluate_model(svm_model, "SVM na base original (não normalizada)", features, labels)
# evaluate_model(svm_model, "SVM na base original (normalizada)", features_scaled, labels)
# evaluate_model(svm_model, "SVM na base PCA (90% da variância)", features_pca_90, labels)
# evaluate_model(svm_model, "SVM na base LDA completo", features_lda_full, labels)
#
# check_overfitting(svm_model, "SVM na base original (não normalizada)", features, labels)
# check_overfitting(svm_model, "SVM na base original (normalizada)", features_scaled, labels)
# check_overfitting(svm_model, "SVM na base PCA (90% da variância)", features_pca_90, labels)
# check_overfitting(svm_model, "SVM na base LDA completo", features_lda_full, labels)
#

# Avaliação dos modelos nas diferentes versões dos dados
# evaluate_model(rf_model, "Random Forest na base original (não normalizada)", features, labels)
# evaluate_model(rf_model, "Random Forest na base original (normalizada)", features_scaled, labels)
# # evaluate_model(rf_model, "Random Forest na base PCA (90% da variância)", features_pca_90, labels)
# evaluate_model(rf_model, "Random Forest na base LDA completo", features_lda_full, labels)
#
# check_overfitting(rf_model, "Random Forest na base original (não normalizada)", features, labels)
# check_overfitting(rf_model, "Random Forest na base original (normalizada)", features_scaled, labels)
# # check_overfitting(rf_model, "Random Forest na base PCA (90% da variância)", features_pca_90, labels)
# check_overfitting(rf_model, "Random Forest na base LDA completo", features_lda_full, labels)

#
# print("\n### MLP (Perceptron de Múltiplas Camadas) ###")
evaluate_model(mlp_model, "MLP na base original (não normalizada)", features, labels)
evaluate_model(mlp_model, "MLP na base original (normalizada)", features_scaled, labels)
#evaluate_model(mlp_model, "MLP na base PCA (90% da variância)", features_pca_90, labels)
evaluate_model(mlp_model, "MLP na base LDA completo", features_lda_full, labels)

check_overfitting(mlp_model, "MLP na base original (não normalizada)", features, labels)
check_overfitting(mlp_model, "MLP na base original (normalizada)", features_scaled, labels)
#check_overfitting(mlp_model, "MLP na base PCA (90% da variância)", features_pca_90, labels)
check_overfitting(mlp_model, "MLP na base LDA completo", features_lda_full, labels)

#TESTANDO LDA NA BASE DE DADOS NÃO NORMALIZADA:
# print("LDA Não normalizado teste")
# evaluate_model(svm_model, "SVM na base LDA N.N (CRUZADO)", LDAnaoNomalizada, labels)
# check_overfitting(svm_model, "SVM na base N.N completo (NORMAL)", LDAnaoNomalizada, labels)
#
#
# print("RF Não normalizado teste")
# evaluate_model(rf_model, "RF na base LDA N.N (CRUZADO)", LDAnaoNomalizada, labels)
# check_overfitting(rf_model, "RF na base N.N completo (NORMAL)", LDAnaoNomalizada, labels)


# print("MLP Não normalizado teste")
evaluate_model(mlp_model, "MLP na base LDA N.N (CRUZADO)", LDAnaoNomalizada, labels)
check_overfitting(mlp_model, "MLP na base N.N completo (NORMAL)", LDAnaoNomalizada, labels)


# Função para plotar gráficos de PCA e LDA
# def plot_comparison(features_pca, features_lda, labels):
#     plt.figure(figsize=(14, 6))
#
#     # Plot PCA
#     plt.subplot(1, 2, 1)
#     for label in set(labels):
#         plt.scatter(features_pca[labels == label, 0], features_pca[labels == label, 1], label=label)
#     plt.title('PCA - 2D Visualization')
#     plt.xlabel('PC1')
#     plt.ylabel('PC2')
#     plt.legend(loc='best')  # Legenda posicionada automaticamente

    # Plot LDA
    # plt.subplot(1, 2, 2)
    # for label in set(labels):
    #     plt.scatter(features_lda[labels == label, 0], features_lda[labels == label, 1], label=label)
    # plt.title('LDA - 2D Visualization')
    # plt.xlabel('LD1')
    # plt.ylabel('LD2')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Legenda à direita do gráfico
    #
    # plt.show()

# print("\n==== Iniciando Visualização 2D ====\n")
# # Plotar a comparação
# plot_comparison(features_pca_2d, features_lda_2d, labels)
# print("\n==== Visualização Concluída ====\n")
