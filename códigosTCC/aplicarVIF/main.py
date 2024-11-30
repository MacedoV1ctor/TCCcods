import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

# Supressão de avisos de convergência
warnings.filterwarnings("ignore", category=UserWarning)

# Definindo a seed para reprodutibilidade
seed = 42
np.random.seed(seed)

# Definindo o caminho do arquivo
file_path = "C:\\Users\\Claudio e Victor\\PycharmProjects\\aplicandoPCA\\dataset\\crops1.csv"

# Carregando o CSV em um DataFrame
df = pd.read_csv(file_path)

print("Base de dados original:")
print(df.head())

# Colocar a coluna da label no final do DataFrame
columns = list(df.columns)
columns.append(columns.pop(columns.index('label')))
df = df[columns]

# Separar as features da coluna 'label'
features = df.drop(columns=['label'])
labels = df['label']

# Função para calcular o VIF e remover features com alto VIF de forma iterativa
def calculate_vif(features):
    vif_data = pd.DataFrame()
    vif_data["feature"] = features.columns
    vif_data["VIF"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
    return vif_data

def remove_high_vif(features, threshold=10.0):
    while True:
        vif_data = calculate_vif(features)
        print("\nValores de VIF antes da remoção:")
        print(vif_data)

        max_vif = vif_data['VIF'].max()
        if max_vif > threshold:
            max_vif_feature = vif_data[vif_data['VIF'] == max_vif]['feature'].values[0]
            print(f"Removendo feature com VIF alto: {max_vif_feature} (VIF = {max_vif:.2f})")
            features = features.drop(columns=[max_vif_feature])
            features = features.reset_index(drop=True)  # Reinicializar os índices após remoção
        else:
            break
    return features

# Calcular e remover VIF iterativamente para as features não normalizadas
print("\nAplicando VIF na base de dados não normalizada:")
features_vif_filtered = remove_high_vif(features)
features_vif_filtered_df = pd.DataFrame(features_vif_filtered, columns=features_vif_filtered.columns)
vif_data_final = calculate_vif(features_vif_filtered)
print("\nValores de VIF após a remoção:")
print(vif_data_final)

print("\nBase de dados após remoção de alto VIF (não normalizada):")
print(features_vif_filtered.head())
print(features_vif_filtered.columns)

# Normalizar a base de dados filtrada pelo VIF (não normalizada)
scaler_vif_filtered = StandardScaler()
features_vif_filtered_scaled = scaler_vif_filtered.fit_transform(features_vif_filtered)
features_vif_filtered_scaled_df = pd.DataFrame(features_vif_filtered_scaled, columns=features_vif_filtered.columns)

# Calcular o VIF para as features normalizadas
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

# Calcular e remover VIF iterativamente para as features normalizadas
print("\nAplicando VIF na base de dados normalizada:")
features_scaled_vif_filtered = remove_high_vif(features_scaled_df)
vif_scaled_data_final = calculate_vif(features_scaled_vif_filtered)
print("\nValores de VIF após a remoção:")
print(vif_scaled_data_final)

print("\nBase de dados após remoção de alto VIF (normalizada):")
print(features_scaled_vif_filtered.head())
print(features_scaled_vif_filtered.columns)

# Aplicar o PCA para reter cerca de 90% da variância dos dados
pca_90 = PCA(n_components=0.90, random_state=seed)
features_pca_90 = pca_90.fit_transform(features_scaled)

# Configurando os modelos
svm_model = SVC(random_state=seed)
rf_model = RandomForestClassifier(random_state=seed)
mlp_model = MLPClassifier(random_state=seed)

# Configurando a validação cruzada com 10 folds
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

precision_macro_scorer = make_scorer(precision_score, average='macro', zero_division=0)
recall_macro_scorer = make_scorer(recall_score, average='macro', zero_division=0)
f1_macro_scorer = make_scorer(f1_score, average='macro', zero_division=0)

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

    print(f'\n{model_name} - Desempenho no Conjunto de Treino:')
    print(f'Acurácia média: {scores["train_accuracy"].mean():.4f}')
    print(f'Precisão média: {scores["train_precision_macro"].mean():.4f}')
    print(f'Recall médio: {scores["train_recall_macro"].mean():.4f}')
    print(f'F1-score médio: {scores["train_f1_macro"].mean():.4f}')

    print(f'\n{model_name} - Desempenho no Conjunto de Teste:')
    print(f'Acurácia média: {scores["test_accuracy"].mean():.4f}')
    print(f'Precisão média: {scores["test_precision_macro"].mean():.4f}')
    print(f'Recall médio: {scores["test_recall_macro"].mean():.4f}')
    print(f'F1-score médio: {scores["test_f1_macro"].mean():.4f}')

def check_overfitting(model, model_name, features, labels):
    print(f"\n{'='*10} Verificação de Overfitting para {model_name} {'='*10}")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=seed,
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

# Avaliação dos modelos nas diferentes versões dos dados

# SVM
# evaluate_model(svm_model, "SVM na base original (não normalizada)", features, labels)
# evaluate_model(svm_model, "SVM na base normalizada", features_scaled, labels)
# evaluate_model(svm_model, "SVM na base com VIF aplicado", features_vif_filtered_scaled_df, labels)
#evaluate_model(svm_model, "SVM na base PCA", features_pca_90, labels)

# check_overfitting(svm_model, "SVM na base original (não normalizada)", features, labels)
# check_overfitting(svm_model, "SVM na base normalizada", features_scaled, labels)
# check_overfitting(svm_model, "SVM na base com VIF aplicado", features_vif_filtered_scaled_df, labels)
#
# evaluate_model(svm_model, "SVM na base com VIF aplicado não normalizado", features_vif_filtered_df, labels)
# check_overfitting(svm_model, "SVM na base com VIF aplicado não normalizadp", features_vif_filtered_df, labels)

#check_overfitting(svm_model, "SVM na base PCA", features_pca_90, labels)


# # Random Forest
# evaluate_model(rf_model, "Random Forest na base original (não normalizada)", features, labels)
# evaluate_model(rf_model, "Random Forest na base normalizada", features_scaled, labels)
# evaluate_model(rf_model, "Random Forest na base com VIF aplicado", features_vif_filtered_scaled_df, labels)

# check_overfitting(rf_model, "Random Forest na base original (não normalizada)", features, labels)
# check_overfitting(rf_model, "Random Forest na base normalizada", features_scaled, labels)
# check_overfitting(rf_model, "Random Forest na base com VIF aplicado", features_vif_filtered_scaled_df, labels)
#check_overfitting(rf_model, "Random Forest na base PCA", features_pca_90, labels)
#
# evaluate_model(rf_model, "RF na base com VIF aplicado não normalizado", features_vif_filtered_df, labels)
# check_overfitting(rf_model, "RF na base com VIF aplicado não normalizadp", features_vif_filtered_df, labels)


# # MLP
evaluate_model(mlp_model, "MLP na base original (não normalizada)", features, labels)
evaluate_model(mlp_model, "MLP na base normalizada", features_scaled, labels)
evaluate_model(mlp_model, "MLP na base com VIF aplicado", features_vif_filtered_scaled_df, labels)
#
check_overfitting(mlp_model, "MLP na base original (não normalizada)", features, labels)
check_overfitting(mlp_model, "MLP na base normalizada", features_scaled, labels)
check_overfitting(mlp_model, "MLP na base com VIF aplicado", features_vif_filtered_scaled_df, labels)
#
evaluate_model(mlp_model, "MLP na base com VIF aplicado não normalizado", features_vif_filtered_df, labels)
check_overfitting(mlp_model, "MLP na base com VIF aplicado não normalizadp", features_vif_filtered_df, labels)