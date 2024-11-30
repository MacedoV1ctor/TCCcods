import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
import time
# Supressão de avisos de convergência
#warnings.filterwarnings("ignore", category=UserWarning)

# Definindo a seed para reprodutibilidade
seed = 42
np.random.seed(seed)

# Carregando o CSV em um DataFrame
file_path = "C:\\Users\\Claudio e Victor\\PycharmProjects\\comparandoAlgoritmoPadrão\\dataset\\crops1.csv"
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

# Aplicar o RFE com o estimador RandomForest
estimator = RandomForestClassifier(random_state=seed)
rfe = RFE(estimator, n_features_to_select=len(features.columns) - 2)  # Remove 2 features
rfe.fit(features_scaled, labels)

# Features selecionadas pelo RFE
selected_features = rfe.support_
selected_columns = features.columns[selected_features]

# Atualizar o DataFrame com as features selecionadas (as features já estão normalizadas)
features_selected = features_scaled[:, selected_features]

# Definir o esquema de validação cruzada com 10 folds e shuffle
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# Instanciar os algoritmos com hiperparâmetros otimizados
random_forest_optimized = RandomForestClassifier(
    criterion='gini',
    max_depth=10,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=10,
    n_estimators=100,
    random_state=seed
)

svm_optimized = SVC(
    C=10,
    degree=2,
    gamma='scale',
    kernel='linear',
    probability=True,
    random_state=seed
)

mlp_optimized = MLPClassifier(
    activation='tanh',
    alpha=0.01,
    hidden_layer_sizes=(50, 50),
    learning_rate_init=0.0005,
    max_iter=200,
    solver='lbfgs',
    random_state=seed
)

# Função personalizada para avaliar as métricas de forma manual e medir o tempo de treinamento
def evaluate_model(model, X, y, cv):
    accuracies, f1s, precisions, recalls, mccs, aucs = [], [], [], [], [], []
    start_time = time.time()  # Início do cronômetro

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Para AUC, precisamos de probabilidades
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = None  # Não há suporte para AUC

        accuracies.append(np.mean(y_pred == y_test))
        f1s.append(f1_score(y_test, y_pred, average='macro'))
        precisions.append(precision_score(y_test, y_pred, average='macro'))
        recalls.append(recall_score(y_test, y_pred, average='macro'))
        mccs.append(matthews_corrcoef(y_test, y_pred))

        # Verificação se o modelo fornece probabilidades para AUC
        if y_proba is not None:
            try:
                aucs.append(roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'))
            except ValueError:  # Caso ocorra algum erro no cálculo do AUC
                aucs.append(np.nan)
        else:
            aucs.append(np.nan)

    end_time = time.time()  # Fim do cronômetro
    training_time = end_time - start_time

    return {
        'accuracy': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'f1_macro': np.mean(f1s),
        'f1_std': np.std(f1s),
        'precision_macro': np.mean(precisions),
        'precision_std': np.std(precisions),
        'recall_macro': np.mean(recalls),
        'recall_std': np.std(recalls),
        'mcc': np.mean(mccs),
        'mcc_std': np.std(mccs),
        'roc_auc': np.mean(aucs),
        'roc_auc_std': np.std(aucs),
        'training_time': training_time  # Adicionando tempo de treinamento
    }

# Avaliar e imprimir resultados de forma formatada
def print_results(model_name, results):
    print(f"\nAvaliação {model_name}:")
    print(f"Accuracy: {results['accuracy']:.4f} ± {results['accuracy_std']:.4f}")
    print(f"F1-Score: {results['f1_macro']:.4f} ± {results['f1_std']:.4f}")
    print(f"Precision: {results['precision_macro']:.4f} ± {results['precision_std']:.4f}")
    print(f"Recall: {results['recall_macro']:.4f} ± {results['recall_std']:.4f}")
    print(f"MCC: {results['mcc']:.4f} ± {results['mcc_std']:.4f}")
    print(f"AUC: {results['roc_auc']:.4f} ± {results['roc_auc_std']:.4f}")
    print(f"Tempo de Treinamento: {results['training_time']:.4f} segundos")

# Avaliar o RandomForest otimizado
rf_optimized_results = evaluate_model(random_forest_optimized, features_selected, labels, cv)
print_results("Random Forest Otimizado", rf_optimized_results)

# Avaliar o SVM otimizado
svm_optimized_results = evaluate_model(svm_optimized, features_selected, labels, cv)
print_results("SVM Otimizado", svm_optimized_results)

# Avaliar o MLP otimizado
mlp_optimized_results = evaluate_model(mlp_optimized, features_selected, labels, cv)
print_results("MLP Otimizado", mlp_optimized_results)
