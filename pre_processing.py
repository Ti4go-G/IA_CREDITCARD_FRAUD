import random
import pandas as pd


def pre_process(
        file: str = "./creditcard.csv"
):
    df = pd.read_csv(f"{file}")
    # Muda o nome de Class para is_fraud
    df.rename(columns={'Class': 'is_fraud'}, inplace=True)
    # Retira valores duplicados
    df = df.drop_duplicates()
    # Remove coluna 'Time' se existir
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])

    # Separa features (todas as colunas exceto 'is_fraud') e alvo
    X = df.drop(columns=['is_fraud']).values.tolist()
    y = df['is_fraud'].values.tolist()

    # Normaliza as features (Min-Max por coluna)
    X_normalized = normalizar_minmax(X)

    # Gera folds de cross-validation embaralhados
    folds = cross_validation_shuffle(X_normalized, y)
    print(len(folds[0][0][0]))
    return X_normalized, y, folds


def normalizar_minmax(X):
    # Transpõe X para trabalhar coluna por coluna
    colunas = list(zip(*X))
    colunas_norm = []

    for coluna in colunas:
        x_min = min(coluna)
        x_max = max(coluna)
        # evita divisão por zero caso max = min
        if x_max == x_min:
            colunas_norm.append([0.0 for _ in coluna])
        else:
            colunas_norm.append([(x - x_min) / (x_max - x_min) for x in coluna])

    # Transpõe de volta (linhas)
    X_norm = list(zip(*colunas_norm))
    return [list(linha) for linha in X_norm]


def cross_validation_shuffle(
        X, y, k: int = 5
):
    # Embaralha os índices
    indices = list(range(len(X)))
    random.seed(42)
    random.shuffle(indices)

    # Calcula tamanho de cada fold
    tamanho_fold = len(X) // k
    folds = []

    for i in range(k):
        inicio = i * tamanho_fold
        fim = inicio + tamanho_fold if i < k - 1 else len(X)
        fold_idx = indices[inicio:fim]
        X_fold = [X[j] for j in fold_idx]
        y_fold = [y[j] for j in fold_idx]
        folds.append((X_fold, y_fold))

    return folds


if __name__ == '__main__':
    pre_process()

