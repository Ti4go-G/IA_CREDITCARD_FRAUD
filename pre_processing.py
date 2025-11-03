import random
import pandas as pd
from typing import Optional


def generate_balanced_csv(
        file: str = "./creditcard.csv",
        out_file: str = "./creditcard_5k_80_20.csv",
        total: int = 5000,
        frac_pos: float = 0.20,
        random_state: int = 42,
        drop_dups: bool = True,
):
    """
    Build a balanced CSV with ~total rows, about 80% non-fraud and 20% fraud.
    - Under-sample class 0 (no replacement)
    - Over-sample class 1 (with replacement if needed)
    """
    df = pd.read_csv(file)

    # Ensure class column is named 'Class' and is integer
    if 'Class' not in df.columns and 'is_fraud' in df.columns:
        df = df.rename(columns={'is_fraud': 'Class'})
    if 'Class' not in df.columns:
        raise ValueError("Column 'Class' not found in dataset.")
    try:
        df['Class'] = df['Class'].astype(int)
    except Exception:
        df['Class'] = pd.to_numeric(df['Class'], errors='coerce').fillna(0).astype(int)

    if drop_dups:
        df = df.drop_duplicates()

    # Target sizes
    pos_target = int(round(total * frac_pos))
    neg_target = total - pos_target

    pos_df = df[df['Class'] == 1]
    neg_df = df[df['Class'] == 0]

    if len(neg_df) < neg_target:
        raise ValueError(f"Class 0 has only {len(neg_df)} rows, smaller than target {neg_target}.")

    # Over-sample positives if needed
    pos_sample = pos_df.sample(
        n=pos_target,
        replace=(len(pos_df) < pos_target),
        random_state=random_state
    )

    # Under-sample negatives
    neg_sample = neg_df.sample(
        n=neg_target,
        replace=False,
        random_state=random_state
    )

    out_df = pd.concat([pos_sample, neg_sample], axis=0)
    out_df = out_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    out_df.to_csv(out_file, index=False)

    counts = out_df['Class'].value_counts().to_dict()
    print(f"Saved: {out_file}")
    print(f"Total: {len(out_df)} | Class 0: {counts.get(0, 0)} | Class 1: {counts.get(1, 0)}")
    return out_file


def generate_balanced_csv_unique(
        file: str = "./creditcard.csv",
        out_file: str = "./creditcard_80_20_unique.csv",
        random_state: int = 42,
        drop_dups: bool = True,
):
    """
    Gera um CSV 80/20 sem oversampling: usa APENAS amostras únicas da classe 1.
    - Define pos_target = número de linhas únicas em Class==1
    - Define neg_target = 4 * pos_target (para 80/20)
    - Amostra neg_target da classe 0 sem reposição
    """
    df = pd.read_csv(file)

    # Garantir coluna 'Class' inteira
    if 'Class' not in df.columns and 'is_fraud' in df.columns:
        df = df.rename(columns={'is_fraud': 'Class'})
    if 'Class' not in df.columns:
        raise ValueError("Column 'Class' not found in dataset.")
    try:
        df['Class'] = df['Class'].astype(int)
    except Exception:
        df['Class'] = pd.to_numeric(df['Class'], errors='coerce').fillna(0).astype(int)

    if drop_dups:
        df = df.drop_duplicates()

    pos_df = df[df['Class'] == 1]
    neg_df = df[df['Class'] == 0]

    # Considera apenas positivos únicos (robusto mesmo sem drop_dups global)
    pos_unique = pos_df.drop_duplicates()
    pos_target = len(pos_unique)
    if pos_target == 0:
        raise ValueError("No positive samples found.")
    neg_target = 4 * pos_target

    if len(neg_df) < neg_target:
        raise ValueError(f"Not enough negatives: have {len(neg_df)}, need {neg_target}.")

    neg_sample = neg_df.sample(n=neg_target, replace=False, random_state=random_state)

    out_df = pd.concat([pos_unique, neg_sample], axis=0)
    out_df = out_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    out_df.to_csv(out_file, index=False)

    counts = out_df['Class'].value_counts().to_dict()
    print(f"Saved: {out_file}")
    print(f"Total: {len(out_df)} | Class 0: {counts.get(0, 0)} | Class 1: {counts.get(1, 0)}")
    return out_file

def pre_process(
        file: str = "./creditcard_5k_80_20.csv"
):
    # Return raw (not-normalized) features and stratified folds.
    # Normalização deve ser feita dentro do loop de CV (fit no conjunto de treino apenas)
    df = pd.read_csv(f"{file}")
    # Muda o nome de Class para is_fraud (compatibilidade)
    df.rename(columns={'Class': 'is_fraud'}, inplace=True)
    # Retira valores duplicados
    df = df.drop_duplicates()
    # Remove coluna 'Time' se existir
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])

    # Separa features (todas as colunas exceto 'is_fraud') e alvo (raw, sem normalizar)
    X = df.drop(columns=['is_fraud']).values.tolist()
    y = df['is_fraud'].values.tolist()

    # Gera folds de cross-validation ESTRATIFICADA (mantém proporção de classes por fold)
    folds = cross_validation_shuffle(X, y)
    # imprime número de atributos
    print(len(folds[0][0][0]))
    return X, y, folds


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
    # Stratified K-Fold: separa índices por classe e divide cada classe em k partes
    n = len(X)
    if n == 0:
        return []

    # índices por classe
    pos_idx = [i for i, yy in enumerate(y) if yy == 1]
    neg_idx = [i for i, yy in enumerate(y) if yy == 0]

    random.seed(42)
    random.shuffle(pos_idx)
    random.shuffle(neg_idx)

    # split each class indices into k folds (some folds may differ by 1)
    def split_to_folds(indices_list, k):
        folds_idx = []
        base = len(indices_list) // k
        rem = len(indices_list) % k
        start = 0
        for i in range(k):
            size = base + (1 if i < rem else 0)
            end = start + size
            folds_idx.append(indices_list[start:end])
            start = end
        return folds_idx

    pos_folds = split_to_folds(pos_idx, k)
    neg_folds = split_to_folds(neg_idx, k)

    folds = []
    for i in range(k):
        fold_idx = []
        if i < len(pos_folds):
            fold_idx.extend(pos_folds[i])
        if i < len(neg_folds):
            fold_idx.extend(neg_folds[i])
        # shuffle within the fold for mixing
        random.shuffle(fold_idx)
        X_fold = [X[j] for j in fold_idx]
        y_fold = [y[j] for j in fold_idx]
        folds.append((X_fold, y_fold))

    return folds


if __name__ == '__main__':
    # First generate the balanced dataset
    generate_balanced_csv()
    # Then process it
    pre_process()


def correlacao_features(
        file: str = "./creditcard.csv",
        target: str = "Class",
        method: str = "pearson",
        top_n: Optional[int] = None,
        drop_time: bool = True,
):
    """
    Analisa correlação entre as colunas numéricas e o alvo, ranqueando por |correlação|.

    Parâmetros:
    - file: caminho do CSV (default: ./creditcard.csv)
    - target: nome da coluna alvo (default: Class)
    - method: método de correlação do pandas ("pearson", "spearman" ou "kendall")
    - top_n: se definido, retorna apenas os top_n mais correlacionados
    - drop_time: remove a coluna 'Time' caso exista (muito dependente de ordem temporal)

    Retorna:
    - Lista de tuplas (coluna, correlacao, correlacao_absoluta) ordenada por |correlação| desc.
    """
    df = pd.read_csv(file)

    # Normaliza nome do alvo, se necessário
    if target not in df.columns and 'is_fraud' in df.columns:
        target = 'is_fraud'
    if target not in df.columns:
        raise ValueError(f"Coluna alvo '{target}' não encontrada no dataset.")

    # Remove 'Time' se solicitado
    if drop_time and 'Time' in df.columns:
        df = df.drop(columns=['Time'])

    # Garante que apenas colunas numéricas sejam usadas na correlação
    # e tenta converter o alvo para numérico
    df[target] = pd.to_numeric(df[target], errors='coerce')
    num_df = df.select_dtypes(include=['number']).copy()

    # Precisa do alvo dentro do subconjunto numérico
    if target not in num_df.columns:
        raise ValueError(f"A coluna alvo '{target}' não é numérica após coerção.")

    # Remove linhas com NaN no alvo
    num_df = num_df.dropna(subset=[target])

    # Calcula correlação
    corr = num_df.corr(method=method)
    if target not in corr.columns:
        raise ValueError("Não foi possível calcular correlação com o alvo.")

    s = corr[target].drop(labels=[target])
    s_abs = s.abs().sort_values(ascending=False)

    if top_n is not None:
        s_abs = s_abs.head(top_n)

    # Monta resultado com sinal original
    resultados = []
    for col in s_abs.index:
        resultados.append((col, float(s[col]), float(s_abs[col])))

    # Impressão amigável
    print("Ranking por |correlação| com o alvo '", target, "' (método=", method, "):", sep='')
    for i, (col, val, aval) in enumerate(resultados, start=1):
        print(f"{i:2d}. {col:>10s}  corr={val:+.4f}  |corr|={aval:.4f}")

    return resultados


def plot_correlacao_top10(
        file: str = "./creditcard.csv",
        target: str = "Class",
        top_n: int = 10,
        drop_time: bool = True,
        save_path: Optional[str] = "correlacao_top10.png",
        figsize=(16, 6),
):
    """
    Gera um gráfico com 3 subplots (Pearson, Spearman e Kendall) mostrando
    o ranking dos Top-N atributos mais correlacionados (por |correlação|) com o alvo.

    Retorna um dict: {metodo: [(coluna, corr, |corr|), ...]} em ordem decrescente de |corr|.
    """
    # Importa matplotlib localmente para evitar dependência global
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise ImportError(
            "matplotlib não está instalado. Instale com: pip install matplotlib"
        ) from e

    df = pd.read_csv(file)

    # Ajusta target se necessário
    if target not in df.columns and 'is_fraud' in df.columns:
        target = 'is_fraud'
    if target not in df.columns:
        raise ValueError(f"Coluna alvo '{target}' não encontrada no dataset.")

    if drop_time and 'Time' in df.columns:
        df = df.drop(columns=['Time'])

    # Trabalha apenas com numéricas e garante alvo numérico
    df[target] = pd.to_numeric(df[target], errors='coerce')
    num_df = df.select_dtypes(include=['number']).copy()
    if target not in num_df.columns:
        raise ValueError(f"A coluna alvo '{target}' não é numérica após coerção.")
    num_df = num_df.dropna(subset=[target])

    metodos = ["pearson", "spearman", "kendall"]
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)
    resultados: dict[str, list[tuple[str, float, float]]] = {}

    for ax, metodo in zip(axes, metodos):
        corr = num_df.corr(method=metodo)
        if target not in corr.columns:
            raise ValueError(f"Não foi possível calcular correlação '{metodo}' com o alvo.")

        s_all = corr[target].drop(labels=[target])
        # Seleciona Top-N por |corr|
        top_abs = s_all.abs().sort_values(ascending=False).head(top_n)
        top_index = top_abs.index
        s_signed = s_all.loc[top_index]

        # Ordena pela magnitude desc e plota de baixo para cima
        s_plot = s_signed.reindex(top_index)[::-1]

        # Guarda resultados
        resultados[metodo] = [(col, float(s_signed[col]), float(top_abs[col])) for col in top_index]

        cores = ['#d62728' if v < 0 else '#1f77b4' for v in s_plot.values]
        ax.barh(s_plot.index, s_plot.values, color=cores)
        ax.set_title(f"{metodo.title()} (Top {top_n})")
        ax.set_xlabel("Correlação")
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        ax.set_xlim(-1.0, 1.0)

        # Anota valores ao lado das barras
        for y, v in enumerate(s_plot.values):
            ax.text(v + (0.01 if v >= 0 else -0.01), y,
                    f"{v:+.2f}", va='center', ha='left' if v >= 0 else 'right', fontsize=8)

    fig.suptitle(f"Top {top_n} correlações com '{target}'", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")

    return resultados

if __name__ == '__main__':
    correlacao_features()
    plot_correlacao_top10()

