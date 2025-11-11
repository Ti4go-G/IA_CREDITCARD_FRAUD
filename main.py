import random
from itertools import product

import perceptron
import pre_processing

# Tentativa de importar Gradient Boosting do scikit-learn
try:
    from sklearn.ensemble import GradientBoostingClassifier
except Exception:  # ImportError ou outros
    GradientBoostingClassifier = None


def acuracia(y_true, y_pred):
    acertos = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return acertos / len(y_true) if y_true else 0.0


def _ranks(vals):
    # Retorna ranks (1..n) com empates recebendo média dos ranks
    # vals: lista de escores
    n = len(vals)
    idx = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[idx[j + 1]] == vals[idx[i]]:
            j += 1
        # média dos ranks para o bloco [i..j]
        r_mean = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[idx[k]] = r_mean
        i = j + 1
    return ranks


def roc_auc_score(y_true, y_score):
    # Implementação baseada em Mann–Whitney U
    pairs = list(zip(y_true, y_score))
    # Remove NaNs
    pairs = [(y, s) for y, s in pairs if s is not None]
    if not pairs:
        return None
    y, s = zip(*pairs)
    P = sum(1 for v in y if v == 1)
    N = sum(1 for v in y if v == 0)
    if P == 0 or N == 0:
        return None
    ranks = _ranks(list(s))
    sum_ranks_pos = sum(r for r, yi in zip(ranks, y) if yi == 1)
    auc = (sum_ranks_pos - P * (P + 1) / 2.0) / (P * N)
    return auc


def pr_auc_score(y_true, y_score):
    # Curva Precision-Recall por ordenação dos scores desc
    pairs = sorted(zip(y_true, y_score), key=lambda t: t[1], reverse=True)
    P = sum(1 for v, _ in pairs if v == 1)
    N = sum(1 for v, _ in pairs if v == 0)
    if P == 0 or N == 0:
        return None
    tp = 0
    fp = 0
    recalls = [0.0]
    precisions = [1.0]
    last_score = None
    for yi, si in pairs:
        if yi == 1:
            tp += 1
        else:
            fp += 1
        # Apenas registra em mudanças (não estritamente necessário)
        recall = tp / P
        precision = tp / (tp + fp)
        recalls.append(recall)
        precisions.append(precision)
        last_score = si

    # Integra por trapézio em recall (ordenado crescente por construção)
    auc = 0.0
    for i in range(1, len(recalls)):
        dr = recalls[i] - recalls[i - 1]
        auc += dr * (precisions[i] + precisions[i - 1]) / 2.0
    return auc


def _fit_minmax(X):
    # X: list of rows
    if not X:
        return [], []
    cols = list(zip(*X))
    mins = [min(c) for c in cols]
    maxs = [max(c) for c in cols]
    return mins, maxs


def _transform_minmax(X, mins, maxs):
    if not X:
        return []
    out = []
    for row in X:
        new = []
        for v, mi, ma in zip(row, mins, maxs):
            if ma == mi:
                new.append(0.0)
            else:
                new.append((v - mi) / (ma - mi))
        out.append(new)
    return out


def avaliar_cv(folds, n_features, metodo, params):
    """Avalia uma configuração com k-fold e retorna métricas por fold e médias.

    metodo: 'MLP' ou 'GB' (Gradient Boosting)
    params: dicionário de hiperparâmetros específicos do método
    """
    accs = []
    aucs = []
    pr_aucs = []
    metodo_l = (metodo or 'MLP').strip().lower()

    for i_valid, (X_valid, y_valid) in enumerate(folds):
        # agrega todos os demais folds para treino
        X_train, y_train = [], []
        for j, (X_fold, y_fold) in enumerate(folds):
            if j == i_valid:
                continue
            X_train.extend(X_fold)
            y_train.extend(y_fold)

        # Normalize using train-only statistics (avoid data leakage)
        mins, maxs = _fit_minmax(X_train)
        X_train_norm = _transform_minmax(X_train, mins, maxs)
        X_valid_norm = _transform_minmax(X_valid, mins, maxs)

        # Treina por método
        if metodo_l == 'mlp':
            n_oculta = params.get('n_oculta', 12)
            taxa_aprendizado = params.get('taxa_aprendizado', 0.1)
            epocas = params.get('epocas', 20)

            rn = perceptron.RedeNeural(
                n_entradas=n_features,
                n_oculta=n_oculta,
                n_saida=1,
                taxa_aprendizado=taxa_aprendizado,
            )
            rn.treinar(X_train_norm, y_train, epocas=epocas)
            y_scores = rn.prever_prob(X_valid_norm)

        elif metodo_l in ('gb', 'gradient_boosting', 'gradient boosting', 'gradientboosting'):
            if GradientBoostingClassifier is None:
                raise ImportError(
                    "scikit-learn não disponível. Instale com `pip install scikit-learn` para usar Gradient Boosting."
                )
            n_estimators = params.get('n_estimators', 200)
            learning_rate = params.get('learning_rate', 0.1)
            max_depth = params.get('max_depth', 3)

            # A profundidade em GB é setada via max_depth do base_estimator (árvore),
            # no GradientBoostingClassifier usa-se 'max_depth' no init dos árvores via 'max_depth' no base tree.
            # Em sklearn, controla-se via 'max_depth' no parâmetro 'max_depth' de DecisionTreeRegressor usado internamente
            # através de 'max_depth' em 'max_depth' de init? Não há parâmetro direto, mas há 'max_depth' em init desde 1.6.
            # Para compatibilidade, passamos 'max_depth' se suportado; caso contrário, ignorado.
            try:
                clf = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=42,
                )
            except TypeError:
                # Versões antigas não aceitam max_depth no init; cai sem max_depth
                clf = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    random_state=42,
                )

            clf.fit(X_train_norm, y_train)
            proba = clf.predict_proba(X_valid_norm)
            y_scores = [p[1] for p in proba]

        else:
            raise ValueError(f"Método desconhecido: {metodo}")

        # métricas comuns
        y_pred = [1 if s >= 0.5 else 0 for s in y_scores]
        acc = acuracia(y_valid, y_pred)
        accs.append(acc)
        auc = roc_auc_score(y_valid, y_scores)
        pr_auc = pr_auc_score(y_valid, y_scores)
        aucs.append(auc if auc is not None else float('nan'))
        pr_aucs.append(pr_auc if pr_auc is not None else float('nan'))

    media_acc = sum(accs) / len(accs) if accs else 0.0
    # médias ignorando NaN
    valid_aucs = [a for a in aucs if isinstance(a, float)]
    valid_prs = [a for a in pr_aucs if isinstance(a, float)]
    media_auc = sum(valid_aucs) / len(valid_aucs) if valid_aucs else float('nan')
    media_pr = sum(valid_prs) / len(valid_prs) if valid_prs else float('nan')
    return accs, media_acc, aucs, media_auc, pr_aucs, media_pr


def grid_search(folds, n_features, grid, opt_metric: str = 'roc_auc', metodo: str = 'MLP'):
    """Executa grid search em k-fold e retorna melhor combinação e resultados.

    opt_metric: métrica usada para seleção ('roc_auc', 'pr_auc', 'acc')
    metodo: 'MLP' (perceptron.RedeNeural) ou 'GB' (Gradient Boosting)
    """
    melhores = None
    melhor_media = -1.0
    historico = []

    metodo_l = (metodo or 'MLP').strip().lower()

    if metodo_l == 'mlp':
        n_ocultas = grid.get('n_oculta', [12])
        taxas = grid.get('taxa_aprendizado', [0.1])
        epocass = grid.get('epocas', [20])

        for n_oculta, taxa, epocas in product(n_ocultas, taxas, epocass):
            params = {'n_oculta': n_oculta, 'taxa_aprendizado': taxa, 'epocas': epocas}
            accs, media_acc, aucs, media_auc, pr_aucs, media_pr = avaliar_cv(
                folds, n_features, metodo, params
            )
            historico.append({
                'metodo': 'MLP',
                **params,
                'accs': accs,
                'media_acc': media_acc,
                'aucs': aucs,
                'media_auc': media_auc,
                'pr_aucs': pr_aucs,
                'media_pr': media_pr,
            })
            print(
                f"GS [MLP] n_oculta={n_oculta}, taxa={taxa}, epocas={epocas} -> "
                f"acc={media_acc:.4f} | roc_auc={media_auc:.4f} | pr_auc={media_pr:.4f}"
            )

            if opt_metric == 'acc':
                score = media_acc
            elif opt_metric == 'pr_auc':
                score = media_pr
            else:
                score = media_auc

            if score > melhor_media:
                melhor_media = score
                melhores = {
                    'metodo': 'MLP',
                    **params,
                    'opt_metric': opt_metric,
                }

    elif metodo_l in ('gb', 'gradient_boosting', 'gradient boosting', 'gradientboosting'):
        n_estimators_list = grid.get('n_estimators', [200])
        learning_rates = grid.get('learning_rate', [0.1])
        max_depths = grid.get('max_depth', [3])

        for n_estimators, lr, md in product(n_estimators_list, learning_rates, max_depths):
            params = {'n_estimators': n_estimators, 'learning_rate': lr, 'max_depth': md}
            accs, media_acc, aucs, media_auc, pr_aucs, media_pr = avaliar_cv(
                folds, n_features, metodo, params
            )
            historico.append({
                'metodo': 'GB',
                **params,
                'accs': accs,
                'media_acc': media_acc,
                'aucs': aucs,
                'media_auc': media_auc,
                'pr_aucs': pr_aucs,
                'media_pr': media_pr,
            })
            print(
                f"GS [GB] n_estimators={n_estimators}, lr={lr}, max_depth={md} -> "
                f"acc={media_acc:.4f} | roc_auc={media_auc:.4f} | pr_auc={media_pr:.4f}"
            )

            if opt_metric == 'acc':
                score = media_acc
            elif opt_metric == 'pr_auc':
                score = media_pr
            else:
                score = media_auc

            if score > melhor_media:
                melhor_media = score
                melhores = {
                    'metodo': 'GB',
                    **params,
                    'opt_metric': opt_metric,
                }
    else:
        raise ValueError(f"Método desconhecido: {metodo}")

    return melhores, melhor_media, historico


def _rand_in_bounds(lo, hi, kind='float'):
    if kind == 'int':
        return random.randint(int(lo), int(hi))
    # float
    return random.uniform(float(lo), float(hi))


def _mutate_value(val, lo, hi, kind='float', strength=0.2):
    # Gaussian-like perturbation within bounds
    if kind == 'int':
        span = max(1, int(round((hi - lo) * strength)))
        new_val = int(val) + random.randint(-span, span)
        return max(int(lo), min(int(hi), new_val))
    else:
        span = (hi - lo) * strength
        new_val = float(val) + random.uniform(-span, span)
        return max(float(lo), min(float(hi), new_val))


def _crossover(a, b, meta):
    # Single-point style per-parameter: for floats blend, for ints average/choose
    child = {}
    for k, spec in meta.items():
        lo, hi, kind = spec[:3]
        if kind == 'float':
            alpha = random.random()
            child[k] = alpha * a[k] + (1 - alpha) * b[k]
            child[k] = max(lo, min(hi, child[k]))
        else:
            if random.random() < 0.5:
                child[k] = int(round((a[k] + b[k]) / 2))
            else:
                child[k] = random.choice([int(a[k]), int(b[k])])
            child[k] = max(int(lo), min(int(hi), int(child[k])))
    return child


def genetic_search(
    folds,
    n_features,
    opt_metric: str = 'roc_auc',
    metodo: str = 'MLP',
    population_size: int = 18,
    generations: int = 12,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.3,
    elitism: int = 2,
):
    """Busca de hiperparâmetros via Algoritmo Genético.

    - metodo: 'MLP' ou 'GB'
    - opt_metric: 'roc_auc' | 'pr_auc' | 'acc'
    Retorna (melhores_params, melhor_score, historico)
    """

    metodo_l = (metodo or 'MLP').strip().lower()

    # Espaço de busca: (lo, hi, kind)
    if metodo_l == 'mlp':
        meta = {
            'n_oculta': (4, 64, 'int'),
            'taxa_aprendizado': (0.001, 0.5, 'float'),
            'epocas': (5, 80, 'int'),
        }
    elif metodo_l in ('gb', 'gradient_boosting', 'gradient boosting', 'gradientboosting'):
        meta = {
            'n_estimators': (50, 600, 'int'),
            'learning_rate': (0.01, 0.3, 'float'),
            'max_depth': (1, 6, 'int'),
        }
    else:
        raise ValueError(f"Método desconhecido: {metodo}")

    def random_individual():
        return {k: _rand_in_bounds(*spec) for k, spec in meta.items()}

    def evaluate(params):
        accs, media_acc, aucs, media_auc, pr_aucs, media_pr = avaliar_cv(
            folds, n_features, metodo, params
        )
        # escolher escore
        if opt_metric == 'acc':
            score = media_acc
        elif opt_metric == 'pr_auc':
            score = media_pr
        else:
            score = media_auc
        return score, {
            'metodo': 'MLP' if metodo_l == 'mlp' else 'GB',
            **params,
            'accs': accs,
            'media_acc': media_acc,
            'aucs': aucs,
            'media_auc': media_auc,
            'pr_aucs': pr_aucs,
            'media_pr': media_pr,
        }

    def mutate(ind):
        child = dict(ind)
        for k, spec in meta.items():
            if random.random() < mutation_rate:
                lo, hi, kind = spec[:3]
                child[k] = _mutate_value(child[k], lo, hi, kind)
        return child

    def tournament_select(pop, fits, k=3):
        best_idx = None
        best_fit = -1e18
        for _ in range(k):
            i = random.randrange(len(pop))
            if fits[i] > best_fit:
                best_fit = fits[i]
                best_idx = i
        return pop[best_idx]

    # Inicialização
    population = [random_individual() for _ in range(population_size)]
    historico = []
    best_params = None
    best_score = -1e18

    for gen in range(generations):
        scores = []
        recs = []
        for ind in population:
            sc, rec = evaluate(ind)
            scores.append(sc)
            recs.append(rec)
        # Log e histórico
        gen_best_idx = max(range(len(scores)), key=lambda i: scores[i])
        gen_best = population[gen_best_idx]
        gen_best_score = scores[gen_best_idx]
        # registro do melhor da geração
        historico.append({**recs[gen_best_idx], 'geracao': gen})
        print(
            f"GA gen={gen} best={gen_best} -> {opt_metric}={gen_best_score:.4f}"
        )
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_params = gen_best.copy()

        # Seleção + reprodução
        # Elitismo
        elite_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:elitism]
        new_population = [population[i] for i in elite_indices]

        # Gerar filhos até completar população
        while len(new_population) < population_size:
            p1 = tournament_select(population, scores, k=3)
            p2 = tournament_select(population, scores, k=3)
            if random.random() < crossover_rate:
                child = _crossover(p1, p2, meta)
            else:
                child = dict(random.choice([p1, p2]))
            child = mutate(child)
            new_population.append(child)

        population = new_population

    # Avalia melhor final para registrar métricas completas
    final_score, final_rec = evaluate(best_params)
    historico.append({**final_rec, 'geracao': generations})

    melhores = {
        'metodo': 'MLP' if metodo_l == 'mlp' else 'GB',
        **best_params,
        'opt_metric': opt_metric,
    }
    return melhores, final_score, historico


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Treino e grid search para MLP e Gradient Boosting')
    parser.add_argument('--metodo', type=str, default='MLP', choices=['MLP', 'GB'], help='Algoritmo de modelagem')
    parser.add_argument('--opt-metric', type=str, default='roc_auc', choices=['roc_auc', 'pr_auc', 'acc'], help='Métrica de seleção do grid search')
    parser.add_argument('--search', type=str, default='grid', choices=['grid', 'ga'], help='Tipo de busca de hiperparâmetros')
    parser.add_argument('--ga-pop', type=int, default=18, help='Tamanho da população do GA')
    parser.add_argument('--ga-gens', type=int, default=12, help='Número de gerações do GA')
    parser.add_argument('--ga-cr', type=float, default=0.9, help='Taxa de crossover do GA')
    parser.add_argument('--ga-mr', type=float, default=0.3, help='Taxa de mutação do GA')
    parser.add_argument('--ga-elit', type=int, default=2, help='Quantidade de elites preservados por geração')
    args = parser.parse_args()

    random.seed(42)
    # Carrega dados e folds embaralhados
    X, y, folds = pre_processing.pre_process()

    # Numero de atributos
    n_features = len(folds[0][0][0])

    # Escolha de método: 'MLP' ou 'GB' (pode ser passado por CLI --metodo)
    metodo = args.metodo

    # Grade de hiperparametros
    if metodo.lower() == 'mlp':
        grid = {
            'n_oculta': [8, 12, 16],
            'taxa_aprendizado': [0.05, 0.1, 0.2],
            'epocas': [10, 20, 50],
        }
    else:  # GB
        grid = {
            'n_estimators': [100, 200, 400],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [2, 3, 4],
        }

    if args.search == 'grid':
        melhores, melhor_media, historico = grid_search(
            folds, n_features, grid, opt_metric=args.opt_metric, metodo=metodo
        )
    else:
        melhores, melhor_media, historico = genetic_search(
            folds,
            n_features,
            opt_metric=args.opt_metric,
            metodo=metodo,
            population_size=args.ga_pop,
            generations=args.ga_gens,
            crossover_rate=args.ga_cr,
            mutation_rate=args.ga_mr,
            elitism=args.ga_elit,
        )

    print("Melhor combinacao:", melhores)

    # Localiza no historico o registro correspondente aos parametros vencedores
    best_metrics = None
    if melhores:
        for rec in historico:
            # confere metodo
            if rec.get('metodo') != melhores.get('metodo'):
                continue
            # compara chaves de params por metodo
            if rec.get('metodo') == 'MLP':
                if (
                    rec.get('n_oculta') == melhores.get('n_oculta') and
                    rec.get('taxa_aprendizado') == melhores.get('taxa_aprendizado') and
                    rec.get('epocas') == melhores.get('epocas')
                ):
                    best_metrics = rec
                    break
            else:  # GB
                if (
                    rec.get('n_estimators') == melhores.get('n_estimators') and
                    rec.get('learning_rate') == melhores.get('learning_rate') and
                    rec.get('max_depth') == melhores.get('max_depth')
                ):
                    best_metrics = rec
                    break

    # Exibe metricas completas do melhor
    if best_metrics is not None:
        print(
            "Metricas do melhor (medias k-fold): "
            f"acc={best_metrics.get('media_acc', float('nan')):.4f} | "
            f"roc_auc={best_metrics.get('media_auc', float('nan')):.4f} | "
            f"pr_auc={best_metrics.get('media_pr', float('nan')):.4f}"
        )
    else:
        # Fallback: manter a saida anterior baseada na metrica otima
        if melhores and melhores.get('opt_metric') == 'acc':
            print(f"Acuracia media (k-fold): {melhor_media:.4f}")
        elif melhores and melhores.get('opt_metric') == 'pr_auc':
            print(f"PR/AUC media (k-fold): {melhor_media:.4f}")
        else:
            print(f"ROC/AUC media (k-fold): {melhor_media:.4f}")
    
