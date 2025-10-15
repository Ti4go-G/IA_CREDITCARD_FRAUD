import random
from itertools import product

import perceptron
import pre_processing


def acuracia(y_true, y_pred):
    acertos = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return acertos / len(y_true) if y_true else 0.0


def avaliar_cv(folds, n_features, n_oculta, taxa_aprendizado, epocas):
    """Avalia uma configuracao com k-fold, retorna acuracias por fold e media."""
    accs = []
    for i_valid, (X_valid, y_valid) in enumerate(folds):
        # agrega todos os demais folds para treino
        X_train, y_train = [], []
        for j, (X_fold, y_fold) in enumerate(folds):
            if j == i_valid:
                continue
            X_train.extend(X_fold)
            y_train.extend(y_fold)

        # cria nova rede para cada validacao
        rn = perceptron.RedeNeural(
            n_entradas=n_features, n_oculta=n_oculta, n_saida=1, taxa_aprendizado=taxa_aprendizado
        )
        rn.treinar(X_train, y_train, epocas=epocas)
        y_pred = rn.prever(X_valid)
        acc = acuracia(y_valid, y_pred)
        accs.append(acc)

    media = sum(accs) / len(accs) if accs else 0.0
    return accs, media


def grid_search(folds, n_features, grid):
    """Executa grid search em k-fold e retorna melhor combinacao e resultados."""
    melhores = None
    melhor_media = -1.0
    historico = []

    for n_oculta, taxa, epocas in product(grid['n_oculta'], grid['taxa_aprendizado'], grid['epocas']):
        accs, media = avaliar_cv(folds, n_features, n_oculta, taxa, epocas)
        historico.append({
            'n_oculta': n_oculta,
            'taxa_aprendizado': taxa,
            'epocas': epocas,
            'accs': accs,
            'media': media,
        })
        print(f"GS n_oculta={n_oculta}, taxa={taxa}, epocas={epocas} -> media={media:.4f}")
        if media > melhor_media:
            melhor_media = media
            melhores = {'n_oculta': n_oculta, 'taxa_aprendizado': taxa, 'epocas': epocas}

    return melhores, melhor_media, historico


if __name__ == '__main__':
    random.seed(42)
    # Carrega dados e folds embaralhados
    X, y, folds = pre_processing.pre_process()

    # Numero de atributos
    n_features = len(folds[0][0][0])

    # Grade de hiperparametros
    grid = {
        'n_oculta': [8, 12, 16],
        'taxa_aprendizado': [0.05, 0.1, 0.2],
        'epocas': [10, 20, 50],
    }

    melhores, melhor_media, historico = grid_search(folds, n_features, grid)

    print("Melhor combinacao:", melhores)
    print(f"Acuracia media (k-fold): {melhor_media:.4f}")
    
