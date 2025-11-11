import math
import random

# ==============================
# Funções auxiliares
# ==============================

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivada(x):
    return x * (1 - x)

def inicializar_pesos(entradas, saidas):
    """Inicializa matriz de pesos aleatoriamente"""
    return [[random.uniform(-1, 1) for _ in range(saidas)] for _ in range(entradas)]

def inicializar_vetor(n):
    return [random.uniform(-1, 1) for _ in range(n)]

# ==============================
# Rede Neural Multicamadas (MLP)
# ==============================
class RedeNeural:
    def __init__(self, n_entradas, n_oculta, n_saida, taxa_aprendizado=0.1):
        self.taxa = taxa_aprendizado

        # Inicialização dos pesos
        self.pesos_entrada_oculta = inicializar_pesos(n_entradas, n_oculta)
        self.bias_oculta = inicializar_vetor(n_oculta)

        self.pesos_oculta_saida = inicializar_pesos(n_oculta, n_saida)
        self.bias_saida = inicializar_vetor(n_saida)

    def feedforward(self, entradas):
        # Camada oculta
        self.oculta = []
        for j in range(len(self.bias_oculta)):
            soma = sum(entradas[i] * self.pesos_entrada_oculta[i][j] for i in range(len(entradas))) + self.bias_oculta[j]
            self.oculta.append(sigmoid(soma))

        # Camada de saída
        self.saida = []
        for k in range(len(self.bias_saida)):
            soma = sum(self.oculta[j] * self.pesos_oculta_saida[j][k] for j in range(len(self.oculta))) + self.bias_saida[k]
            self.saida.append(sigmoid(soma))

        return self.saida

    def backpropagation(self, entradas, alvo):
        # Cálculo do erro na saída
        erros_saida = [alvo[k] - self.saida[k] for k in range(len(self.saida))]
        deltas_saida = [erros_saida[k] * sigmoid_derivada(self.saida[k]) for k in range(len(self.saida))]

        # Erro na camada oculta
        erros_oculta = [sum(deltas_saida[k] * self.pesos_oculta_saida[j][k] for k in range(len(deltas_saida))) for j in range(len(self.oculta))]
        deltas_oculta = [erros_oculta[j] * sigmoid_derivada(self.oculta[j]) for j in range(len(self.oculta))]

        # Atualizar pesos da camada oculta → saída
        for j in range(len(self.oculta)):
            for k in range(len(deltas_saida)):
                self.pesos_oculta_saida[j][k] += self.taxa * deltas_saida[k] * self.oculta[j]

        # Atualizar bias da saída
        for k in range(len(self.bias_saida)):
            self.bias_saida[k] += self.taxa * deltas_saida[k]

        # Atualizar pesos da camada entrada → oculta
        for i in range(len(entradas)):
            for j in range(len(deltas_oculta)):
                self.pesos_entrada_oculta[i][j] += self.taxa * deltas_oculta[j] * entradas[i]

        # Atualizar bias da oculta
        for j in range(len(self.bias_oculta)):
            self.bias_oculta[j] += self.taxa * deltas_oculta[j]

    def treinar(self, X, y, epocas=100):
        for e in range(epocas):
            erro_total = 0
            for i in range(len(X)):
                saida = self.feedforward(X[i])
                alvo = [y[i]]
                self.backpropagation(X[i], alvo)
                erro_total += sum((alvo[k] - saida[k]) ** 2 for k in range(len(alvo)))
            if e % 10 == 0:
                print(f"Época {e}: Erro total = {erro_total:.5f}")

    def prever(self, X):
        pred = []
        for amostra in X:
            s = self.feedforward(amostra)[0]
            pred.append(1 if s >= 0.5 else 0)
        return pred

    def prever_prob(self, X):
        """Retorna probabilidades (valor contínuo entre 0 e 1) para cada amostra em X.

        Útil para calcular ROC AUC / PR AUC ou quando o código externo espera probabilidades.
        """
        probs = []
        for amostra in X:
            s = self.feedforward(amostra)[0]
            probs.append(s)
        return probs

    # Alias compatível com APIs que chamam 'predict_proba'
    def predict_proba(self, X):
        """Retorna lista de [probabilidade_da_classe_0, probabilidade_da_classe_1] por amostra.

        Mantemos compatibilidade mínima com sklearn-style: retorna apenas a probabilidade da classe 1
        embutida em cada elemento como [1-p, p] para cada amostra.
        """
        probs = self.prever_prob(X)
        return [[1.0 - p, p] for p in probs]

