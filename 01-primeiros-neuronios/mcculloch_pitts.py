# Neuronio de McCulloch e Pitts (1943)
# Entradas binarias, pesos fixos e um limiar: ou o neuronio dispara (1) ou fica inativo (0)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

ENTRADAS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# cada porta é (pesos, limiar), peso 1 = entrada excitatoria e -1 = inibitoria
PORTAS = {
    'AND': ([1, 1], 2),
    'OR': ([1, 1], 1),
}


def neuronio(x, pesos, limiar):
    # soma ponderada das entradas, dispara se chegar no limiar
    soma = np.dot(x, pesos)
    return int(soma >= limiar)


def tabela(pesos, limiar):
    return [neuronio(x, pesos, limiar) for x in ENTRADAS]


def plota_porta(ax, nome, pesos, limiar, n):
    # n = quantas entradas ja passaram pelo neuronio
    ax.clear()
    resp = np.array(tabela(pesos, limiar))
    x1 = np.linspace(-0.5, 1.5)
    # reta onde a soma é igual ao limiar: w1*x1 + w2*x2 = limiar
    x2 = (limiar - pesos[0]*x1) / pesos[1]

    ax.plot(x1, x2, 'k--', label='limiar')
    # regiao onde o neuronio dispara
    ax.fill_between(x1, x2, 1.5, color='orange', alpha=0.2)
    ax.scatter(*ENTRADAS[n:].T, s=120, c='lightgray', label='nao testada')
    ax.scatter(*ENTRADAS[:n][resp[:n] == 0].T, s=120, c='white', edgecolors='k', label='inativo (0)')
    ax.scatter(*ENTRADAS[:n][resp[:n] == 1].T, s=120, c='orange', edgecolors='k', label='dispara (1)')

    ax.set_title(f'{nome} (limiar = {limiar})')
    if n > 0:
        # mostra a conta da ultima entrada testada
        x = ENTRADAS[n-1]
        ax.set_xlabel(f'x = ({x[0]}, {x[1]})   soma = {np.dot(x, pesos)}   saida = {resp[n-1]}')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])


def frame(i):
    # a cada frame o neuronio recebe mais uma entrada, depois recomeca
    for ax, nome in zip(eixos, PORTAS):
        pesos, limiar = PORTAS[nome]
        plota_porta(ax, nome, pesos, limiar, i % 5)


if __name__ == '__main__':
    for nome in PORTAS:
        pesos, limiar = PORTAS[nome]
        print(nome)
        for x, s in zip(ENTRADAS, tabela(pesos, limiar)):
            print(x[0], x[1], '->', s)
        print()

    # NOT usa uma entrada so, inibitoria
    print('NOT')
    for x in [0, 1]:
        print(x, '->', neuronio([x], [-1], 0))

    fig, eixos = plt.subplots(1, 2, figsize=(9, 4.4))
    frame(0)
    fig.legend(*eixos[0].get_legend_handles_labels(), loc='lower center', ncol=4)
    fig.subplots_adjust(bottom=0.24, wspace=0.25)
    anim = animation.FuncAnimation(fig, frame, frames=10, interval=900)
    # anim.save('mcculloch_pitts.mp4', fps=1, dpi=200)
    plt.show()

# Observamos que uma reta separa as saidas do AND e do OR, mas nao existe
# reta que separe o XOR, por isso um neuronio so nao resolve esse caso
