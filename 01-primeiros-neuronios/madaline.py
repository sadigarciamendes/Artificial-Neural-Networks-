# Madaline (Widrow e Hoff) - Many Adalines
# Dois Adalines na camada escondida e uma saida fixa que faz OR entre eles
# Com duas retas da pra resolver o XOR, que um neuronio sozinho nao resolve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap

TAXA = 0.05
N = 40
FRAMES = 80
PASSOS = 4      # amostras apresentadas por frame


def gerar_dados(n, seed):
    # XOR com 4 nuvens: cantos iguais sao -1 e cantos diferentes sao 1
    r = np.random.default_rng(seed)
    cantos = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]]) * 0.5
    X = np.vstack([r.normal(c, 0.13, (n, 2)) for c in cantos])
    d = np.r_[-np.ones(2*n), np.ones(2*n)]
    ordem = r.permutation(4*n)
    return X[ordem], d[ordem]


def sinal(v):
    return np.where(v >= 0, 1, -1)


def somas(x, W, b):
    # cada coluna de W é um Adaline
    return x @ W + b


def saida(x, W, b):
    z = sinal(somas(x, W, b))
    # OR fixo: a rede dispara se pelo menos um Adaline disparar
    return sinal(z.sum(-1) + 1)


def corrige(x, alvo, W, b):
    # regra MRI: so mexe nos Adalines quando a rede erra
    s = somas(x, W, b)
    if saida(x, W, b) == alvo:
        return W, b
    if alvo == 1:
        # ninguem disparou: puxa pra 1 o Adaline que estava mais perto de disparar
        j = [np.argmax(s)]
    else:
        # disparou quem nao devia: empurra pra -1 todos que dispararam
        j = np.where(s >= 0)[0]
    W, b = W.copy(), b.copy()
    W[:, j] += TAXA * np.outer(x, alvo - s[j])
    b[j] += TAXA * (alvo - s[j])
    return W, b


def reta(w, b, x1):
    # reta de um Adaline: w1*x1 + w2*x2 + b = 0
    return -(w[0]*x1 + b) / w[1]


X, d = gerar_dados(N, seed=0)

rng = np.random.default_rng(3)
est = {'W': rng.normal(0, 0.3, (2, 2)), 'b': rng.normal(0, 0.3, 2), 'n': 0}

g = np.linspace(-1.05, 1.05, 300)
gx, gy = np.meshgrid(g, g)
grid = np.c_[gx.ravel(), gy.ravel()]
cmap = LinearSegmentedColormap.from_list('n', ['#2456d6', '#0c1734', '#0a0a0a', '#33100a', '#d9541f'])

plt.style.use('dark_background')
fig = plt.figure(figsize=(5.4, 6.2), facecolor='#0a0a0a')
ax = fig.add_axes([0.04, 0.12, 0.92, 0.86])
ax.set_axis_off()

im = ax.imshow(np.zeros(gx.shape), cmap=cmap, vmin=-1, vmax=1, alpha=0.7,
               extent=(-1.05, 1.05, -1.05, 1.05), origin='lower')
ax.scatter(*X[d == -1].T, s=14, c='#cfe0ff', edgecolors='#0a0a0a', lw=.4)
ax.scatter(*X[d == 1].T, s=20, c='#ffe0cc', marker='^', edgecolors='#0a0a0a', lw=.4)
# uma reta pra cada Adaline
linhas = [ax.plot([], [], 'w', lw=1.2)[0] for _ in range(2)]
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)

info = fig.text(.5, .06, '', ha='center', va='center', color='#cfcfcf',
                family='monospace', fontsize=10)


def frame(i):
    # o primeiro frame mostra os pesos iniciais, sem treinar
    if i > 0:
        for _ in range(PASSOS):
            k = est['n'] % len(X)
            est['W'], est['b'] = corrige(X[k], d[k], est['W'], est['b'])
            est['n'] += 1
    W, b = est['W'], est['b']

    im.set_data(saida(grid, W, b).reshape(gx.shape))
    for j in range(2):
        linhas[j].set_data(g, reta(W[:, j], b[j], g))
    acc = (saida(X, W, b) == d).mean()
    info.set_text(f"amostras {est['n']:>4} | acc {acc:.1%}")
    return [im, info] + linhas


def inicio():
    # sem isso o matplotlib chama frame(0) duas vezes
    return []


anim = animation.FuncAnimation(fig, frame, init_func=inicio, frames=FRAMES, interval=100, repeat=False)
# anim.save('madaline.mp4', fps=10, dpi=200)
plt.show()

# Observamos que cada Adaline cuida de um canto do XOR, e o OR na saida
# junta as duas regioes
