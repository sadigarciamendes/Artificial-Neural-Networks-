# Adaline (Widrow e Hoff, 1960)
# Parecido com o Perceptron, mas o erro é medido na saida linear (antes do degrau)
# e os pesos descem o gradiente do erro quadratico (regra delta)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap

TAXA = 0.002
N = 60
EPOCAS = 80


def gerar_dados(n, seed):
    # mesmas nuvens do Perceptron, mas com as classes em -1 e 1
    r = np.random.default_rng(seed)
    a = r.normal([-0.4, -0.4], 0.15, (n, 2))
    b = r.normal([0.4, 0.4], 0.15, (n, 2))
    return np.vstack([a, b]), np.r_[-np.ones(n), np.ones(n)]


def soma(x, w, b):
    return x @ w + b


def treina_epoca(X, d, w, b):
    # regra delta: corrige em toda amostra, proporcional ao tamanho do erro
    for x, alvo in zip(X, d):
        erro = alvo - soma(x, w, b)
        w = w + TAXA*erro*x
        b = b + TAXA*erro
    return w, b


def eqm(X, d, w, b):
    return np.mean((d - soma(X, w, b))**2)


def reta(w, b, x1):
    # fronteira de decisao: w1*x1 + w2*x2 + b = 0
    return -(w[0]*x1 + b) / w[1]


X, d = gerar_dados(N, seed=0)

# mesmos pesos iniciais ruins do Perceptron
est = {'w': np.array([-0.3, 0.2]), 'b': 0.1, 'erros': []}

g = np.linspace(-1.05, 1.05, 300)
gx, gy = np.meshgrid(g, g)
grid = np.c_[gx.ravel(), gy.ravel()]
cmap = LinearSegmentedColormap.from_list('n', ['#2456d6', '#0c1734', '#0a0a0a', '#33100a', '#d9541f'])

plt.style.use('dark_background')
fig = plt.figure(figsize=(10, 5.6), facecolor='#0a0a0a')
ax = fig.add_axes([0.02, 0.12, 0.47, 0.84])
ax.set_axis_off()

# o fundo mostra a saida linear, por isso é um degrade e nao duas cores secas
im = ax.imshow(np.zeros(gx.shape), cmap=cmap, vmin=-1, vmax=1,
               extent=(-1.05, 1.05, -1.05, 1.05), origin='lower')
ax.scatter(*X[d == -1].T, s=14, c='#cfe0ff', edgecolors='#0a0a0a', lw=.4)
ax.scatter(*X[d == 1].T, s=20, c='#ffe0cc', marker='^', edgecolors='#0a0a0a', lw=.4)
linha, = ax.plot([], [], 'w', lw=1.2)
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)

# curva do erro quadratico medio
ax2 = fig.add_axes([0.58, 0.2, 0.38, 0.7], facecolor='#0a0a0a')
curva, = ax2.plot([], [], color='#d9541f')
ax2.set_xlim(0, EPOCAS)
ax2.set_ylim(0, eqm(X, d, est['w'], est['b'])*1.05)
ax2.set_xlabel('epoca')
ax2.set_ylabel('erro quadratico medio')

info = fig.text(.255, .06, '', ha='center', va='center', color='#cfcfcf',
                family='monospace', fontsize=10)


def frame(i):
    # o primeiro frame mostra os pesos iniciais, sem treinar
    if i > 0:
        est['w'], est['b'] = treina_epoca(X, d, est['w'], est['b'])
    w, b = est['w'], est['b']
    est['erros'].append(eqm(X, d, w, b))

    im.set_data(np.clip(soma(grid, w, b), -1, 1).reshape(gx.shape))
    linha.set_data(g, reta(w, b, g))
    curva.set_data(range(len(est['erros'])), est['erros'])
    acc = (np.sign(soma(X, w, b)) == d).mean()
    info.set_text(f"epoca {i:>3} | eqm {est['erros'][-1]:.3f} | acc {acc:.1%}")
    return [im, linha, curva, info]


def inicio():
    # sem isso o matplotlib chama frame(0) duas vezes
    return []


anim = animation.FuncAnimation(fig, frame, init_func=inicio, frames=EPOCAS+1, interval=80, repeat=False)
# anim.save('adaline.mp4', fps=15, dpi=200)
plt.show()

# Observamos que o Adaline continua ajustando a reta mesmo depois de acertar
# todas as amostras, porque ele minimiza o erro quadratico e nao so o numero de erros
