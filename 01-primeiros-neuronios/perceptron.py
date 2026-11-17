# Perceptron (Rosenblatt, 1958)
# Mesmo neuronio de McCulloch e Pitts, mas agora os pesos sao aprendidos com os erros
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap

TAXA = 0.02
N = 60
FRAMES = 40


def gerar_dados(n, seed):
    # duas nuvens de pontos que uma reta consegue separar
    r = np.random.default_rng(seed)
    a = r.normal([-0.4, -0.4], 0.15, (n, 2))
    b = r.normal([0.4, 0.4], 0.15, (n, 2))
    return np.vstack([a, b]), np.r_[np.zeros(n), np.ones(n)]


def saida(x, w, b):
    # degrau: dispara se a soma ponderada passar de zero
    return (x @ w + b >= 0).astype(int)


def corrige(x, d, w, b):
    # regra do perceptron: erro = desejado - obtido, se acertou o erro é 0 e nada muda
    erro = d - saida(x, w, b)
    return w + TAXA*erro*x, b + TAXA*erro


def prox_erro(X, d, w, b, ini):
    # procura a proxima amostra classificada errada, comecando em ini
    for k in range(len(X)):
        i = (ini + k) % len(X)
        if saida(X[i], w, b) != d[i]:
            return i
    return -1


def reta(w, b, x1):
    # fronteira de decisao: w1*x1 + w2*x2 + b = 0
    return -(w[0]*x1 + b) / w[1]


X, d = gerar_dados(N, seed=0)

# pesos iniciais ruins de proposito, pra dar pra ver o aprendizado
est = {'w': np.array([-0.3, 0.2]), 'b': 0.1, 'i': 0, 'n': 0}

g = np.linspace(-1.05, 1.05, 300)
gx, gy = np.meshgrid(g, g)
grid = np.c_[gx.ravel(), gy.ravel()]
cmap = LinearSegmentedColormap.from_list('n', ['#2456d6', '#0c1734', '#0a0a0a', '#33100a', '#d9541f'])

plt.style.use('dark_background')
fig = plt.figure(figsize=(5.4, 6.2), facecolor='#0a0a0a')
ax = fig.add_axes([0.04, 0.12, 0.92, 0.86])
ax.set_axis_off()

im = ax.imshow(np.full(gx.shape, .5), cmap=cmap, vmin=0, vmax=1, alpha=0.7,
               extent=(-1.05, 1.05, -1.05, 1.05), origin='lower')
ax.scatter(*X[d == 0].T, s=14, c='#cfe0ff', edgecolors='#0a0a0a', lw=.4)
ax.scatter(*X[d == 1].T, s=20, c='#ffe0cc', marker='^', edgecolors='#0a0a0a', lw=.4)
linha, = ax.plot([], [], 'w', lw=1.2)
# circulo na amostra que causou a ultima correcao
marca = ax.scatter([], [], s=160, facecolors='none', edgecolors='yellow', lw=1.5)
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)

info = fig.text(.5, .06, '', ha='center', va='center', color='#cfcfcf',
                family='monospace', fontsize=10)


def frame(_):
    w, b = est['w'], est['b']
    i = prox_erro(X, d, w, b, est['i'])
    if i >= 0:
        w, b = corrige(X[i], d[i], w, b)
        est.update(w=w, b=b, i=i+1, n=est['n']+1)
        marca.set_offsets([X[i]])
    else:
        marca.set_offsets(np.empty((0, 2)))

    im.set_data(saida(grid, w, b).reshape(gx.shape))
    linha.set_data(g, reta(w, b, g))
    erros = (saida(X, w, b) != d).sum()
    txt = f"correcoes {est['n']:>3} | erros {erros:>3}"
    if erros == 0:
        txt += ' | convergiu'
    info.set_text(txt)
    return [im, linha, marca, info]


def inicio():
    # sem isso o matplotlib chama frame(0) duas vezes
    return []


anim = animation.FuncAnimation(fig, frame, init_func=inicio, frames=FRAMES, interval=150, repeat=False)
# anim.save('perceptron.mp4', fps=8, dpi=200)
plt.show()

# Observamos que o perceptron so mexe na reta quando erra, e para assim que
# todas as amostras ficam do lado certo
