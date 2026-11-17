# Regra de Hebb (1949)
# o peso cresce quando entrada e saida concordam
# Nao calcula erro nenhum, so reforça a conexao. Aqui aprendendo o AND bipolar (-1 e 1)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap

X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
D = np.array([1, -1, -1, -1])


def corrige(x, d, w, b):
    # regra de Hebb: soma entrada*saida desejada no peso, sem olhar se acertou
    return w + x*d, b + d


def saida(x, w, b):
    return np.sign(x @ w + b)


def reta(w, b, x1):
    # fronteira de decisao: w1*x1 + w2*x2 + b = 0
    return -(w[0]*x1 + b) / w[1]


est = {'w': np.zeros(2), 'b': 0}

g = np.linspace(-1.8, 1.8, 300)
gx, gy = np.meshgrid(g, g)
grid = np.c_[gx.ravel(), gy.ravel()]
cmap = LinearSegmentedColormap.from_list('n', ['#2456d6', '#0c1734', '#0a0a0a', '#33100a', '#d9541f'])

plt.style.use('dark_background')
fig = plt.figure(figsize=(5.4, 6.2), facecolor='#0a0a0a')
ax = fig.add_axes([0.04, 0.12, 0.92, 0.86])
ax.set_axis_off()

im = ax.imshow(np.zeros(gx.shape), cmap=cmap, vmin=-1, vmax=1, alpha=0.7,
               extent=(-1.8, 1.8, -1.8, 1.8), origin='lower')
ax.scatter(*X[D == -1].T, s=90, c='#cfe0ff', edgecolors='#0a0a0a')
ax.scatter(*X[D == 1].T, s=110, c='#ffe0cc', marker='^', edgecolors='#0a0a0a')
linha, = ax.plot([], [], 'w', lw=1.2)
# circulo na amostra que acabou de ser apresentada
marca = ax.scatter([], [], s=420, facecolors='none', edgecolors='yellow', lw=1.5)
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)

info = fig.text(.5, .06, '', ha='center', va='center', color='#cfcfcf',
                family='monospace', fontsize=10)


def frame(i):
    if i == 0:
        # comeca (e recomeca) com os pesos zerados
        est['w'], est['b'] = np.zeros(2), 0
        marca.set_offsets(np.empty((0, 2)))
    else:
        est['w'], est['b'] = corrige(X[i-1], D[i-1], est['w'], est['b'])
        marca.set_offsets([X[i-1]])
    w, b = est['w'], est['b']

    im.set_data(saida(grid, w, b).reshape(gx.shape))
    if w[1] != 0:
        linha.set_data(g, reta(w, b, g))
    else:
        linha.set_data([], [])
    acertos = (saida(X, w, b) == D).sum()
    info.set_text(f'amostra {i} | w = ({w[0]:.0f}, {w[1]:.0f})  b = {b:.0f} | acertos {acertos}/4')
    return [im, linha, marca, info]


def inicio():
    # sem isso o matplotlib chama frame(0) duas vezes
    return []


anim = animation.FuncAnimation(fig, frame, init_func=inicio, frames=5, interval=1200)
# anim.save('hebb.mp4', fps=1, dpi=200)
plt.show()

# Observamos que uma passada pelas 4 amostras ja resolve o AND, mas como a regra
# nao olha o erro ela nao garante acerto em outros problemas
