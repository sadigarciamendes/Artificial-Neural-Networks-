import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation 
from matplotlib.colors import LinearSegmentedColormap

def spirals(n, seed):
    r = np.random.default_rng(seed)
    t = np.sqrt(r.random(n) * 3 * np.pi)
    a = np.c_[t*np.cos(t), t*np.sin(t)] / 10 
    x = np.vstack([a, -a]) + r.normal(0, 0.02, (2*n, 2))
    y = np.r_[np.zeros(n), np.ones(n)] [:, None]
    return x, y

X, y = spirals(200, seed=0) # train 
Xt, yt = spirals(200, seed=1) # test 

rng = np.random.default_rng(42)
sizes = [2, 16, 16, 1] # 2, 32, 32, 1 
W = [rng.normal(0,1/np.sqrt(i), (i, j))
     for i, j in zip(sizes[:-1], sizes[1:])]
b = [np.zeros((1, j)) for j in sizes[1:]]

def forward(x):
    hs = [x]
    for k in range(3):
        z = hs[-1] @ W[k] + b[k]
        hs.append(np.tanh(z) if k < 2 else 1/(1 + np.exp(-z)))
    return hs

def step(lr=0.5):
    hs = forward(X)
    d = (hs[-1] - y) /len(X)
    for k in range(2, -1, -1):
        gW, gb = hs[k].T @ d, d.sum(0)
        if k: d = (d @ W[k].T) * (1 - hs[k]**2)
        W[k] -= lr * gW
        b[k] -= lr * gb
    p = hs[-1]
    return -np.mean(y*np.log(p+1e-9) + (1-y)*np.log(1-p + 1e-9))

acc = lambda: (forward(Xt)[-1].round() == yt).mean()

g = np.linspace(-1.05, 1.05, 300)
gx, gy = np.meshgrid(g, g)
grid = np.c_[gx.ravel(), gy.ravel()]
cmap = LinearSegmentedColormap.from_list('n', ['#2456d6', '#0c1734', '#0a0a0a', '#33100a','#d9541f'])

# --------------------------------------------------------------------------------------------------------------

plt.style.use("dark_background")
fig = plt.figure(figsize=(5.4, 7.2), facecolor="#0a0a0a")
ax = fig.add_axes([0.04, 0.30, 0.92, 0.69])
ax.set_axis_off()

im = ax.imshow(np.full(gx.shape, .5), cmap=cmap, vmin=0,
               vmax=1, extent=(-1.05, 1.05, -1.05, 1.05),
               origin="lower", interpolation="bilinear")

ax.scatter(*X[y[:, 0] == 0].T, s=10, c="#fce0ff",
           edgecolors="#0a0a0a", lw=.4)
ax.scatter(*X[y[:, 0] == 1].T, s=10, c="#ffe0cc",
           edgecolors="#0a0a0a", lw=.4)

info = fig.text(.5, .265, "", ha="center", va="center", color="#cfcfcf",
                family="monospace", fontsize=10)

msg = "every neuron does three things: weigh, add, decide."
bits = [f"{ord(c):08b}" for c in msg]
rows = [" ".join(bits[i:i+7]) for i in range(0, len(bits), 7)]
fig.text(.5, .035, "\n".join(rows), ha="center", va="bottom",
         color="#d8d8d8", family="serif", fontsize=8.6,
         linespacing=1.35)

state = {"n": 0, "lines": []}


def frame(i):
    for _ in range(1 + i // 15):
        L = step()
        state["n"] += 1
    P = forward(grid)[-1].reshape(gx.shape)
    im.set_data(P)
    for c in state["lines"]:
        c.remove()
    state["lines"] = [ax.contour(gx, gy, P, [.5],
                                 colors="white", linewidths=1.1)]
    info.set_text(f"step {state['n']:>4} | loss {L:.3f}"
                  f"   test acc {acc():.1%}")
    return [im, info]


anim = animation.FuncAnimation(fig, frame, frames=300)
anim.save("spiral.mp4", fps=30, dpi=200)