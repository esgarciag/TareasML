

# %%
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

n = 50
X_pos = rng.normal([2.5, 2.5], 0.7, size=(n, 2))
X_neg = rng.normal([-2.5, -2.5], 0.7, size=(n, 2))
X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(n), -np.ones(n)])           # etiquetas en {+1, -1}

plt.figure(figsize=(5, 5))
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='tab:blue', label='+1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='tab:red', label='-1')
plt.legend(); plt.title('Dos gaussianas separables'); plt.gca().set_aspect('equal'); plt.show()

# %% [markdown]
# ## El PLA
#
# Se agrega una coordenada constante 1 para el sesgo. Mientras haya un punto mal clasificado
# (y_i (w·x_i) <= 0), se corrige con  w <- w + y_i x_i.

# %%
def pla(X, y, max_iter=10000):
    Xb = np.hstack([np.ones((len(X), 1)), X])      # x0 = 1 (sesgo)
    w = np.zeros(Xb.shape[1])
    updates = 0
    snaps = [w.copy()]
    for _ in range(max_iter):
        pred = np.sign(Xb @ w); pred[pred == 0] = -1
        mis = np.where(pred != y)[0]
        if len(mis) == 0:
            break                                  # separó todo: converge
        i = mis[0]
        w = w + y[i] * Xb[i]                        # regla del PLA
        updates += 1
        snaps.append(w.copy())
    return w, updates, snaps

# %%
w, updates, snaps = pla(X, y)
Xb = np.hstack([np.ones((len(X), 1)), X])
acc = np.mean(np.sign(Xb @ w) == y)
print('actualizaciones hasta converger:', updates)
print('exactitud:', acc)
print('w (sesgo, w1, w2) =', w)

# %%
def line_pts(w, xlim):                             # w0 + w1*x + w2*y = 0
    xx = np.linspace(*xlim, 100)
    return xx, -(w[0] + w[1] * xx) / w[2]

xlim = (X[:, 0].min() - 1, X[:, 0].max() + 1)
plt.figure(figsize=(5, 5))
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='tab:blue', label='+1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='tab:red', label='-1')

# algunas rectas intermedias (la evolución del PLA)
for k in np.linspace(1, len(snaps) - 1, min(4, len(snaps) - 1)).astype(int):
    if snaps[k][2] != 0:
        xx, yy = line_pts(snaps[k], xlim)
        plt.plot(xx, yy, color='gray', alpha=0.4)

xx, yy = line_pts(w, xlim)
plt.plot(xx, yy, 'k-', lw=2, label='frontera final')
plt.legend(); plt.title('PLA: convergencia'); plt.gca().set_aspect('equal')
plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1); plt.show()
