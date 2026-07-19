



# %%
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# Dos clases 2D linealmente separables
n = 40
X_pos = rng.normal([2.0, 2.0], 0.6, size=(n, 2))
X_neg = rng.normal([-2.0, -2.0], 0.6, size=(n, 2))
X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(n), -np.ones(n)])           # etiquetas en {+1, -1}

plt.figure(figsize=(5, 5))
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='tab:blue', label='+1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='tab:red', label='-1')
plt.legend(); plt.title('Datos'); plt.gca().set_aspect('equal'); plt.show()

# %%
def svm_subgradient(X, y, C=1.0, lr=1e-3, epochs=3000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    hist = []
    for _ in range(epochs):
        margins = y * (X @ w + b)                   # y_i (w·x_i + b)
        mask = margins < 1                          # dentro del margen o mal clasificados
        grad_w = w - C * (X[mask] * y[mask, None]).sum(axis=0)
        grad_b = -C * y[mask].sum()
        w -= lr * grad_w
        b -= lr * grad_b
        hist.append(0.5 * w @ w + C * np.maximum(0, 1 - margins).sum())
    return w, b, hist

# %%
w, b, hist = svm_subgradient(X, y, C=1.0, lr=1e-3, epochs=3000)
print('w =', w, '  b = %.3f' % b)
print('margen 1/||w|| = %.3f' % (1 / np.linalg.norm(w)))

margins = y * (X @ w + b)
sv = np.where(margins < 1.05)[0]                    # vectores de soporte (aprox.)
print('num. vectores de soporte ~', len(sv))

plt.figure()
plt.plot(hist); plt.xlabel('epoch'); plt.ylabel('perdida'); plt.title('Convergencia'); plt.show()

# %%
plt.figure(figsize=(5, 5))
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='tab:blue', label='+1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='tab:red', label='-1')
plt.scatter(X[sv, 0], X[sv, 1], s=140, facecolors='none', edgecolors='k', label='vec. soporte')

xx = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
for c, style in [(0, 'k-'), (1, 'k--'), (-1, 'k--')]:   # w·x + b = 0, 1, -1
    plt.plot(xx, (c - b - w[0] * xx) / w[1], style)
plt.legend(); plt.title('SVM: frontera y margen'); plt.gca().set_aspect('equal'); plt.show()
