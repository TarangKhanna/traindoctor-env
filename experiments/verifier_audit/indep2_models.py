"""indep2: numpy-only learners (no sklearn available): softmax boosted stumps, random forest, kNN; Wilson CI."""
import numpy as np

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n); r = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return [(c - r)/d, (c + r)/d]

def bins_fit(X, nb=8):
    qs = np.linspace(0, 1, nb + 1)[1:-1]
    edges = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
    return edges

def bins_apply(X, edges):
    return np.stack([np.searchsorted(e, X[:, j], side="left") for j, e in enumerate(edges)], 1).astype(np.int16)

NB = 9

class StumpGB:
    def __init__(self, rounds=60, lr=0.3, seed=0): self.rounds, self.lr = rounds, lr
    def fit(self, X, y, K):
        self.K = K; self.edges = bins_fit(X); B = bins_apply(X, self.edges); n, f = B.shape
        self.f0 = np.log(np.bincount(y, minlength=K) + 1.0) ; F = np.tile(self.f0, (n, 1)); Y = np.eye(K)[y]
        flat = (B + (np.arange(f) * NB)[None, :]).ravel()
        self.trees = []
        for _ in range(self.rounds):
            P = np.exp(F - F.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
            rd = []
            for k in range(K):
                g = P[:, k] - Y[:, k]; h = P[:, k] * (1 - P[:, k]) + 1e-6
                G = np.bincount(flat, np.repeat(g, f), f * NB).reshape(f, NB); H = np.bincount(flat, np.repeat(h, f), f * NB).reshape(f, NB)
                cg, ch = G.cumsum(1)[:, :-1], H.cumsum(1)[:, :-1]; tg, th = G.sum(1, keepdims=True), H.sum(1, keepdims=True)
                gain = cg**2/(ch+1) + (tg-cg)**2/(th-ch+1) - tg**2/(th+1)
                j, t = np.unravel_index(np.argmax(gain), gain.shape)
                lv = -cg[j, t]/(ch[j, t]+1); rv = -(tg[j, 0]-cg[j, t])/(th[j, 0]-ch[j, t]+1)
                left = B[:, j] <= t
                F[:, k] += self.lr * np.where(left, lv, rv)
                rd.append((j, t, lv, rv))
            self.trees.append(rd)
        return self
    def decision(self, X):
        B = bins_apply(X, self.edges); F = np.tile(self.f0, (len(X), 1)).astype(float)
        for rd in self.trees:
            for k, (j, t, lv, rv) in enumerate(rd): F[:, k] += self.lr * np.where(B[:, j] <= t, lv, rv)
        return F
    def predict(self, X): return self.decision(X).argmax(1)

class RF:
    def __init__(self, trees=100, depth=6, seed=0): self.T, self.D, self.rng = trees, depth, np.random.default_rng(seed)
    def fit(self, X, y, K):
        self.K = K; self.edges = bins_fit(X); B = bins_apply(X, self.edges); n, f = B.shape; self.trees = []
        m = max(1, int(np.sqrt(f)))
        for _ in range(self.T):
            idx = self.rng.integers(0, n, n); self.trees.append(self._grow(B[idx], y[idx], 0, m))
        return self
    def _grow(self, B, y, d, m):
        cnt = np.bincount(y, minlength=self.K)
        if d >= self.D or (cnt > 0).sum() == 1 or len(y) < 4: return cnt / cnt.sum()
        fs = self.rng.choice(B.shape[1], m, replace=False); best = (0, None)
        Y = np.eye(self.K)[y]
        for j in fs:
            H = np.zeros((NB, self.K)); np.add.at(H, B[:, j], Y); c = H.cumsum(0)[:-1]; tot = H.sum(0)
            nl = c.sum(1); nr = len(y) - nl
            ok = (nl > 0) & (nr > 0)
            if not ok.any(): continue
            gl = 1 - ((c/np.maximum(nl[:, None], 1))**2).sum(1); r = tot - c; gr = 1 - ((r/np.maximum(nr[:, None], 1))**2).sum(1)
            imp = np.where(ok, (nl*gl + nr*gr)/len(y), 9)
            t = imp.argmin(); gain = (1 - ((tot/len(y))**2).sum()) - imp[t]
            if gain > best[0]: best = (gain, (j, t))
        if best[1] is None: return cnt / cnt.sum()
        j, t = best[1]; L = B[:, j] <= t
        return (j, t, self._grow(B[L], y[L], d+1, m), self._grow(B[~L], y[~L], d+1, m))
    def _pred(self, tr, b):
        while isinstance(tr, tuple): tr = tr[2] if b[tr[0]] <= tr[1] else tr[3]
        return tr
    def predict(self, X):
        B = bins_apply(X, self.edges)
        P = np.array([[self._pred(tr, b) for tr in self.trees] for b in B]).mean(1)
        return P.argmax(1)

def folds(groups, k=5):
    ug = sorted(set(groups)); rng = np.random.default_rng(1); rng.shuffle(ug)
    fid = {g: i % k for i, g in enumerate(ug)}
    return np.array([fid[g] for g in groups])

def oof(model_fn, X, y, groups, K, k=5):
    f = folds(groups, k); pred = np.zeros(len(y), int)
    for i in range(k):
        tr, te = f != i, f == i
        pred[te] = model_fn().fit(X[tr], y[tr], K).predict(X[te])
    return pred

def oof_majority(y, groups, K, k=5):
    f = folds(groups, k); pred = np.zeros(len(y), int)
    for i in range(k): pred[f == i] = np.bincount(y[f != i], minlength=K).argmax()
    return pred

def knn_oof(S, y, groups, K, kk=7, k=5):
    f = folds(groups, k); pred = np.zeros(len(y), int)
    for i in range(k):
        tr, te = np.where(f != i)[0], np.where(f == i)[0]
        for a in te:
            nn = tr[np.argsort(-S[a, tr])[:kk]]; pred[a] = np.bincount(y[nn], minlength=K).argmax()
    return pred
