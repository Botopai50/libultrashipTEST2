# BANCADA: ambiente simulado a partir do arquivo de captura, COM verdade de referencia.
#
# O que faltava em tudo o que tentei antes era referencia. Comparar dois kernels entre si
# so diz qual e diferente, nao qual esta certo, e foi assim que mandei uma piora achando
# que era melhoria.
#
# A verdade sai do proprio arquivo. O mapa capturado tem 2048; reduzido para 512 ele vira
# o que um mapa grosseiro teria guardado (minimo por bloco, que e o que um depth buffer com
# LESS registra). Entao:
#
#     verdade = a sombra calculada no mapa de 2048, amostrada fino
#     teste   = a mesma cena, mesmo receptor, kernel rodando no mapa de 512
#
# Quatro vezes mais fino nao e infinito, mas e geometria REAL e uma referencia REAL, e o
# erro que ele expoe e exatamente o que a reconstrucao deveria estar recuperando.
import numpy as np

Q = 1.0/65535.0

def reduzir(mapa, k):
    """Mapa grosseiro: minimo por bloco k x k -- o oclusor mais proximo, que e o que o
       depth buffer guardaria naquela resolucao."""
    h, w = mapa.shape
    return mapa[:h//k*k, :w//k*k].reshape(h//k, k, w//k, k).min((1, 3))

class Cena:
    """Um pedaco do mundo: mapa, receptor ajustado a ele, e a grade de amostragem."""
    def __init__(self, mapa2048, x0, y0, n, up, atras=40.0):
        self.up, self.n = up, n
        crop = mapa2048[y0:y0+n, x0:x0+n]
        ok = crop < 1.0
        self.coef = ajustar_plano(crop, ok)
        a, b, c = self.coef
        gy, gx = np.mgrid[0:n*up, 0:n*up]
        # coordenada CONTINUA em texels do mapa fino, com origem no recorte
        self.fx = (gx + 0.5)/up
        self.fy = (gy + 0.5)/up
        self.z = a*self.fx + b*self.fy + c + atras*Q
        self.fino = crop
        self.x0, self.y0 = x0, y0
        # gradiente do receptor por texel FINO
        self.grad_fino = np.array([a, b])

    def nivel(self, k):
        """A mesma cena vista por um mapa k vezes mais grosseiro."""
        return Nivel(self, k)

def ajustar_plano(dep, mask, passos=6):
    """O RECEPTOR: a superficie de tras, que e o chao. Ajustada so aos texels mais LONGES
       que a mediana, e mantendo-os -- a versao anterior reajustava de forma a empurrar o
       plano para tras de tudo, e entao a cena inteira ficava em sombra e nao havia borda
       nenhuma para medir. Doze das dezesseis regioes do primeiro teste eram isso."""
    ys, xs = np.mgrid[0:dep.shape[0], 0:dep.shape[1]]
    if not mask.any():
        return np.array([0.0, 0.0, 0.6])
    med = float(np.median(dep[mask]))
    keep = mask & (dep >= med)
    coef = np.array([0.0, 0.0, med])
    for _ in range(passos):
        if keep.sum() < 40: break
        A = np.stack([xs[keep], ys[keep], np.ones(keep.sum())], 1)
        coef, *_ = np.linalg.lstsq(A, dep[keep], rcond=None)
        r = dep - (coef[0]*xs + coef[1]*ys + coef[2])
        sd = np.std(r[keep]) + 1e-12
        keep = mask & (dep >= med) & (np.abs(r) < 2.5*sd)
    return coef

class Nivel:
    """O mapa numa resolucao, com as contas de amostragem que o kernel precisa."""
    def __init__(self, cena, k):
        self.cena, self.k = cena, k
        self.m = reduzir(cena.fino, k) if k > 1 else cena.fino
        self.size = self.m.shape[0]
        # coordenada em texels DESTE nivel
        self.tx = cena.fx/k
        self.ty = cena.fy/k
        self.z = cena.z
        self.grad = cena.grad_fino*k          # por texel deste nivel
        self.ox = np.floor(self.tx).astype(np.int64)
        self.oy = np.floor(self.ty).astype(np.int64)

    def prof(self, ix, iy):
        ix = np.clip(ix, 0, self.size-1); iy = np.clip(iy, 0, self.size-1)
        return self.m[iy, ix]

    def aceso(self, ix, iy, eps=0.0):
        """Teste por texel, com o plano do receptor avaliado no centro daquele texel."""
        dentro = (ix >= 0) & (ix < self.size) & (iy >= 0) & (iy < self.size)
        zl = self.prof(ix, iy)
        zc = self.z + self.grad[0]*((ix+0.5) - self.tx) + self.grad[1]*((iy+0.5) - self.ty)
        out = np.where(zc - eps <= zl, 1.0, 0.0)
        out = np.where(zl >= 1.0, 1.0, out)
        return np.where(dentro, out, 1.0)

    def duro(self, eps=0.0):
        return self.aceso(self.ox, self.oy, eps)

    def pcf4(self, eps=0.0):
        tp_x, tp_y = self.tx-0.5, self.ty-0.5
        bx = np.floor(tp_x).astype(np.int64); by = np.floor(tp_y).astype(np.int64)
        sx, sy = tp_x-bx, tp_y-by
        s00 = self.aceso(bx, by, eps);     s10 = self.aceso(bx+1, by, eps)
        s01 = self.aceso(bx, by+1, eps);   s11 = self.aceso(bx+1, by+1, eps)
        return s00*(1-sx)*(1-sy) + s10*sx*(1-sy) + s01*(1-sx)*sy + s11*sx*sy

    # ---- SmsrDiscontinuity ----
    def disc(self, ox, oy, s, eps):
        L = np.abs(self.aceso(ox-1, oy, eps) - s)
        R = np.abs(self.aceso(ox+1, oy, eps) - s)
        T = np.abs(self.aceso(ox, oy-1, eps) - s)
        B = np.abs(self.aceso(ox, oy+1, eps) - s)
        return (2.0*L + R)*0.25, (T + 2.0*B)*0.25

    # ---- SmsrTrace ----
    def trace(self, ox, oy, dx, dy, limit, init_dir, along_x, eps):
        dist_out = np.full(ox.shape, -(limit+1.0))
        found    = np.zeros(ox.shape)
        active   = np.ones(ox.shape, bool)
        ax, ay = (0,1) if along_x else (1,0)
        for dist in range(1, limit+1):
            cx, cy = ox + dx*dist, oy + dy*dist
            oob = (cx<0)|(cx>=self.size)|(cy<0)|(cy>=self.size)
            hit = active & oob
            dist_out[hit] = -float(dist); found[hit] = 0.0; active &= ~hit
            if not active.any(): break
            s = self.aceso(cx, cy, eps)
            end = active & (s < 0.5)
            dist_out[end] = float(dist); found[end] = 1.0; active &= ~end
            if not active.any(): break
            minus = 1.0 - self.aceso(cx-ax, cy-ay, eps)
            plus  = 1.0 - self.aceso(cx+ax, cy+ay, eps)
            dc = (minus + 2.0*plus)*0.25 if along_x else (2.0*minus + plus)*0.25
            beg = active & (dc != init_dir)
            dist_out[beg] = -float(dist); found[beg] = 1.0; active &= ~beg
            if not active.any(): break
        return dist_out, found

    @staticmethod
    def normalize(na, nf, pa, pf, p):
        trunc = (nf < 0.5) | (pf < 0.5)
        a1, a2 = na, pa
        kind = (a1>0).astype(float) + (a2>0).astype(float) - 1.0
        L = np.maximum(np.abs(a1)+np.abs(a2)-1.0, 1.0)
        po = np.where(a1 > a2, 1.0-p, p)
        don = (1.0 - np.maximum(a1,a2)/L) + po/L
        return np.where(trunc, 0.0, np.clip(don,0,1)), np.where(trunc, -1.0, kind)

    @staticmethod
    def vsmsr(dcx, dcy, domx, domy, donx, dony, donz, donw, px, py):
        out = np.ones(dcx.shape)
        vertical   = np.where((np.where(dcy==0.5, 1.0-py, py) < donx), 0.0, 1.0)
        horizontal = np.where((np.where(dcx==0.5, px, 1.0-px) < dony), 0.0, 1.0)
        # de baixo para cima, na ordem inversa do shader, para o primeiro caso vencer
        py2 = np.where(dcy==0.5, py, 1.0-py)
        out = np.where(1.0-donx < py2, 0.0, 1.0)                      # casos 6/7
        out = np.where(domy>0.5, vertical, out)                       # caso 5
        out = np.where(domx>0.5, horizontal, out)                     # caso 4
        out = np.where((domx>0.5)&(domy>0.5), np.minimum(horizontal,vertical), out)  # caso 8
        out = np.where(dcy==0.0, horizontal, out)                     # casos 11/12
        out = np.where(dcx==0.0, vertical, out)                       # casos 9/10
        out = np.where((dcx==0.75)|(dcy==0.75), 0.0, out)             # caso 3
        out = np.where((donz>0.0)|(donw>0.0), 0.0, out)               # caso 2
        out = np.where((donz<0.0)|(donw<0.0), 1.0, out)               # caso 1
        out = np.where((dcx==0.0)&(dcy==0.0), 1.0, out)
        return out

    def smsr(self, eps=0.0, limit=16):
        ox, oy = self.ox, self.oy
        s = self.aceso(ox, oy, eps)
        dcx, dcy = self.disc(ox, oy, s, eps)
        px, py = self.tx-ox, self.ty-oy
        corner = (dcx>0)&(dcy>0)
        domx = np.where(dcx>0, 1.0, 0.0); domy = np.where(dcy>0, 1.0, 0.0)
        if True:
            awayYy = np.where(dcy==0.25, 1, -1); awayXx = np.where(dcx==0.5, 1, -1)
            nYx,_ = self.disc(ox, oy+awayYy, 1.0, eps)
            _,nXy = self.disc(ox+awayXx, oy, 1.0, eps)
            domx = np.where(corner, np.where(nYx==dcx, 1.0, 0.0), domx)
            domy = np.where(corner, np.where(nXy==dcy, 1.0, 0.0), domy)
        donx=np.zeros(s.shape); dony=np.zeros(s.shape)
        donz=np.zeros(s.shape); donw=np.zeros(s.shape)
        need_h = (domy>0.5) | (corner & (domx==0) & (domy==0))
        if need_h.any():
            na,nf = self.trace(ox,oy,-1,0,limit,dcy,True,eps)
            pa,pf = self.trace(ox,oy, 1,0,limit,dcy,True,eps)
            e,k = self.normalize(na,nf,pa,pf,px)
            donx = np.where(need_h, e, donx); donz = np.where(need_h, k, donz)
        need_v = domx>0.5
        if need_v.any():
            na,nf = self.trace(ox,oy,0,-1,limit,dcx,False,eps)
            pa,pf = self.trace(ox,oy,0, 1,limit,dcx,False,eps)
            e,k = self.normalize(na,nf,pa,pf,py)
            dony = np.where(need_v, e, dony); donw = np.where(need_v, k, donw)
        v = self.vsmsr(dcx,dcy,domx,domy,donx,dony,donz,donw,px,py)
        res = s * np.where((dcx==0)&(dcy==0), 1.0, v)
        res = np.where(s<0.5, s, res)
        return res


class Complementado:
    """O mesmo nivel com aceso/apagado trocados. Serve para rodar a reconstrucao a partir
       do lado ESCURO: o SMSR so sabe comer luz (`s * vSMSR`, e s=0 devolve 0 direto), entao
       o lado escuro nunca e reconstruido. Rodando no complemento e devolvendo 1-v, ele
       reconstroi os dois lados com o mesmo codigo."""
    def __init__(self, nivel):
        self._n = nivel
        for a in ("size","tx","ty","ox","oy","z","grad","m","k","cena"):
            setattr(self, a, getattr(nivel, a))
    def prof(self, ix, iy): return self._n.prof(ix, iy)
    def aceso(self, ix, iy, eps=0.0): return 1.0 - self._n.aceso(ix, iy, eps)
    disc = Nivel.disc
    trace = Nivel.trace
    normalize = staticmethod(Nivel.normalize)
    vsmsr = staticmethod(Nivel.vsmsr)
    smsr = Nivel.smsr


def smsr_dois_lados(nivel, eps=0.0, limit=16):
    """Reconstrucao dos DOIS lados. Onde o texel esta aceso, o SMSR de sempre; onde esta
       apagado, a mesma reconstrucao rodada no complemento e devolvida invertida."""
    s = nivel.aceso(nivel.ox, nivel.oy, eps)
    claro = nivel.smsr(eps, limit)
    escuro = 1.0 - Complementado(nivel).smsr(eps, limit)
    return np.where(s >= 0.5, claro, escuro)


def smsr_guarda(nivel, eps=0.0, limit=16, modo="fina"):
    """SMSR que se recusa a agir onde suas premissas nao valem.

       O metodo supoe uma BORDA: de um lado sombra, do outro luz, e um segmento de reta a
       recuperar dentro do texel. Numa fresta fina -- um triangulo de luz de um ou dois
       texels de largura, que e o que sobra de um oclusor pequeno num mapa grosseiro -- nao
       ha borda unica, ha duas coladas. Reconstruir uma reta ali corta a fresta ao meio, e
       foi isso que a bancada mostrou: a verdade e um triangulo reto, o SMSR devolve uma
       curva concava menor.

       modo 'fina'   : a vizinhanca de 4 alterna muito (fresta ou salpico) -> nao mexe
       modo 'oposto' : os dois vizinhos no mesmo eixo concordam entre si e discordam do
                       centro -> o centro e uma lasca de um texel -> nao mexe
    """
    ox, oy = nivel.ox, nivel.oy
    s = nivel.aceso(ox, oy, eps)
    v = nivel.smsr(eps, limit)
    l = nivel.aceso(ox-1, oy, eps); r = nivel.aceso(ox+1, oy, eps)
    t = nivel.aceso(ox, oy-1, eps); b = nivel.aceso(ox, oy+1, eps)
    if modo == "fina":
        trocas = np.abs(l-s) + np.abs(r-s) + np.abs(t-s) + np.abs(b-s)
        return np.where(trocas >= 3.0, s, v)
    lasca = ((np.abs(l-s) > 0.5) & (np.abs(r-s) > 0.5)) | ((np.abs(t-s) > 0.5) & (np.abs(b-s) > 0.5))
    return np.where(lasca, s, v)
