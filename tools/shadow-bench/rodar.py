import numpy as np, banco
import os
M = np.load(os.environ.get('SHADOW_NPY', 'world.npy')).astype(np.float64)/65535.0

def _ero(a,n):
    for _ in range(n): a = a & np.roll(a,1,0) & np.roll(a,-1,0) & np.roll(a,1,1) & np.roll(a,-1,1)
    return a
def _dil(a,n):
    for _ in range(n): a = a | np.roll(a,1,0) | np.roll(a,-1,0) | np.roll(a,1,1) | np.roll(a,-1,1)
    return a
def espeto(img, r=2):
    s = img < 0.5
    if s.sum()==0 or (~s).sum()==0: return np.nan
    return float((s & ~_dil(_ero(s,r),r)).sum()/max(s.sum(),1) +
                 (_ero(_dil(s,r),r) & ~s).sum()/max((~s).sum(),1))

def regioes(sl, n=48, passo=64, minacesa=0.12, maxacesa=0.88, maximo=40):
    """Regioes que TEM borda. O criterio unico e esse: a verdade nao pode ser quase toda
       clara nem quase toda escura, senao a medida mede o vazio e todo kernel empata.
       A primeira versao filtrava por fracao de oclusor no mapa e deixou passar doze
       regioes 100% em sombra, nas quais o SMSR nao tinha o que fazer e empatava com tudo."""
    d = M[sl]; out=[]
    for y0 in range(0, 2048-n, passo):
        for x0 in range(0, 2048-n, passo):
            c = d[y0:y0+n, x0:x0+n]
            if (c < 1.0).mean() < 0.5: continue
            cn = banco.Cena(d, x0, y0, n, 4)
            a = float(cn.nivel(1).duro(0.0).mean())
            if minacesa < a < maxacesa:
                out.append((x0, y0, a))
    out.sort(key=lambda t: abs(t[2]-0.5))     # as mais equilibradas primeiro
    return [(x,y) for x,y,_ in out[:maximo]]

def avaliar(sl, regs, k=4, up=8, n=64, kernels=None):
    tot = {}
    for x0,y0 in regs:
        c = banco.Cena(M[sl], x0, y0, n, up)
        verdade = c.nivel(1).duro(0.0)
        g = c.nivel(k)
        for nome, fn in kernels.items():
            img = fn(g)
            e = float(np.abs(img-verdade).mean())
            d = img - verdade
            esc = float(np.maximum(-d,0).mean()); cla = float(np.maximum(d,0).mean())
            sp = espeto(img)
            a = tot.setdefault(nome, [0,0.0,0.0,0.0,0.0,0])
            a[0]+=1; a[1]+=e; a[2]+=esc; a[3]+=cla
            if np.isfinite(sp): a[4]+=sp; a[5]+=1
    return tot

if __name__ == "__main__":
    kernels = {
        "por texel": lambda g: g.duro(0.0),
        "filtro 4":  lambda g: g.pcf4(0.0),
        "SMSR":      lambda g: g.smsr(0.0, 16),
    }
    for sl in (0,1):
        regs = regioes(sl)
        print(f"\n=== fatia {sl}: {len(regs)} regioes, mapa 512 contra a verdade de 2048 ===")
        t = avaliar(sl, regs, kernels=kernels)
        print(f"  {'':12s} {'erro':>8} {'escurece':>10} {'clareia':>9} {'espeto':>9}")
        for nome,a in t.items():
            print(f"  {nome:12s} {a[1]/a[0]:8.4f} {100*a[2]/a[0]:9.2f}% {100*a[3]/a[0]:8.2f}%"
                  f" {a[4]/max(a[5],1):9.4f}")
