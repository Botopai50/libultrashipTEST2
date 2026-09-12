import struct, numpy as np, sys
raw = open('world.sds','rb').read()
magic,w,h,slices = struct.unpack('<4I', raw[:16])
assert magic == 0x31534453, magic
print("SDS1  %dx%d  %d fatias" % (w,h,slices))
off = 16
out = np.zeros((slices,h,w), dtype=np.uint16)
for s in range(slices):
    for y in range(h):
        (runs,) = struct.unpack_from('<I', raw, off); off += 4
        # runs consecutivos: (u16 valor, u32 comprimento)
        buf = np.frombuffer(raw, dtype=np.uint8, count=runs*6, offset=off).reshape(runs,6)
        off += runs*6
        vals = buf[:,0].astype(np.uint16) | (buf[:,1].astype(np.uint16) << 8)
        lens = (buf[:,2].astype(np.uint32) | (buf[:,3].astype(np.uint32) << 8)
                | (buf[:,4].astype(np.uint32) << 16) | (buf[:,5].astype(np.uint32) << 24))
        row = np.repeat(vals, lens)
        assert row.size == w, (s,y,row.size)
        out[s,y] = row
print("consumidos %d de %d bytes" % (off, len(raw)))
np.save('world.npy', out)
for s in range(slices):
    d = out[s]
    naomax = d < 65535
    print("fatia %d: %.1f%% com conteudo, profundidade %d..%d (%.5f..%.5f)"
          % (s, 100.0*naomax.mean(), d.min(), d.max(), d.min()/65535, d.max()/65535))
