# Shadow bench: the capture, replayed offline, against a reference

A shadow capture (`world.sds` + `capture.json`, written by the in-game button) is replayed
here as a scene: the stored depths, a receiver fitted to them, and any kernel run over the
result at whatever magnification you ask for. It exists so a reported artefact can be
studied without a build, and so a proposed fix is measured before it ships.

    python3 decode.py            # world.sds -> world.npy
    python3 rodar.py             # every kernel, every region with a boundary, both slices

## Why it needs a reference, and where the reference comes from

Comparing two kernels against each other says which is different, not which is right. That
gap is not theoretical: a change was measured, shipped, played, and came back worse, because
every metric used scored *where* the boundary landed and none scored whether it landed
softly.

The reference comes out of the capture itself. The map is 2048; reduced 4x by taking the
minimum of each block -- which is what a depth buffer with LESS would have stored at that
resolution -- it becomes what a coarse map would hold. So:

    truth = the shadow computed on the 2048 map
    test  = the same scene, same receiver, a kernel running on the 512 map

Four times finer is not infinite, but it is real geometry and a real reference, and the error
it exposes is exactly what a reconstruction is supposed to be recovering.

## Two mistakes this bench made first, both worth not repeating

**The receiver was fitted so badly that twelve of sixteen regions were entirely in shadow.**
The plane fit iterated in a way that pushed the receiver behind every surface in the crop, so
there was no boundary anywhere, every kernel scored identically, and the conclusion drawn
from it ("SMSR changes nothing") was an artefact of the harness. `ajustar_plano` now fits
only to texels farther than the median and keeps them.

**Regions were selected by how much occluder was in the map**, which does not imply a visible
boundary. They are now selected by the one thing that matters -- the truth being neither
almost all lit nor almost all shadowed -- and sorted so the most balanced come first.

**Magnification has to be checked too.** With `up=8` and `k=4` there are two screen pixels per
coarse texel, and a sub-texel reconstruction has no room to show anything. Use `up >= 4*k`.

## What it found

Twenty regions per slice, all with a real boundary, 512 against the 2048 truth:

| slice 0            | error  | darkens | lightens |
|--------------------|--------|---------|----------|
| per texel          | 0.0657 |   5.90% |    0.68% |
| four-tap filter    | 0.0979 |   6.22% |    3.57% |
| SMSR               | 0.0968 |   9.40% |    0.28% |
| SMSR, thin guard   | 0.0939 |   9.10% |    0.29% |
| SMSR, two-sided    | 0.1139 |   7.92% |    3.48% |

| slice 1            | error  | darkens | lightens |
|--------------------|--------|---------|----------|
| per texel          | 0.0996 |   9.71% |    0.25% |
| four-tap filter    | 0.1185 |   9.55% |    2.30% |
| SMSR               | 0.1204 |  11.86% |    0.18% |
| SMSR, thin guard   | 0.1198 |  11.79% |    0.19% |
| SMSR, two-sided    | 0.1128 |   9.74% |    1.54% |

SMSR adds two to three and a half points of shadow that is not in the truth, and returns
light essentially never -- 0.18% and 0.28% against the truth's own 0.25% and 0.68%. That is
`return s * vSMSR(...)`: where the texel reads lit the reconstruction can only take light
away, and where it reads shadowed the function returns before the reconstruction is used at
all. On a thin lit sliver -- what a small occluder leaves in a coarse map -- it cuts the
sliver in half and bends the straight edge into a concave curve. Rendered side by side
against the truth, that curve is the reported "spike".

Making it two-sided removes the bias exactly on slice 1 (9.74% against the truth's 9.71%) and
lowers the total error there, but costs on slice 0. It is not yet a fix, and it is not
shipped.
