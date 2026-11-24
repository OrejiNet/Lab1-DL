# -*- coding: utf-8 -*-
"""Banner lo-fi del repositorio: una colision de particulas en clave azul.

La imagen cita los event displays del CMS ---trazas curvas saliendo del vertice,
hits del detector dispersos alrededor--- pero en una paleta apagada de azules
con grano de pelicula, en vez del azul electrico saturado del original.

Salida: higgsnet_banner_lofi.png (1200x627, el formato de portada de LinkedIn).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

W, H = 1200, 627
DPI = 100
SEED = 42

# --- paleta -----------------------------------------------------------------
FONDO_HONDO = "#0a0e24"   # indigo casi negro, esquinas
FONDO_MEDIO = "#1a1f4d"   # indigo con algo de violeta en el centro

# Paleta synthwave apagada: el azul sigue mandando, pero acompanado.
FRIOS = ["#4c6ef5", "#5ec8e5", "#2ac3de", "#73daca", "#7aa2f7", "#89ddff"]
VIOLETAS = ["#bb9af7", "#9d7cd8", "#c3a6ff"]
CALIDOS = ["#ff9e64", "#e0af68", "#f7768e", "#e06c9f"]
HUMO = "#d5dcf7"

rng = np.random.default_rng(SEED)

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

# --- fondo: degradado radial suave hacia el vertice -------------------------
CX, CY = W * 0.30, H * 0.52
yy, xx = np.mgrid[0:H, 0:W]
dist = np.sqrt(((xx - CX) / W) ** 2 + ((yy - CY) / H) ** 2)
campo = np.clip(1 - dist / 0.85, 0, 1) ** 1.6

c_hondo = np.array(matplotlib.colors.to_rgb(FONDO_HONDO))
c_medio = np.array(matplotlib.colors.to_rgb(FONDO_MEDIO))
fondo = c_hondo[None, None, :] + campo[..., None] * (c_medio - c_hondo)[None, None, :]
ax.imshow(fondo, extent=[0, W, 0, H], origin="lower", zorder=0, interpolation="bilinear")


def traza(theta0, largo, curvatura, color, grosor, alpha, zorder=3):
    """Traza de una particula cargada: una espiral que sale del vertice."""
    r = np.linspace(0, largo, 260)
    ang = theta0 + curvatura * r
    x = CX + r * np.cos(ang)
    y = CY + r * np.sin(ang) * 0.72          # el detector se ve achatado
    # halo tenue debajo, para el brillo suave
    ax.plot(x, y, color=color, lw=grosor * 3.2, alpha=alpha * 0.16,
            solid_capstyle="round", zorder=zorder - 1)
    ax.plot(x, y, color=color, lw=grosor, alpha=alpha,
            solid_capstyle="round", zorder=zorder)


# --- hits del detector: cuadraditos dispersos, mas densos cerca del vertice --
n_hits = 520
radio = np.abs(rng.normal(0, 0.30, n_hits)) * W * 0.95
ang = rng.uniform(0, 2 * np.pi, n_hits)
hx = CX + radio * np.cos(ang)
hy = CY + radio * np.sin(ang) * 0.72
for x, y in zip(hx, hy):
    if -40 < x < W + 40 and -40 < y < H + 40:
        lado = rng.uniform(2.5, 6.0)
        paleta_hit = FRIOS if rng.random() < 0.78 else VIOLETAS
        ax.add_patch(Rectangle((x, y), lado, lado,
                               facecolor=rng.choice(paleta_hit),
                               alpha=rng.uniform(0.16, 0.48),
                               edgecolor="none", zorder=2))

# --- trazas -----------------------------------------------------------------
N = 170
for i in range(N):
    theta0 = rng.uniform(0, 2 * np.pi)
    largo = rng.uniform(H * 0.30, W * 0.72)
    curv = rng.normal(0, 1.0) * 0.0011
    p = rng.random()
    if p < 0.20:
        color, grosor, alpha = (rng.choice(CALIDOS), rng.uniform(1.3, 2.6),
                                rng.uniform(0.45, 0.78))
    elif p < 0.40:
        color, grosor, alpha = (rng.choice(VIOLETAS), rng.uniform(1.0, 2.3),
                                rng.uniform(0.38, 0.68))
    elif p < 0.47:
        color, grosor, alpha = HUMO, rng.uniform(0.7, 1.4), rng.uniform(0.22, 0.42)
    else:
        color, grosor, alpha = (rng.choice(FRIOS), rng.uniform(0.9, 2.5),
                                rng.uniform(0.34, 0.74))
    traza(theta0, largo, curv, color, grosor, alpha)

# --- algunos "loopers": particulas de bajo momento que se enrollan -----------
for _ in range(5):
    theta0 = rng.uniform(0, 2 * np.pi)
    traza(theta0, rng.uniform(W * 0.30, W * 0.55),
          rng.choice([-1, 1]) * rng.uniform(0.0065, 0.0105),
          rng.choice(FRIOS + VIOLETAS), rng.uniform(1.0, 1.8),
          rng.uniform(0.30, 0.52))

# --- vertice ----------------------------------------------------------------
for radio_glow, alfa, tono in [(120, 0.05, "#7aa2f7"), (95, 0.07, "#89ddff"),
                               (62, 0.10, "#c3a6ff"), (34, 0.16, "#ffc9a0"),
                               (16, 0.26, "#ffe0c0")]:
    ax.scatter([CX], [CY], s=radio_glow ** 2 / 2.4, color=tono,
               alpha=alfa, edgecolors="none", zorder=5)
ax.scatter([CX], [CY], s=52, color="#fff4e8", alpha=0.95, edgecolors="none", zorder=6)

# --- velo degradado bajo el texto -------------------------------------------
# Va por encima de las trazas (zorder 2-6) y por debajo del texto (8-9): el
# lado derecho se oscurece lo justo para que el texto se lea sin tapar la
# imagen.
rampa = np.clip((xx / W - 0.40) / 0.42, 0, 1) ** 1.4
velo = np.zeros((H, W, 4))
velo[..., 0] = 0.039
velo[..., 1] = 0.055
velo[..., 2] = 0.141
velo[..., 3] = rampa * 0.62
ax.imshow(velo, extent=[0, W, 0, H], origin="lower", zorder=7,
          interpolation="bilinear")

# --- texto ------------------------------------------------------------------
TX = W * 0.545
ax.text(TX, H * 0.63, "HiggsNet", color="#eef2ff", fontsize=52,
        fontweight="bold", family="DejaVu Sans", ha="left", va="center",
        zorder=9, alpha=0.96)
ax.text(TX, H * 0.63, "HiggsNet", color="#bb9af7", fontsize=52,
        fontweight="bold", family="DejaVu Sans", ha="left", va="center",
        zorder=8, alpha=0.30)

ax.plot([TX, TX + 168], [H * 0.505, H * 0.505], color="#2ac3de", lw=2.6,
        alpha=0.80, zorder=9, solid_capstyle="round")

ax.text(TX, H * 0.425, "Deep ensembles sobre 11 millones", color="#c5d2f5",
        fontsize=15.5, family="DejaVu Sans", ha="left", va="center", zorder=9, alpha=0.92)
ax.text(TX, H * 0.345, "de colisiones de partículas", color="#c5d2f5",
        fontsize=15.5, family="DejaVu Sans", ha="left", va="center", zorder=9, alpha=0.92)
ax.text(TX, H * 0.235, "Bagging  ·  Optuna  ·  Stacking Ensemble", color="#ff9e64",
        fontsize=11.5, family="DejaVu Sans", ha="left", va="center",
        zorder=9, alpha=0.72)

# --- grano de pelicula y vineta: la firma lo-fi -----------------------------
grano = rng.normal(0, 1, (H, W, 1)).repeat(3, axis=2)
grano = (grano - grano.min()) / (grano.max() - grano.min())
ax.imshow(grano, extent=[0, W, 0, H], origin="lower", cmap=None,
          alpha=0.055, zorder=10, interpolation="nearest")

vineta = np.clip(dist / 0.95, 0, 1) ** 2.1
negro = np.zeros((H, W, 4))
negro[..., 3] = vineta * 0.38
ax.imshow(negro, extent=[0, W, 0, H], origin="lower", zorder=11,
          interpolation="bilinear")

# lineas de barrido horizontales, muy tenues
barrido = np.zeros((H, W, 4))
barrido[::3, :, 3] = 0.035
ax.imshow(barrido, extent=[0, W, 0, H], origin="lower", zorder=12,
          interpolation="nearest")

SALIDA = Path(__file__).with_name("higgsnet_banner_lofi.png")
fig.savefig(SALIDA, dpi=DPI, facecolor=FONDO_HONDO, pad_inches=0)
plt.close(fig)

# El grano arruina la compresion PNG y deja el archivo sobre 1 MB. Cuantizar a
# una paleta de 256 colores con difuminado lo deja a la mitad sin diferencia
# apreciable: la imagen ya usa pocos tonos.
from PIL import Image

im = Image.open(SALIDA).convert("RGB")
im.quantize(colors=256, method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.FLOYDSTEINBERG).save(SALIDA, optimize=True)
print(f"banner generado: {SALIDA}  ({SALIDA.stat().st_size/1024:.0f} KB)")
