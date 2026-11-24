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
FONDO_HONDO = "#0b1026"   # indigo casi negro, esquinas
FONDO_MEDIO = "#141b3d"   # indigo del centro
AZULES = ["#4c6ef5", "#5e8bd8", "#7aa2f7", "#5ec8e5", "#8ab4e8"]
LAVANDA = "#9d7cd8"
AMBAR = "#d9a66c"         # unico acento calido, muy medido
HUMO = "#c8d3f5"

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
        ax.add_patch(Rectangle((x, y), lado, lado,
                               facecolor=rng.choice(AZULES), alpha=rng.uniform(0.10, 0.34),
                               edgecolor="none", zorder=2))

# --- trazas -----------------------------------------------------------------
N = 170
for i in range(N):
    theta0 = rng.uniform(0, 2 * np.pi)
    largo = rng.uniform(H * 0.30, W * 0.72)
    curv = rng.normal(0, 1.0) * 0.0011
    p = rng.random()
    if p < 0.06:
        color, grosor, alpha = AMBAR, rng.uniform(1.4, 2.4), rng.uniform(0.40, 0.62)
    elif p < 0.16:
        color, grosor, alpha = LAVANDA, rng.uniform(1.0, 2.0), rng.uniform(0.28, 0.48)
    elif p < 0.26:
        color, grosor, alpha = HUMO, rng.uniform(0.7, 1.4), rng.uniform(0.18, 0.34)
    else:
        color, grosor, alpha = rng.choice(AZULES), rng.uniform(0.8, 2.2), rng.uniform(0.22, 0.52)
    traza(theta0, largo, curv, color, grosor, alpha)

# --- algunos "loopers": particulas de bajo momento que se enrollan -----------
for _ in range(5):
    theta0 = rng.uniform(0, 2 * np.pi)
    traza(theta0, rng.uniform(W * 0.30, W * 0.55),
          rng.choice([-1, 1]) * rng.uniform(0.0065, 0.0105),
          rng.choice(AZULES), rng.uniform(0.9, 1.6), rng.uniform(0.20, 0.38))

# --- vertice ----------------------------------------------------------------
for radio_glow, alfa in [(95, 0.05), (62, 0.07), (34, 0.11), (16, 0.18)]:
    ax.scatter([CX], [CY], s=radio_glow ** 2 / 2.4, color="#bcd3ff",
               alpha=alfa, edgecolors="none", zorder=5)
ax.scatter([CX], [CY], s=42, color="#eaf1ff", alpha=0.85, edgecolors="none", zorder=6)

# --- texto ------------------------------------------------------------------
TX = W * 0.545
ax.text(TX, H * 0.63, "HiggsNet", color="#eef2ff", fontsize=52,
        fontweight="bold", family="DejaVu Sans", ha="left", va="center",
        zorder=9, alpha=0.96)
ax.text(TX, H * 0.63, "HiggsNet", color="#7aa2f7", fontsize=52,
        fontweight="bold", family="DejaVu Sans", ha="left", va="center",
        zorder=8, alpha=0.30)

ax.plot([TX, TX + 168], [H * 0.505, H * 0.505], color="#5ec8e5", lw=2.2,
        alpha=0.55, zorder=9, solid_capstyle="round")

ax.text(TX, H * 0.425, "Deep ensembles sobre 11 millones", color="#9fb3e8",
        fontsize=15.5, family="DejaVu Sans", ha="left", va="center", zorder=9, alpha=0.92)
ax.text(TX, H * 0.345, "de colisiones de partículas", color="#9fb3e8",
        fontsize=15.5, family="DejaVu Sans", ha="left", va="center", zorder=9, alpha=0.92)
ax.text(TX, H * 0.235, "Bagging  ·  Optuna  ·  Stacking Ensemble", color="#d9a66c",
        fontsize=11.5, family="DejaVu Sans", ha="left", va="center",
        zorder=9, alpha=0.72)

# --- grano de pelicula y vineta: la firma lo-fi -----------------------------
grano = rng.normal(0, 1, (H, W, 1)).repeat(3, axis=2)
grano = (grano - grano.min()) / (grano.max() - grano.min())
ax.imshow(grano, extent=[0, W, 0, H], origin="lower", cmap=None,
          alpha=0.055, zorder=10, interpolation="nearest")

vineta = np.clip(dist / 0.95, 0, 1) ** 2.1
negro = np.zeros((H, W, 4))
negro[..., 3] = vineta * 0.55
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
