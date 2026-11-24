# -*- coding: utf-8 -*-
"""
Genera el banner de presentación del proyecto (1200x627 px, medida estándar de LinkedIn).

Uso:
    python assets/generar_banner.py

Todo se dibuja con matplotlib: no hay imágenes externas ni fuentes que instalar.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Circle

# ---------------------------------------------------------------- paleta
FONDO_SUP = "#0d1628"
FONDO_INF = "#060b16"
BLANCO = "#f8fafc"
GRIS = "#94a3b8"
GRIS_CLARO = "#cbd5e1"
CIAN = "#38bdf8"
AZUL = "#3b82f6"
VIOLETA = "#818cf8"

W, H = 1200, 627

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

# ---------------------------------------------------------------- fondo
cmap = LinearSegmentedColormap.from_list("fondo", [FONDO_INF, FONDO_SUP])
ax.imshow(
    np.linspace(0, 1, 256).reshape(-1, 1),
    extent=(0, W, 0, H),
    aspect="auto",
    cmap=cmap,
    zorder=0,
)

# Polvo de partículas de fondo (determinista)
rng = np.random.default_rng(7)
ax.scatter(
    rng.uniform(0, W, 220),
    rng.uniform(0, H, 220),
    s=rng.uniform(0.3, 5.0, 220),
    c=CIAN,
    alpha=0.18,
    zorder=1,
    linewidths=0,
)

# Halo suave detrás del diagrama
for r, a in [(300, 0.030), (210, 0.035), (130, 0.040)]:
    ax.add_patch(Circle((905, 320), r, color=AZUL, alpha=a, zorder=1, linewidth=0))


# ---------------------------------------------------------------- helpers
def chip(x, y, texto, color_borde, color_texto):
    """Etiqueta redondeada; devuelve el x donde termina, para encadenarlas."""
    ancho = 26 + 10.2 * len(texto)
    ax.add_patch(
        FancyBboxPatch(
            (x, y), ancho, 40,
            boxstyle="round,pad=0,rounding_size=20",
            facecolor="#ffffff", alpha=0.055,
            edgecolor=color_borde, linewidth=1.3, zorder=3,
        )
    )
    ax.text(x + ancho / 2, y + 20, texto, color=color_texto, fontsize=12.5,
            ha="center", va="center", fontweight="medium", zorder=4)
    return x + ancho + 13


def caja(cx, cy, ancho, alto, texto, color, fs=13, grosor=1.6, alpha_fondo=0.09):
    ax.add_patch(
        FancyBboxPatch(
            (cx - ancho / 2, cy - alto / 2), ancho, alto,
            boxstyle="round,pad=0,rounding_size=9",
            facecolor=color, alpha=alpha_fondo,
            edgecolor=color, linewidth=grosor, zorder=3,
        )
    )
    ax.text(cx, cy, texto, color=BLANCO, fontsize=fs, ha="center", va="center",
            fontweight="bold", zorder=4, linespacing=1.35)


# ---------------------------------------------------------------- texto (izquierda)
ax.text(70, 556, "M A G Í S T E R   E N   I N F O R M Á T I C A", color=CIAN,
        fontsize=12.5, fontweight="bold", va="center", zorder=4)
ax.text(70, 530, "Fundamentos de Aprendizaje Profundo y Aplicaciones", color=GRIS,
        fontsize=12.5, va="center", zorder=4)

ax.text(68, 445, "HiggsNet", color=BLANCO, fontsize=66, fontweight="bold",
        va="center", zorder=4)

ax.add_patch(FancyBboxPatch((70, 392), 96, 5, boxstyle="round,pad=0,rounding_size=2.5",
                            facecolor=CIAN, edgecolor="none", zorder=4))

ax.text(70, 340, "Deep ensembles sobre 11 millones de", color=GRIS_CLARO,
        fontsize=22, va="center", zorder=4)
ax.text(70, 304, "colisiones de partículas", color=GRIS_CLARO,
        fontsize=22, va="center", zorder=4)

ax.text(70, 228, "Clasificar señal de bosón de Higgs frente al ruido de fondo,\n"
                 "entrenando bajo las restricciones reales de memoria y\n"
                 "cómputo de un entorno cloud gratuito.",
        color=GRIS, fontsize=13.5, va="center", zorder=4, linespacing=1.75)

x = chip(70, 96, "1.000.000 × 28", CIAN, BLANCO)
x = chip(x, 96, "PyTorch 2.x", AZUL, GRIS_CLARO)
x = chip(x, 96, "Bagging + Stacking", VIOLETA, GRIS_CLARO)
chip(x, 96, "Google Colab", "#475569", GRIS)

# ---------------------------------------------------------------- diagrama (derecha)
BURST = (726, 320)

# Colisión: líneas radiales
ang = np.linspace(0, 2 * np.pi, 15, endpoint=False)
for i, a in enumerate(ang):
    largo = 30 if i % 2 == 0 else 19
    ax.plot([BURST[0], BURST[0] + largo * np.cos(a)],
            [BURST[1], BURST[1] + largo * np.sin(a)],
            color=CIAN, linewidth=1.5, alpha=0.75, zorder=3,
            solid_capstyle="round")
ax.add_patch(Circle(BURST, 9, color=BLANCO, zorder=4, linewidth=0))
ax.add_patch(Circle(BURST, 17, color=CIAN, alpha=0.30, zorder=3, linewidth=0))
ax.text(BURST[0], BURST[1] - 56, "datos", color=GRIS, fontsize=11.5,
        ha="center", va="center", zorder=4)

# Modelos base
ys = [470, 395, 320, 245, 170]
CX_BASE, ANCHO_B, ALTO_B = 898, 132, 46
for i, y in enumerate(ys):
    ax.plot([BURST[0] + 34, CX_BASE - ANCHO_B / 2], [BURST[1], y],
            color=AZUL, linewidth=1.25, alpha=0.5, zorder=2)
    caja(CX_BASE, y, ANCHO_B, ALTO_B, f"FFNN {i+1}", AZUL, fs=13.5)

ax.text(CX_BASE, 520, "modelos base · bootstrap", color=GRIS, fontsize=11.5,
        ha="center", va="center", zorder=4)

# Meta-red
CX_META = 1086
for y in ys:
    ax.plot([CX_BASE + ANCHO_B / 2, CX_META - 56], [y, 320],
            color=VIOLETA, linewidth=1.25, alpha=0.5, zorder=2)
caja(CX_META, 320, 112, 104, "Meta-red\nStacking", VIOLETA, fs=13.5,
     grosor=2.0, alpha_fondo=0.13)

ax.text(CX_META, 236, "señal / fondo", color=CIAN, fontsize=11.5,
        ha="center", va="center", fontweight="bold", zorder=4)

# ---------------------------------------------------------------- salida
destino = Path(__file__).resolve().parent / "higgsnet_banner.png"
fig.savefig(destino, dpi=100, facecolor=FONDO_INF)
plt.close(fig)
print(f"Banner generado: {destino}")
print(f"Dimensiones: {W}x{H} px (formato recomendado por LinkedIn)")
