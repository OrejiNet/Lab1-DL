"""
Codigo base del Laboratorio 1 - Fundamentos de Aprendizaje Profundo y Aplicaciones (2-2026).

Extraido sin modificaciones desde el notebook de ejemplo entregado por el profesor
(Laboratorio_1_Deep_Learning_2026_codigo_ejemplo.ipynb). Sirve como motor reutilizable:
FFNN con BatchNorm/Dropout, entrenamiento AdamW + Early Stopping, Bagging Bootstrap
y Stacking Ensemble con meta-red entrenable.

Las adaptaciones del grupo (dataset masivo, pos_weight, metricas, etc.) van en los
notebooks de notebooks/ o en modulos nuevos, no aqui.
"""



# ======================================================================================
# [Celda 1: Imports, Configuración de Hardware y Reproducibilidad]
# ======================================================================================

import os
import math
import random
from typing import Tuple, List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from ydata_profiling import ProfileReport, compare

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
    make_classification,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Configuración de precisión de multiplicación matricial en PyTorch 2.x.
# Permite aprovechar TensorFloat-32 (TF32) en GPUs modernas para acelerar el cálculo.
torch.set_float32_matmul_precision("high")

# Detección automática del acelerador de hardware disponible
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

os.makedirs("checkpoints", exist_ok=True)
print(f"[INFO] Dispositivo de cómputo seleccionado: {DEVICE}")


def set_seed(seed: int = 42) -> None:
    """
    Fija la semilla de todos los generadores de números aleatorios (Python, NumPy y PyTorch).
    Garantiza que la partición de datos, el muestreo bootstrap y la inicialización
    de pesos (He/Xavier) sean completamente reproducibles entre ejecuciones.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================================
# [Celda 2: Definición de Datasets y Prevención de Data Leakage]
# ======================================================================================

class BaseTabularDataset(Dataset):
    """
    Clase base compatible con PyTorch Dataset para datos tabulares.
    Conserva tanto los tensores para el entrenamiento como los metadatos (nombres de
    columnas y clases) para enriquecer las visualizaciones del EDA.
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        target_names: List[str],
        is_binary: bool = False,
    ):
        self.X_raw = np.asarray(X, dtype=np.float32)
        self.y_raw = np.asarray(y, dtype=np.float32 if is_binary else np.int64)
        self.feature_names = list(feature_names)
        self.target_names = list(target_names)
        self.is_binary = is_binary
        self.num_features = self.X_raw.shape[1]
        self.num_classes = 2 if is_binary else int(np.unique(y).size)

        self.X = torch.tensor(self.X_raw, dtype=torch.float32)
        self.y = torch.tensor(
            self.y_raw, dtype=torch.float32 if is_binary else torch.long
        )

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class SyntheticStructuredDataset(BaseTabularDataset):
    """
    Dataset Sintético con Estructura Estadística y Geométrica Coherente.
    --------------------------------------------------------------------
    Genera datos mediante distribuciones gaussianas ubicadas en los vértices de un hipercubo:
    - Variables Informativas ('Info_1'..'Info_4'): Definen geométricamente el límite de decisión.
    - Variables Colineales ('Corr_1'..'Corr_4'): Combinaciones lineales de las informativas
      (generan correlaciones fuertes visibles en el mapa de calor).
    - Variables de Ruido ('Ruido_1'..'Ruido_4'): Ruido gaussiano independiente (correlación ~ 0).
    """
    def __init__(
        self,
        n_samples: int = 900,
        n_informative: int = 4,
        n_redundant: int = 4,
        n_noise: int = 4,
        class_sep: float = 1.3,
        seed: int = 42,
    ):
        n_total = n_informative + n_redundant + n_noise
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_total,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1,
            class_sep=class_sep,
            flip_y=0.04,          # 4% de ruido en las etiquetas para simular datos reales imperfectos
            shuffle=False,
            random_state=seed,
        )

        feature_names = (
            [f"Info_{i+1}" for i in range(n_informative)]
            + [f"Corr_{i+1}" for i in range(n_redundant)]
            + [f"Ruido_{i+1}" for i in range(n_noise)]
        )
        target_names = ["Clase_0", "Clase_1"]
        super().__init__(X, y, feature_names, target_names, is_binary=True)


class BreastCancerDataset(BaseTabularDataset):
    """Dataset clínico real para diagnóstico binario de cáncer de mama (30 características)."""
    def __init__(self):
        data = load_breast_cancer()
        super().__init__(
            data.data, data.target, data.feature_names, data.target_names, is_binary=True
        )


class IrisDataset(BaseTabularDataset):
    """Dataset botánico clásico para clasificación multiclase de flores Iris (4 características, 3 clases)."""
    def __init__(self):
        data = load_iris()
        super().__init__(
            data.data, data.target, data.feature_names, data.target_names, is_binary=False
        )


class WineDataset(BaseTabularDataset):
    """Dataset químico para clasificación multiclase de vinos (13 características, 3 clases)."""
    def __init__(self):
        data = load_wine()
        super().__init__(
            data.data, data.target, data.feature_names, data.target_names, is_binary=False
        )


def prepare_splits(
    dataset: BaseTabularDataset,
    test_size: float = 0.2,
    val_size: float = 0.2,
    standardize: bool = True,
    seed: int = 42,
) -> Tuple[TensorDataset, TensorDataset, TensorDataset, Optional[StandardScaler]]:
    """
    Divide los datos en Entrenamiento (Train), Validación (Val) y Prueba (Test),
    aplicando estandarización de variables de entrada (StandardScaler) sin Data Leakage.

    Nota pedagógica sobre StandardScaler vs. BatchNorm:
    - `StandardScaler` normaliza las características de entrada (X) para que todas tengan
      media 0 y varianza 1 antes de entrar a la primera capa de la red. Esto evita que variables
      con escalas numéricas grandes (ej. área en cientos vs. suavidad en decimales) dominen
      el producto punto inicial.
    - El ajuste (`.fit()`) se calcula ÚNICAMENTE sobre `X_train`.
    """
    X, y = dataset.X_raw, dataset.y_raw

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, stratify=y_train_val, random_state=seed
    )

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    y_dtype = torch.float32 if dataset.is_binary else torch.long

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=y_dtype),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=y_dtype),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=y_dtype),
    )

    return train_ds, val_ds, test_ds, scaler


# ======================================================================================
# [Celda 3: Funciones de Análisis Exploratorio de Datos (EDA)]
# ======================================================================================

def eda_hist_grid(
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    bins: int = 25,
    title_prefix: str = "",
) -> None:
    """Grafica una cuadrícula de histogramas con los nombres reales de cada variable."""
    Xn = np.asarray(X)
    n_features = Xn.shape[1]
    if feature_names is None:
        feature_names = [f"Var {i+1}" for i in range(n_features)]

    cols = min(5, n_features)
    rows = int(np.ceil(n_features / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.5 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(n_features):
        ax = axes[i]
        ax.hist(Xn[:, i], bins=bins, color="#3b82f6", edgecolor="black", alpha=0.8)
        ax.set_title(feature_names[i], fontsize=10, fontweight="bold")
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    for j in range(n_features, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{title_prefix} — Distribución de Variables (Histogramas)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def eda_corr_heatmap(
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    title_prefix: str = "",
) -> None:
    """
    Visualiza la matriz de correlación de Pearson incluyendo el valor numérico en cada celda
    y ajustando dinámicamente el tamaño de fuente para mantener la legibilidad.
    """
    Xn = np.asarray(X)
    n_features = Xn.shape[1]
    if feature_names is None:
        feature_names = [f"Var {i+1}" for i in range(n_features)]

    corr = np.corrcoef(Xn, rowvar=False)

    fig_size = max(7, min(16, n_features * 0.55))
    font_size = 9 if n_features <= 15 else (6.5 if n_features <= 25 else 5)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Coeficiente de Pearson (r)", fontsize=10)

    ax.set_xticks(np.arange(n_features))
    ax.set_yticks(np.arange(n_features))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=max(7, font_size))
    ax.set_yticklabels(feature_names, fontsize=max(7, font_size))

    for i in range(n_features):
        for j in range(n_features):
            val = corr[i, j]
            text_color = "white" if abs(val) > 0.55 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color=text_color, fontsize=font_size,
            )

    ax.set_title(f"{title_prefix} — Matriz de Correlación de Pearson", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.show()


def eda_feature_bars(
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    title_prefix: str = "",
) -> None:
    """Compara la media y desviación estándar de cada característica para evidenciar diferencias de escala."""
    Xn = np.asarray(X)
    n_features = Xn.shape[1]
    if feature_names is None:
        feature_names = [f"Var {i+1}" for i in range(n_features)]

    means = Xn.mean(axis=0)
    stds = Xn.std(axis=0)
    idx = np.arange(n_features)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(9, n_features * 0.45), 6.2), sharex=True)

    ax1.bar(idx, means, color="#10b981", edgecolor="black", alpha=0.85)
    ax1.set_title(f"{title_prefix} — Media por Variable", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Media")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    ax2.bar(idx, stds, color="#f59e0b", edgecolor="black", alpha=0.85)
    ax2.set_title(f"{title_prefix} — Desviación Estándar por Variable", fontsize=12, fontweight="bold")
    ax2.set_ylabel("STD")
    ax2.set_xticks(idx)
    ax2.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


def eda_class_balance(
    y: np.ndarray,
    target_names: Optional[List[str]] = None,
    title_prefix: str = "",
) -> None:
    """Grafica la proporción de muestras por cada clase."""
    y_int = np.asarray(y).astype(int)
    vals, counts = np.unique(y_int, return_counts=True)
    labels = [target_names[v] if target_names and v < len(target_names) else f"Clase {v}" for v in vals]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    bars = ax.bar(labels, counts, color="#6366f1", edgecolor="black", width=0.5, alpha=0.85)
    ax.set_title(f"{title_prefix} — Balance de Clases", fontsize=12, fontweight="bold")
    ax.set_ylabel("Número de muestras")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    total = counts.sum()
    for bar, count in zip(bars, counts):
        pct = (count / total) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (total * 0.01),
            f"{count} ({pct:.1f}%)",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    plt.show()

def eda_ydata_split_comparison(
    dataset: BaseTabularDataset,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    test_ds: TensorDataset,
    title_prefix: str = "",
    output_dir: str = "reportes_ydata",
    minimal: bool = True,
    max_samples_per_split: int = 10000,
    show_in_notebook: bool = True,
    seed: int = 42,
) -> str:
    """
    Auditoría Estadística de Particiones (Train vs. Val vs. Test) con YData Profiling.
    ----------------------------------------------------------------------------------
    Genera perfiles estadísticos de los conjuntos de Entrenamiento, Validación y Prueba
    y los superpone en un único reporte comparativo HTML para:
    1. Verificar que las distribuciones de probabilidad se mantengan consistentes (sin Data Drift).
    2. Auditar la ausencia de Data Leakage y comprobar el correcto escalamiento (StandardScaler).
    3. Confirmar que la estratificación de la variable objetivo ('Target') sea idéntica en los 3 splits.

    Nota de Ingeniería (Big Data / Colab):
    Se activa `minimal=True` y un tope `max_samples_per_split` para evitar desbordamientos de
    memoria RAM (OOM) cuando se analizan datasets masivos (>40 columnas y cientos de miles de filas).
    """
    os.makedirs(output_dir, exist_ok=True)

    def _tensor_to_df(tensor_ds: TensorDataset) -> pd.DataFrame:
        X_t, y_t = tensor_ds.tensors
        X_np = X_t.cpu().numpy()
        y_np = y_t.cpu().numpy()

        # Submuestreo preventivo si la partición supera el límite seguro de memoria para el reporte
        if len(X_np) > max_samples_per_split:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X_np), size=max_samples_per_split, replace=False)
            X_np = X_np[idx]
            y_np = y_np[idx]

        df = pd.DataFrame(X_np, columns=dataset.feature_names)
        df["Target"] = y_np.astype(int)
        return df

    print(f"\n[YData Profiling] Generando auditoría comparativa Train / Val / Test para: {title_prefix}...")
    df_train = _tensor_to_df(train_ds)
    df_val = _tensor_to_df(val_ds)
    df_test = _tensor_to_df(test_ds)

    # Creación de los perfiles individuales para cada subconjunto
    train_report = ProfileReport(df_train, title="1. Train Set", minimal=minimal, progress_bar=False)
    val_report = ProfileReport(df_val, title="2. Validation Set", minimal=minimal, progress_bar=False)
    test_report = ProfileReport(df_test, title="3. Test Set", minimal=minimal, progress_bar=False)

    # Superposición estadística de los 3 conjuntos de datos
    comparison_report = compare([train_report, val_report, test_report])

    safe_name = title_prefix.lower().replace(" ", "_")
    html_path = os.path.join(output_dir, f"ydata_comparacion_{safe_name}.html")
    comparison_report.to_file(html_path)
    print(f"  -> [OK] Reporte comparativo exportado en: '{html_path}'")

    if show_in_notebook:
        comparison_report.to_notebook_iframe()

    # Liberación explícita de memoria (buena práctica para datasets de gran escala)
    del df_train, df_val, df_test, train_report, val_report, test_report, comparison_report
    return html_path


# ======================================================================================
# [Celda 4: Análisis No Supervisado (PCA, t-SNE y K-Means)]
# ======================================================================================

def eda_unsupervised_analysis(
    dataset: BaseTabularDataset,
    title_prefix: str = "",
    seed: int = 42,
) -> None:
    """
    Fase de Aprendizaje No Supervisado previa a las Redes Neuronales.
    -----------------------------------------------------------------
    Permite entender la topología y estructura oculta de los datos sin usar las etiquetas:
    1. Reducción de Dimensionalidad:
       - PCA (lineal): Proyecta los datos en las direcciones ortogonales de mayor varianza.
       - t-SNE (no lineal): Preserva relaciones de vecindad local para descubrir variedades complejas.
    2. Agrupamiento (Clustering con K-Means):
       - Particiona los datos en K grupos por cercanía euclidiana sin conocer las clases reales.
       - Evaluamos con Silueta (compacidad geométrica) y Adjusted Rand Index (coherencia real).
    """
    X_scaled = StandardScaler().fit_transform(dataset.X_raw)
    y_true = dataset.y_raw.astype(int)
    n_classes = dataset.num_classes
    target_names = dataset.target_names

    # 1. PCA a 2 componentes
    pca = PCA(n_components=2, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100

    # 2. t-SNE a 2 dimensiones
    perplexity = min(30, max(5, len(X_scaled) // 5))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X_scaled)

    # 3. K-Means (sin acceso a 'y_true')
    kmeans = KMeans(n_clusters=n_classes, n_init=10, random_state=seed)
    cluster_labels = kmeans.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, cluster_labels)
    ari_score = adjusted_rand_score(y_true, cluster_labels)

    print(f"\n--- Resumen No Supervisado [{title_prefix}] ---")
    print(f"  * Varianza explicada por PCA (2D)          : {var_exp.sum():.2f}% (PC1: {var_exp[0]:.1f}%, PC2: {var_exp[1]:.1f}%)")
    print(f"  * Coeficiente de Silueta (K-Means, k={n_classes})    : {sil_score:.3f}")
    print(f"  * Adjusted Rand Index (Coherencia con 'y') : {ari_score:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    cmap_classes = plt.cm.get_cmap("tab10", n_classes)

    # Panel 1: Proyección PCA con clases reales
    for c in range(n_classes):
        mask = y_true == c
        axes[0].scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            label=target_names[c], alpha=0.75, edgecolors="k", s=40, color=cmap_classes(c)
        )
    axes[0].set_title(f"PCA 2D — Clases Reales\n(Varianza explicada: {var_exp.sum():.1f}%)", fontweight="bold")
    axes[0].set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Proyección t-SNE con clases reales
    for c in range(n_classes):
        mask = y_true == c
        axes[1].scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            label=target_names[c], alpha=0.75, edgecolors="k", s=40, color=cmap_classes(c)
        )
    axes[1].set_title("t-SNE 2D — Clases Reales\n(Separabilidad Local No Lineal)", fontweight="bold")
    axes[1].set_xlabel("Dimensión t-SNE 1")
    axes[1].set_ylabel("Dimensión t-SNE 2")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # Panel 3: Clústeres K-Means proyectados sobre PCA
    cmap_clusters = plt.cm.get_cmap("Set2", n_classes)
    for k in range(n_classes):
        mask = cluster_labels == k
        axes[2].scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            label=f"Clúster {k}", alpha=0.75, edgecolors="k", s=40, color=cmap_clusters(k)
        )
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    axes[2].scatter(
        centroids_pca[:, 0], centroids_pca[:, 1],
        marker="X", s=220, color="red", edgecolors="black", linewidths=1.5, label="Centroides"
    )
    axes[2].set_title(
        f"K-Means sobre PCA (Ciego a Etiquetas)\nARI = {ari_score:.2f} | Silueta = {sil_score:.2f}",
        fontweight="bold",
    )
    axes[2].set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    axes[2].set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"{title_prefix} — Análisis Exploratorio No Supervisado", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def run_complete_eda(dataset: BaseTabularDataset, title: str, seed: int = 42) -> None:
    """Función wrapper que ejecuta todo el flujo de EDA, auditoría YData Profiling y aprendizaje no supervisado."""
    print(f"\n{'='*75}\nFASE A — EXPLORACIÓN DE DATOS, AUDITORÍA YDATA Y NO SUPERVISADO: {title.upper()}\n{'='*75}")
    eda_class_balance(dataset.y_raw, dataset.target_names, title_prefix=title)
    eda_feature_bars(dataset.X_raw, dataset.feature_names, title_prefix=title)
    eda_hist_grid(dataset.X_raw, dataset.feature_names, title_prefix=title)
    eda_corr_heatmap(dataset.X_raw, dataset.feature_names, title_prefix=title)

    # Partición preventiva para auditar y comparar las distribuciones de Train, Val y Test con YData Profiling
    train_ds, val_ds, test_ds, _ = prepare_splits(
        dataset, test_size=0.2, val_size=0.2, standardize=True, seed=seed
    )
    eda_ydata_split_comparison(
        dataset=dataset,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        title_prefix=title,
        minimal=True,
        seed=seed,
    )

    eda_unsupervised_analysis(dataset, title_prefix=title, seed=seed)


# ======================================================================================
# [Celda 5: Arquitectura FFNN Avanzada (BatchNorm, Dropout e Inicialización He/Xavier)]
# ======================================================================================

def init_weights_explicit(module: nn.Module, is_output_layer: bool = False) -> None:
    """
    Inicialización explícita de pesos según la naturaleza de cada capa.
    -------------------------------------------------------------------
    ¿Por qué NUNCA debemos inicializar todos los pesos en cero (W = 0)?
    Si todos los pesos son cero (o tienen exactamente el mismo valor constante), todas las
    neuronas de una capa oculta calcularán exactamente la misma salida en el Forward Pass y
    recibirán exactamente el mismo gradiente durante el Backpropagation. Esto se conoce como
    el "Problema de la Simetría": las neuronas nunca se diferencian entre sí y la red entera
    actúa como si tuviera una sola neurona por capa.

    Estrategias modernas para romper la simetría manteniendo estable la varianza:
    1. Inicialización de He / Kaiming (Kaiming He et al., 2015):
       Diseñada específicamente para funciones de activación asimétricas como ReLU (que anulan
       los valores negativos, reduciendo la varianza a la mitad en cada capa). Extrae los pesos
       de una distribución normal con desviación estándar `sqrt(2 / fan_in)`.
    2. Inicialización de Xavier / Glorot (Glorot & Bengio, 2010):
       Diseñada para capas lineales de salida o activaciones simétricas (Sigmoide, Tanh).
       Mantiene constante la varianza de activaciones y gradientes considerando tanto las
       entradas (`fan_in`) como las salidas (`fan_out`) de la capa: varianza `2 / (fan_in + fan_out)`.
    """
    if isinstance(module, nn.Linear):
        if is_output_layer:
            # Capa lineal de salida (o asociada a Sigmoide/Softmax): Inicialización Xavier/Glorot
            nn.init.xavier_uniform_(module.weight)
        else:
            # Capas ocultas con activación ReLU: Inicialización Kaiming/He
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")

        # Los sesgos (biases) sí pueden inicializarse en cero porque la asimetría ya la
        # rompen las matrices de pesos W.
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.BatchNorm1d):
        # En Batch Normalization, gamma (weight) se inicializa en 1 (escala unitaria)
        # y beta (bias) se inicializa en 0 (desplazamiento nulo).
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class FFNN(nn.Module):
    """
    Red Neuronal Feed-Forward (MLP) mejorada con Batch Normalization, Dropout e Inicialización He/Xavier.
    -----------------------------------------------------------------------------------------------------
    Flujo de cada bloque oculto:
        Linear (sin bias si hay BatchNorm) -> BatchNorm1d -> ReLU -> Dropout

    1. Batch Normalization (`nn.BatchNorm1d`):
       Normaliza las activaciones intermedias de cada mini-lote para que tengan media 0 y varianza 1,
       y luego aplica dos parámetros entrenables (escala `gamma` y desplazamiento `beta`).
       Beneficios clave:
       - Mitiga el desplazamiento de covariable interna (Internal Covariate Shift).
       - Suaviza la superficie de pérdida, permitiendo usar tasas de aprendizaje (lr) más altas
         y acelerando la convergencia.
       - Nota técnica: Cuando se usa `BatchNorm1d` justo después de `nn.Linear`, el parámetro `bias`
         de la capa lineal se vuelve redundante porque BatchNorm resta la media del lote y suma su
         propio parámetro `beta`. Por ello usamos `bias=not use_batchnorm`.

    2. Dropout (`nn.Dropout`):
       Actúa como un potente regularizador estocástico. En cada paso de entrenamiento, "apaga"
       (pone en cero) una fracción aleatoria `p` de las neuronas. Esto impide la co-adaptación
       compleja de neuronas (evita que la red memorice ejemplos específicos de entrenamiento) y
       obliga a aprender representaciones distribuidas y robustas. En evaluación (`model.eval()`),
       el Dropout se desactiva automáticamente.
    """
    def __init__(
        self,
        d_in: int,
        hidden_dims: Tuple[int, ...] = (64, 32),
        d_out: int = 1,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        d_prev = d_in

        # Construcción modular de los bloques ocultos
        for h in hidden_dims:
            block = []
            # Si usamos BatchNorm antes de la activación, desactivamos el bias de Linear por eficiencia
            linear_layer = nn.Linear(d_prev, h, bias=not use_batchnorm)
            block.append(linear_layer)

            if use_batchnorm:
                block.append(nn.BatchNorm1d(h))

            block.append(nn.ReLU())

            if dropout > 0.0:
                block.append(nn.Dropout(p=dropout))

            self.hidden_layers.append(nn.Sequential(*block))
            d_prev = h

        # Capa de salida: transformación lineal que entrega logits crudos
        self.out_layer = nn.Linear(d_prev, d_out)

        # Aplicar nuestra inicialización matemática explícita (He en ocultas, Xavier en salida)
        self._initialize_network()

    def _initialize_network(self) -> None:
        """Recorre las capas aplicando Kaiming/He a las capas con ReLU y Xavier a la salida."""
        for block in self.hidden_layers:
            for layer in block:
                init_weights_explicit(layer, is_output_layer=False)
        init_weights_explicit(self.out_layer, is_output_layer=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagación hacia adelante a través de los bloques ocultos y la capa de salida."""
        for block in self.hidden_layers:
            x = block(x)
        logits = self.out_layer(x)
        return logits


# ======================================================================================
# [Celda 6: Motor de Entrenamiento (AdamW + L2, Early Stopping) y Evaluación]
# ======================================================================================

@torch.no_grad()
def compute_accuracy(logits: torch.Tensor, y_true: torch.Tensor, is_binary: bool) -> float:
    """Calcula la exactitud (accuracy) a partir de los logits sin almacenar el grafo de gradientes."""
    if is_binary:
        preds = (logits.view(-1) >= 0.0).float()
        return (preds == y_true.view(-1)).float().mean().item()
    else:
        preds = logits.argmax(dim=1)
        return (preds == y_true).float().mean().item()


def fit(
    model: nn.Module,
    train_ds: TensorDataset,
    val_ds: Optional[TensorDataset] = None,
    *,
    is_binary: bool = True,
    epochs: int = 150,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    patience: int = 20,
    min_delta: float = 1e-4,
    ckpt_path: Optional[str] = None,
    verbose_every: int = 15,
) -> Tuple[nn.Module, Dict[str, List[float]], Optional[int]]:
    """
    Bucle de entrenamiento supervisado con Optimización Moderna y Regularización.
    -----------------------------------------------------------------------------
    1. Optimizador AdamW y Regularización L2 (`weight_decay`):
       - La Regularización L2 (también llamada penalización de Tikhonov o Ridge) añade un costo
         proporcional al cuadrado de la magnitud de los pesos: `Loss_total = Loss_datos + (lambda/2) * ||W||^2`.
       - Al derivar este término, en cada paso de actualización los pesos se contraen hacia cero:
         `W = W - lr * (dLoss/dW + weight_decay * W)`.
       - ¿Por qué penalizar los pesos grandes? Porque pesos de gran magnitud hacen que pequeños
         cambios o ruido en las variables de entrada produzcan oscilaciones bruscas en la salida,
         generando fronteras de decisión excesivamente sinuosas (sobreajuste / overfitting).
         Al mantener los pesos pequeños, obligamos a la red a encontrar funciones suaves y estables.
       - Usamos `AdamW` en lugar del `Adam` clásico porque `AdamW` aplica el decaimiento de pesos
         de forma desacoplada del momento adaptativo, logrando una regularización L2 matemáticamente exacta.

    2. Paro Temprano (Early Stopping):
       - A medida que avanzan las épocas, el error de entrenamiento (`train_loss`) casi siempre
         disminuye, pero llega un punto de inflexión donde la red empieza a memorizar el ruido del
         conjunto de entrenamiento y el error de validación (`val_loss`) comienza a subir.
       - El Early Stopping monitorea `val_loss` en cada época, guarda en memoria una instantánea
         (`best_state`) de los pesos cuando `val_loss` alcanza un nuevo mínimo y detiene el
         entrenamiento si transcurren `patience` épocas sin mejora, previniendo el sobreajuste.
    """
    model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()

    # Instanciación explícita del optimizador AdamW con Regularización L2 (weight_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,  # Coeficiente lambda de penalización L2 sobre los pesos
    )

    # drop_last=False, pero evitamos lotes de tamaño 1 en entrenamiento porque BatchNorm1d
    # requiere al menos 2 muestras para calcular la varianza del lote.
    drop_last_train = (len(train_ds) % batch_size == 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last_train)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False) if val_ds else None

    best_val_loss = math.inf
    best_epoch = None
    best_state = None
    history = {"epoch": [], "loss": [], "acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # --- MODO ENTRENAMIENTO (BatchNorm usa estadísticas del lote, Dropout está activo) ---
        model.train()
        total_loss = 0.0
        total_samples = 0
        batch_accs = []

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            # 1. Propagación hacia adelante (Forward Pass)
            logits = model(xb)

            # 2. Evaluación de la función de costo (Loss)
            if is_binary:
                loss = criterion(logits.view(-1), yb.float().view(-1))
            else:
                loss = criterion(logits, yb.long())

            # 3. Retropropagación (Backpropagation) y actualización con decaimiento L2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_sz = xb.size(0)
            total_loss += loss.item() * batch_sz
            total_samples += batch_sz
            batch_accs.append(compute_accuracy(logits.detach(), yb, is_binary))

        tr_loss = total_loss / total_samples
        tr_acc = float(np.mean(batch_accs))

        # --- MODO EVALUACIÓN (BatchNorm usa medias móviles globales, Dropout se apaga) ---
        val_loss, val_acc = None, None
        if val_loader is not None:
            model.eval()
            v_loss_sum = 0.0
            v_samples = 0
            v_accs = []

            with torch.no_grad():
                for xvb, yvb in val_loader:
                    xvb, yvb = xvb.to(DEVICE), yvb.to(DEVICE)
                    v_logits = model(xvb)
                    if is_binary:
                        v_loss = criterion(v_logits.view(-1), yvb.float().view(-1))
                    else:
                        v_loss = criterion(v_logits, yvb.long())

                    v_loss_sum += v_loss.item() * xvb.size(0)
                    v_samples += xvb.size(0)
                    v_accs.append(compute_accuracy(v_logits, yvb, is_binary))

            val_loss = v_loss_sum / v_samples
            val_acc = float(np.mean(v_accs))

            # Control de mejor modelo para Early Stopping
            if val_loss + min_delta < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if ckpt_path:
                    torch.save(best_state, ckpt_path)

        history["epoch"].append(epoch)
        history["loss"].append(tr_loss)
        history["acc"].append(tr_acc)
        if val_loss is not None:
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

        if verbose_every > 0 and (epoch % verbose_every == 0 or epoch == 1):
            msg = f"  [Época {epoch:03d}/{epochs:03d}] Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:5.1f}%"
            if val_loss is not None:
                msg += f" || Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.1f}%"
            print(msg)

        # Disparo del Paro Temprano si se supera la paciencia sin mejoras en validación
        if val_loader is not None and best_epoch is not None and (epoch - best_epoch) >= patience:
            print(f"  [EARLY STOPPING] Entrenamiento detenido en época {epoch}. Mejor época: {best_epoch} (Val Loss: {best_val_loss:.4f})")
            break

    # Restaurar los pesos óptimos encontrados antes de que comenzara el sobreajuste
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_epoch


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: TensorDataset,
    is_binary: bool = True,
    target_names: Optional[List[str]] = None,
    batch_size: int = 256,
) -> Dict[str, object]:
    """Evalúa un modelo o ensamble de PyTorch en el conjunto de prueba."""
    model.eval()
    model.to(DEVICE)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()

    total_samples = 0
    total_loss = 0.0
    all_y, all_preds = [], []

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)

        if is_binary:
            loss = criterion(logits.view(-1), yb.float().view(-1))
            preds = (logits.view(-1) >= 0.0).long()
        else:
            loss = criterion(logits, yb.long())
            preds = logits.argmax(dim=1)

        total_loss += loss.item() * xb.size(0)
        total_samples += xb.size(0)
        all_y.append(yb.view(-1).cpu().numpy().astype(int))
        all_preds.append(preds.cpu().numpy().astype(int))

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_preds)

    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=target_names, digits=4, zero_division=0
    )

    return {
        "acc": acc,
        "loss": total_loss / total_samples,
        "cm": cm,
        "report": report,
    }


def plot_training_history(
    history: Dict[str, List[float]],
    best_epoch: Optional[int] = None,
    title: str = "Entrenamiento",
) -> None:
    """Grafica la evolución de la Pérdida (Loss) y la Exactitud (Accuracy) marcando el Early Stopping."""
    epochs = history["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.3))

    ax1.plot(epochs, history["loss"], label="Train Loss", color="#2563eb", linewidth=2)
    if history.get("val_loss"):
        ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#dc2626", linewidth=2)
    if best_epoch is not None:
        ax1.axvline(best_epoch, linestyle="--", color="green", label=f"Early Stop (Época {best_epoch})")
    ax1.set_title(f"{title} — Evolución de la Pérdida (Loss)", fontweight="bold")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    tr_acc_pct = [a * 100 for a in history["acc"]]
    ax2.plot(epochs, tr_acc_pct, label="Train Acc", color="#2563eb", linewidth=2)
    if history.get("val_acc"):
        val_acc_pct = [a * 100 for a in history["val_acc"]]
        ax2.plot(epochs, val_acc_pct, label="Val Acc", color="#dc2626", linewidth=2)
    if best_epoch is not None:
        ax2.axvline(best_epoch, linestyle="--", color="green", label=f"Early Stop (Época {best_epoch})")
    ax2.set_title(f"{title} — Evolución de la Exactitud (Accuracy)", fontweight="bold")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Exactitud (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


def train_and_evaluate_single_model(
    dataset: BaseTabularDataset,
    title: str,
    hidden_dims: Tuple[int, ...] = (64, 32),
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 150,
    patience: int = 20,
    seed: int = 42,
) -> Tuple[nn.Module, Dict[str, object]]:
    """Ejecuta el flujo completo de entrenamiento supervisado para un modelo FFNN individual."""
    print(f"\n{'='*75}\nFASE B — ENTRENAMIENTO FFNN INDIVIDUAL: {title.upper()}\n{'='*75}")
    set_seed(seed)

    train_ds, val_ds, test_ds, _ = prepare_splits(
        dataset, test_size=0.2, val_size=0.2, standardize=True, seed=seed
    )

    d_out = 1 if dataset.is_binary else dataset.num_classes
    model = FFNN(
        d_in=dataset.num_features,
        hidden_dims=hidden_dims,
        d_out=d_out,
        dropout=dropout,
        use_batchnorm=True,
    )

    ckpt_file = os.path.join("checkpoints", f"best_{title.lower().replace(' ', '_')}.pt")
    model, history, best_epoch = fit(
        model,
        train_ds,
        val_ds,
        is_binary=dataset.is_binary,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        ckpt_path=ckpt_file,
        verbose_every=20,
    )

    plot_training_history(history, best_epoch=best_epoch, title=title)

    eval_metrics = evaluate(
        model, test_ds, is_binary=dataset.is_binary, target_names=dataset.target_names
    )
    print(f"\n---> Evaluación en Test Set (Modelo Individual) [{title}]:")
    print(f"Exactitud (Test Accuracy): {eval_metrics['acc']*100:.2f}% | Pérdida (Test Loss): {eval_metrics['loss']:.4f}")
    print("\nReporte de Clasificación:")
    print(eval_metrics["report"])
    print("Matriz de Confusión:")
    print(eval_metrics["cm"])

    return model, eval_metrics


# ======================================================================================
# [Celda 7: Métodos de Ensamblaje (Bootstrap Bagging + Trainable Stacking)]
# ======================================================================================

def create_bootstrap_dataset(train_ds: TensorDataset, seed: int) -> TensorDataset:
    """
    Muestreo Bootstrap (con reemplazo) para Bagging.
    ------------------------------------------------
    Dado un conjunto de entrenamiento de tamaño N, extrae N muestras al azar CON reemplazo.
    Estadísticamente, cada subconjunto bootstrap contiene alrededor del 63.2% de las muestras
    originales únicas (algunas repetidas), mientras que el 36.8% restante queda fuera (Out-Of-Bag).
    Esto hace que cada modelo base vea una distribución ligeramente distinta y cometa errores
    diferentes, reduciendo la varianza al combinarlos.
    """
    X_orig, y_orig = train_ds.tensors
    n_samples = X_orig.size(0)

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(low=0, high=n_samples, size=n_samples)
    boot_indices_t = torch.tensor(boot_indices, dtype=torch.long)

    return TensorDataset(X_orig[boot_indices_t], y_orig[boot_indices_t])


class StackingEnsemble(nn.Module):
    """
    Meta-Modelo de Stacking con Pesos Entrenables en PyTorch.
    ---------------------------------------------------------
    En lugar de promediar de manera ingenua las predicciones de los modelos base (votación simple),
    el Stacking utiliza un "Meta-Aprendiz" (Meta-Learner) que aprende cómo ponderar y combinar
    óptimamente las salidas de los M modelos expertos.

    Arquitectura del Meta-Modelo:
    1. Congela los pesos de todos los modelos base (`requires_grad = False`) para que actúen
       únicamente como extractores de predicciones (logits).
    2. Durante el `forward(x)`, evalúa la entrada `x` en cada uno de los M modelos base y
       concatena sus logits formando un vector de dimensión `M * d_out`.
    3. Pasa ese vector concatenado por una capa lineal entrenable (`self.meta_learner`) inicializada
       con Xavier/Glorot, la cual descubre qué modelos son más confiables para cada clase.
    """
    def __init__(self, base_models: List[nn.Module], d_out: int):
        super().__init__()
        self.num_models = len(base_models)
        self.d_out = d_out
        self.base_models = nn.ModuleList(base_models)

        # Congelar los parámetros de todos los modelos base (ya fueron entrenados vía Bootstrap)
        for bm in self.base_models:
            bm.eval()
            for param in bm.parameters():
                param.requires_grad = False

        # Meta-capa lineal entrenable: recibe las salidas concatenadas de los M modelos (M * d_out)
        # y produce la predicción combinada final (d_out).
        self.meta_learner = nn.Linear(self.num_models * d_out, d_out)
        init_weights_explicit(self.meta_learner, is_output_layer=True)

    def train(self, mode: bool = True):
        """
        Sobrescribimos el método `.train()` de PyTorch para asegurar que, aunque el
        `StackingEnsemble` esté en modo entrenamiento al ajustar el `meta_learner`,
        los modelos base congelados permanezcan SIEMPRE en modo `.eval()` (para no alterar
        sus estadísticas de BatchNorm ni activar su Dropout).
        """
        super().train(mode)
        for bm in self.base_models:
            bm.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Obtener los logits de cada modelo base sin calcular gradientes en ellos
        with torch.no_grad():
            base_outputs = [bm(x) for bm in self.base_models]

        # Concatenar a lo largo de la dimensión de características: shape -> (batch_size, M * d_out)
        meta_features = torch.cat(base_outputs, dim=1)

        # El meta-modelo combina linealmente las predicciones con sus pesos entrenables
        final_logits = self.meta_learner(meta_features)
        return final_logits


def run_ensemble_pipeline(
    dataset: BaseTabularDataset,
    title: str,
    n_estimators: int = 3,
    hidden_dims: Tuple[int, ...] = (64, 32),
    dropout: float = 0.2,
    lr_base: float = 2e-3,
    lr_meta: float = 5e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 32,
    epochs_base: int = 120,
    epochs_meta: int = 80,
    seed: int = 42,
) -> Tuple[StackingEnsemble, Dict[str, object]]:
    """
    Pipeline completo de Aprendizaje por Ensamblaje (Bagging + Trainable Stacking):
    1. Entrena un modelo individual de referencia sobre el conjunto de entrenamiento completo.
    2. Genera `n_estimators` subconjuntos mediante Bootstrap (muestreo con reemplazo) y entrena
       un modelo FFNN independiente en cada uno.
    3. Instancia el `StackingEnsemble` congelando los modelos base y entrena el meta-modelo lineal.
    4. Compara el desempeño en el conjunto de prueba (Test Set) de cada componente frente al Ensamble.
    """
    print(f"\n{'='*75}\nFASE C — ENSEMBLE LEARNING (BOOTSTRAP + STACKING): {title.upper()}\n{'='*75}")
    set_seed(seed)

    train_ds, val_ds, test_ds, _ = prepare_splits(
        dataset, test_size=0.2, val_size=0.2, standardize=True, seed=seed
    )
    d_out = 1 if dataset.is_binary else dataset.num_classes

    # --- 1. Modelo Individual Estándar (Línea Base / Baseline) ---
    print("\n[Paso 1/3] Entrenando Modelo Individual de Referencia (Sin Bootstrap)...")
    baseline_model = FFNN(dataset.num_features, hidden_dims, d_out, dropout=dropout, use_batchnorm=True)
    baseline_model, _, _ = fit(
        baseline_model, train_ds, val_ds,
        is_binary=dataset.is_binary, epochs=epochs_base, batch_size=batch_size,
        lr=lr_base, weight_decay=weight_decay, patience=20, verbose_every=0
    )
    base_eval = evaluate(baseline_model, test_ds, is_binary=dataset.is_binary, target_names=dataset.target_names)

    # --- 2. Entrenamiento de M Modelos Base mediante Bootstrap (Bagging) ---
    print(f"[Paso 2/3] Entrenando {n_estimators} Modelos Base con Muestreo Bootstrap (con reemplazo)...")
    base_models = []
    individual_boot_accs = []

    for i in range(n_estimators):
        model_seed = seed + (i + 1) * 100
        set_seed(model_seed)

        # Generar muestra bootstrap exclusiva para el modelo i
        boot_train_ds = create_bootstrap_dataset(train_ds, seed=model_seed)

        bm = FFNN(dataset.num_features, hidden_dims, d_out, dropout=dropout, use_batchnorm=True)
        bm, _, _ = fit(
            bm, boot_train_ds, val_ds,
            is_binary=dataset.is_binary, epochs=epochs_base, batch_size=batch_size,
            lr=lr_base, weight_decay=weight_decay, patience=20, verbose_every=0
        )
        bm_metrics = evaluate(bm, test_ds, is_binary=dataset.is_binary, target_names=dataset.target_names)
        individual_boot_accs.append(bm_metrics["acc"])
        base_models.append(bm)
        print(f"  -> Modelo Bootstrap #{i+1} | Test Accuracy: {bm_metrics['acc']*100:.2f}% | Test Loss: {bm_metrics['loss']:.4f}")

    # --- 3. Creación y Entrenamiento del Meta-Modelo (Stacking) ---
    print("\n[Paso 3/3] Entrenando Meta-Modelo (StackingEnsemble) sobre los modelos base congelados...")
    set_seed(seed)
    stacking_model = StackingEnsemble(base_models=base_models, d_out=d_out)

    # Entrenamos la capa lineal del meta-modelo. Combinamos train_ds y val_ds o usamos val_ds
    # para que el meta-aprendiz pondere a los modelos según su capacidad de generalización.
    stacking_model, meta_hist, best_meta_ep = fit(
        stacking_model,
        val_ds,  # Entrenar el meta-modelo en validación evita que sobreajuste a los modelos que memorizaron train
        val_ds=None,
        is_binary=dataset.is_binary,
        epochs=epochs_meta,
        batch_size=batch_size,
        lr=lr_meta,
        weight_decay=1e-4,
        verbose_every=0,
    )

    # Evaluación final del Ensamble Stacking en el conjunto de Prueba (Test Set)
    ens_eval = evaluate(stacking_model, test_ds, is_binary=dataset.is_binary, target_names=dataset.target_names)

    # Inspeccionar los pesos aprendidos por el Meta-Modelo
    meta_weights = stacking_model.meta_learner.weight.detach().cpu().numpy()
    meta_bias = stacking_model.meta_learner.bias.detach().cpu().numpy()

    print("\n" + "-" * 65)
    print(f"RESUMEN COMPARATIVO EN TEST SET — {title.upper()}")
    print("-" * 65)
    print(f"  * Modelo Individual Estándar (Baseline) : {base_eval['acc']*100:.2f}% (Loss: {base_eval['loss']:.4f})")
    for i, acc_i in enumerate(individual_boot_accs):
        print(f"  * Modelo Base Bootstrap #{i+1}              : {acc_i*100:.2f}%")
    print(f"  * Promedio de Modelos Bootstrap         : {np.mean(individual_boot_accs)*100:.2f}%")
    print(f"  >>> ENSAMBLE STACKING (Meta-Modelo)     : {ens_eval['acc']*100:.2f}% (Loss: {ens_eval['loss']:.4f}) <<<")
    print("-" * 65)
    print(f"Pesos aprendidos por la capa Meta-Learner (shape {meta_weights.shape}):\n{np.round(meta_weights, 3)}")
    print(f"Bias del Meta-Learner: {np.round(meta_bias, 3)}")

    # Gráfico comparativo de barras
    labels = ["Individual\n(Baseline)"] + [f"Bootstrap #{i+1}" for i in range(n_estimators)] + ["Stacking\nEnsemble"]
    accs = [base_eval["acc"] * 100] + [a * 100 for a in individual_boot_accs] + [ens_eval["acc"] * 100]
    colors = ["#64748b"] + ["#3b82f6"] * n_estimators + ["#10b981"]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(labels, accs, color=colors, edgecolor="black", width=0.55, alpha=0.9)
    ax.set_title(f"{title} — Comparación de Exactitud en Test: Modelos vs. Stacking Ensemble", fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)")
    min_y = max(0.0, min(accs) - 8.0)
    ax.set_ylim(min_y, 102.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f" {val:.2f}%",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold"
        )
    plt.tight_layout()
    plt.show()

    return stacking_model, ens_eval


# ======================================================================================
# [Celda 8: Ejecución Principal del Laboratorio (Fases A, B y C)]
# ======================================================================================

if __name__ == "__main__":
    set_seed(42)

    # Definición de los conjuntos de datos para el laboratorio
    # Formato: (Nombre, Instancia Dataset, Learning Rate sugerido)
    datasets_lab = [
        ("Sintético Estructurado", SyntheticStructuredDataset(n_samples=900, class_sep=1.25, seed=42), 2e-3),
        ("Breast Cancer", BreastCancerDataset(), 1e-3),
        ("Iris", IrisDataset(), 4e-3),
        ("Wine", WineDataset(), 2e-3),
    ]

    # =========================================================================
    # FASE A: Análisis Exploratorio de Datos (EDA) + Aprendizaje No Supervisado
    # =========================================================================
    for name, ds, _ in datasets_lab:
        run_complete_eda(ds, title=name)

    # =========================================================================
    # FASE B: Entrenamiento Supervisado de Redes Neuronales (FFNN Mejorada)
    # =========================================================================
    # Incluye Batch Normalization, Dropout, Inicialización He/Xavier,
    # Optimizador AdamW con Regularización L2 (weight_decay) y Early Stopping.
    single_models = {}
    for name, ds, lr in datasets_lab:
        model, metrics = train_and_evaluate_single_model(
            dataset=ds,
            title=name,
            hidden_dims=(64, 32),
            dropout=0.2,
            lr=lr,
            weight_decay=1e-3,  # Penalización L2 explícita
            batch_size=32,
            epochs=150,
            patience=20,
            seed=42,
        )
        single_models[name] = (model, metrics)

    # =========================================================================
    # FASE C: Métodos de Ensamblaje (Bootstrap Bagging + Trainable Stacking)
    # =========================================================================
    # Demostración sobre el Dataset Sintético Estructurado y Breast Cancer
    # entrenando 3 modelos base con muestreo con reemplazo y combinándolos con Stacking.
    ensemble_targets = [
        datasets_lab[0],  # Sintético Estructurado (Binario)
        datasets_lab[1],  # Breast Cancer (Binario)
        datasets_lab[3],  # Wine (Multiclase)
    ]

    ensemble_results = {}
    for name, ds, lr in ensemble_targets:
        ens_model, ens_metrics = run_ensemble_pipeline(
            dataset=ds,
            title=name,
            n_estimators=3,
            hidden_dims=(64, 32),
            dropout=0.2,
            lr_base=lr,
            lr_meta=5e-3,
            weight_decay=1e-3,
            batch_size=32,
            epochs_base=120,
            epochs_meta=80,
            seed=42,
        )
        ensemble_results[name] = (ens_model, ens_metrics)
