import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from camera          import Camera
from projection      import Projection
from graphicPipeline import GraphicPipeline
from readply         import readply
from mipmap          import build_mipmaps, mipmap_atlas

#  Modes disponibles :
#    "single"   -> rendu avec un seul filtre (FILTER_MODE)
#    "compare"  -> rendu avec tous les filtres + grille de comparaison
#    "mipmap_vis" -> visualisation de la pyramide MIP
#
#  Filtres de sampling (FILTER_MODE) :
#    "nearest", "bilinear", "trilinear", "anisotropic"
#
#  Filtres de downsampling (DOWNSAMPLE_FILTER) :
#    "box", "gaussian", "lanczos"
# 

# CONFIGURATION
# 
MODE             = "compare"      # "single" | "compare" | "mipmap_vis"
FILTER_MODE      = "trilinear"    # pour le mode "single"
DOWNSAMPLE_FILTER = "box"         # "box" | "gaussian" | "lanczos"
SAVE_IMAGES      = True           # sauvegarder les images en PNG
OUTPUT_DIR       = "output"

WIDTH  = 512
HEIGHT = 288


_dir         = os.path.dirname(os.path.abspath(__file__))
PLY_PATH     = os.path.join(_dir, "ply/suzanne.ply")
TEXTURE_PATH = os.path.join(_dir, "texture/suzanne.png")

# ------------------------------------------------------------------
# SCENE
# ------------------------------------------------------------------
position = np.array([1.1, 1.1, 1.1])
lookAt   = np.array([-0.577, -0.577, -0.577])
up       = np.array([ 0.33333333,  0.33333333, -0.66666667])
right    = np.array([-0.57735027,  0.57735027,  0.0        ])
cam      = Camera(position, lookAt, up, right)

nearPlane   = 0.1
farPlane    = 10.0
fov         = 1.91986
aspectRatio = WIDTH / HEIGHT
proj        = Projection(nearPlane, farPlane, fov, aspectRatio)

lightPosition = np.array([10.0, 0.0, 10.0])

# ------------------------------------------------------------------
# CHARGEMENT
# ------------------------------------------------------------------
vertices, triangles = readply(PLY_PATH)
print(f"[INFO] Mesh : {vertices.shape[0]} sommets, "
      f"{triangles.shape[0]} triangles")

texture = np.asarray(Image.open(TEXTURE_PATH))
print(f"[INFO] Texture : {texture.shape[1]}x{texture.shape[0]} px")

data = {
    'viewMatrix'    : cam.getMatrix(),
    'projMatrix'    : proj.getMatrix(),
    'cameraPosition': position,
    'lightPosition' : lightPosition,
    'texture'       : texture,
}

os.makedirs(os.path.join(_dir, OUTPUT_DIR), exist_ok=True)

# ------------------------------------------------------------------
# FONCTION UTILITAIRE
# ------------------------------------------------------------------

def render_with_filter(filter_mode, downsample_filter="box"):
    """Execute le pipeline complet et retourne (image, duree_sec)."""
    pipeline = GraphicPipeline(WIDTH, HEIGHT,
                                filter_mode=filter_mode,
                                downsample_filter=downsample_filter)
    t0 = time.time()
    pipeline.draw(vertices, triangles, data)
    return pipeline.image, time.time() - t0


def save_image(img, filename):
    """Sauvegarde une image float[0,1] en PNG 8 bits."""
    out_path = os.path.join(_dir, OUTPUT_DIR, filename)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(out_path)
    print(f"[SAVE] {out_path}")


# ------------------------------------------------------------------
# MODE SINGLE
# ------------------------------------------------------------------
if MODE == "single":
    print(f"\n=== Rendu : {FILTER_MODE} / {DOWNSAMPLE_FILTER} ===")
    img, dt = render_with_filter(FILTER_MODE, DOWNSAMPLE_FILTER)
    print(f"[INFO] Rendu termine en {dt:.2f}s")

    if SAVE_IMAGES:
        save_image(img, f"render_{FILTER_MODE}_{DOWNSAMPLE_FILTER}.png")

    plt.figure(figsize=(10, 6))
    plt.title(f"Filtre : {FILTER_MODE}  |  Downsample : {DOWNSAMPLE_FILTER}  "
              f"|  {dt:.1f}s")
    plt.imshow(img); plt.axis('off'); plt.tight_layout(); plt.show()


# ------------------------------------------------------------------
# MODE COMPARE  (grille 2x2 + temps)
# ------------------------------------------------------------------
elif MODE == "compare":
    filters  = ["nearest", "bilinear", "trilinear", "anisotropic"]
    results  = {}
    times    = {}

    for fm in filters:
        print(f"\n=== Rendu : {fm} ===")
        img, dt = render_with_filter(fm, DOWNSAMPLE_FILTER)
        results[fm] = img
        times[fm]   = dt
        if SAVE_IMAGES:
            save_image(img, f"render_{fm}.png")

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(
        f"Comparaison des filtres de texture (downsample={DOWNSAMPLE_FILTER})",
        fontsize=14, fontweight='bold')

    for ax, fm in zip(axes.flat, filters):
        ax.imshow(results[fm])
        ax.set_title(f"{fm.capitalize()}  ({times[fm]:.1f}s)", fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    if SAVE_IMAGES:
        fig.savefig(os.path.join(_dir, OUTPUT_DIR, "comparison_filters.png"),
                    dpi=150, bbox_inches='tight')
        print(f"[SAVE] comparison_filters.png")
    plt.show()

    # Comparaison des filtres de downsampling (uniquement trilinear)
    print("\n=== Comparaison des filtres de downsampling ===")
    ds_filters = ["box", "gaussian", "lanczos"]
    ds_results = {}; ds_times = {}
    for dsf in ds_filters:
        print(f"  -> downsampling={dsf}")
        img, dt = render_with_filter("trilinear", dsf)
        ds_results[dsf] = img; ds_times[dsf] = dt
        if SAVE_IMAGES:
            save_image(img, f"render_trilinear_{dsf}.png")

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle("Comparaison des filtres de downsampling (sampling=trilinear)",
                  fontsize=13, fontweight='bold')
    for ax, dsf in zip(axes2, ds_filters):
        ax.imshow(ds_results[dsf])
        ax.set_title(f"Downsample : {dsf}  ({ds_times[dsf]:.1f}s)", fontsize=11)
        ax.axis('off')
    plt.tight_layout()
    if SAVE_IMAGES:
        fig2.savefig(os.path.join(_dir, OUTPUT_DIR,
                                  "comparison_downsample.png"),
                     dpi=150, bbox_inches='tight')
        print(f"[SAVE] comparison_downsample.png")
    plt.show()


# ------------------------------------------------------------------
# MODE MIPMAP_VIS  (visualisation de la pyramide MIP)
# ------------------------------------------------------------------
elif MODE == "mipmap_vis":
    print("\n=== Visualisation de la pyramide MIP ===")
    mips  = build_mipmaps(texture, DOWNSAMPLE_FILTER)
    atlas = mipmap_atlas(mips)

    print(f"[INFO] {len(mips)} niveaux MIP :")
    for i, m in enumerate(mips):
        print(f"  Level {i:2d} : {m.shape[1]:4d} x {m.shape[0]:4d} px")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                              gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(f"Pyramide MIP — filtre : {DOWNSAMPLE_FILTER}", fontsize=13)

    axes[0].imshow(atlas); axes[0].axis('off')
    axes[0].set_title("Atlas des niveaux MIP (L0 → Lmax)")

    # Graphique : taille en pixels par niveau
    lvls   = list(range(len(mips)))
    pixels = [m.shape[0] * m.shape[1] for m in mips]
    axes[1].semilogy(lvls, pixels, 'o-', color='steelblue', linewidth=2)
    axes[1].set_xlabel("Niveau MIP"); axes[1].set_ylabel("Pixels (log)")
    axes[1].set_title("Taille par niveau"); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if SAVE_IMAGES:
        fig.savefig(os.path.join(_dir, OUTPUT_DIR,
                                 f"mipmap_pyramid_{DOWNSAMPLE_FILTER}.png"),
                    dpi=150, bbox_inches='tight')
        print(f"[SAVE] mipmap_pyramid_{DOWNSAMPLE_FILTER}.png")
    plt.show()
