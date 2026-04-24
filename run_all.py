"""
run_all.py  —  Benchmark automatique sur toutes les textures du dossier texture/

Usage :
    python run_all.py                    # tous les filtres, toutes les textures
    python run_all.py --filter trilinear # un seul filtre de sampling
    python run_all.py --ds gaussian      # un seul filtre de downsampling
    python run_all.py --width 512        # résolution personnalisée

Structure des sorties :
    output/
    └── <nom_texture>/
        ├── comparison_filters.png       ← grille 4 filtres de sampling
        ├── comparison_downsample.png    ← grille 3 filtres de downsampling
        ├── mipmap_pyramid.png           ← atlas des niveaux MIP
        ├── nearest/
        │   └── render.png
        ├── bilinear/
        │   └── render.png
        ├── trilinear/
        │   └── render.png
        └── anisotropic/
            └── render.png

Ajouter une nouvelle texture :
    1. Mettre le fichier .png dans texture/
    2. Mettre le fichier .ply correspondant dans ply/
       (ou copier damier.ply si c'est une surface plane)
    3. Relancer python run_all.py
    → Tout le reste est automatique.
"""

import os, sys, time, argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

from camera          import Camera
from projection      import Projection
from graphicPipeline import GraphicPipeline
from readply         import readply
from mipmap          import build_mipmaps, mipmap_atlas

# ═══════════════════════════════════════════════════════════════════
#  PARAMÈTRES PAR DÉFAUT
# ═══════════════════════════════════════════════════════════════════

WIDTH  = 512
HEIGHT = 288

SAMPLING_FILTERS   = ["nearest", "bilinear", "trilinear", "anisotropic"]
DOWNSAMPLE_FILTERS = ["box", "gaussian", "lanczos"]

LIGHT_POSITION = np.array([10.0, 0.0, 10.0])

# Couleur de bord pour chaque filtre de sampling dans les figures
FILTER_COLORS = {
    "nearest":     "#e74c3c",
    "bilinear":    "#e67e22",
    "trilinear":   "#27ae60",
    "anisotropic": "#2980b9",
}
DS_COLORS = {
    "box":      "#8e44ad",
    "gaussian": "#16a085",
    "lanczos":  "#c0392b",
    "median":   "#d35400",
}

# ═══════════════════════════════════════════════════════════════════
#  CAMÉRAS  (une par type de mesh)
# ═══════════════════════════════════════════════════════════════════
#
#  → damier.ply  : surface plane vue en perspective rasante
#                  → la caméra "plane" convient à toutes les textures
#                    appliquées sur ce mesh
#  → suzanne.ply : tête 3D → caméra dédiée
#
#  Si tu ajoutes un nouveau mesh, ajoute sa caméra ici.

def _camera_plane():
    """Caméra vue en perspective rasante — pour tous les meshes plats."""
    pos    = np.array([0.0, 15.0, -10.0])
    lookAt = np.array([0.0, -0.555,  0.832])
    up     = np.array([0.0,  0.832,  0.555])
    right  = np.array([1.0,  0.0,    0.0  ])
    return Camera(pos, lookAt, up, right), pos

def _camera_suzanne():
    """Caméra face à Suzanne."""
    pos    = np.array([1.1,   1.1,   1.1  ])
    lookAt = np.array([-0.577,-0.577,-0.577])
    up     = np.array([ 0.333, 0.333,-0.667])
    right  = np.array([-0.577, 0.577, 0.0  ])
    return Camera(pos, lookAt, up, right), pos

# Association mesh → caméra
MESH_CAMERA = {
    "suzanne": _camera_suzanne,
    # tous les autres meshes (surfaces planes) utilisent _camera_plane
}

def get_camera(mesh_name):
    fn = MESH_CAMERA.get(mesh_name, _camera_plane)
    return fn()

# ═══════════════════════════════════════════════════════════════════
#  DÉCOUVERTE AUTOMATIQUE DES TEXTURES
# ═══════════════════════════════════════════════════════════════════

def discover_textures():
    """
    Parcourt texture/ et ply/ et retourne la liste des scènes
    disponibles (= paires texture + mesh du même nom).

    Pour ajouter une texture : déposer <nom>.png dans texture/
    et <nom>.ply dans ply/ (ou copier damier.ply).
    """
    tex_dir = os.path.join(_dir, "texture")
    ply_dir = os.path.join(_dir, "ply")

    textures = {os.path.splitext(f)[0]
                for f in os.listdir(tex_dir)
                if f.endswith(".png") and not f.endswith("Identifier")}
    plys     = {os.path.splitext(f)[0]
                for f in os.listdir(ply_dir)
                if f.endswith(".ply")}

    available = sorted(textures & plys)   # intersection : les deux fichiers existent
    missing   = textures - plys
    if missing:
        print(f"[WARN] Textures sans .ply correspondant (ignorées) : {missing}")

    return available

# ═══════════════════════════════════════════════════════════════════
#  CHARGEMENT D'UNE SCÈNE
# ═══════════════════════════════════════════════════════════════════

def load_scene(name, width, height):
    ply_path = os.path.join(_dir, "ply",     f"{name}.ply")
    tex_path = os.path.join(_dir, "texture", f"{name}.png")

    vertices, triangles = readply(ply_path)
    texture = np.asarray(Image.open(tex_path).convert("RGB"))

    cam, cam_pos = get_camera(name)
    proj = Projection(0.1, 100.0, 1.91986, width / height)

    data = {
        'viewMatrix'    : cam.getMatrix(),
        'projMatrix'    : proj.getMatrix(),
        'cameraPosition': cam_pos,
        'lightPosition' : LIGHT_POSITION,
        'texture'       : texture,
    }
    return vertices, triangles, texture, data

# ═══════════════════════════════════════════════════════════════════
#  RENDU
# ═══════════════════════════════════════════════════════════════════

def render(vertices, triangles, data, width, height,
           filter_mode, downsample_filter="box"):
    """Rend la scène et retourne (image_float, durée_sec)."""
    pipeline = GraphicPipeline(width, height,
                                filter_mode=filter_mode,
                                downsample_filter=downsample_filter)
    t0 = time.time()
    pipeline.draw(vertices, triangles, data)
    return pipeline.image, time.time() - t0


def save_png(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)

# ═══════════════════════════════════════════════════════════════════
#  FIGURES DE COMPARAISON
# ═══════════════════════════════════════════════════════════════════

def make_comparison_filters(tex_name, texture, results, times, out_dir):
    """
    Grille 1×5 : texture originale + 4 filtres de sampling.
    """
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle(
        f"« {tex_name} »  —  Comparaison des filtres de sampling",
        fontsize=13, fontweight='bold', y=1.02)

    # Colonne 0 : texture originale
    axes[0].imshow(texture)
    axes[0].set_title("Texture\noriginale", fontsize=9, fontweight='bold')
    axes[0].axis('off')

    for ax, fm in zip(axes[1:], SAMPLING_FILTERS):
        if fm not in results:
            ax.axis('off'); continue
        ax.imshow(results[fm])
        ax.set_title(f"{fm.capitalize()}\n{times[fm]:.1f}s", fontsize=9)
        ax.axis('off')
        for sp in ax.spines.values():
            sp.set_edgecolor(FILTER_COLORS[fm]); sp.set_linewidth(3)

    plt.tight_layout()
    path = os.path.join(out_dir, "comparison_filters.png")
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"    [fig] comparison_filters.png")


def make_comparison_downsample(tex_name, ds_results, ds_times, out_dir):
    """
    Grille 1×3 : 3 filtres de downsampling (sampling=trilinear).
    """
    n = len(DOWNSAMPLE_FILTERS)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 4))
    fig.suptitle(
        f"« {tex_name} »  —  Filtres de downsampling (sampling=trilinear)",
        fontsize=12, fontweight='bold')

    for ax, dsf in zip(axes, DOWNSAMPLE_FILTERS):
        if dsf not in ds_results:
            ax.axis('off'); continue
        ax.imshow(ds_results[dsf])
        ax.set_title(f"Downsample : {dsf}\n{ds_times[dsf]:.1f}s", fontsize=10)
        ax.axis('off')
        for sp in ax.spines.values():
            sp.set_edgecolor(DS_COLORS.get(dsf, 'gray')); sp.set_linewidth(3)

    plt.tight_layout()
    path = os.path.join(out_dir, "comparison_downsample.png")
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"    [fig] comparison_downsample.png")


def make_mipmap_vis(tex_name, texture, downsample_filter, out_dir):
    """Atlas de la pyramide MIP."""
    mips  = build_mipmaps(texture, downsample_filter)
    atlas = mipmap_atlas(mips)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4),
                                    gridspec_kw={'width_ratios': [4, 1]})
    fig.suptitle(
        f"« {tex_name} »  —  Pyramide MIP ({downsample_filter})  "
        f"|  {len(mips)} niveaux",
        fontsize=12)
    ax1.imshow(atlas); ax1.axis('off')
    ax1.set_title("Atlas (L0 → L_max)")
    lvls   = list(range(len(mips)))
    pixels = [m.shape[0]*m.shape[1] for m in mips]
    ax2.semilogy(lvls, pixels, 'o-', color='steelblue', linewidth=2)
    ax2.set_xlabel("Niveau"); ax2.set_ylabel("Pixels (log)")
    ax2.set_title("Taille par niveau"); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "mipmap_pyramid.png")
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"    [fig] mipmap_pyramid.png")

# ═══════════════════════════════════════════════════════════════════
#  TRAITEMENT D'UNE TEXTURE
# ═══════════════════════════════════════════════════════════════════

def process_texture(name, width, height,
                    sampling_filters=None, downsample_filters=None):
    """
    Charge la scène, rend avec tous les filtres demandés,
    sauvegarde les images individuelles et les figures de comparaison.
    """
    if sampling_filters  is None: sampling_filters  = SAMPLING_FILTERS
    if downsample_filters is None: downsample_filters = DOWNSAMPLE_FILTERS

    out_base = os.path.join(_dir, "output", name)
    os.makedirs(out_base, exist_ok=True)

    print(f"\n{'─'*55}")
    print(f"  Texture : {name}")
    print(f"{'─'*55}")

    vertices, triangles, texture, data = load_scene(name, width, height)
    print(f"  Mesh   : {vertices.shape[0]} sommets, "
          f"{triangles.shape[0]} triangles")
    print(f"  Texture: {texture.shape[1]}×{texture.shape[0]} px")

    # ── Filtres de sampling (downsampling=box par défaut)
    results = {}; times = {}
    for fm in sampling_filters:
        print(f"  → sampling={fm} ...", end="", flush=True)
        img, dt = render(vertices, triangles, data, width, height, fm, "box")
        results[fm] = img; times[fm] = dt
        save_png(img, os.path.join(out_base, fm, "render.png"))
        print(f" {dt:.1f}s  [saved]")

    make_comparison_filters(name, texture, results, times, out_base)

    # ── Filtres de downsampling (sampling=trilinear)
    ds_results = {}; ds_times = {}
    for dsf in downsample_filters:
        print(f"  → downsample={dsf} (trilinear) ...", end="", flush=True)
        img, dt = render(vertices, triangles, data, width, height,
                         "trilinear", dsf)
        ds_results[dsf] = img; ds_times[dsf] = dt
        save_png(img, os.path.join(out_base, f"trilinear_{dsf}", "render.png"))
        print(f" {dt:.1f}s  [saved]")

    make_comparison_downsample(name, ds_results, ds_times, out_base)

    # ── Visualisation pyramide MIP
    make_mipmap_vis(name, texture, "box", out_base)

# ═══════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark mipmap sur toutes les textures")
    p.add_argument("--filter", default=None,
                   choices=SAMPLING_FILTERS,
                   help="Un seul filtre de sampling (défaut: tous)")
    p.add_argument("--ds",     default=None,
                   choices=DOWNSAMPLE_FILTERS,
                   help="Un seul filtre de downsampling (défaut: tous)")
    p.add_argument("--width",  type=int, default=WIDTH)
    p.add_argument("--height", type=int, default=HEIGHT)
    p.add_argument("--only",   default=None,
                   help="Traiter une seule texture (ex: --only brick)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    sf  = [args.filter] if args.filter else SAMPLING_FILTERS
    dsf = [args.ds]     if args.ds     else DOWNSAMPLE_FILTERS

    textures = discover_textures()
    if args.only:
        if args.only not in textures:
            print(f"[ERROR] '{args.only}' introuvable. "
                  f"Disponibles : {textures}")
            sys.exit(1)
        textures = [args.only]

    print(f"[INFO] {len(textures)} texture(s) à traiter : {textures}")
    print(f"[INFO] Filtres sampling   : {sf}")
    print(f"[INFO] Filtres downsampling : {dsf}")
    print(f"[INFO] Résolution : {args.width}×{args.height}")

    t_total = time.time()
    for name in textures:
        process_texture(name, args.width, args.height, sf, dsf)

    print(f"\n{'═'*55}")
    print(f"  TERMINÉ — {len(textures)} texture(s) traitée(s)")
    print(f"  Durée totale : {time.time()-t_total:.1f}s")
    print(f"  Résultats dans : {os.path.join(_dir, 'output')}/")
    print(f"{'═'*55}")
