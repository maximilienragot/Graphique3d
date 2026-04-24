import numpy as np
import time
import os
import matplotlib.pyplot as plt
from PIL import Image

from camera          import Camera
from projection      import Projection
from graphicPipeline import GraphicPipeline
from readply         import readply
from mipmap          import build_mipmaps, mipmap_atlas

_dir = os.path.dirname(os.path.abspath(__file__))

# positions de cameras

def make_camera_damier():
    position = np.array([0.0, 15.0, -10.0])
    lookAt   = np.array([0.0, -0.555, 0.832])
    up       = np.array([0.0,  0.832, 0.555])
    right    = np.array([1.0,  0.0,   0.0  ])
    return Camera(position, lookAt, up, right), position

def make_camera_damier2():
    position = np.array([0.0, 15.0, -1.0])
    lookAt   = np.array([0.0, -0.555, 0.832])
    up       = np.array([0.0,  0.832, 0.555])
    right    = np.array([1.0,  0.0,   0.0  ])
    return Camera(position, lookAt, up, right), position

def make_camera_damier3():
    position = np.array([0.0, 15.0, -10.0])
    lookAt   = np.array([0.0, -0.555, 0.832])
    up       = np.array([0.0,  0.832, 0.555])
    right    = np.array([1.0,  0.0,   0.0  ])
    return Camera(position, lookAt, up, right), position


def make_camera_wall():
    position = np.array([0.96, 15.0, -10.0])
    lookAt   = np.array([0.0, -0.555,  0.832])
    up       = np.array([0.0,  0.832,  0.555])
    right    = np.array([1.0,  0.0,    0.0  ])
    return Camera(position, lookAt, up, right), position

def make_camera_suzanne():
    position = np.array([1.1, 1.1, 1.1])
    lookAt   = np.array([-0.577, -0.577, -0.577])
    up       = np.array([ 0.333,  0.333, -0.667])
    right    = np.array([-0.577,  0.577,  0.0  ])
    return Camera(position, lookAt, up, right), position

def make_camera_plane():
    """Caméra générique pour toutes les surfaces planes (damier, briques, etc.)"""
    position = np.array([0.0, 15.0, -10.0])
    lookAt   = np.array([0.0, -0.555, 0.832])
    up       = np.array([0.0,  0.832, 0.555])
    right    = np.array([1.0,  0.0,   0.0  ])
    return Camera(position, lookAt, up, right), position

cameras = {
    # Scènes originales
    "damier"         : make_camera_damier,
    "damier2"        : make_camera_damier2,
    "damier3"        : make_camera_damier3,
    "wall"           : make_camera_wall,
    "wall2"          : make_camera_wall,
    "suzanne"        : make_camera_suzanne,
    # Nouvelles textures (surfaces planes)
    "carrelage"      : make_camera_plane,
    "checkerboard"   : make_camera_plane,
    "brick"          : make_camera_plane,
    "wood"           : make_camera_plane,
    "marble"         : make_camera_plane,
    "stripes_diag"   : make_camera_plane,
    "grid"           : make_camera_plane,
    "fabric"         : make_camera_plane,
    "radial_gradient": make_camera_plane,
    "noise"          : make_camera_plane,
}


def make_projection(width, height, near=0.1, far=100.0, fov=1.91986):
    return Projection(near, far, fov, width / height)


def load_scene(name, cam, cam_position, proj, light_position):
    ply_path     = os.path.join(_dir, "ply",     f"{name}.ply")
    texture_path = os.path.join(_dir, "texture", f"{name}.png")
    vertices, triangles = readply(ply_path)
    texture = np.asarray(Image.open(texture_path).convert('RGB'))
    data = {
        'viewMatrix'    : cam.getMatrix(),
        'projMatrix'    : proj.getMatrix(),
        'cameraPosition': cam_position,
        'lightPosition' : light_position,
        'texture'       : texture,
    }
    return vertices, triangles, texture, data


def render_with_filter(vertices, triangles, data, width, height,
                        filter_mode, downsample_filter="box"):
    pipeline = GraphicPipeline(width, height,
                                filter_mode=filter_mode,
                                downsample_filter=downsample_filter)
    t0 = time.time()
    pipeline.draw(vertices, triangles, data)
    return pipeline.image, time.time() - t0


def save_image(img, directory, filename):
    os.makedirs(directory, exist_ok=True)
    out_path = os.path.join(directory, filename)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(out_path)


def _filter_dir(image_name, filter_mode, downsample_filter=None):
    parts = [_dir, "output", image_name]
    if filter_mode == "anisotropic":
        parts.append("anisotropic")
    else:
        parts += ["isotropic", filter_mode]
    if downsample_filter:
        parts.append(downsample_filter)
    return os.path.join(*parts)


def _mipmap_dir(image_name, downsample_filter):
    return os.path.join(_dir, "output", image_name, "mipmap", downsample_filter)


def mode_single(vertices, triangles, data, width, height,
                filter_mode, downsample_filter, image_name="", ax=None):
    img, dt = render_with_filter(vertices, triangles, data, width, height,
                                  filter_mode, downsample_filter)
    save_image(img, _filter_dir(image_name, filter_mode, downsample_filter), "render.png")
    if ax is not None:
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{image_name}  |  {filter_mode}  |  {downsample_filter}  |  {dt:.1f}s")


def mode_all(vertices, triangles, data, width, height,
              downsample_filter="box", image_name=""):

    for fm in ["nearest", "bilinear", "trilinear", "anisotropic"]:
        for dsf in ["box", "gaussian", "lanczos", "median"]:
            img, dt = render_with_filter(vertices, triangles, data, width, height,
                                        fm, downsample_filter)
            save_image(img, _filter_dir(image_name, fm, dsf), f"render ({dt:.1f} s).png")



def mode_mipmap_vis(texture, downsample_filter="box", image_name="", ax=None):
    mips  = build_mipmaps(texture, downsample_filter)
    atlas = mipmap_atlas(mips)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))
        created_fig = True
    else:
        fig = ax.get_figure()
        created_fig = False

    fig.suptitle(f"{image_name}  —  Pyramide MIP  |  filtre : {downsample_filter}", fontsize=13)
    ax.imshow(atlas)
    ax.axis('off')
    ax.set_title("Atlas des niveaux MIP (L0 -> Lmax)")

    if created_fig:
        plt.tight_layout()
        d = _mipmap_dir(image_name, downsample_filter)
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, "pyramid.png"), dpi=150, bbox_inches='tight')


def mode_mipmap_vis_all(downsample_filters=None, axes=None):
    if downsample_filters not in [None, "box", "gaussian", "lanczos", "median"]:
        RaiseValueError(f"Valeurs possibles : {"None", "box", "gaussian", "lanczos", "median"}")

    if downsample_filters is None:
        downsample_filters = ["box", "gaussian", "lanczos", "median"]
    for name in cameras:    # pour avoir les noms des images
        texture_path = os.path.join(_dir, "texture", f"{name}.png")
        texture = np.asarray(Image.open(texture_path).convert('RGB'))
        for i, dsf in enumerate(downsample_filters):
            ax = axes[i] if axes is not None else None
            mode_mipmap_vis(texture, dsf, image_name=name, ax=ax)
        if axes is None:
            plt.close('all')


def main(
    mode="single",
    filter_mode="trilinear",
    downsample_filter="box",
    width=512,
    height=288,
    name="damier",
    light_position=None,
    ax=None,
):
    if light_position is None:
        light_position = np.array([10.0, 0.0, 10.0])

    if name not in cameras: # pour avoir le nom des images
        raise ValueError(f"Valeurs possibles : {list(cameras)}")

    cam, cam_pos = cameras[name]()
    proj = make_projection(width, height)
    vertices, triangles, texture, data = load_scene(name, cam, cam_pos, proj, light_position)

    if mode == "single":
        mode_single(vertices, triangles, data, width, height,
                    filter_mode, downsample_filter, image_name=name, ax=ax)
    elif mode == "all":
        mode_all(vertices, triangles, data, width, height,
                 downsample_filter=downsample_filter, image_name=name)
    elif mode == "mipmap_vis":
        mode_mipmap_vis(texture, downsample_filter, image_name=name, ax=ax)
    elif mode == "mipmap_vis_all":
        mode_mipmap_vis_all(axes=ax)
    else:
        raise ValueError(f"Mode inconnu. Valeurs possibles : ['single', 'all', 'mipmap_vis', 'mipmap_vis_all']")



def discover_textures():
    """
    Retourne la liste de toutes les scènes disponibles :
    = les noms présents à la fois dans texture/ et ply/.
    Utile pour lancer main() sur toutes les textures d'un coup.
    """
    tex_dir = os.path.join(_dir, "texture")
    ply_dir = os.path.join(_dir, "ply")
    textures = {os.path.splitext(f)[0]
                for f in os.listdir(tex_dir)
                if f.endswith(".png") and "Identifier" not in f}
    plys     = {os.path.splitext(f)[0]
                for f in os.listdir(ply_dir)
                if f.endswith(".ply")}
    return sorted(textures & plys & cameras.keys())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Renderer 3D — Mipmapping",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Exemples :
  # Rendu unique : trilinear sur damier
  python main2.py --name damier --mode single --filter trilinear --ds box

  # Tous les filtres sur une seule texture
  python main2.py --name brick --mode all

  # Toutes les textures disponibles, mode all
  python main2.py --all --mode all

  # Visualisation pyramide MIP
  python main2.py --name carrelage --mode mipmap_vis --ds gaussian
        """
    )

    parser.add_argument("--name",   default="damier",
                        help=f"Nom de la texture. Disponibles : {discover_textures()}")
    parser.add_argument("--mode",   default="single",
                        choices=["single", "all", "mipmap_vis", "mipmap_vis_all"],
                        help="Mode de rendu (défaut: single)")
    parser.add_argument("--filter", default="trilinear",
                        choices=["nearest", "bilinear", "trilinear", "anisotropic"],
                        help="Filtre de sampling (défaut: trilinear)")
    parser.add_argument("--ds",     default="box",
                        choices=["box", "gaussian", "lanczos", "median"],
                        help="Filtre de downsampling (défaut: box)")
    parser.add_argument("--width",  type=int, default=512,
                        help="Largeur de l'image (défaut: 512)")
    parser.add_argument("--height", type=int, default=288,
                        help="Hauteur de l'image (défaut: 288)")
    parser.add_argument("--all",    action="store_true",
                        help="Traiter toutes les textures disponibles")

    args = parser.parse_args()

    if args.all:
        # Lance main() sur chaque texture disponible
        available = discover_textures()
        print(f"[INFO] Traitement de {len(available)} texture(s) : {available}")
        for name in available:
            print(f"\n{'─'*50}\n  → {name}\n{'─'*50}")
            main(
                mode=args.mode,
                filter_mode=args.filter,
                downsample_filter=args.ds,
                width=args.width,
                height=args.height,
                name=name,
            )
    else:
        main(
            mode=args.mode,
            filter_mode=args.filter,
            downsample_filter=args.ds,
            width=args.width,
            height=args.height,
            name=args.name,
        )
