import numpy as np

# ============================================================
#  projection.py
#  Matrice de projection perspective.
#  Transforme les coordonnées repère caméra → Clip Space,
#  puis la division par w donne le NDC ([-1,1]^3).
# ============================================================

class Projection:
    """
    Projection perspective définie par :
      - near        : distance du plan de découpe proche (near plane)
      - far         : distance du plan de découpe lointain (far plane)
      - fov         : champ de vision vertical en RADIANS
      - aspectRatio : largeur / hauteur de l'image (ex: 1280/720)
    """

    def __init__(self, near, far, fov, aspectRatio):
        self.nearPlane   = near
        self.farPlane    = far
        self.fov         = fov          # en radians
        self.aspectRatio = aspectRatio

    def getMatrix(self):
        """
        Construit la matrice de projection perspective (4×4).

        s = 1/tan(fov/2) : facteur de zoom vertical.
          Plus fov est petit → s grand → zoom avant (télé).
          Plus fov est grand → grand angle (fisheye).

        La 4e ligne [0,0,1,0] copie z dans w.
        Après le vertex shader, on effectue x/w, y/w, z/w :
        c'est la "division de perspective" qui crée l'effet
        de profondeur (objets lointains semblent plus petits).

        La ligne z mappe [near, far] → [0, 1] en NDC.
        """

        f = self.farPlane
        n = self.nearPlane
        s = 1.0 / np.tan(self.fov / 2.0)

        perspective = np.array([
            [s / self.aspectRatio,  0,  0,              0           ],
            [0,                     s,  0,              0           ],
            [0,                     0,  f / (f - n),   -(f*n)/(f-n)],
            [0,                     0,  1,              0           ]
        ], dtype=float)

        return perspective
