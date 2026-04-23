import numpy as np

# ============================================================
#  camera.py
#  Représente la caméra dans la scène 3D.
#  Produit la View Matrix : transforme les coordonnées du
#  repère monde vers le repère caméra.
# ============================================================

class Camera:
    """
    La caméra est définie par 4 vecteurs :
      - position : où se trouve la caméra dans le monde
      - lookAt   : direction vers laquelle elle regarde (normalisé)
      - up       : vecteur "haut" de la caméra (normalisé)
      - right    : vecteur "droite" de la caméra (normalisé)

    Ces 3 vecteurs (right, up, lookAt) forment une base
    orthonormée qui décrit l'orientation de la caméra.
    """

    def __init__(self, position, lookAt, up, right):
        self.position = np.array(position, dtype=float)

        # On normalise chaque direction : un vecteur non unitaire
        # fausserait la rotation et déformerait la scène.
        self.lookAt  = np.array(lookAt, dtype=float)
        self.lookAt /= np.linalg.norm(self.lookAt)

        self.up  = np.array(up, dtype=float)
        self.up /= np.linalg.norm(self.up)

        self.right  = np.array(right, dtype=float)
        self.right /= np.linalg.norm(self.right)

    def getMatrix(self):
        """
        Construit la View Matrix (4×4) = Rotation × Translation.

        Étape 1 — Translation : ramène la caméra à l'origine
          en décalant tout le monde de -position.

        Étape 2 — Rotation : projette chaque axe monde sur les
          axes de la caméra (right, up, lookAt).

        L'ordre est crucial : translater d'abord, puis tourner.
        """

        # Matrice de translation : déplace le monde de -position
        translation = np.array([
            [1, 0, 0, -self.position[0]],
            [0, 1, 0, -self.position[1]],
            [0, 0, 1, -self.position[2]],
            [0, 0, 0,  1               ]
        ], dtype=float)

        # Matrice de rotation : chaque ligne = un axe de la caméra.
        # Multiplier un vecteur par cette matrice le re-exprime
        # dans le repère caméra.
        rotation = np.array([
            [self.right[0],  self.right[1],  self.right[2],  0],
            [self.up[0],     self.up[1],     self.up[2],     0],
            [self.lookAt[0], self.lookAt[1], self.lookAt[2], 0],
            [0,              0,              0,              1]
        ], dtype=float)

        # View Matrix finale
        return np.matmul(rotation, translation)
