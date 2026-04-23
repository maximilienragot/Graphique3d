import numpy as np

# ============================================================
#  readply.py
#  Lecteur de fichiers PLY (Polygon File Format / Stanford).
#
#  Format attendu (ASCII) :
#    ply
#    format ascii 1.0
#    element vertex N
#      property float x / y / z
#      property float nx / ny / nz
#      property float s / t  (ou u / v)
#    element face M
#      property list uchar int vertex_indices
#    end_header
#    ... données ...
#
#  Retourne :
#    vertices  : np.ndarray (N, 8) → [x, y, z, nx, ny, nz, u, v]
#    triangles : np.ndarray (M, 3) → indices des 3 sommets
# ============================================================

def readply(filename):
    """
    Lit un fichier PLY ASCII et retourne les tableaux numpy
    de sommets et de triangles.

    Corrections apportées vs version initiale :
      - .strip() global → supprime \\r Windows et espaces parasites
      - Fan triangulation → gère les polygones avec plus de 3 côtés
      - Gestion des lignes vides dans l'en-tête
    """

    with open(filename, 'r') as f:
        lines = f.readlines()

    vertices  = []
    triangles = []
    nbVertices = 0
    nbFaces    = 0

    # Machine à états :
    #   0 = lecture en-tête
    #   1 = lecture sommets
    #   2 = lecture faces
    state   = 0
    counter = 0

    for raw_line in lines:

        # strip() supprime \n, \r (Windows), et espaces en tête/queue
        line = raw_line.strip()

        # ------- EN-TÊTE -------------------------------------------
        if state == 0:
            parts = line.split()
            if not parts:
                continue   # ligne vide → on ignore

            if parts[0] == 'element':
                if parts[1] == 'vertex':
                    nbVertices = int(parts[2])
                elif parts[1] == 'face':
                    nbFaces = int(parts[2])

            if parts[0] == 'end_header':
                state = 1
                continue

        # ------- SOMMETS -------------------------------------------
        elif state == 1:
            parts = line.split()
            if not parts:
                continue
            # Chaque ligne = 8 floats : x y z nx ny nz u v
            vertex = [float(v) for v in parts]
            vertices.append(vertex)
            counter += 1
            if counter == nbVertices:
                counter = 0
                state   = 2

        # ------- FACES (triangles ou polygones) --------------------
        elif state == 2:
            parts = line.split()
            if not parts:
                continue
            # Format : "3 i0 i1 i2"  ou  "4 i0 i1 i2 i3" (quad)
            n = int(parts[0])
            indices = [int(p) for p in parts[1:n+1]]

            # Fan triangulation : découpe tout polygone en triangles
            # depuis le sommet 0.
            # Triangle k = (indices[0], indices[k], indices[k+1])
            for k in range(1, len(indices) - 1):
                triangles.append([indices[0], indices[k], indices[k+1]])

    return (np.array(vertices,  dtype=np.float64),
            np.array(triangles, dtype=np.int32))
