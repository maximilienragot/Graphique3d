import numpy as np
from mipmap import (build_mipmaps, sample_nearest, sample_bilinear,
                    sample_trilinear, sample_anisotropic,
                    compute_lod_accurate)

# ============================================================
#  graphicPipeline.py
#  Pipeline graphique 3D "software renderer" complet.
#
#  Etapes du pipeline :
#    1. Pre-calcul mipmaps  -> pyramide de textures
#    2. Vertex Shader       -> transforme les sommets monde -> NDC
#    3. Rasterizer          -> triangle -> fragments (pixels candidats)
#                             + calcul du LOD precis (4 derivees partielles)
#    4. Fragment Shader     -> Phong + filtre de texture configurable
#    5. Depth Test          -> ne garde que le fragment le plus proche
#
#  Filtres disponibles (parametre FILTER_MODE dans main.py) :
#    "nearest"      -> nearest-neighbour (baseline, reference)
#    "bilinear"     -> bilineaire sur un seul niveau MIP
#    "trilinear"    -> bilineaire x2 niveaux + interpolation (standard)
#    "anisotropic"  -> anisotropique simplifie (meilleur sur surfaces obliques)
#
#  Filtres de downsampling (parametre DOWNSAMPLE_FILTER dans main.py) :
#    "box"          -> box filter (standard, rapide)
#    "gaussian"     -> gaussien 4 taps (moins d'aliasing residuel)
#    "lanczos"      -> Lanczos ordre 2 (plus net, plus lent)
#
#  Donnees par sommet (8 valeurs dans le .ply) :
#    [0] x, [1] y, [2] z   -> position monde
#    [3] nx,[4] ny,[5] nz  -> normale
#    [6] u, [7] v          -> coordonnees de texture
#
#  Donnees apres vertex shader (18 valeurs) :
#    [0:3]  position NDC
#    [3:6]  normale monde
#    [6:9]  vecteur V (camera - sommet)
#    [9:12] vecteur L (lumiere - sommet)
#    [12:14] coordonnees UV
#    [14:16] coordonnees NDC (x,y) -> pour LOD
#    [16:18] coordonnees ecran (px, py) -> pour LOD precis
# ============================================================


def edgeSide(p, v0, v1):
    """
    Produit vectoriel 2D de (v0->v1) x (v0->p).
    Positif -> p a gauche de l'arete ; negatif -> a droite ; 0 -> sur l'arete.
    """
    return ((p[0]  - v0[0]) * (v1[1] - v0[1])
          - (p[1]  - v0[1]) * (v1[0] - v0[0]))


class Fragment:
    """
    Un Fragment = un pixel candidat emis par le rasterizer.

    Attributs :
      x, y              : coordonnees pixel (entiers)
      depth             : profondeur z NDC (pour le depth test)
      interpolated_data : attributs interpoles (N, V, L, UV, ...)
      lod               : LOD isotropique calcule
      dudx,dvdx,dudy,dvdy : derivees partielles UV (pour anisotropique)
      output            : couleur finale RGB [0,1]
    """

    def __init__(self, x, y, depth, interpolated_data, lod=0.0,
                 dudx=0.0, dvdx=0.0, dudy=0.0, dvdy=0.0):
        self.x                = x
        self.y                = y
        self.depth            = depth
        self.interpolated_data = interpolated_data
        self.lod              = lod
        self.dudx             = dudx
        self.dvdx             = dvdx
        self.dudy             = dudy
        self.dvdy             = dvdy
        self.output           = np.zeros(3, dtype=float)


class GraphicPipeline:
    """
    Pipeline graphique complet avec filtres de texture configurables.
    """

    def __init__(self, width, height,
                 filter_mode="trilinear",
                 downsample_filter="box"):
        """
        Parametres :
          width, height      : resolution de l'image de sortie
          filter_mode        : filtre de sampling ("nearest", "bilinear",
                               "trilinear", "anisotropic")
          downsample_filter  : filtre de downsampling pour la pyramide
                               ("box", "gaussian", "lanczos")
        """
        self.width             = width
        self.height            = height
        self.filter_mode       = filter_mode
        self.downsample_filter = downsample_filter

        self.image       = np.zeros((height, width, 3), dtype=float)
        self.depthBuffer = np.ones((height, width), dtype=float)
        self.mips        = None

    # ==============================================================
    #  ETAPE 1 — VERTEX SHADER
    # ==============================================================

    def VertexShader(self, vertex, data):
        """
        Transforme un sommet monde -> NDC et calcule N, V, L.

        Entree  (vertex, 8 valeurs) :
          [0:3]  position monde
          [3:6]  normale
          [6:8]  UV texture

        Sortie (18 valeurs) :
          [0:3]  position NDC
          [3:6]  normale monde
          [6:9]  vecteur V (vue)
          [9:12] vecteur L (lumiere)
          [12:14] UV
          [14:16] NDC (x, y) -> LOD
          [16:18] position ecran en pixels (px, py) -> LOD precis
        """
        out = np.zeros(18, dtype=float)

        pos  = np.array([vertex[0], vertex[1], vertex[2], 1.0])
        clip = data['projMatrix'] @ (data['viewMatrix'] @ pos)
        w    = clip[3]

        out[0] = clip[0] / w
        out[1] = clip[1] / w
        out[2] = clip[2] / w

        out[3] = vertex[3]; out[4] = vertex[4]; out[5] = vertex[5]

        out[6]  = data['cameraPosition'][0] - vertex[0]
        out[7]  = data['cameraPosition'][1] - vertex[1]
        out[8]  = data['cameraPosition'][2] - vertex[2]

        out[9]  = data['lightPosition'][0] - vertex[0]
        out[10] = data['lightPosition'][1] - vertex[1]
        out[11] = data['lightPosition'][2] - vertex[2]

        out[12] = vertex[6]   # u
        out[13] = vertex[7]   # v

        out[14] = out[0]      # NDC x
        out[15] = out[1]      # NDC y

        # Position ecran en pixels (pour derivees partielles precises)
        out[16] = (out[0] + 1.0) * 0.5 * self.width
        out[17] = (out[1] + 1.0) * 0.5 * self.height

        return out

    # ==============================================================
    #  ETAPE 2 — RASTERIZER avec LOD precis (4 derivees partielles)
    # ==============================================================

    def Rasterizer(self, v0, v1, v2):
        """
        Convertit un triangle en liste de Fragments.

        Calcul du LOD precis (formule OpenGL) :
          On estime les 4 derivees partielles des UV :
            dU/dx, dV/dx (variation selon x ecran)
            dU/dy, dV/dy (variation selon y ecran)
          via les rapports entre les sommets du triangle.

          LOD = log2( max( ||dUV/dx||, ||dUV/dy|| ) x tex_size )

          Cette formule est plus correcte que l'approximation diagonale
          max(|dU/dx|, |dV/dy|) qui ignorait dV/dx et dU/dy.
        """
        fragments = []

        # ---- Back-face culling --------------------------------
        area = edgeSide(v0, v1, v2)
        if area <= 0:
            return fragments

        # ---- Conversion NDC -> pixels --------------------------
        def ndc_to_px(v):
            return np.array([(v[0] + 1.0) * 0.5 * self.width,
                             (v[1] + 1.0) * 0.5 * self.height])

        p0 = ndc_to_px(v0); p1 = ndc_to_px(v1); p2 = ndc_to_px(v2)

        # ---- AABB clippee aux bords de l'image ----------------
        A = np.floor(np.min([p0, p1, p2], axis=0)).astype(int)
        B = np.ceil( np.max([p0, p1, p2], axis=0)).astype(int)
        A = np.clip(A, [0, 0], [self.width - 1, self.height - 1])
        B = np.clip(B, [0, 0], [self.width - 1, self.height - 1])

        # ---- Calcul LOD precis (4 derivees partielles) ----------
        u0, v_0 = v0[12], v0[13]
        u1, v_1 = v1[12], v1[13]
        u2, v_2 = v2[12], v2[13]

        dx_01 = p1[0] - p0[0]; dy_01 = p1[1] - p0[1]
        dx_02 = p2[0] - p0[0]; dy_02 = p2[1] - p0[1]
        du_01 = u1 - u0;        dv_01 = v_1 - v_0
        du_02 = u2 - u0;        dv_02 = v_2 - v_0

        # Resolution du systeme 2x2 pour dU/dx, dU/dy, dV/dx, dV/dy
        det = dx_01 * dy_02 - dx_02 * dy_01
        if abs(det) > 1e-10:
            inv_det = 1.0 / det
            dudx = (du_01 * dy_02 - du_02 * dy_01) * inv_det
            dudy = (dx_01 * du_02 - dx_02 * du_01) * inv_det
            dvdx = (dv_01 * dy_02 - dv_02 * dy_01) * inv_det
            dvdy = (dx_01 * dv_02 - dx_02 * dv_01) * inv_det
        else:
            dudx = dvdx = dudy = dvdy = 0.0

        if self.mips is not None:
            tex_size = max(self.mips[0].shape[0], self.mips[0].shape[1])
        else:
            tex_size = 1

        lod = compute_lod_accurate(dudx, dvdx, dudy, dvdy, tex_size)

        # ---- Boucle sur les pixels de la AABB ----------------
        for j in range(A[1], B[1] + 1):
            for i in range(A[0], B[0] + 1):

                x = (i + 0.5) / self.width  * 2.0 - 1.0
                y = (j + 0.5) / self.height * 2.0 - 1.0
                p = np.array([x, y])

                a0 = edgeSide(p, v0, v1)
                a1 = edgeSide(p, v1, v2)
                a2 = edgeSide(p, v2, v0)

                if a0 >= 0 and a1 >= 0 and a2 >= 0:
                    l0 = a1 / area; l1 = a2 / area; l2 = a0 / area

                    iz0 = 1.0 / v0[2] if abs(v0[2]) > 1e-8 else 1.0
                    iz1 = 1.0 / v1[2] if abs(v1[2]) > 1e-8 else 1.0
                    iz2 = 1.0 / v2[2] if abs(v2[2]) > 1e-8 else 1.0

                    z = l0 * v0[2] + l1 * v1[2] + l2 * v2[2]

                    denom = l0*iz0 + l1*iz1 + l2*iz2
                    if abs(denom) < 1e-12:
                        continue
                    w0 = (l0 * iz0) / denom
                    w1 = (l1 * iz1) / denom
                    w2 = (l2 * iz2) / denom

                    n = v0.shape[0]
                    interp = v0[3:n] * w0 + v1[3:n] * w1 + v2[3:n] * w2

                    fragments.append(Fragment(i, j, z, interp, lod,
                                              dudx, dvdx, dudy, dvdy))

        return fragments

    # ==============================================================
    #  ETAPE 3 — FRAGMENT SHADER (Phong + filtre configurable)
    # ==============================================================

    def fragmentShader(self, fragment, data):
        """
        Calcule la couleur finale : eclairage de Phong + texture.

        Le filtre de texture est selectionne via self.filter_mode :
          "nearest"     -> sample_nearest sur le niveau floor(LOD)
          "bilinear"    -> sample_bilinear sur le niveau floor(LOD)
          "trilinear"   -> sample_trilinear (standard)
          "anisotropic" -> sample_anisotropic (multi-tap directionnel)

        Modele de Phong :
          phong = ka*ambiant + kd*diffus + ks*speculaire
          avec toon shading (quantification en niveaux discrets).
        """
        N = fragment.interpolated_data[0:3]
        n_len = np.linalg.norm(N)
        if n_len < 1e-8:
            fragment.output = np.zeros(3); return
        N = N / n_len

        V = fragment.interpolated_data[3:6]
        v_len = np.linalg.norm(V)
        if v_len < 1e-8:
            fragment.output = np.zeros(3); return
        V = V / v_len

        L = fragment.interpolated_data[6:9]
        l_len = np.linalg.norm(L)
        if l_len < 1e-8:
            fragment.output = np.zeros(3); return
        L = L / l_len

        NdotL    = np.dot(N, L)
        R        = 2.0 * NdotL * N - L
        ambient  = 1.0
        diffuse  = max(NdotL, 0.0)
        specular = pow(max(np.dot(R, V), 0.0), 64)

        ka = 0.1; kd = 0.9; ks = 0.3
        phong = ka * ambient + kd * diffuse + ks * specular
        phong = np.ceil(phong * 4 + 1) / 6.0  # toon shading

        u = fragment.interpolated_data[9]
        v = fragment.interpolated_data[10]

        # --- Echantillonnage de texture -------------------------
        fm = self.filter_mode

        if fm == "nearest":
            lvl = min(int(np.floor(fragment.lod)), len(self.mips) - 1)
            tex_color = sample_nearest(self.mips[lvl], u, v)

        elif fm == "bilinear":
            lvl = min(int(np.floor(fragment.lod)), len(self.mips) - 1)
            tex_color = sample_bilinear(self.mips[lvl], u, v)

        elif fm == "trilinear":
            tex_color = sample_trilinear(self.mips, u, v, fragment.lod)

        elif fm == "anisotropic":
            tex_color = sample_anisotropic(
                self.mips, u, v, fragment.lod,
                fragment.dudx, fragment.dvdx,
                fragment.dudy, fragment.dvdy,
                max_samples=8)
        else:
            tex_color = sample_trilinear(self.mips, u, v, fragment.lod)

        fragment.output = np.array([phong, phong, phong]) * tex_color

    # ==============================================================
    #  BOUCLE PRINCIPALE
    # ==============================================================

    def draw(self, vertices, triangles, data):
        """
        Execute le pipeline complet.

          1. Construction de la pyramide MIP (pré-calcul)
          2. Vertex Shader sur tous les sommets
          3. Rasterization de chaque triangle
          4. Fragment Shader + Depth Test sur chaque fragment
        """
        print(f"[INFO] Construction pyramide MIP "
              f"(downsample={self.downsample_filter})...")
        self.mips = build_mipmaps(data['texture'], self.downsample_filter)
        print(f"[INFO] {len(self.mips)} niveaux MIP generes "
              f"(de {self.mips[0].shape[1]}x{self.mips[0].shape[0]} "
              f"a {self.mips[-1].shape[1]}x{self.mips[-1].shape[0]})")
        print(f"[INFO] Filtre de sampling : {self.filter_mode}")

        nb_vertices = vertices.shape[0]
        self.newVertices = np.zeros((nb_vertices, 18), dtype=float)
        for i in range(nb_vertices):
            self.newVertices[i] = self.VertexShader(vertices[i], data)

        all_fragments = []
        for tri in triangles:
            v0 = self.newVertices[tri[0]]
            v1 = self.newVertices[tri[1]]
            v2 = self.newVertices[tri[2]]
            all_fragments.extend(self.Rasterizer(v0, v1, v2))

        print(f"[INFO] {len(all_fragments)} fragments generes")

        for f in all_fragments:
            self.fragmentShader(f, data)
            if self.depthBuffer[f.y, f.x] > f.depth:
                self.depthBuffer[f.y, f.x] = f.depth
                self.image[f.y, f.x]       = f.output
