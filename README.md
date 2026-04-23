# Software 3D Renderer — Python (Mipmapping)

Renderer 3D complet implémenté en Python pur (NumPy).
Simule le pipeline graphique d'un GPU en logiciel.
Projet ENSIMAG — Amélioration du pipeline de rendu par **texture mipmapping**.

---

## Structure du projet

```
renderer/
├── main.py             → Point d'entrée + modes de comparaison
├── camera.py           → View Matrix (monde → caméra)
├── projection.py       → Projection Matrix (perspective)
├── graphicPipeline.py  → Pipeline complet (VS, Rasterizer, FS)
├── mipmap.py           → Pyramide MIP + tous les filtres
├── readply.py          → Lecteur fichiers PLY ASCII
├── suzanne.ply         → Mesh 3D
├── suzanne.png         → Texture
└── output/             → Images générées
```

---

## Pipeline graphique

```
Sommets .ply (x, y, z, nx, ny, nz, u, v)
        │
        ▼
┌──────────────────┐
│  build_mipmaps() │  Pyramide MIP pré-calculée (une fois)
│  mipmap.py       │  Filtres: box | gaussian | lanczos
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Vertex Shader   │  monde → NDC (clip space)
│                  │  calcul N, V, L par sommet
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Rasterizer     │  back-face culling
│                  │  AABB + test pixel-dans-triangle
│                  │  interpolation perspective-correcte
│                  │  LOD précis (4 dérivées partielles)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Fragment Shader  │  Phong (ambiant + diffus + spéculaire)
│                  │  Toon shading (quantification)
│                  │  Filtre texture configurable :
│                  │    nearest | bilinear | trilinear | anisotropic
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Depth Test     │  dépthBuffer → ne garde que le plus proche
└────────┬─────────┘
         │
         ▼
    Image finale (H × W × 3) float [0,1]
```

---

## Mipmapping — méthodes implémentées

### Filtres de downsampling (construction de la pyramide)

| Filtre | Description | Coût |
|--------|-------------|------|
| `box` | Moyenne uniforme 2×2 — standard, rapide | ★ |
| `gaussian` | Noyau gaussien séparable 4 taps (σ≈0.85) — moins d'aliasing résiduel | ★★ |
| `lanczos` | Filtre sinc fenêtré (ordre 2) — plus net, professionnel | ★★★ |

### Filtres de sampling (interpolation lors du rendu)

| Filtre | Description | Qualité | Coût |
|--------|-------------|---------|------|
| `nearest` | Nearest-neighbour — référence/baseline | ★ | ★ |
| `bilinear` | Interpolation 4 voisins sur un niveau | ★★ | ★★ |
| `trilinear` | Bilinéaire ×2 niveaux + lerp — standard OpenGL | ★★★ | ★★★ |
| `anisotropic` | Multi-tap directionnel — meilleur sur surfaces obliques | ★★★★ | ★★★★ |

### Calcul du LOD (Level of Detail)

Formule précise selon OpenGL (EXT_texture_lod) :

```
rho   = max( ||dUV/dx||, ||dUV/dy|| ) × tex_size
LOD   = log2(rho)

avec  ||dUV/dx|| = sqrt(dudx² + dvdx²)
      ||dUV/dy|| = sqrt(dudy² + dvdy²)
```

Les 4 dérivées partielles `dudx, dvdx, dudy, dvdy` sont calculées
par résolution d'un système linéaire 2×2 à partir des coordonnées
des sommets du triangle.

---

## Modes de rendu (`main.py`)

```python
MODE = "single"      # Rendu avec un seul filtre
MODE = "compare"     # Grille 2x2 (4 filtres) + comparaison downsampling
MODE = "mipmap_vis"  # Visualisation de la pyramide MIP (atlas + courbe)
```

---

## Dépendances

```bash
pip install numpy Pillow matplotlib
```

## Lancement

```bash
cd renderer/
python main.py
```

---

## Résultats et comparaison

| Méthode | Anti-aliasing | Flou | Surfaces obliques | Coût |
|---------|--------------|------|-------------------|------|
| Sans mipmap (nearest) | ✗ Aliasing fort | ✗ Pixelisé | ✗ | Très faible |
| Bilinéaire (niveau fixe) | ✗ Aliasing lointain | ✓ | ✗ | Faible |
| Trilinéaire (standard) | ✓ | ✓ | ✗ Légèrement flou | Modéré |
| Anisotropique | ✓ | ✓ | ✓ | Élevé |

---

## Corrections et améliorations

| Fichier | Amélioration |
|---------|-------------|
| `mipmap.py` | **Nouveau** : 3 filtres de downsampling (box, gaussian, lanczos) |
| `mipmap.py` | **Nouveau** : 4 filtres de sampling (nearest, bilinear, trilinear, anisotropic) |
| `mipmap.py` | **Nouveau** : `mipmap_atlas()` pour visualisation de la pyramide |
| `mipmap.py` | **Nouveau** : `compute_lod_accurate()` — formule OpenGL précise |
| `graphicPipeline.py` | **Amélioré** : LOD précis avec 4 dérivées partielles |
| `graphicPipeline.py` | **Amélioré** : vertex shader étendu à 18 attributs |
| `graphicPipeline.py` | **Amélioré** : filtre de sampling configurable (filter_mode) |
| `main.py` | **Nouveau** : 3 modes (single, compare, mipmap_vis) |
| `main.py` | **Nouveau** : comparaison automatique filtres + sauvegarde PNG |
| `readply.py` | `.strip()` global + fan triangulation pour quads |
