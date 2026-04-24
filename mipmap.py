import numpy as np

# ──────────────────────────────────────────────────────────────
#  FILTRES DE DOWNSAMPLING  (construction de la pyramide)
# ──────────────────────────────────────────────────────────────

def _box_downsample(img):
    """
    Box filter : moyenne uniforme de blocs 2x2.
    """
    H, W = img.shape[0], img.shape[1]
    H2, W2 = H - (H % 2), W - (W % 2)
    cropped = img[:H2, :W2]
    H2, W2  = H2 // 2, W2 // 2
    return (cropped
            .reshape(H2, 2, W2, 2, 3)
            .mean(axis=(1, 3))
            .astype(np.float32))


def _gaussian_downsample(img):
    """
    Filtre gaussien (sigma ~0.85)
    Le noyau separe 1D est applique horizontalement puis verticalement.
    """
    kernel_1d = np.array([0.0625, 0.4375, 0.4375, 0.0625], dtype=np.float32)
    H, W = img.shape[:2]
    out  = np.zeros_like(img)

    # Convolution horizontale
    for k, w in enumerate(kernel_1d):
        shift = k - 1
        x0 = max(0, shift);  x1 = min(W, W + shift)
        xd0 = max(0, -shift); xd1 = min(W, W - shift)
        out[:, xd0:xd1] += w * img[:, x0:x1]

    tmp = out.copy(); out[:] = 0.0

    # Convolution verticale
    for k, w in enumerate(kernel_1d):
        shift = k - 1
        y0 = max(0, shift);  y1 = min(H, H + shift)
        yd0 = max(0, -shift); yd1 = min(H, H - shift)
        out[yd0:yd1, :] += w * tmp[y0:y1, :]

    H2, W2 = H - (H % 2), W - (W % 2)
    return out[:H2:2, :W2:2].astype(np.float32)


def _lanczos_kernel_vals(x, a=2):
    """Noyau de Lanczos : sinc(x) * sinc(x/a), |x| < a."""
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    mask = np.abs(x) < a
    xm = x[mask]
    pi_xm = np.pi * xm
    result[mask] = np.where(
        np.abs(xm) < 1e-10,
        1.0,
        (np.sin(pi_xm) / pi_xm) * (np.sin(pi_xm / a) / (pi_xm / a))
    )
    return result.astype(np.float32)


def _lanczos_downsample(img, a=2):
    """
    Filtre de Lanczos (ordre a=2)
    """
    H, W    = img.shape[:2]
    Ho, Wo  = H // 2, W // 2
    out     = np.zeros((Ho, Wo, 3), dtype=np.float32)

    xs_all = np.arange(W, dtype=np.float32)
    ys_all = np.arange(H, dtype=np.float32)

    for j in range(Ho):
        cy = 2.0 * j + 0.5
        ys = np.arange(max(0, int(cy) - a + 1), min(H, int(cy) + a + 1))
        wy = _lanczos_kernel_vals((ys - cy), a)

        for i in range(Wo):
            cx = 2.0 * i + 0.5
            xs = np.arange(max(0, int(cx) - a + 1), min(W, int(cx) + a + 1))
            wx = _lanczos_kernel_vals((xs - cx), a)

            W2d     = np.outer(wy, wx)
            W2d_sum = W2d.sum()
            if W2d_sum < 1e-8:
                continue
            W2d    /= W2d_sum
            patch   = img[np.ix_(ys, xs)]
            out[j, i] = (W2d[:, :, np.newaxis] * patch).sum(axis=(0, 1))

    return np.clip(out, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────
#  CONSTRUCTION DE LA PYRAMIDE MIP
# ──────────────────────────────────────────────────────────────

DOWNSAMPLE_METHODS = {
    "box":      _box_downsample,
    "gaussian": _gaussian_downsample,
    "lanczos":  _lanczos_downsample,
}


def build_mipmaps(texture, downsample_filter="box"):
    """
    Construit la pyramide MIP complete a partir de la texture originale.
    """
    fn      = DOWNSAMPLE_METHODS.get(downsample_filter, _box_downsample)
    current = texture.astype(np.float32) / 255.0
    mips    = [current]

    while current.shape[0] > 1 and current.shape[1] > 1:
        down    = fn(current)
        mips.append(down)
        current = down

    return mips


# ──────────────────────────────────────────────────────────────
#  FILTRES DE SAMPLING  (interpolation lors du rendu)
# ──────────────────────────────────────────────────────────────

def sample_nearest(mip_level, u, v):
    """
    Nearest-neighbour : texel le plus proche, sans interpolation.
    """
    u = u % 1.0; v = v % 1.0
    H, W = mip_level.shape[:2]
    tx = int(round(u * (W - 1)))
    ty = int(round((1.0 - v) * (H - 1)))
    return mip_level[np.clip(ty, 0, H-1), np.clip(tx, 0, W-1)]


def sample_bilinear(mip_level, u, v):
    """
    Filtrage bilineaire : interpolation ponderee des 4 texels voisins.
    """
    u = u % 1.0; v = v % 1.0
    H, W = mip_level.shape[:2]

    tx = u * (W - 1);       ty = (1.0 - v) * (H - 1)
    x0 = int(tx);           x1 = min(x0 + 1, W - 1)
    y0 = int(ty);           y1 = min(y0 + 1, H - 1)
    fx = tx - x0;           fy = ty - y0

    c00 = mip_level[y0, x0]; c10 = mip_level[y0, x1]
    c01 = mip_level[y1, x0]; c11 = mip_level[y1, x1]

    top    = c00 * (1.0 - fx) + c10 * fx
    bottom = c01 * (1.0 - fx) + c11 * fx
    return top * (1.0 - fy) + bottom * fy


def sample_trilinear(mips, u, v, lod):
    """
    Filtrage trilineaire : bilineaire sur deux niveaux MIP adjacents
    + interpolation lineaire entre les deux.
    """
    max_level = len(mips) - 1
    lod = max(0.0, min(lod, float(max_level)))

    l0   = int(np.floor(lod))
    l1   = min(l0 + 1, max_level)
    frac = lod - l0

    c0 = sample_bilinear(mips[l0], u, v)
    if frac < 1e-6 or l0 == l1:
        return c0

    c1 = sample_bilinear(mips[l1], u, v)
    return c0 * (1.0 - frac) + c1 * frac


def sample_anisotropic(mips, u, v, lod, dudx, dvdx, dudy, dvdy,
                        max_samples=8):
    """
    Filtrage anisotropique simplifie (approximation EWA multi-tap).
    """
    max_level = len(mips) - 1
    tex_size  = max(mips[0].shape[0], mips[0].shape[1])

    mag_x = np.sqrt(dudx**2 + dvdx**2) * tex_size + 1e-10
    mag_y = np.sqrt(dudy**2 + dvdy**2) * tex_size + 1e-10

    lod_min = max(0.0, min(np.log2(min(mag_x, mag_y)), float(max_level)))
    ratio   = max(mag_x, mag_y) / min(mag_x, mag_y)
    n_samp  = int(np.clip(round(ratio), 1, max_samples))

    if mag_x >= mag_y:
        step_u = dudx / (n_samp * tex_size + 1e-10)
        step_v = dvdx / (n_samp * tex_size + 1e-10)
    else:
        step_u = dudy / (n_samp * tex_size + 1e-10)
        step_v = dvdy / (n_samp * tex_size + 1e-10)

    color = np.zeros(3, dtype=np.float32)
    u_s   = u - step_u * (n_samp - 1) / 2.0
    v_s   = v - step_v * (n_samp - 1) / 2.0

    l0 = int(np.floor(lod_min))
    for _ in range(n_samp):
        color += sample_bilinear(mips[l0], u_s, v_s)
        u_s   += step_u
        v_s   += step_v

    return color / n_samp


# ──────────────────────────────────────────────────────────────
#  LOD PRECIS  (formule OpenGL / EXT_texture_lod)
# ──────────────────────────────────────────────────────────────

def compute_lod_accurate(dudx, dvdx, dudy, dvdy, tex_size):
    """
    Calcul precis du LOD selon la formule OpenGL.

    rho = max( ||dUV/dx||, ||dUV/dy|| ) x tex_size
    LOD = log2(rho)

    ou ||dUV/dx|| = sqrt(dudx^2 + dvdx^2)
       ||dUV/dy|| = sqrt(dudy^2 + dvdy^2)

    Plus correct que max(|dU/dx|, |dV/dy|) qui ignore
    les derivees croisees dV/dx et dU/dy.
    """
    rho_x = np.sqrt(dudx**2 + dvdx**2) * tex_size
    rho_y = np.sqrt(dudy**2 + dvdy**2) * tex_size
    return max(0.0, np.log2(max(rho_x, rho_y, 1e-10)))


# ──────────────────────────────────────────────────────────────
#  UTILITAIRE : atlas de visualisation de la pyramide MIP
# ──────────────────────────────────────────────────────────────

def mipmap_atlas(mips):
    """
    Assemble tous les niveaux MIP en une seule image horizontale.
    """
    H_max   = mips[0].shape[0]
    gap     = 2
    W_total = sum(m.shape[1] for m in mips) + gap * (len(mips) - 1)
    atlas   = np.zeros((H_max, W_total, 3), dtype=np.float32)
    x_off   = 0
    for m in mips:
        h, w = m.shape[:2]
        atlas[:h, x_off:x_off + w] = m
        x_off += w + gap
    return atlas
