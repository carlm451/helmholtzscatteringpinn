"""Hexagonal lattice generator for honeycomb structured scatterer."""

import math


def generate_honeycomb_lattice(a=1.0, r_s=None, d=None):
    """Generate 19 circle centers in hexagonal arrangement fitting within radius a.

    Three concentric rings:
      Ring 0: 1 center at origin
      Ring 1: 6 centers at distance d, angles n*60 deg
      Ring 2: 6 at distance 2d, angles n*60 deg
             + 6 at distance d*sqrt(3), angles n*60 + 30 deg

    Args:
        a: overall cluster radius (all circles fit within this)
        r_s: small circle radius (default: 0.15*a)
        d: lattice constant / nearest-neighbor distance (default: 0.4*a)

    Returns:
        list of (cx, cy, r_s) tuples
    """
    if r_s is None:
        r_s = 0.15 * a
    if d is None:
        d = 0.4 * a

    centers = []

    # Ring 0: origin
    centers.append((0.0, 0.0))

    # Ring 1: 6 neighbors at distance d
    for n in range(6):
        angle = n * math.pi / 3
        centers.append((d * math.cos(angle), d * math.sin(angle)))

    # Ring 2: 6 at distance 2d (same angles as ring 1)
    for n in range(6):
        angle = n * math.pi / 3
        centers.append((2 * d * math.cos(angle), 2 * d * math.sin(angle)))

    # Ring 2: 6 at distance d*sqrt(3), offset by 30 degrees
    r2 = d * math.sqrt(3)
    for n in range(6):
        angle = n * math.pi / 3 + math.pi / 6
        centers.append((r2 * math.cos(angle), r2 * math.sin(angle)))

    assert len(centers) == 19, f"Expected 19 centers, got {len(centers)}"

    # Validate: all circles fit within radius a
    for cx, cy in centers:
        dist_from_origin = math.sqrt(cx**2 + cy**2) + r_s
        assert dist_from_origin <= a + 1e-10, (
            f"Circle at ({cx:.3f}, {cy:.3f}) with r_s={r_s:.3f} "
            f"extends to {dist_from_origin:.3f} > a={a:.3f}"
        )

    # Validate: no overlapping circles
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            cx1, cy1 = centers[i]
            cx2, cy2 = centers[j]
            sep = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            assert sep >= 2 * r_s - 1e-10, (
                f"Circles {i} and {j} overlap: separation={sep:.4f}, "
                f"min required={2*r_s:.4f}"
            )

    return [(cx, cy, r_s) for cx, cy in centers]
