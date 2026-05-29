#!/usr/bin/env python
"""Convert a helmet mesh (e.g. a Cerca OPM ``.ply``) into an MNE helmet file.

MNE stores the "standard" MEG/OPM helmet surfaces that are drawn by functions
such as :func:`mne.viz.plot_alignment` and :class:`mne.gui.coregistration` in
``mne/data/helmets/<SYSTEM>.fif.gz``. Each file is a BEM-style surface (vertices
``rr`` and triangles ``tris``) tagged with ``FIFFV_MNE_SURF_MEG_HELMET`` and
stored in the device coordinate frame, in meters.

This helper reads an arbitrary mesh file (``.ply``, ``.obj``, ``.stl``, ``.vtk``,
... anything PyVista can open) and writes it back out in that format, optionally
installing it directly into your MNE installation so it becomes a built-in
helmet.

Examples
--------
Write a helmet file next to the source mesh (input mesh is in millimeters)::

    python tools/import_helmet.py cerca_helmet.ply --system Cerca --units mm

Convert and install straight into the active MNE package so it is picked up
automatically (input mesh in meters)::

    python tools/import_helmet.py cerca_helmet.ply --system Cerca --units m \
        --install --overwrite

See the module docstring of ``mne/surface.py`` and ``_get_meg_system`` in
``mne/channels/channels.py`` for how the resulting file is loaded and linked to
acquired data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import mne
from mne._fiff.constants import FIFF

# millimeters per unit, used to bring the mesh into mm before handing it to
# ``_surfaces_to_bem`` (which rescales mm -> m for on-disk storage)
_UNIT_SCALE = {"m": 1000.0, "cm": 10.0, "mm": 1.0}
_COORD_FRAMES = {
    "device": FIFF.FIFFV_COORD_DEVICE,
    "head": FIFF.FIFFV_COORD_HEAD,
    "mri": FIFF.FIFFV_COORD_MRI,
}


def _kabsch(a, b):
    """Rigid transform (no scale) mapping points ``a`` onto points ``b``.

    Returns a 4x4 matrix ``T`` such that ``apply_trans(T, a) ~= b``.
    """
    ca, cb = a.mean(0), b.mean(0)
    h = (a - ca).T @ (b - cb)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    t = cb - r @ ca
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = t
    return out


def _register_points_to_mesh(points, mesh_rr, *, n_iter=50):
    """Rigidly register device-frame points into the mesh's CAD frame via ICP.

    Finds the rigid transform that lays the device-frame ``points`` onto the
    mesh surface. Tries several PCA-based initial orientations (sign
    ambiguities) plus a plain centroid initialization, runs ICP for each, and
    keeps the lowest-residual solution. Returns ``(T_points_to_mesh, rms_mm)``.
    """
    from itertools import product

    from scipy.spatial import cKDTree

    from mne.transforms import apply_trans

    tree = cKDTree(mesh_rr)
    cp, cm = points.mean(0), mesh_rr.mean(0)

    # candidate initial rotations: identity + PCA axis alignments (det +1 only)
    up = np.linalg.svd((points - cp).T @ (points - cp))[0]
    um = np.linalg.svd((mesh_rr - cm).T @ (mesh_rr - cm))[0]
    inits = [np.eye(4)]  # centroid-only (translation) init handled below
    for signs in product((1, -1), repeat=3):
        r = um @ np.diag(signs) @ up.T
        if np.linalg.det(r) > 0:
            t = np.eye(4)
            t[:3, :3] = r
            t[:3, 3] = cm - r @ cp
            inits.append(t)
    inits[0][:3, 3] = cm - cp  # translation-only init

    best_t, best_rms = None, np.inf
    for t in inits:
        for _ in range(n_iter):
            moved = apply_trans(t, points)
            _, idx = tree.query(moved)
            t_new = _kabsch(points, mesh_rr[idx])
            if np.allclose(t_new, t, atol=1e-9):
                t = t_new
                break
            t = t_new
        rms = np.sqrt(np.mean(tree.query(apply_trans(t, points))[0] ** 2))
        if rms < best_rms:
            best_t, best_rms = t, rms
    return best_t, best_rms * 1000.0


def _device_positions(fname):
    """Read device-frame sensor positions from a fif/raw/epochs or JSON file.

    Returns MEG ``loc[:3]`` from a fif/raw/epochs file, or the ``[x, y, z]``
    values from a JSON ch_pos file.
    """
    fname = Path(fname)
    if fname.suffix == ".txt" or fname.suffix == ".json":
        import json

        return np.array(list(json.loads(fname.read_text()).values()), float)
    info = mne.io.read_info(fname)
    picks = mne.pick_types(info, meg=True, ref_meg=False, exclude=())
    pos = np.array([info["chs"][p]["loc"][:3] for p in picks], float)
    return np.unique(pos, axis=0)  # one point per physical sensor


def _read_mesh(fname):
    """Read a mesh file and return (points, triangles).

    Uses PyVista (already an MNE 3D dependency) so that ``.ply`` and many other
    formats are supported. ``points`` are returned in the mesh's native units.
    """
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Reading helmet meshes requires PyVista. Install it with "
            "`pip install pyvista` (it is part of the standard MNE 3D stack)."
        ) from exc

    mesh = pv.read(fname)
    # ensure we only have triangles, then drop the leading "3" of each face
    mesh = mesh.extract_surface().triangulate()
    rr = np.asarray(mesh.points, dtype=float)
    faces = np.asarray(mesh.faces).reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise RuntimeError("Mesh could not be triangulated into triangles.")
    tris = faces[:, 1:].astype(np.int64)
    return rr, tris


def import_helmet(
    mesh_fname,
    *,
    system,
    out_fname=None,
    units="mm",
    coord_frame="device",
    align_to=None,
    install=False,
    overwrite=False,
    verbose=None,
):
    """Convert a helmet mesh into an MNE ``<system>.fif.gz`` helmet file.

    Parameters
    ----------
    mesh_fname : path-like
        The source mesh (e.g. a Cerca OPM ``.ply`` file).
    system : str
        Name of the system. The output file is ``<system>.fif.gz`` and this is
        also the name MNE uses to look the helmet up (see ``_get_meg_system``).
    out_fname : path-like | None
        Where to write the helmet. Defaults to ``<system>.fif.gz`` next to the
        input mesh. Ignored when ``install=True``.
    units : "mm" | "cm" | "m"
        Units of the input mesh's vertex coordinates. Helmet files are stored in
        meters, so the vertices are scaled accordingly.
    align_to : path-like | None
        If given, rigidly register the mesh into the device frame so it matches
        these reference sensor positions, and bake the transform into the saved
        helmet. Accepts a fif/raw/epochs file (uses MEG ``loc[:3]``) or a JSON
        ch_pos file. This is a one-time, subject-independent step: OPM sensors
        sit in fixed helmet slots, so their device-frame positions are the same
        for every recording, and ``get_meg_helmet_surf`` applies the per-subject
        ``dev_head_t`` afterwards. Use this when the mesh is authored in a
        different (CAD/world) frame than the sensors.
    coord_frame : "device" | "head" | "mri"
        Coordinate frame the mesh vertices live in. Built-in helmets are stored
        in the ``"device"`` frame (i.e. relative to the sensor array), which is
        the right choice if your mesh is defined in the same frame as the OPM
        sensor positions. ``get_meg_helmet_surf`` applies ``dev_head_t`` (and
        the head<->MRI ``trans``) at load time.
    install : bool
        If True, write directly into the active MNE installation's
        ``data/helmets`` directory so it becomes a built-in helmet.
    overwrite : bool
        Overwrite an existing output file.
    verbose : bool | str | int | None
        Control verbosity.

    Returns
    -------
    out_fname : pathlib.Path
        Path to the written helmet file.
    """
    if units not in _UNIT_SCALE:
        raise ValueError(f"units must be one of {sorted(_UNIT_SCALE)}, got {units!r}")
    if coord_frame not in _COORD_FRAMES:
        raise ValueError(
            f"coord_frame must be one of {sorted(_COORD_FRAMES)}, got {coord_frame!r}"
        )

    mesh_fname = Path(mesh_fname)
    rr, tris = _read_mesh(mesh_fname)
    mne.utils.logger.info(
        f"Read {len(rr)} vertices and {len(tris)} triangles from {mesh_fname.name}"
    )

    rr = rr * _UNIT_SCALE[units] / 1000.0  # -> meters for registration

    if align_to is not None:
        points = _device_positions(align_to)
        t_pts_to_mesh, rms = _register_points_to_mesh(points, rr)
        # mesh -> device is the inverse of (device points -> mesh) map
        t_mesh_to_dev = np.linalg.inv(t_pts_to_mesh)
        rr = mne.transforms.apply_trans(t_mesh_to_dev, rr)
        mne.utils.logger.info(
            f"Registered mesh to {len(points)} sensor positions from "
            f"{Path(align_to).name} (sensor-to-surface RMS = {rms:.1f} mm)."
        )
        coord_frame = "device"

    # bring vertices into mm; _surfaces_to_bem rescales mm -> m for storage
    surf = dict(rr=rr * 1000.0, tris=tris)
    mne.surface.complete_surface_info(surf, copy=False, do_neighbor_tri=False)
    surf["coord_frame"] = _COORD_FRAMES[coord_frame]

    surfs = mne.bem._surfaces_to_bem(
        [surf],
        ids=[FIFF.FIFFV_MNE_SURF_MEG_HELMET],
        sigmas=[1.0],
        incomplete="ignore",
    )
    del surfs[0]["sigma"]  # a helmet is not a conductor boundary

    if install:
        helmet_dir = Path(mne.__file__).parent / "data" / "helmets"
        out_fname = helmet_dir / f"{system}.fif.gz"
    elif out_fname is None:
        out_fname = mesh_fname.with_name(f"{system}.fif.gz")
    out_fname = Path(out_fname)

    mne.write_bem_surfaces(out_fname, surfs, overwrite=overwrite)
    mne.utils.logger.info(f"Wrote helmet surface to {out_fname}")
    return out_fname


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mesh", help="Input mesh file (e.g. a .ply helmet).")
    parser.add_argument(
        "--system",
        required=True,
        help="System name; output is <system>.fif.gz and MNE looks it up by "
        "this name (e.g. 'Cerca').",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output .fif.gz path (default: <system>.fif.gz next to the mesh).",
    )
    parser.add_argument(
        "--units",
        default="mm",
        choices=sorted(_UNIT_SCALE),
        help="Units of the input mesh vertices (default: mm).",
    )
    parser.add_argument(
        "--coord-frame",
        default="device",
        choices=sorted(_COORD_FRAMES),
        help="Coordinate frame of the mesh (default: device).",
    )
    parser.add_argument(
        "--align-to",
        default=None,
        help="Register the mesh to the sensor positions in this fif/raw/epochs "
        "file (or JSON ch_pos file) and bake the transform in. One-time, "
        "subject-independent.",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install into the active MNE package's data/helmets directory.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite an existing output file."
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the command-line interface."""
    args = _parse_args(argv)
    out = import_helmet(
        args.mesh,
        system=args.system,
        out_fname=args.out,
        units=args.units,
        coord_frame=args.coord_frame,
        align_to=args.align_to,
        install=args.install,
        overwrite=args.overwrite,
        verbose=True,
    )
    print(f"Helmet written to: {out}")


if __name__ == "__main__":
    main()
