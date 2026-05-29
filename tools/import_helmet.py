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

    # bring vertices into mm; _surfaces_to_bem rescales mm -> m for storage
    surf = dict(rr=rr * _UNIT_SCALE[units], tris=tris)
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
        "--install",
        action="store_true",
        help="Install into the active MNE package's data/helmets directory.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite an existing output file."
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    out = import_helmet(
        args.mesh,
        system=args.system,
        out_fname=args.out,
        units=args.units,
        coord_frame=args.coord_frame,
        install=args.install,
        overwrite=args.overwrite,
        verbose=True,
    )
    print(f"Helmet written to: {out}")


if __name__ == "__main__":
    main()
