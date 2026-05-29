#!/usr/bin/env python
r"""Build a ``<system>_ch_pos.txt`` file from a helmet-configuration TSV.

MNE can deform a built-in helmet mesh so that it conforms to the sensor layout
of an actual recording. This is done by :func:`mne.surface._scale_helmet_to_sensors`,
which looks for a file ``mne/data/helmets/<system>_ch_pos.txt`` next to the
``<system>.fif.gz`` helmet. That file is a JSON dictionary mapping a *channel
name prefix* to the *nominal* (CAD/design) position of that sensor, in meters,
in the same coordinate frame as the helmet mesh::

    {"F1 ": [-0.023264, 0.0796, 0.03634], "F3 ": [-0.042006, 0.082033, 0.0034043], ...}

At plot time MNE matches every key against the recording's channel names with
``str.startswith``, reads the *actual* sensor positions from each channel's
``info["chs"][i]["loc"][:3]``, and warps the mesh (affine + scale, then a
nonlinear displacement field) from the nominal positions onto the actual ones.

This helper turns a Cerca-style helmet config TSV (columns
``Name Px Py Pz Ox Oy Oz Layx Layy Sensor``) into that JSON file.

Examples
--------
Key the entries by the helmet-slot ``Name`` column and install next to the
helmet so it activates automatically::

    python tools/helmet_ch_pos_from_tsv.py helmet_config.tsv --system Cerca \
        --name-column Name --install --overwrite

Key by the physical ``Sensor`` column instead (use whichever column matches the
channel names in your raw recording)::

    python tools/helmet_ch_pos_from_tsv.py helmet_config.tsv --system Cerca \
        --name-column Sensor --out Cerca_ch_pos.txt
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

_AXES = {"X", "Y", "Z"}


def _split_axis(name):
    """Return (key, axis) splitting a trailing X/Y/Z token off ``name``.

    The trailing separator is kept on the key (e.g. ``"F1 X" -> ("F1 ", "X")``)
    so that the key is matched against full channel names via ``startswith``
    without colliding with similarly-named sensors (``"F1 "`` will not match
    ``"F10 X"``).
    """
    parts = name.rsplit(maxsplit=1)
    if len(parts) == 2 and parts[1].upper() in _AXES:
        axis = parts[1]
        return name[: len(name) - len(axis)], axis
    return name, None


def ch_pos_from_helmet_tsv(
    tsv_fname,
    *,
    system,
    out_fname=None,
    name_column="Name",
    drop_axis=True,
    strip_keys=False,
    install=False,
    overwrite=False,
):
    """Convert a helmet-configuration TSV into a ``<system>_ch_pos.txt`` file.

    Parameters
    ----------
    tsv_fname : path-like
        The helmet config TSV with at least the columns ``<name_column>``,
        ``Px``, ``Py``, ``Pz``.
    system : str
        System name. The output file is ``<system>_ch_pos.txt`` and must sit
        next to ``<system>.fif.gz`` for MNE to use it.
    out_fname : path-like | None
        Output path. Defaults to ``<system>_ch_pos.txt`` next to the TSV.
        Ignored when ``install=True``.
    name_column : str
        Which column to derive the channel-name prefixes from. Use whichever
        column matches the channel names in your recording (commonly the
        helmet-slot ``"Name"`` or the physical ``"Sensor"`` column).
    drop_axis : bool
        If True (default), strip a trailing ``X``/``Y``/``Z`` axis token so that
        each physical sensor location yields a single reference point (the three
        axis channels of one sensor share a position). This is required: the
        deformation interpolator fails on duplicate reference positions.
    strip_keys : bool
        If True, ``.strip()`` the keys (drops the trailing separator kept by
        ``drop_axis``). Only enable this if your channel names have no separator
        before the axis (e.g. ``"F1X"``). Risks prefix collisions otherwise.
    install : bool
        If True, write into the active MNE package's ``data/helmets`` directory.
    overwrite : bool
        Overwrite an existing output file.

    Returns
    -------
    out_fname : pathlib.Path
        Path to the written JSON file.
    """
    tsv_fname = Path(tsv_fname)
    positions = {}  # key -> list of [x, y, z]
    with open(tsv_fname, newline="") as fid:
        reader = csv.DictReader(fid, delimiter="\t")
        for col in (name_column, "Px", "Py", "Pz"):
            if col not in reader.fieldnames:
                raise ValueError(
                    f"Column {col!r} not found in {tsv_fname.name}. "
                    f"Available columns: {reader.fieldnames}"
                )
        for row in reader:
            name = row[name_column].strip()
            key = _split_axis(name)[0] if drop_axis else name
            if strip_keys:
                key = key.strip()
            pos = [float(row["Px"]), float(row["Py"]), float(row["Pz"])]
            positions.setdefault(key, []).append(pos)

    # collapse each key to a single position (axis channels share a location)
    ch_pos = {}
    for key, poss in positions.items():
        poss = np.array(poss, float)
        spread = float(np.ptp(poss, axis=0).max())
        if spread > 1e-4:  # 0.1 mm
            print(
                f"  warning: positions for key {key!r} span {spread * 1000:.1f} mm; "
                "using the mean (is --name-column correct?)"
            )
        ch_pos[key] = np.round(poss.mean(axis=0), 6).tolist()

    if len(ch_pos) < 4:
        raise ValueError(
            f"Only {len(ch_pos)} reference points produced; MNE needs at least 4 "
            "to deform the helmet. Check --name-column / --drop-axis."
        )

    # warn about prefix collisions (str.startswith matching is used by MNE)
    keys = sorted(ch_pos)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if b.startswith(a):
                print(
                    f"  warning: key {a!r} is a prefix of {b!r}; both will match "
                    f"channels starting with {a!r}. Consider --no-strip-keys or a "
                    "more specific name column."
                )

    if install:
        import mne

        out_fname = (
            Path(mne.__file__).parent / "data" / "helmets" / f"{system}_ch_pos.txt"
        )
    elif out_fname is None:
        out_fname = tsv_fname.with_name(f"{system}_ch_pos.txt")
    out_fname = Path(out_fname)
    if out_fname.is_file() and not overwrite:
        raise FileExistsError(f"{out_fname} exists; pass overwrite=True to replace it.")

    with open(out_fname, "w") as fid:
        json.dump(ch_pos, fid, indent=0)
    print(f"Wrote {len(ch_pos)} reference positions to {out_fname}")
    return out_fname


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("tsv", help="Helmet config TSV file.")
    parser.add_argument("--system", required=True, help="System name, e.g. 'Cerca'.")
    parser.add_argument("--out", default=None, help="Output JSON path.")
    parser.add_argument(
        "--name-column",
        default="Name",
        help="Column to key on; must match your raw channel names (default: Name).",
    )
    parser.add_argument(
        "--no-drop-axis",
        dest="drop_axis",
        action="store_false",
        help="Keep the X/Y/Z axis token in the keys (not recommended).",
    )
    parser.add_argument(
        "--strip-keys",
        action="store_true",
        help="Strip trailing whitespace from keys (only if channel names have no "
        "separator before the axis).",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install into the active MNE package's data/helmets directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    """Run the command-line interface."""
    args = _parse_args(argv)
    out = ch_pos_from_helmet_tsv(
        args.tsv,
        system=args.system,
        out_fname=args.out,
        name_column=args.name_column,
        drop_axis=args.drop_axis,
        strip_keys=args.strip_keys,
        install=args.install,
        overwrite=args.overwrite,
    )
    print(f"ch_pos written to: {out}")


if __name__ == "__main__":
    main()
