# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.io import loadmat

import mne
from mne._fiff.pick import pick_channels, pick_info, pick_types
from mne._fiff.proj import make_projector
from mne.datasets import testing
from mne.io import RawArray, read_info, read_raw_fil
from mne.io.constants import FIFF
from mne.preprocessing import apply_amm, compute_proj_amm
from mne.preprocessing.hfc import (
    _SPHEROID_ROTATIONS,
    _amm_spaces,
    _prolate_angular,
    _prolate_coords,
    _prolate_fields,
    _prolate_radial,
    compute_proj_hfc,
)
from mne.transforms import Transform

fil_path = testing.data_path(download=False) / "FIL"
fname_root = "sub-noise_ses-001_task-noise220622_run-001"

io_dir = Path(__file__).parents[2] / "io"
ctf_fname = io_dir / "tests" / "data" / "test_ctf_raw.fif"
fif_fname = io_dir / "tests" / "data" / "test_raw.fif"

# The below channels in the test data do not have positions, set to bad
bads = ["G2-DS-Y", "G2-DS-Z", "G2-DT-Y", "G2-DT-Z", "G2-MW-Y", "G2-MW-Z"]

# TODO: Ignore this warning in all these tests until we deal with this properly
pytestmark = pytest.mark.filterwarnings(
    "ignore:No fiducials.*problems later!:RuntimeWarning",
)


def _unpack_mat(matin):
    """Extract relevant entries from unstructred readmat."""
    data = matin["data"]
    grad = data[0][0]["grad"]
    label = list()
    coil_label = list()
    for ii in range(len(data[0][0]["label"])):
        label.append(str(data[0][0]["label"][ii][0][0]))
    for ii in range(len(grad[0][0]["label"])):
        coil_label.append(str(grad[0][0]["label"][ii][0][0]))

    matout = {
        "label": label,
        "trial": data["trial"][0][0][0][0],
        "coil_label": coil_label,
        "coil_pos": grad[0][0]["coilpos"],
        "coil_ori": grad[0][0]["coilori"],
    }
    return matout


def _angle_between_each(A):
    """Measure the angle between each row vector in a matrix."""
    assert A.ndim == 2
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    d = (A @ A.T).ravel()
    np.clip(d, -1, 1, out=d)
    ang = np.abs(np.arccos(d))
    return ang


@testing.requires_testing_data
@pytest.mark.parametrize("order", [1, 2, 3])
def test_correction(order):
    """Apply HFC and compare to previous computed solutions."""
    binname = fil_path / "sub-noise_ses-001_task-noise220622_run-001_meg.bin"
    raw = read_raw_fil(binname)
    raw.load_data()
    raw.info["bads"].extend([b for b in bads])
    projs = compute_proj_hfc(raw.info, order=order, accuracy="point")
    raw.add_proj(projs).apply_proj()

    mat = _unpack_mat(loadmat(fil_path / f"{fname_root}_hfc_l{order}.mat"))

    proj_list = projs[0]["data"]["col_names"]
    picks = pick_channels(raw.ch_names, proj_list, ordered=True)
    mat_list = mat["coil_label"]
    mat_inds = pick_channels(mat_list, proj_list, ordered=True)

    want = mat["trial"][mat_inds]
    got = raw.copy().add_proj(projs).apply_proj()[picks, 0:300][0] * 1e15
    assert_allclose(got, want, rtol=1e-7)

    # Now with default accuracy: not super close with tol but corr is good
    projs = compute_proj_hfc(raw.info, order=order)
    got = raw.copy().add_proj(projs).apply_proj()[picks, 0:300][0] * 1e15
    corr = np.corrcoef(got.ravel(), want.ravel())[0, 1]
    assert 0.999999 < corr <= 1.0


@testing.requires_testing_data
def test_l1_basis_orientations():
    """Test that angles between the basis components matches orientations."""
    binname = fil_path / "sub-noise_ses-001_task-noise220622_run-001_meg.bin"
    raw = read_raw_fil(binname)
    raw.info["bads"].extend([b for b in bads])
    projs = compute_proj_hfc(raw.info, accuracy="point")
    basis = np.hstack([p["data"]["data"].T for p in projs])
    picks = pick_types(raw.info, meg="mag")
    assert len(picks) == 68
    assert basis.shape == (len(picks), 3)
    ang_model = _angle_between_each(basis)
    n_ang = len(picks) ** 2
    assert ang_model.shape == (n_ang,)

    chs = pick_info(raw.info, picks)["chs"]
    ori_sens = np.array([ch["loc"][-3:] for ch in chs])
    # match the normalization that our projectors get
    ori_sens /= np.linalg.norm(ori_sens, axis=0, keepdims=True)
    assert ori_sens.shape == (len(picks), 3)
    ang_sens = _angle_between_each(ori_sens)
    assert ang_sens.shape == (n_ang,)

    assert_allclose(ang_sens, ang_model, atol=1e-7)


def test_ref_degenerate():
    """Test reference channel handling and degenerate conditions."""
    info = read_info(ctf_fname)
    # exclude ref by default
    projs = compute_proj_hfc(info)
    meg_names = [
        info["ch_names"][pick]
        for pick in pick_types(info, meg=True, ref_meg=False, exclude=[])
    ]
    assert len(projs) == 3
    assert projs[0]["desc"] == "HFC: l=1 m=-1"
    assert projs[1]["desc"] == "HFC: l=1 m=0"
    assert projs[2]["desc"] == "HFC: l=1 m=1"
    assert projs[0]["data"]["col_names"] == meg_names
    meg_ref_names = [
        info["ch_names"][pick]
        for pick in pick_types(info, meg=True, ref_meg=True, exclude=[])
    ]
    projs = compute_proj_hfc(info, picks=("meg", "ref_meg"))
    assert projs[0]["data"]["col_names"] == meg_ref_names

    # Degenerate
    info = read_info(fif_fname)
    compute_proj_hfc(info)  # smoke test
    with pytest.raises(ValueError, match="Only.*could be interpreted as MEG"):
        compute_proj_hfc(info, picks=[0, 330])  # one MEG, one EEG
    info["chs"][0]["loc"][:] = np.nan  # first MEG proj
    with pytest.raises(ValueError, match="non-finite projectors"):
        compute_proj_hfc(info)
    info_eeg = pick_info(info, pick_types(info, meg=False, eeg=True))
    with pytest.raises(ValueError, match=r"picks \(\'meg\'\) could not be"):
        compute_proj_hfc(info_eeg)


HEAD_CENTER = np.array([0.0, 0.01, 0.04])


def _opm_info(rotation=np.eye(3)):
    """Triaxial point magnetometers 15 mm outside the top of a head-like ellipsoid."""
    radii = np.array([0.075, 0.095, 0.085])
    k = np.arange(160) + 0.5  # Fibonacci points on the sphere
    z = 1 - 2 * k / len(k)
    phi = np.pi * (1 + 5**0.5) * k
    unit = np.column_stack(
        [np.sqrt(1 - z**2) * np.cos(phi), np.sqrt(1 - z**2) * np.sin(phi), z]
    )
    unit = unit[unit[:, 2] > -0.25][::2][:40]
    normal = unit / radii
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    pos = unit * radii + 0.015 * normal + HEAD_CENTER
    tangent = np.cross(normal, [0.3, 0.2, 1.0])
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    axes = np.stack([normal, tangent, np.cross(normal, tangent)], axis=1)
    info = mne.create_info([f"OPM{ii:03d}" for ii in range(120)], 200.0, "mag")
    for ii, ch in enumerate(info["chs"]):
        ez = axes[ii // 3, ii % 3]
        ex = np.cross(ez, [0.3, 0.2, 1.0])
        ex /= np.linalg.norm(ex)
        loc = np.r_[pos[ii // 3], ex, np.cross(ez, ex), ez].reshape(4, 3)
        ch["loc"][:] = (loc @ rotation.T).ravel()
        ch["coil_type"] = FIFF.FIFFV_COIL_POINT_MAGNETOMETER
    with info._unlock():
        info["dev_head_t"] = Transform("meg", "head")
    return info


def _dipole_field(info, r0, moment):
    """Field of a magnetic dipole along each channel's orientation (arbitrary units)."""
    pos = np.array([ch["loc"][:3] for ch in info["chs"]]) - r0
    ori = np.array([ch["loc"][9:12] for ch in info["chs"]])
    dist = np.linalg.norm(pos, axis=1, keepdims=True)
    b = 3 * pos * (pos @ moment)[:, np.newaxis] / dist**5 - moment / dist**3
    return np.sum(b * ori, axis=1)


@pytest.mark.parametrize("kind", ["internal", "external"])
@pytest.mark.parametrize("b", [0.08, 0.1])
def test_amm_basis(kind, b):
    """Each AMM basis vector is the field of a harmonic potential."""
    rng = np.random.default_rng(0)
    a, c = 0.1, np.sqrt(0.1**2 - b**2)
    v = rng.normal(size=(20, 3))
    v *= 0.13 / np.linalg.norm(v, axis=1, keepdims=True)
    n = rng.normal(size=(20, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)

    def potential(x):
        mu, nu, phi = _prolate_coords(x, c)
        return np.column_stack(
            [
                _prolate_radial(kind, mu, a, c, degree, abs(order))[0]
                * _prolate_angular(nu, phi, degree, order)[0]
                for degree in range(1, 5)
                for order in range(-degree, degree + 1)
            ]
        )

    h, grad, lap = 1e-5, 0, 0
    for step in np.eye(3) * 1e-5:
        plus, minus = potential(v + step), potential(v - step)
        grad = grad + (plus - minus) / (2 * h) * (n @ step / h)[:, np.newaxis]
        lap = lap + (plus - 2 * potential(v) + minus) / h**2
    fields = _prolate_fields(v, n, a, b, 4, kind)
    assert_allclose(grad, fields, atol=1e-6 * np.abs(fields).max())
    assert np.abs(lap).max() < 1e-5 * np.abs(fields).max() / h  # Laplace

    # The coordinate poles have finite transverse gradients, including the
    # uniform external fields. Check the limit from either side of the axis.
    v = np.tile([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]], (3, 1))
    n = np.repeat(np.eye(3), 2, axis=0)
    grad = 0
    for step in np.eye(3) * h:
        grad = (
            grad
            + (potential(v + step) - potential(v - step))
            / (2 * h)
            * (n @ step / h)[:, np.newaxis]
        )
    fields = _prolate_fields(v, n, a, b, 4, kind)
    assert_allclose(fields, grad, atol=1e-6 * np.abs(grad).max())
    if kind == "external":
        for offset in (0.0, -1e-10, 1e-10):
            v[:, 0] = offset
            fields = _prolate_fields(v, n, a, b, 1, kind)
            assert_allclose(
                fields, np.c_[-n[:, 2] / b, n[:, 1] / a, -n[:, 0] / b], atol=1e-12
            )


@pytest.mark.parametrize("kind", ["internal", "external"])
@pytest.mark.parametrize("c", [0.0, 0.06, 0.09])
def test_amm_radial(kind, c):
    """Radial derivatives include every term, up to the default internal order."""
    mu, a, step = np.array([0.101, 0.12, 0.2]), 0.1, 1e-7
    for degree in range(1, 10):
        for order in range(degree + 1):
            _, derivative = _prolate_radial(kind, mu, a, c, degree, order)
            plus = _prolate_radial(kind, mu + step, a, c, degree, order)[0]
            minus = _prolate_radial(kind, mu - step, a, c, degree, order)[0]
            assert_allclose(
                derivative, (plus - minus) / (2 * step), rtol=1e-7, atol=1e-8
            )


def test_compute_proj_amm():
    """Test AMM projectors."""
    info = _opm_info()
    projs = compute_proj_amm(info, accuracy="point")
    assert len(projs) == info["nchan"] - 99  # 99 internal harmonics of order 9
    assert projs[0]["desc"] == "AMM: 1"
    proj = make_projector(projs, info["ch_names"])[0]
    # uniform fields and linear gradients are external harmonics of order <= 2
    pos = np.array([ch["loc"][:3] for ch in info["chs"]])
    ori = np.array([ch["loc"][9:12] for ch in info["chs"]])
    gradient = pos[:, [0]] * ori[:, [2]] + pos[:, [2]] * ori[:, [0]]
    assert_allclose(proj @ np.c_[ori, gradient], 0, atol=1e-10)
    # current dipoles in a spherical head are kept
    rng = np.random.default_rng(0)
    d = rng.normal(size=(30, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    rr = HEAD_CENTER + d * rng.uniform(0.02, 0.06, (30, 1)) * [0.8, 1.0, 0.9]
    src = mne.setup_volume_source_space(pos=dict(rr=rr, nn=d))
    sphere = mne.make_sphere_model(r0=HEAD_CENTER, head_radius=0.09)
    gain = mne.make_forward_solution(info, None, src, sphere)["sol"]["data"]
    gain = gain[:, np.linalg.norm(gain, axis=0) > 1e-3 * np.abs(gain).max()]

    def kept(proj):
        return np.linalg.norm(proj @ gain, axis=0) / np.linalg.norm(gain, axis=0)

    assert np.median(kept(proj)) > 0.97 and kept(proj).min() > 0.9
    # SPM's centring removes the channel mean too, and more of the brain signal
    centered = compute_proj_amm(info, accuracy="point", center_basis=True)
    proj_c = make_projector(centered, info["ch_names"])[0]
    assert_allclose(proj_c.sum(axis=1), 0, atol=1e-10)
    assert np.median(kept(proj_c)) < np.median(kept(proj)) - 0.02
    # the spheroid's long axis can be any axis of the frame
    info_x = _opm_info(rotation=_SPHEROID_ROTATIONS[0].T)
    proj_x = make_projector(
        compute_proj_amm(info_x, accuracy="point"), info["ch_names"]
    )
    assert_allclose(proj_x[0], proj, atol=1e-10)
    meg = compute_proj_amm(info, coord_frame="meg", accuracy="point")
    assert_allclose(make_projector(meg, info["ch_names"])[0], proj, atol=1e-12)
    # Changing a channel's polarity cannot change the physical projection.
    signs = rng.choice([-1, 1], info["nchan"])
    flipped = info.copy()
    for ch, sign in zip(flipped["chs"], signs):
        ch["loc"][6:] *= sign  # flip ey and ez, preserving a right-handed frame
    flipped = make_projector(compute_proj_amm(flipped), info["ch_names"])[0]
    assert_allclose(signs[:, None] * flipped * signs, proj, atol=1e-10)
    with info._unlock():
        info["dev_head_t"] = None
    with pytest.raises(ValueError, match='coord_frame="head" needs'):
        compute_proj_amm(info)


@pytest.mark.parametrize("n_channels", [36, 72, 120])
def test_amm_rank(n_channels):
    """Rank is bounded by the external complement; empty residuals are valid."""
    info = pick_info(_opm_info(), np.arange(n_channels))
    _, u_in, u_out, u_res = _amm_spaces(
        info, 9, 2, "meg", "bads", "head", False, "point"
    )
    assert u_in.shape[1] == min(99, n_channels - 8)
    assert u_res.shape[1] == max(n_channels - 107, 0)
    basis = np.c_[u_in, u_out, u_res]
    assert_allclose(basis.T @ basis, np.eye(n_channels), atol=1e-12)
    data = np.random.default_rng(0).normal(size=(n_channels, 400))
    raw = RawArray(data, info)
    spatial = apply_amm(raw, accuracy="point")
    temporal = apply_amm(raw, accuracy="point", st_duration=1.0)
    if u_res.shape[1] == 0:
        assert_allclose(temporal.get_data(), spatial.get_data(), atol=1e-12)
    with pytest.raises(ValueError, match="external harmonics.*all.*channels"):
        compute_proj_amm(info, ext_order=12)


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_amm_existing_proj(active, mixed):
    """Temporal reconstruction honors existing active and inactive SSPs."""
    info, u_in, u_out, _ = _amm_spaces(
        _opm_info(), 9, 2, "meg", "bads", "head", False, "point"
    )
    vec = u_in[:, 0] + mixed * u_out[:, 0]
    proj = mne.Projection(
        active=False,
        desc="existing SSP",
        data=dict(
            col_names=info["ch_names"],
            row_names=None,
            data=vec[None],
            ncol=len(vec),
            nrow=1,
        ),
    )
    raw = RawArray(np.random.default_rng(0).normal(size=(120, 800)), info).add_proj(
        proj
    )
    if active:
        raw.apply_proj()
    spatial = apply_amm(raw, accuracy="point")
    temporal = apply_amm(raw, accuracy="point", st_duration=2.0)
    combined = make_projector(temporal.info["projs"], temporal.ch_names)[0]
    assert all(p["active"] for p in temporal.info["projs"])
    assert_allclose(combined @ temporal.get_data(), temporal.get_data(), atol=1e-12)
    assert_allclose(temporal.get_data(), spatial.get_data(), atol=1e-12)


def test_amm_annotations():
    """Temporal fitting skips BAD/EDGE spans and never crosses a boundary."""
    info = _opm_info()
    rng = np.random.default_rng(0)
    raw = RawArray(rng.normal(size=(120, 300)), info, first_samp=10000)
    raw.set_annotations(
        mne.Annotations([0.4, 0.9], [0.2, 0.1], ["BAD_artifact", "EDGE_motion"])
    )
    raw = mne.concatenate_raws([raw, RawArray(rng.normal(size=(120, 300)), info)])
    expected = apply_amm(raw, accuracy="point").get_data()
    spatial = expected.copy()
    for start, stop in [(0, 80), (120, 180), (200, 300), (300, 600)]:
        segment = raw.copy().crop(start / 200.0, (stop - 1) / 200.0)
        expected[:, start:stop] = apply_amm(
            segment, accuracy="point", st_duration=10.0
        ).get_data()
    actual = apply_amm(raw, accuracy="point", st_duration=10.0).get_data()
    assert np.linalg.norm(expected - spatial) > 1.0
    assert_allclose(actual, expected, atol=1e-12)
    for start, stop in [(80, 120), (180, 200)]:
        assert_allclose(actual[:, start:stop], spatial[:, start:stop], atol=1e-12)


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"int_order": value}, ValueError, "int_order must be at least 1")
        for value in (0, -1)
    ]
    + [
        ({"ext_order": value}, ValueError, "ext_order must be at least 1")
        for value in (0, -1)
    ]
    + [
        ({"st_duration": value}, ValueError, "st_duration must be finite and positive")
        for value in (0, -1, np.nan, np.inf)
    ]
    + [
        ({"st_correlation": value}, ValueError, "st_correlation must satisfy")
        for value in (0, -1, 1.5, np.nan, np.inf)
    ]
    + [
        ({"st_duration": "1"}, TypeError, "st_duration"),
        ({"st_correlation": "0.98"}, TypeError, "st_correlation"),
        ({"center_basis": "false"}, TypeError, "center_basis"),
    ],
)
def test_amm_validation(kwargs, error, match):
    """Reject invalid options before filtering, including when temporal is off."""
    raw = RawArray(np.zeros((120, 20)), _opm_info())
    with pytest.raises(error, match=match):
        apply_amm(raw, copy=False, **kwargs)
    assert not raw.info["projs"]
    assert not raw.get_data().any()


def test_apply_amm():
    """Test applying AMM and its temporal extension."""
    info = _opm_info()
    t = np.arange(400) / info["sfreq"]
    brain = _dipole_field(info, HEAD_CENTER + [0.03, 0.02, 0.03], np.array([1, 0, 0]))
    near = _dipole_field(info, HEAD_CENTER + [0, 0.14, 0], np.array([0, 0, 1]))
    near *= 50 * np.abs(brain).max() / np.abs(near).max()  # too close for ext_order=2
    s_brain, s_near = np.sin(2 * np.pi * 11 * t), np.sin(2 * np.pi * 7 * t)
    noise = np.random.default_rng(0).normal(
        scale=np.abs(brain).max() / 3, size=(120, 400)
    )
    raw = RawArray(np.outer(brain, s_brain) + np.outer(near, s_near) + noise, info)
    proj = make_projector(compute_proj_amm(info, accuracy="point"), info["ch_names"])[0]

    def amplitude(out, topo, s):
        """Amplitude of P topo at the time course s in out, relative to P topo."""
        ref = proj @ topo
        return (out.get_data() @ s / (s @ s)) @ ref / (ref @ ref)

    out = apply_amm(raw, accuracy="point")
    assert not raw.info["projs"]
    assert all(p["active"] for p in out.info["projs"])
    assert_allclose(
        out.get_data(), proj @ raw.get_data(), atol=1e-10 * np.abs(near).max()
    )
    assert amplitude(out, near, s_near) > 0.95
    # the temporal extension removes the near source and keeps the brain signal
    out = apply_amm(raw, accuracy="point", st_duration=1.0)
    assert abs(amplitude(out, near, s_near)) < 0.1
    assert 0.8 < amplitude(out, brain, s_brain) < 1.2
    assert_allclose(
        proj @ out.get_data(), out.get_data(), atol=1e-10 * np.abs(near).max()
    )
