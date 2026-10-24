# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
from scipy import linalg
from scipy.special import eval_gegenbauer, gammaln, hyp2f1

from .._fiff.pick import _picks_to_idx, pick_info
from .._fiff.proj import Projection, make_projector
from ..annotations import _annotations_starts_stops
from ..io import BaseRaw
from ..transforms import apply_trans
from ..utils import (
    _check_option,
    _check_preload,
    _ensure_int,
    _validate_type,
    logger,
    verbose_static,
)
from .maxwell import _orth_overwrite, _overlap_projector, _prep_mf_coils, _sss_basis


@verbose_static("info", "projs")
def compute_proj_hfc(
    info, order=1, picks="meg", exclude="bads", *, accuracy="accurate", verbose=None
):
    """Generate projectors to perform homogeneous/harmonic correction to data.

    Remove environmental fields from magnetometer data by assuming it is
    explained as a homogeneous :footcite:`TierneyEtAl2021` or harmonic field
    :footcite:`TierneyEtAl2022`. Useful for arrays of OPMs.

    Parameters
    ----------
    info : mne.Info | None
        The :class:`mne.Info` object with information about the
        sensors and methods of measurement.
    order : int
        The order of the spherical harmonic basis set to use. Set to 1 to use
        only the homogeneous field component (default), 2 to add gradients, 3
        to add quadrature terms, etc.
    picks : str | array_like | slice | None
        Channels to include. Default of ``'meg'`` (same as None) will select
        all non-reference MEG channels. Use ``('meg', 'ref_meg')`` to include
        reference sensors as well.
    exclude : list | 'bads'
        List of channels to exclude from HFC, only used when picking
        based on types (e.g., exclude="bads" when picks="meg").
        Specify ``'bads'`` (the default) to exclude all channels marked as bad.
    accuracy : str
        Can be ``"point"``, ``"normal"`` or ``"accurate"`` (default), defines
        which level of coil definition accuracy is used to generate model.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    projs : list of Projection
        List of computed projection vectors.

    See Also
    --------
    mne.io.Raw.add_proj
    mne.io.Raw.apply_proj

    Notes
    -----
    To apply the projectors to a dataset, use
    ``inst.add_proj(projs).apply_proj()``.

    .. versionadded:: 1.4

    References
    ----------
    .. footbibliography::
    """
    picks = _picks_to_idx(info, picks, none="meg", exclude=exclude, with_ref_meg=False)
    info = pick_info(info, picks)
    del picks
    exp = dict(origin=(0.0, 0.0, 0.0), int_order=0, ext_order=order)
    coils = _prep_mf_coils(info, ignore_ref=False, accuracy=accuracy)
    n_chs = len(coils[5])
    if n_chs != info["nchan"]:
        raise ValueError(
            f"Only {n_chs}/{info['nchan']} picks could be interpreted as MEG channels."
        )
    S = _sss_basis(exp, coils)
    del coils
    bad_chans = [
        info["ch_names"][pick] for pick in np.where((~np.isfinite(S)).any(axis=1))[0]
    ]
    if bad_chans:
        raise ValueError(
            "The following channel(s) generate non-finite projectors:\n"
            f"    {bad_chans}\nPlease exclude from picks!"
        )
    S /= np.linalg.norm(S, axis=0)
    labels = _label_basis(order)
    assert len(labels) == S.shape[1]
    projs = []
    for label, vec in zip(labels, S.T):
        proj_data = dict(
            col_names=info["ch_names"],
            row_names=None,
            data=vec[np.newaxis, :],
            ncol=info["nchan"],
            nrow=1,
        )
        projs.append(Projection(active=False, data=proj_data, desc=label))
    return projs


def _label_basis(order):
    """Give basis vectors names for Projection() class."""
    return [
        f"HFC: l={L} m={m}"
        for L in np.arange(1, order + 1)
        for m in np.arange(-1 * L, L + 1)
    ]


@verbose_static("info", "projs")
def compute_proj_amm(
    info,
    int_order=9,
    ext_order=2,
    picks="meg",
    exclude="bads",
    *,
    coord_frame="head",
    center_basis=False,
    accuracy="accurate",
    verbose=None,
):
    """Generate projectors for adaptive multipole models (AMM).

    Model the field measured by an array of OPMs as prolate spheroidal
    harmonics defined on a spheroid fitted to the array
    :footcite:`TierneyEtAl2024`: internal harmonics for sources inside the
    spheroid (the brain) and external harmonics for sources outside the array
    (interference). The projectors keep the part of the data in the internal
    space once the external space has been removed (an orthogonal projection)
    and remove everything else.

    Parameters
    ----------
    info : mne.Info | None
        The :class:`mne.Info` object with information about the
        sensors and methods of measurement.
    int_order : int
        Positive order of the internal harmonics (``int_order * (int_order + 2)``
        harmonics; 99 for the default of 9).
    ext_order : int
        Positive order of the external harmonics. An order that spans all
        selected channels leaves no internal space and raises an error.
    picks : str | array_like | slice | None
        Channels to include. Default of ``'meg'`` (same as None) will select
        all non-reference MEG channels.
    exclude : list | 'bads'
        List of channels to exclude, only used when picking based on types
        (e.g., exclude="bads" when picks="meg"). Specify ``'bads'`` (the
        default) to exclude all channels marked as bad.
    coord_frame : str
        The coordinate frame the spheroid is fitted in, ``'head'`` (default)
        or ``'meg'``; its axes are aligned with the axes of that frame. For
        empty-room recordings that do not have a head<->meg transform
        ``info['dev_head_t']``, the MEG coordinate frame should be used.
    center_basis : bool
        If True, remove the mean across channels from every basis vector
        before building the projectors, as SPM's ``spm_opm_amm`` does; the
        projectors then also remove the mean across channels. The default
        (False) follows the equations of :footcite:`TierneyEtAl2024`.
    accuracy : str
        Can be ``"point"``, ``"normal"`` or ``"accurate"`` (default), defines
        which level of coil definition accuracy is used to generate model.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    projs : list of Projection
        List of computed projection vectors.

    See Also
    --------
    apply_amm
    compute_proj_hfc

    Notes
    -----
    The steps follow SPM's ``spm_opm_amm`` by Tim Tierney. An axis-aligned
    spheroid is fitted to the channel positions with each axis in turn as the
    long axis (the equal-radii case of the least-squares parameterisation of
    Yury Petrov's ``ellipsoid_fit``); the prolate fit with the smallest
    spread is kept, and its radii are shrunk in steps of 0.5 % of the largest
    one until every channel lies outside. The basis vectors are the field of
    each harmonic potential, integrated over each coil: the radial parts are
    associated Legendre functions of the second kind (internal) and of the
    first kind (external), the angular parts real spherical harmonics.

    This implementation was written from the mathematics and checked against
    SPM's. The radial derivative of the internal harmonics includes the
    derivative of the hypergeometric factor of :math:`Q_l^m`, a term
    ``spm_ipharm.m`` leaves out. Unlike SPM, the bases are not centered across
    channels by default: centering changes the physical field spaces and
    imposes an additional rejection of the equal-channel vector.

    To apply the projectors to a dataset, use
    ``inst.add_proj(projs).apply_proj()``, or :func:`apply_amm`, which can
    also apply the temporal extension.

    .. versionadded:: 1.14

    References
    ----------
    .. footbibliography::
    """
    info, _, u_out, u_res = _amm_spaces(
        info, int_order, ext_order, picks, exclude, coord_frame, center_basis, accuracy
    )
    return _amm_projs(info, np.concatenate([u_out, u_res], axis=1))


@verbose_static()
def apply_amm(
    raw,
    int_order=9,
    ext_order=2,
    *,
    st_duration=None,
    st_correlation=0.98,
    coord_frame="head",
    center_basis=False,
    picks="meg",
    exclude="bads",
    accuracy="accurate",
    copy=True,
    verbose=None,
):
    """Apply adaptive multipole models (AMM) to raw data.

    Add and apply the projectors of :func:`compute_proj_amm` and, if
    ``st_duration`` is given, the temporal extension of
    :footcite:`TierneyEtAl2024`: in each window, the time courses of the
    internal space whose canonical correlation with those of the
    intermediate space (what neither the internal nor the external
    harmonics explain) reaches ``st_correlation`` are projected out, as in
    tSSS :footcite:`TauluSimola2006`. The temporal extension removes time
    courses only, so the data stay in the space the projectors keep, and the
    projectors alone describe the result for forward modelling.

    Parameters
    ----------
    raw : instance of Raw
        Raw data with MEG channels, preloaded unless ``copy=True``.
    int_order : int
        Positive order of the internal harmonics (see :func:`compute_proj_amm`).
    ext_order : int
        Positive order of the external harmonics.
    st_duration : float | None
        If not None, a finite positive duration in seconds for temporal
        windows; a shorter remainder joins the last window in each contiguous
        segment. Temporal fitting skips annotations beginning with ``BAD``
        or ``EDGE`` (case insensitive) and splits at recording boundaries.
        Spatial projection still applies to all samples. If the intermediate
        space is empty, only spatial projection is applied.
    st_correlation : float
        Finite correlation limit satisfying ``0 < st_correlation <= 1``;
        :footcite:`TierneyEtAl2024` used 0.98.
    coord_frame : str
        The coordinate frame the spheroid is fitted in, ``'head'`` (default)
        or ``'meg'`` (see :func:`compute_proj_amm`).
    center_basis : bool
        If True, centre the basis across channels as SPM does (see
        :func:`compute_proj_amm`).
    picks : str | array_like | slice | None
        Channels to include. Default of ``'meg'`` (same as None) will select
        all non-reference MEG channels.
    exclude : list | 'bads'
        List of channels to exclude, only used when picking based on types.
        Specify ``'bads'`` (the default) to exclude all channels marked as bad.
    accuracy : str
        Can be ``"point"``, ``"normal"`` or ``"accurate"`` (default), defines
        which level of coil definition accuracy is used to generate model.
    copy : bool
        If True (default), operate on a copy of ``raw``; otherwise modify it
        in place.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    raw : instance of Raw
        The data with the AMM projectors added and applied.

    See Also
    --------
    compute_proj_amm

    Notes
    -----
    The time courses are taken from the coordinates of the data in the
    internal and intermediate spaces, so their dimension never exceeds that
    of each space.

    .. versionadded:: 1.14

    References
    ----------
    .. footbibliography::
    """
    _validate_type(raw, BaseRaw, "raw")
    _validate_type(st_correlation, "numeric", "st_correlation")
    if not np.isfinite(st_correlation) or not 0 < st_correlation <= 1:
        raise ValueError("st_correlation must satisfy 0 < st_correlation <= 1")
    if st_duration is not None:
        _validate_type(st_duration, "numeric", "st_duration")
        if not np.isfinite(st_duration) or st_duration <= 0:
            raise ValueError("st_duration must be finite and positive")
    raw = raw.copy().load_data() if copy else raw
    _check_preload(raw, "apply_amm")
    info, u_in, u_out, u_res = _amm_spaces(
        raw.info,
        int_order,
        ext_order,
        picks,
        exclude,
        coord_frame,
        center_basis,
        accuracy,
    )
    picks = [raw.ch_names.index(name) for name in info["ch_names"]]
    do_temporal = st_duration is not None and u_res.shape[1] > 0
    if do_temporal:
        data = raw._data[picks]
        coef, res = u_in.T @ data, u_res.T @ data
        duration = min(st_duration, raw.n_times / raw.info["sfreq"])
        n_win = max(int(round(duration * raw.info["sfreq"])), 1)
        onsets, ends = _annotations_starts_stops(raw, ("bad", "edge"), invert=True)
        for onset, end in zip(onsets, ends):
            if onset == end:
                continue
            starts = list(range(onset, end - n_win + 1, n_win)) or [onset]
            for start, stop in zip(starts, starts[1:] + [end]):
                t_proj = _overlap_projector(
                    coef[:, start:stop], res[:, start:stop], st_correlation
                )
                coef[:, start:stop] -= (coef[:, start:stop] @ t_proj) @ t_proj.T
                logger.info(
                    f"    Projecting {t_proj.shape[1]:2d} intersecting component(s) "
                    f"for samples {start} - {stop - 1}"
                )
    raw.add_proj(_amm_projs(info, np.concatenate([u_out, u_res], axis=1)))
    raw.apply_proj()
    if do_temporal:
        raw._data[picks] = u_in @ coef
        # Reconstruction can restore existing SSP components. Their flags are
        # already active, so apply_proj() would not apply them again.
        projector = make_projector(raw.info["projs"], raw.ch_names, raw.info["bads"])[0]
        raw._data[:] = projector @ raw._data
    return raw


def _amm_projs(info, removed):
    """One projector per removed direction."""
    return [
        Projection(
            active=False,
            desc=f"AMM: {ii + 1}",
            data=dict(
                col_names=info["ch_names"],
                row_names=None,
                data=vec[np.newaxis, :],
                ncol=info["nchan"],
                nrow=1,
            ),
        )
        for ii, vec in enumerate(removed.T)
    ]


def _amm_spaces(
    info, int_order, ext_order, picks, exclude, coord_frame, center_basis, accuracy
):
    """Orthonormal bases of the internal, external and remaining spaces."""
    int_order = _ensure_int(int_order, "int_order")
    ext_order = _ensure_int(ext_order, "ext_order")
    for name, order in (("int_order", int_order), ("ext_order", ext_order)):
        if order < 1:
            raise ValueError(f"{name} must be at least 1, got {order}")
    _validate_type(center_basis, bool, "center_basis")
    _check_option("coord_frame", coord_frame, ("head", "meg"))
    picks = _picks_to_idx(info, picks, none="meg", exclude=exclude, with_ref_meg=False)
    info = pick_info(info, picks)
    del picks
    rmags, cosmags, bins, n_coils = _prep_mf_coils(
        info, ignore_ref=False, accuracy=accuracy
    )[:4]
    if n_coils != info["nchan"]:
        raise ValueError(
            f"Only {n_coils}/{info['nchan']} picks could be interpreted as MEG "
            "channels."
        )
    pos = np.array([ch["loc"][:3] for ch in info["chs"]])
    if coord_frame == "head":
        if info["dev_head_t"] is None:
            raise ValueError(
                'coord_frame="head" needs info["dev_head_t"]; use coord_frame="meg" '
                "for recordings without one"
            )
        trans = info["dev_head_t"]["trans"]
        rmags, pos = apply_trans(trans, rmags), apply_trans(trans, pos)
        cosmags = cosmags @ trans[:3, :3].T
    rotation, center, radii = _fit_spheroid(pos)
    rmags = rmags @ rotation.T - center
    cosmags = cosmags @ rotation.T
    starts = np.flatnonzero(np.diff(bins, prepend=-1))  # bins is sorted
    a, b = radii.max(), radii.min()
    bases = list()
    for order, kind in ((ext_order, "external"), (int_order, "internal")):
        basis = _prolate_fields(rmags, cosmags, a, b, order, kind)
        basis = np.add.reduceat(basis, starts, axis=0)
        bad = ~np.isfinite(basis).all(axis=1)
        if bad.any():
            raise ValueError(
                "The following channel(s) generate non-finite projectors:\n"
                f"    {[info['ch_names'][ii] for ii in np.where(bad)[0]]}\n"
                "Please exclude from picks!"
            )
        if center_basis:
            basis -= basis.mean(axis=0)
        norms = np.linalg.norm(basis, axis=0)
        bases.append(basis / np.where(norms == 0, 1, norms))
    u_out = _orth_overwrite(bases[0])
    complement = linalg.null_space(u_out.T)
    if complement.shape[1] == 0:
        raise ValueError(
            "The external harmonics span all selected channels; reduce ext_order "
            "or include more channels"
        )
    u, s, _ = linalg.svd(complement.T @ bases[1], full_matrices=True)
    # Use the original basis scale, not the size of subtraction roundoff.
    tol = max(bases[1].shape) * np.finfo(float).eps * np.linalg.norm(bases[1])
    rank = np.sum(s > tol)
    if rank == 0:
        raise ValueError("The harmonics leave no internal space; reduce ext_order")
    u_in = complement @ u[:, :rank]
    u_res = complement @ u[:, rank:]
    logger.info(
        f"    AMM: spheroid a={1e3 * a:0.1f} mm, b={1e3 * b:0.1f} mm; keeping "
        f"{u_in.shape[1]} of {info['nchan']} dimensions"
    )
    return info, u_in, u_out, u_res


# SPM's rotations bringing the long axis of the spheroid to the second axis
_SPHEROID_ROTATIONS = (
    np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    np.eye(3),
    np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
)


def _fit_spheroid_axis(pos, axis):
    """Least-squares axis-aligned spheroid whose axis ``axis`` is the distinct one.

    ``|v|^2 = beta_0 (sum of the other squared coordinates - 2 v_axis^2)
    + 2 beta_{1..3} . v + beta_4`` is linear in beta. Returns ``(center,
    radii, spread)``, ``spread`` being the standard deviation over points of
    ``sum(((v - center) / radii) ** 2)``.
    """
    sq = pos**2
    shape = np.delete(sq, axis, axis=1).sum(axis=1) - 2 * sq[:, axis]
    design = np.column_stack([shape, 2 * pos, np.ones(len(pos))])
    beta = np.linalg.pinv(design) @ sq.sum(axis=1)
    quad = np.full(3, beta[0] - 1.0)  # quadric sum(quad v^2) + 2 lin . v + const
    quad[axis] = -2 * beta[0] - 1.0
    lin, const = beta[1:4], beta[4]
    center = -lin / quad
    scaled = -quad / (const - np.sum(lin**2 / quad))
    radii = np.sign(scaled) / np.sqrt(np.abs(scaled))
    spread = np.std(np.sum(((pos - center) / radii) ** 2, axis=1), ddof=1)
    return center, radii, spread


def _fit_spheroid(pos):
    """Fit AMM's prolate spheroid; returns ``(rotation, center, radii)``.

    ``rotation`` maps coordinates to a frame whose second axis is the long
    axis, in which ``center`` and ``radii`` are given.
    """
    fits = [_fit_spheroid_axis(pos, axis) for axis in range(3)]
    prolate = [
        axis
        for axis, (_, radii, _) in enumerate(fits)
        if radii[axis] > radii[1 if axis != 1 else 0]
    ]
    if not prolate:
        raise ValueError("The sensor array is not well modelled by a prolate spheroid")
    rotation = _SPHEROID_ROTATIONS[min(prolate, key=lambda axis: fits[axis][2])]
    pos = pos @ rotation.T
    center, radii, _ = _fit_spheroid_axis(pos, 1)
    d2 = (pos - center) ** 2
    step = 0.005 * radii.max()
    for n_step in range(201):
        shrunk = radii - n_step * step
        if np.any(shrunk <= 0):
            break
        if np.all(np.sum(d2 / shrunk**2, axis=1) > 1):
            return rotation, center, shrunk
    raise ValueError("No spheroid inside the sensor array leaves every sensor outside")


def _prolate_coords(v, c):
    """``(mu, nu, phi)`` of points ``v`` (long axis second) for focal distance ``c``.

    ``v = (s sin(nu) cos(phi), mu cos(nu), s sin(nu) sin(phi))`` with
    ``s = sqrt(mu**2 - c**2)``.
    """
    t = np.sum(v**2, axis=1) + c**2
    mu = np.sqrt((t + np.sqrt(np.maximum(t**2 - 4 * v[:, 1] ** 2 * c**2, 0))) / 2)
    # atan2 preserves the small transverse coordinate near either pole.
    nu = np.arctan2(
        np.hypot(v[:, 0], v[:, 2]) * mu,
        v[:, 1] * np.sqrt(mu**2 - c**2),
    )
    return mu, nu, np.arctan2(v[:, 2], v[:, 0])


def _legendre_p_series(w, degree, order):
    """``q(w)`` and ``q'(w)`` with ``P_l^m(x) ∝ (x^2 - 1)^(m/2) x^(l-m) q(1/x^2)``."""
    k = np.arange((degree - order) // 2 + 1)
    log_c = (
        gammaln(2 * degree - 2 * k + 1)
        - gammaln(k + 1)
        - gammaln(degree - k + 1)
        - gammaln(degree - 2 * k - order + 1)
    )
    coef = (-1.0) ** k * np.exp(log_c - log_c[0])
    powers = np.asarray(w)[..., np.newaxis] ** k
    return powers @ coef, powers[..., :-1] @ (coef[1:] * k[1:])


def _hypergeometric_ratio(mu, a, c, degree, order):
    """``F(c^2/mu^2) / F(c^2/a^2)`` and its derivative in ``mu``.

    ``Q_l^m(x) ∝ (x^2 - 1)^(m/2) x^(-l-m-1) F(1/x^2)`` with
    ``F = 2F1((l+m+1)/2, (l+m+2)/2; l+3/2; .)``.
    """
    alpha, beta, gamma = (
        (degree + order + 1) / 2,
        (degree + order + 2) / 2,
        degree + 1.5,
    )
    w = (c / mu) ** 2
    ref = hyp2f1(alpha, beta, gamma, (c / a) ** 2)
    dfdw = alpha * beta / gamma * hyp2f1(alpha + 1, beta + 1, gamma + 1, w)
    return hyp2f1(alpha, beta, gamma, w) / ref, dfdw * (-2 * c**2 / mu**3) / ref


def _prolate_radial(kind, mu, a, c, degree, order):
    """``R(mu) / R(a)`` and its derivative: ``Q_l^m(mu/c)`` or ``P_l^m(mu/c)``."""
    p = degree - order if kind == "external" else -degree - order - 1
    g = ((mu**2 - c**2) / (a**2 - c**2)) ** (order / 2) * (mu / a) ** p
    dg = g * (order * mu / (mu**2 - c**2) + p / mu)
    if kind == "external":
        q, dq = _legendre_p_series((c / mu) ** 2, degree, order)
        q_ref = _legendre_p_series((c / a) ** 2, degree, order)[0]
        h, dh = q / q_ref, dq * (-2 * c**2 / mu**3) / q_ref
    else:
        h, dh = _hypergeometric_ratio(mu, a, c, degree, order)
    return g * h, dg * h + g * dh


def _prolate_angular(nu, phi, degree, order):
    """Real spherical harmonic of ``(nu, phi)`` and its derivatives in phi and nu."""
    m = abs(order)
    x, sine = np.cos(nu), np.sin(nu)
    # P_l^m(x) = (-sin(nu))^m (2m-1)!! C_{l-m}^{m+1/2}(x).
    # Unlike lpmv(cos(nu)), this retains sin(nu) when cos(nu) rounds to +/-1.
    factor = (-1.0) ** m * np.exp(gammaln(2 * m + 1) - m * np.log(2) - gammaln(m + 1))
    poly = eval_gegenbauer(degree - m, m + 0.5, x)
    p = factor * sine**m * poly
    dp = np.zeros_like(nu)
    if degree > m:
        dp -= (
            factor
            * sine ** (m + 1)
            * (2 * m + 1)
            * eval_gegenbauer(degree - m - 1, m + 1.5, x)
        )
    if m:
        dp += factor * m * sine ** (m - 1) * x * poly
    if order > 0:
        trig, dtrig = np.cos(m * phi), -m * np.sin(m * phi)
    elif order < 0:
        trig, dtrig = np.sin(m * phi), m * np.cos(m * phi)
    else:
        trig, dtrig = np.ones_like(phi), np.zeros_like(phi)
    return p * trig, p * dtrig, dp * trig


def _prolate_fields(v, n, a, b, max_degree, kind):
    """Field along ``n`` at points ``v`` of each potential ``R_l^m(mu) S_l^m(nu, phi)``.

    ``v`` is relative to the spheroid center, long axis second; columns are
    ordered ``l = 1..max_degree``, ``m = -l..l``.
    """
    c = np.sqrt(a**2 - b**2)
    mu, nu, phi = _prolate_coords(v, c)
    s = np.sqrt(mu**2 - c**2)
    d = np.sqrt(mu**2 - c**2 * np.cos(nu) ** 2)
    n_xz = np.cos(phi) * n[:, 0] + np.sin(phi) * n[:, 2]
    # orientation . unit vectors, divided by the scale factors h_mu, h_nu, h_phi
    w_mu = (mu * np.sin(nu) * n_xz + s * np.cos(nu) * n[:, 1]) / d * (s / d)
    w_nu = (s * np.cos(nu) * n_xz - mu * np.sin(nu) * n[:, 1]) / d**2
    h_phi = s * np.sin(nu)
    pole = (v[:, 0] == 0) & (v[:, 2] == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_phi = np.where(
            h_phi > 0, (np.cos(phi) * n[:, 2] - np.sin(phi) * n[:, 0]) / h_phi, 0.0
        )
    columns = list()
    for degree in range(1, max_degree + 1):
        for order in range(-degree, degree + 1):
            r, dr = _prolate_radial(kind, mu, a, c, degree, abs(order))
            ang, dang_phi, dang_nu = _prolate_angular(nu, phi, degree, order)
            field = w_mu * dr * ang + w_nu * r * dang_nu + w_phi * r * dang_phi
            # At a pole only m=0 (axial) and |m|=1 (transverse) survive.
            sign = np.sign(v[pole, 1])
            if order == 0:
                field[pole] = dr[pole] * sign ** (degree + 1) * n[pole, 1]
            elif abs(order) == 1:
                axis = 2 if order < 0 else 0
                field[pole] = (
                    -r[pole]
                    / s[pole]
                    * degree
                    * (degree + 1)
                    / 2
                    * sign ** (degree + 1)
                    * n[pole, axis]
                )
            else:
                field[pole] = 0
            columns.append(field)
    return np.column_stack(columns)
