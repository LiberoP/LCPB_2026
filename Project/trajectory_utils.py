from ase.io import read, write
import numpy as np
from ase.geometry import find_mic
from ase.geometry.rdf import get_rdf
from ase.data import atomic_numbers
import ase.units as units
import matplotlib.pyplot as plt
import os
import pickle
from collections import defaultdict


# reads local OUTCAR file for analysis (see "general analysis" notebook) from a certain index on
def clean_dataset(outcar_file, destination_file, start_idx):
    try:
        atoms_list = read(outcar_file, index=":")
        atoms_list = atoms_list[start_idx:]
        print(f"Retrieved {len(atoms_list)} structures from OUTCAR.")

        # usable file
        write(destination_file, atoms_list, format="extxyz")
        print(f"File saved as {destination_file}")

    except Exception as e:
        print(f"Error cleaning OUTCAR: {e}")

    return 0


def read_merged(paths):
    """Concatenate OUTCAR files that are sequential continuations of one run.
    Assumes the first frame of each subsequent file duplicates the last frame
    of the previous file, so it is dropped."""
    frames = []
    for i, p in enumerate(paths):
        traj = read(p, index=":")
        if i > 0:
            traj = traj[1:]   # drop duplicated boundary frame
        frames.extend(traj)
    return frames



def _per_file(value, n_files, name):
    """Return a per-file list from either a single value or a list."""
    if isinstance(value, (list, tuple)):
        if len(value) != n_files:
            raise ValueError(f"{name} must have one entry per file ({n_files}), got {len(value)}")
        return list(value)
    return [value] * n_files


def _carve_test_blocks(n_frames, t0, n_blocks, block_length, buffer, rng):
    """
    Place n_blocks non-overlapping test blocks of block_length frames, all
    starting at or after t0, separated by at least `buffer` frames.
    Returns (test_indices, buffer_indices) as sorted integer arrays.
    """
    free = (n_frames - t0) - n_blocks * block_length - (n_blocks - 1) * buffer
    if free < 0:
        raise ValueError(
            f"{n_blocks} blocks of {block_length} frames with buffer {buffer} do not fit "
            f"in the {n_frames - t0} frames after t0={t0}"
        )
    offsets = np.sort(rng.integers(0, free + 1, size=n_blocks))
    starts = t0 + offsets + np.arange(n_blocks) * (block_length + buffer)

    test_mask = np.zeros(n_frames, dtype=bool)
    near_mask = np.zeros(n_frames, dtype=bool)
    for s in starts:
        e = s + block_length
        test_mask[s:e] = True
        near_mask[max(s - buffer, 0):min(e + buffer, n_frames)] = True

    buffer_mask = near_mask & ~test_mask

    return np.flatnonzero(test_mask), np.flatnonzero(buffer_mask)


def _min_distance(a, b):
    """Smallest |a_i - b_j| over all pairs, for integer arrays, b sorted."""
    pos = np.searchsorted(b, a)
    left = b[np.clip(pos - 1, 0, len(b) - 1)]
    right = b[np.clip(pos, 0, len(b) - 1)]
    return np.minimum(np.abs(a - left), np.abs(a - right)).min()


def _subsample_stride(indices, stride=5, seed=42):
    """Take every `stride`-th index, then shuffle (decorrelation subsampling)."""
    subsampled = np.array(indices[::stride])
    rng = np.random.RandomState(seed)
    rng.shuffle(subsampled)
    return subsampled


def _subsample_fps(frames, n_samples=500, pool_stride=10, r_cut=5.0, n_max=4, l_max=4):
    """
    Select `n_samples` structurally diverse frames via SOAP descriptors +
    Farthest Point Sampling (FPS). `pool_stride` thins the trajectory first
    to keep the SOAP computation cheap. Returns the selected positions in
    the input list. Deterministic (always starts from the first pool frame).
    """
    from dscribe.descriptors import SOAP

    pool_positions = list(range(0, len(frames), pool_stride))
    pool_frames = [frames[i] for i in pool_positions]
    species = list(set(pool_frames[0].get_chemical_symbols()))

    soap = SOAP(species=species, periodic=True, r_cut=r_cut, n_max=n_max, l_max=l_max)

    fingerprints = []
    for frame in pool_frames:
        atom_soaps = soap.create(frame)
        fingerprints.append(np.mean(atom_soaps, axis=0))
    features = np.array(fingerprints)

    selected_indices = [0]
    min_distances = np.linalg.norm(features - features[0], axis=1)
    for _ in range(1, n_samples):
        next_idx = np.argmax(min_distances)
        selected_indices.append(next_idx)
        new_distances = np.linalg.norm(features - features[next_idx], axis=1)
        min_distances = np.minimum(min_distances, new_distances)

    return [pool_positions[i] for i in selected_indices]


def _plot_splits(split_indices, n_frames):
    """One row per file, colored by role of each frame index."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    names = ["excluded", "buffer", "test", "train"]
    colors = ["lightgrey", "orange", "red", "tab:blue"]
    cmap = ListedColormap(colors)

    fig, axes = plt.subplots(
    len(split_indices), 1, figsize=(30, 1.3 * len(split_indices) + 1), squeeze=False, dpi=150
)
    for ax, (name, split) in zip(axes[:, 0], split_indices.items()):
        codes = np.zeros(n_frames[name], dtype=int)
        codes[split["buffer"]] = 1
        codes[split["test"]] = 2
        codes[split["train"]] = 3
        ax.imshow(codes[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax.set_yticks([])
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("frame index")
    fig.legend(
        handles=[Patch(color=c, label=n) for c, n in zip(colors, names)],
        loc="upper right",
    )
    fig.tight_layout()
    plt.show()



def tail_band_t0(x, tail_frac=0.3, window=50, k=3):
    x = np.asarray(x)
    n = len(x)
    s = int((1 - tail_frac) * n)
    roll = np.convolve(x, np.ones(window) / window, mode="valid")
    ref = x[s:].mean()
    band = k * roll[s:].std()
    outside = np.where(np.abs(roll - ref) > band)[0]
    if len(outside) == 0:
        return 0
    return outside[-1] + window


def plot_convergence(
    temperatures,
    energies,
    msd,
    peak_height_history,
    time_steps,
    t0_per_metric,
    t0_consensus,
    pairs=("Sn-I", "N-H"),
    label=None,
    save_dir="/content/convergence_check_tailband",
):
    """
    Plot convergence diagnostics: temperature, energy (if given), MSD, and
    RDF peak heights. Vertical lines mark per-metric t0 and the consensus t0.
    If save_dir is given, the figure is saved to
    convergence_<label>.png there.
    """
    x_rdf = np.array(time_steps)

    t0s = [t0_per_metric["temperature"]]
    data_list = [temperatures]
    x_list = [np.arange(len(temperatures))]
    names = ["Temperature (K)"]

    if energies is not None:
        t0s.append(t0_per_metric["energy"])
        data_list.append(energies)
        x_list.append(np.arange(len(energies)))
        names.append("Energy (eV)")

    t0s.append(t0_per_metric["msd"])
    data_list.append(msd)
    x_list.append(np.arange(len(msd)))
    names.append("MSD, non-H (A^2)")

    for name in pairs:
        key = f"{name}_height"
        if key not in t0_per_metric:
            continue
        t0s.append(t0_per_metric[key])
        data_list.append(peak_height_history[name])
        x_list.append(x_rdf)
        names.append(f"{name} peak height")

    fig, axs = plt.subplots(
        len(data_list), 1, figsize=(10, 2.4 * len(data_list)), sharex=True
    )
    for ax, x, data, name, t0 in zip(axs, x_list, data_list, names, t0s):
        ax.plot(x, data, lw=1)
        ax.axvline(t0, color="red", ls="--", label=f"{name} t0 = {t0}")
        ax.axvline(t0_consensus, color="black", ls=":",
                   label=f"consensus t0 = {t0_consensus}")
        ax.set_ylabel(name)
        ax.legend()
    axs[-1].set_xlabel("Frame index")
    if label is not None:
        fig.suptitle(f"Convergence check: {label}")
    plt.tight_layout()

    if save_dir is not None:
        safe_name = (label or "trajectory").replace("/", "_")
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"convergence_{safe_name}.png")
        fig.savefig(fig_path, dpi=150)
        print(f"Saved convergence plot to {fig_path}")

    plt.show()
    return fig

def check_convergence(
    atoms_list,
    rmax=None,
    nbins=1000,
    window_size=20,
    timestep_in_fs=0.5,
    tail_frac=0.3,
    tail_k=3,
    tail_window=50,
    rdf_smooth_windows=5,
    plot=False,
    label=None,
    save_dir=None,
):
    """
    Assess equilibration of a trajectory using a tail-band estimator on
    temperature, total energy (if available), non-H MSD, and element-pair
    RDF peaks (height) for Sn-I (inorganic) and N-H (organic).

    tail_band_t0 finds the last point where a rolling mean of the series
    departs from the tail-region mean by more than tail_k standard
    deviations of the tail spread; that point is the per-metric t0.
    tail_window is the rolling window in frames, used for temperature,
    energy and MSD. rdf_smooth_windows is the same idea for the RDF peak
    histories, expressed in number of RDF windows instead of raw frames,
    since those histories are already binned by window_size frames.

    This is the computationally heavy step (RDF over the whole
    trajectory), so whenever save_dir is given, every intermediate metric
    is pickled there, independent of whether a plot is requested, so a
    later manual convergence decision never needs to recompute any of
    this.

    atoms_list must come from an OUTCAR-based read (so that
    atoms.get_potential_energy() works) if the energy criterion is to be
    used.
    """
    n_frames = len(atoms_list)

    # --- temperature (all atoms, MIC-corrected finite-difference velocities) ---
    all_velocities = []
    for i in range(n_frames - 1):
        delta_p = atoms_list[i + 1].get_positions() - atoms_list[i].get_positions()
        delta_p, _ = find_mic(
            delta_p, atoms_list[i].get_cell(), pbc=atoms_list[i].get_pbc()
        )
        all_velocities.append(delta_p / timestep_in_fs)
    all_velocities = np.array(all_velocities)

    n_atoms = len(atoms_list[0])
    if rmax is None:
        min_height_overall = np.inf
        for atoms in atoms_list:
            cell = atoms.get_cell()
            min_height = min(
                cell.volume / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
                for i in range(3)
            )
            min_height_overall = min(min_height_overall, min_height)  # NPT: cell volume varies
        rmax = 0.99 * min_height_overall / 2
        print(f"Auto-selected rmax = {rmax:.3f} A (smallest cell over full trajectory)")

    dof = 3 * n_atoms - 3  # only remove 3 dofs for center-of-mass translation

    temperatures = []
    for i, atoms in enumerate(atoms_list[:-1]):
        m = atoms.get_masses()
        v_ase = all_velocities[i] / units.fs
        ek = 0.5 * np.sum(m[:, np.newaxis] * v_ase**2)
        temperatures.append((2 * ek) / (dof * units.kB))
    temperatures = np.array(temperatures)

    # --- element-pair RDF peaks: Sn-I,  N-H (height), rolling windows ---
    pairs = {
        "Sn-I": (atomic_numbers["Sn"], atomic_numbers["I"]),
        "N-H": (atomic_numbers["N"], atomic_numbers["H"]),
    }

    peak_height_history = {name: [] for name in pairs}
    time_steps = []

    distances = None
    for start_idx in range(0, n_frames - window_size + 1, window_size):
        window_atoms = atoms_list[start_idx : start_idx + window_size]
        window_rdfs = {name: [] for name in pairs}
        for atoms in window_atoms:
            for name, (z1, z2) in pairs.items():
                rdf_vals, distances = get_rdf(atoms, rmax=rmax, nbins=nbins, elements=(z1, z2))
                window_rdfs[name].append(rdf_vals)
        for name in pairs:
            window_avg_rdf = np.mean(window_rdfs[name], axis=0)
            max_idx = np.argmax(window_avg_rdf)
            peak_height_history[name].append(window_avg_rdf[max_idx])
        time_steps.append(start_idx + window_size // 2)

    for name in pairs:
        peak_height_history[name] = np.array(peak_height_history[name])

    # --- MSD relative to frame 0, MIC-unwrapped, hydrogens excluded ---
    pbc = atoms_list[0].get_pbc()
    cell = atoms_list[0].get_cell()
    symbols = np.array(atoms_list[0].get_chemical_symbols())
    non_h_mask = symbols != "H"

    unwrapped_disp = np.zeros_like(atoms_list[0].get_positions())
    msd = [0.0]
    prev_pos = atoms_list[0].get_positions()
    for atoms in atoms_list[1:]:
        pos = atoms.get_positions()
        step_disp = pos - prev_pos
        step_disp, _ = find_mic(step_disp, cell, pbc=pbc)
        unwrapped_disp += step_disp
        msd.append(np.mean(np.sum(unwrapped_disp[non_h_mask] ** 2, axis=1)))
        prev_pos = pos
    msd = np.array(msd)

    # --- tail-band t0 per observable ---
    t0_per_metric = {
        "temperature": tail_band_t0(temperatures, tail_frac=tail_frac, window=tail_window, k=tail_k),
        "msd": tail_band_t0(msd, tail_frac=tail_frac, window=tail_window, k=tail_k),
    }
    for name in pairs:
        t0_height_steps = tail_band_t0(
            peak_height_history[name], tail_frac=tail_frac, window=rdf_smooth_windows, k=tail_k
        )
        t0_per_metric[f"{name}_height"] = t0_height_steps * window_size

    # energy criterion only if the calculator attached to these atoms supports it
    # (true for atoms read straight from OUTCAR; not for a stripped extxyz file)
    have_energy = False
    energies = None
    try:
        energies = np.array([atoms.get_potential_energy() for atoms in atoms_list])
        t0_per_metric["energy"] = tail_band_t0(energies, tail_frac=tail_frac, window=tail_window, k=tail_k)
        have_energy = True
    except Exception:
        print("No energies found! Skipping this when determining convergence")

    t0_consensus = max(t0_per_metric.values())

    # --- save every metric computed above, independent of `plot`: this step is the expensive one ---
    safe_name = (label or "trajectory").replace("/", "_")
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        metrics = {
            "label": label,
            "rmax": rmax,
            "nbins": nbins,
            "window_size": window_size,
            "distances": distances,
            "temperatures": temperatures,
            "peak_height_history": peak_height_history,
            "msd": msd,
            "energies": energies,
            "time_steps": np.array(time_steps),
            "t0_per_metric": t0_per_metric,
            "t0_consensus": t0_consensus,
        }
        pickle_path = os.path.join(save_dir, f"convergence_{safe_name}.pickle")
        with open(pickle_path, "wb") as f:
            pickle.dump(metrics, f)
        print(f"Saved convergence metrics to {pickle_path}")

    if plot:
        plot_convergence(
            temperatures=temperatures,
            energies=energies if have_energy else None,
            msd=msd,
            peak_height_history=peak_height_history,
            time_steps=time_steps,
            t0_per_metric=t0_per_metric,
            t0_consensus=t0_consensus,
            pairs=tuple(pairs.keys()),
            label=label,
            save_dir=save_dir,
        )

    return t0_consensus, t0_per_metric


def _detrend(x, order=2):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= order + 1:
        return x - x.mean()
    t = np.arange(n)
    return x - np.polyval(np.polyfit(t, x, order), t)


def _acf(x, max_lag):
    x = np.asarray(x, dtype=float)
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    if acf[0] <= 0:
        return np.zeros(max_lag + 1)
    acf = acf / acf[0]
    return acf[: max_lag + 1]


def _tau_samples(acf, thresh=0.05):
    below = np.flatnonzero(acf < thresh)
    return int(below[0]) if len(below) else len(acf) - 1


def estimate_buffer_from_pickle(
    pickle_path,
    n_tau=5.0,
    tail_frac=0.5,
    thresh=0.05,
    max_lag_frac=0.25,
    detrend_order=2,
    verbose=True,
):
    """
    Estimate a decorrelation buffer (in frames) from saved convergence metrics.

    For each observable (temperature, energy, MSD, RDF peak heights),
    we detrend the equilibrated tail, compute the autocorrelation function, and
    take the first lag where ACF < `thresh` as the correlation length tau.
    The suggested buffer is n_tau * max(tau) across all metrics and files.

    Parameters
    ----------
    pickle_path : str or list[str]
        Paths to convergence_*.pickle files from check_convergence.
    n_tau : float
        Multiplier on tau. 3-5 is typical; 5 is conservative.
    tail_frac : float
        Fraction of the trajectory tail used (must be equilibrated).
    thresh : float
        ACF threshold defining the correlation length.
    max_lag_frac : float
        Cap on ACF lag as a fraction of the segment length.
    detrend_order : int
        Polynomial order removed before ACF (handles MSD's linear growth).

    Returns
    -------
    buffer_frames : int
    details : dict  {label: {metric: tau_frames}}
    """
    with open(pickle_path, "rb") as f:
        m = pickle.load(f)

    label = m.get("label", os.path.basename(pickle_path))
    window_size = m.get("window_size", 1)

    candidates = [
        ("temperature", m.get("temperatures"), 1),
        ("energy",      m.get("energies"),     1),

    ]
    for name, hist in (m.get("peak_height_history") or {}).items():
        candidates.append((f"{name}_height", hist, window_size))
    for name, hist in (m.get("peak_pos_history") or {}).items():
        candidates.append((f"{name}_pos", hist, window_size))

    file_taus = {}
    for name, ts, unit in candidates:
        if ts is None:
            continue
        ts = np.asarray(ts, dtype=float)
        if len(ts) < 20:
            continue
        s = int((1 - tail_frac) * len(ts))
        seg = _detrend(ts[s:], order=detrend_order)
        if seg.std() == 0:
            continue
        max_lag = max(5, int(max_lag_frac * len(seg)))
        acf = _acf(seg, max_lag)
        tau_samp = _tau_samples(acf, thresh=thresh)
        tau_frames = tau_samp * unit
        file_taus[name] = tau_frames
        if verbose:
            print(f"  [{label:>24s}] {name:>18s}: "
                  f"tau = {tau_samp:4d} samples = {tau_frames:7.1f} frames")

    if not file_taus:
        raise RuntimeError(f"No usable timeseries in {pickle_path}")

    tau_max = float(np.max(list(file_taus.values())))
    buffer_frames = int(np.ceil(n_tau * tau_max))
    if verbose:
        print(f"  [{label}] tau_max = {tau_max:.1f} frames -> "
              f"buffer = {buffer_frames} frames")
    return buffer_frames, file_taus



def load_trajectories(
    trajectories,
    train_region="all",
    start_frames=None,
    test_n_blocks=None,
    test_block_length=None,
    test_buffer=None,
    train_fraction=None,
    subsample_method=None,
    subsample_kwargs=None,
    n_train_target=None,
    seed=42,
    plot_splits=False,
    plot_convergence=False,
    convergence_plot_dir="./convergence_plots",
):
    """
    Load one or more OUTCAR trajectories and split each into train / test /
    buffer / excluded frames. All bookkeeping is done on frame indices
    (positions in read(outcar_file, ":"), before any trimming); the frames
    are only taken from the original file at the end, using those indices.

    Parameters
    ----------
    outcar_files : list[str]
        Path to one OUTCAR file per trajectory.
    start_frames : "auto", None, or list[int or None]
        - "auto": auto-detect the equilibration frame t0 for every file (via
          check_convergence).
        - None: t0 = 0 for every file.
        - list, same length as outcar_files: fixed t0 per file (None = 0).
    test_n_blocks, test_block_length, test_buffer : int, None, or list
        Per-file (or single value for all files) test carve-out. Each file
        gets test_n_blocks blocks of test_block_length consecutive frames,
        placed randomly, entirely at or after t0, non-overlapping and at
        least test_buffer frames apart. test_buffer frames on each side of
        every block (after t0) are discarded for both train and test.
        test_n_blocks = None means no test frames for that file.
    train_fraction : None, float in (0, 1], or list
        Fraction of the remaining training pool (after removing pre-t0,
        test and buffer frames) that is randomly kept. None keeps all.
    subsample_method : None, "stride", or "fps"
        Applied after train_fraction.
        - "stride": every Nth index of each file's pool + shuffle. kwargs:
          stride (int, or list with one value per file).
        - "fps": SOAP + Farthest Point Sampling on the combined pool of all
          files. kwargs: n_samples, pool_stride, r_cut, n_max, l_max.
    subsample_kwargs : dict, optional
        Extra keyword arguments forwarded to the chosen subsampling method.
    seed : int
        Base seed. File k uses seed + k for block placement, train_fraction
        and the stride shuffle.
    plot_splits : bool
        Show one row per file, colored by train / test / buffer / excluded.
    plot_convergence, convergence_plot_dir :
        Forwarded to check_convergence for files using start_frames == "auto".

    Returns
    -------
    train_frames : list[ase.Atoms]
        Training frames from all files, in the same order as the train
        indices (file by file, unless FPS reorders them).
    test_frames : list[ase.Atoms]
        Test frames from all files, ascending index within each file.
    convergence_info : dict
        {outcar_file: t0_per_metric} for files using "auto".
    split_indices : dict
        {outcar_file: {"t0", "train", "test", "buffer", "excluded"}} with
        frame indices relative to the original file. "train" is in the same
        order as that file's frames in train_frames. "excluded" is every
        frame that is not train, test or buffer (pre-t0 and unused frames).
    """
    n_files = len(trajectories)
    outcar_files = [f"traj_{k}" for k in range(n_files)]

    start_frames = _per_file(start_frames, n_files, "start_frames")
    n_blocks_list = _per_file(test_n_blocks, n_files, "test_n_blocks")
    block_len_list = _per_file(test_block_length, n_files, "test_block_length")
    buffer_list = _per_file(test_buffer, n_files, "test_buffer")
    fraction_list = _per_file(train_fraction, n_files, "train_fraction")

    subsample_kwargs = dict(subsample_kwargs or {})
    n_train_target_list = _per_file(n_train_target, n_files, "n_train_target") if n_train_target is not None else [None] * n_files
    if subsample_method == "stride" and n_train_target is None:
        stride_arg = subsample_kwargs.pop("stride", 5)
        train_strides = _per_file(stride_arg, n_files, "stride")
    else:
        train_strides = [None] * n_files
    if subsample_method not in (None, "stride", "fps"):
        raise ValueError(f"Unknown subsample_method: {subsample_method}")

    convergence_info = {}
    n_frames = {}
    t0_per_file = {}
    test_idx_per_file = {}
    buffer_idx_per_file = {}
    test_frames = []
    candidate_frames = []
    candidate_labels = []

    for k, (outcar_file, start, n_blocks, block_len, buffer, fraction, train_stride) in enumerate(
        zip(outcar_files, start_frames, n_blocks_list, block_len_list, buffer_list,
            fraction_list, train_strides)
    ):
        frames = trajectories[k]
        n = len(frames)
        print(f"{outcar_file}: read {n} frames")

        if start == "auto":
            t0, t0_per_metric = check_convergence(
                frames,
                plot=plot_convergence,
                label=outcar_file,
                save_dir=convergence_plot_dir if plot_convergence else None,
            )
            t0 = int(t0)
            print(f"{outcar_file}: per-metric t0s = {t0_per_metric}")
            convergence_info[outcar_file] = t0_per_metric
            print(f"{outcar_file}: auto-detected equilibration at frame {t0}")
        elif start is None:
            t0 = 0
        else:
            t0 = int(start)
            print(f"{outcar_file}: using fixed start frame {t0}")

        if t0 >= n:
            raise ValueError(f"{outcar_file}: t0={t0} is not smaller than the number of frames {n}")

        # index-level split
        rng = np.random.default_rng(seed + k)

        if n_blocks is None:
            test_idx = np.array([], dtype=int)
            buffer_idx = np.array([], dtype=int)
        else:
            if block_len is None or buffer is None:
                raise ValueError(f"{outcar_file}: test_block_length and test_buffer are required")
            test_idx, buffer_idx = _carve_test_blocks(n, t0, n_blocks, block_len, buffer, rng)

        blocked = np.zeros(n, dtype=bool)
        blocked[test_idx] = True
        blocked[buffer_idx] = True

        pool_idx = np.arange(n)
        if train_region == "after":
            pool_idx = pool_idx[pool_idx >= t0]
        elif train_region == "before":
            pool_idx = pool_idx[pool_idx < t0]
        pool_idx = pool_idx[~blocked[pool_idx]]

        if fraction is not None:
            n_keep = int(round(fraction * len(pool_idx)))
            pool_idx = np.sort(rng.choice(pool_idx, size=n_keep, replace=False))

        if subsample_method == "stride" and n_train_target_list[k] is not None:
            train_stride = max(1, len(pool_idx) // n_train_target_list[k])
            print(f"{outcar_file}: derived stride={train_stride} for target N_train={n_train_target_list[k]}")

        if train_stride is not None:
            pool_idx = _subsample_stride(pool_idx, stride=train_stride, seed=seed + k)

        # consistency checks on indices
        assert len(np.intersect1d(pool_idx, test_idx)) == 0, "train and test indices overlap"
        assert len(np.intersect1d(pool_idx, buffer_idx)) == 0, "train and buffer indices overlap"
        assert test_idx.min() >= t0 if len(test_idx) else True, "test frame before t0"
        if len(pool_idx) and len(test_idx) and buffer is not None:
            dist = _min_distance(pool_idx, test_idx)
            assert dist > buffer, f"train frame only {dist} frames from a test frame"

        print(
            f"{outcar_file}: t0={t0}, test={len(test_idx)}, buffer={len(buffer_idx)}, "
            f"train candidates={len(pool_idx)}"
        )

        # only now take the frames from the original file
        test_frames.extend(frames[i] for i in test_idx)
        candidate_frames.extend(frames[i] for i in pool_idx)
        candidate_labels.extend((outcar_file, int(i)) for i in pool_idx)

        n_frames[outcar_file] = n
        t0_per_file[outcar_file] = t0
        test_idx_per_file[outcar_file] = test_idx
        buffer_idx_per_file[outcar_file] = buffer_idx

    print(f"Total training-pool frames from all files (before FPS, if any): {len(candidate_frames)}")
    print(f"Total test frames from all files: {len(test_frames)}")

    if subsample_method == "fps":
        positions = _subsample_fps(candidate_frames, **subsample_kwargs)
        candidate_frames = [candidate_frames[p] for p in positions]
        candidate_labels = [candidate_labels[p] for p in positions]
        print(f"Total training-pool frames after FPS subsampling: {len(candidate_frames)}")

    train_frames = candidate_frames

    split_indices = {}
    for outcar_file in outcar_files:
        train_idx = np.array(
            [i for f, i in candidate_labels if f == outcar_file], dtype=int
        )
        used = np.zeros(n_frames[outcar_file], dtype=bool)
        used[train_idx] = True
        used[test_idx_per_file[outcar_file]] = True
        used[buffer_idx_per_file[outcar_file]] = True
        split_indices[outcar_file] = {
            "t0": t0_per_file[outcar_file],
            "train": train_idx,
            "test": test_idx_per_file[outcar_file],
            "buffer": buffer_idx_per_file[outcar_file],
            "excluded": np.flatnonzero(~used),
        }

    if plot_splits:
        _plot_splits(split_indices, n_frames)

    return train_frames, test_frames, convergence_info, split_indices
