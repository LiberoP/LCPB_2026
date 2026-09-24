import ase.io
from ase.io import read, write
import numpy as np
from ase.geometry import find_mic
from ase.geometry.rdf import get_rdf
import ase.units as units
import matplotlib.pyplot as plt
import os
import pickle


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


import os
from collections import defaultdict
from ase.io import read


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
    #buffer_mask[:t0] = False
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
        len(split_indices), 1, figsize=(12, 1.3 * len(split_indices) + 1), squeeze=False
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


def check_convergence(
    atoms_list,
    rmax=None,
    nbins=1000,
    window_size=20,
    timestep_in_fs=0.5,
    plot=False,
    label=None,
    save_dir=None,
):
    """
    Assess equilibration of a trajectory: runs pymbar.timeseries.detect_equilibration
    on temperature, RDF peak height (g_max), MSD, and (if available) total
    energy. Returns the consensus (max) equilibration frame, plus the per-metric t0
    and statistical inefficiency g.

    This is the computationally heavy step (RDF over the whole trajectory), so
    whenever save_dir is given, every intermediate metric (raw arrays, t0/g/Neff
    per metric) is pickled there -- independent of whether a plot is requested --
    so a later manual convergence decision never needs to recompute any of this.

    atoms_list must come from an OUTCAR-based read (so that
    atoms.get_potential_energy() works) if the energy criterion is to be used.
    """
    from pymbar import timeseries

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
        print(f"Auto-selected rmax = {rmax:.3f} Å (smallest cell over full trajectory)")

    dof = 3 * n_atoms - 3  # only remove 3 dofs for center-of-mass translation

    temperatures = []
    for i, atoms in enumerate(atoms_list[:-1]):
        m = atoms.get_masses()
        v_ase = all_velocities[i] / units.fs
        ek = 0.5 * np.sum(m[:, np.newaxis] * v_ase**2)
        temperatures.append((2 * ek) / (dof * units.kB))
    temperatures = np.array(temperatures)

# FWHM excluded from consensus bc two maxes of similar height later in runs would make this meaningless
# - and identifying the first max would be useless work, as we already have plenty of metrics
    # --- RDF first-peak height (g_max) and width (FWHM), rolling windows ---
    g_max_history = []
    #fwhm_history = []
    time_steps = []

    #peak_r_history = [] # for debugging
    distances = None
    for start_idx in range(0, n_frames - window_size + 1, window_size):
        window_rdfs = []
        for atoms in atoms_list[start_idx : start_idx + window_size]:
            rdf_vals, distances = get_rdf(atoms, rmax=rmax, nbins=nbins)
            window_rdfs.append(rdf_vals)
        window_avg_rdf = np.mean(window_rdfs, axis=0)

        max_idx = np.argmax(window_avg_rdf)

        g_max = window_avg_rdf[max_idx]
        #half_max = g_max / 2.0

        #left_indices = np.where(window_avg_rdf[:max_idx] < half_max)[0]
        #right_indices = np.where(window_avg_rdf[max_idx:] < half_max)[0]
        #if len(left_indices) > 0 and len(right_indices) > 0:
            #fwhm = distances[max_idx + right_indices[0]] - distances[left_indices[-1]]
        #else:
            #fwhm = np.nan

        g_max_history.append(g_max)
        #fwhm_history.append(fwhm)
        time_steps.append(start_idx + window_size // 2)
    g_max_history = np.array(g_max_history)
    #fwhm_history = np.array(fwhm_history)

    # --- MSD relative to frame 0, MIC-unwrapped ---
    pbc = atoms_list[0].get_pbc()
    cell = atoms_list[0].get_cell()
    unwrapped_disp = np.zeros_like(atoms_list[0].get_positions())
    msd = [0.0]
    prev_pos = atoms_list[0].get_positions()
    for atoms in atoms_list[1:]:
        pos = atoms.get_positions()
        step_disp = pos - prev_pos
        step_disp, _ = find_mic(step_disp, cell, pbc=pbc)
        unwrapped_disp += step_disp
        msd.append(np.mean(np.sum(unwrapped_disp**2, axis=1)))
        prev_pos = pos
    msd = np.array(msd)

    # --- pymbar detect_equilibration per observable: t0, g (statistical inefficiency), Neff ---
    t0_T, g_T, Neff_T = timeseries.detect_equilibration(temperatures)
    t0_gmax, g_gmax, Neff_gmax = timeseries.detect_equilibration(g_max_history)
    #t0_fwhm, g_fwhm, Neff_fwhm = timeseries.detect_equilibration(fwhm_history)
    t0_msd, g_msd, Neff_msd = timeseries.detect_equilibration(msd)

    t0_per_metric = {
        "temperature": t0_T,
        "g_max": t0_gmax * window_size,
        #"fwhm": t0_fwhm * window_size,
        "msd": t0_msd,
    }
    g_per_metric = {"temperature": g_T, "g_max": g_gmax, "msd": g_msd}# "fwhm": g_fwhm, 
    Neff_per_metric = {
        "temperature": Neff_T, "g_max": Neff_gmax, "msd": Neff_msd #"fwhm": Neff_fwhm, 
    }
    t0_frames = list(t0_per_metric.values())

    # energy criterion only if the calculator attached to these atoms supports it
    # (true for atoms read straight from OUTCAR; not for a stripped extxyz file)
    have_energy = False
    energies = None
    try:
        energies = np.array([atoms.get_potential_energy() for atoms in atoms_list])
        t0_E, g_E, Neff_E = timeseries.detect_equilibration(energies)
        t0_per_metric["energy"] = t0_E
        g_per_metric["energy"] = g_E
        Neff_per_metric["energy"] = Neff_E
        t0_frames.append(t0_E)
        have_energy = True
    except Exception:
        print("No energies found! Skipping this when determining convergence")

    t0_consensus = max(t0_frames)

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
            "g_max_history": g_max_history,
            #"fwhm_history": fwhm_history,
            "msd": msd,
            "energies": energies,
            "time_steps": np.array(time_steps),
            "t0_per_metric": t0_per_metric,
            "g_per_metric": g_per_metric,
            "Neff_per_metric": Neff_per_metric,
            "t0_consensus": t0_consensus,
        }
        pickle_path = os.path.join(save_dir, f"convergence_{safe_name}.pickle")
        with open(pickle_path, "wb") as f:
            pickle.dump(metrics, f)
        print(f"Saved convergence metrics to {pickle_path}")

    if plot:

        # for debugging:
        # plt.plot(time_steps, peak_r_history)
        # plt.xlabel("frame")
        # plt.ylabel("RDF peak position")
        # plt.show()
        ################################

        x_T = np.arange(len(temperatures))
        x_msd = np.arange(len(msd))
        x_gmax = np.array(time_steps)
        #x_fwhm = np.array(time_steps)

        t0s = [t0_T, t0_gmax * window_size, t0_msd]# t0_fwhm * window_size, 
        data_list = [temperatures, g_max_history,msd]# fwhm_history, 
        x_list = [x_T, x_gmax,  x_msd]#x_fwhm,
        names = ["Temperature (K)", "g_max", "MSD (Å²)"] #"FWHM (Å)"

        if have_energy:
            t0s.insert(1, t0_E)
            data_list.insert(1, energies)
            x_list.insert(1, np.arange(len(energies)))
            names.insert(1, "Energy (eV)")

        fig, axs = plt.subplots(len(data_list), 1, figsize=(10, 2.4 * len(data_list)), sharex=True)
        for ax, x, data, name, t0 in zip(axs, x_list, data_list, names, t0s):
            ax.plot(x, data, lw=1)
            ax.axvline(t0, color="red", ls="--", label=f"{name} t0 = {t0}")
            ax.axvline(t0_consensus, color="black", ls=":", label=f"consensus t0 = {t0_consensus}")
            ax.set_ylabel(name)
            ax.legend()
        axs[-1].set_xlabel("Frame index")
        if label is not None:
            fig.suptitle(f"Convergence check: {label}")
        plt.tight_layout()

        if save_dir is not None:
            fig_path = os.path.join(save_dir, f"convergence_{safe_name}.png")
            fig.savefig(fig_path, dpi=150)
            print(f"Saved convergence plot to {fig_path}")

        plt.show()

    return t0_consensus, t0_per_metric, g_per_metric






def load_trajectories(
    #outcar_files,
    trajectories,
    train_region="all",
    start_frames=None,
    test_n_blocks=None,
    test_block_length=None,
    test_buffer=None,
    train_fraction=None,
    subsample_method=None,
    subsample_kwargs=None,
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
        {outcar_file: (t0_per_metric, g_per_metric)} for files using "auto".
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
    if subsample_method == "stride":
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
            t0, t0_per_metric, g_per_metric = check_convergence(
                frames,
                plot=plot_convergence,
                label=outcar_file,
                save_dir=convergence_plot_dir if plot_convergence else None,
            )
            t0 = int(t0)
            print(f"{outcar_file}: per-metric t0s = {t0_per_metric}")
            print(f"{outcar_file}: per-metric g's = {g_per_metric}")
            convergence_info[outcar_file] = (t0_per_metric, g_per_metric)
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

        #pool_idx = np.arange(t0, n)
        #pool_idx = pool_idx[~blocked[t0:]]

        pool_idx = np.arange(n)
        if train_region == "after":
            pool_idx = pool_idx[pool_idx >= t0]
        elif train_region == "before":
            pool_idx = pool_idx[pool_idx < t0]
        pool_idx = pool_idx[~blocked[pool_idx]]

        if fraction is not None:
            n_keep = int(round(fraction * len(pool_idx)))
            pool_idx = np.sort(rng.choice(pool_idx, size=n_keep, replace=False))

        if train_stride is not None:
            pool_idx = _subsample_stride(pool_idx, stride=train_stride, seed=seed + k)

        # consistency checks on indices
        assert len(np.intersect1d(pool_idx, test_idx)) == 0, "train and test indices overlap"
        assert len(np.intersect1d(pool_idx, buffer_idx)) == 0, "train and buffer indices overlap"
        #assert pool_idx.min() >= t0 if len(pool_idx) else True, "train frame before t0"
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