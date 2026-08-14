import ase.io
from ase.io import read, write
import numpy as np


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


# splits data in train and validation for MACE
def split_train_valid(
    full_cleaned_extxyz_file,
    destination_train_file,
    destination_valid_file,
    train_frac,
    seed=1,
):
    traj = ase.io.read(full_cleaned_extxyz_file, index=":")
    print(f"Total frames uploaded: {len(traj)}")

    rng = np.random.RandomState(seed)
    indices = np.arange(len(traj))
    rng.shuffle(indices)
    traj = [traj[i] for i in indices]

    split = int(train_frac * len(traj))
    train_frames = traj[:split]
    valid_frames = traj[split:]

    ase.io.write(destination_train_file, train_frames)
    ase.io.write(destination_valid_file, valid_frames)

    print(
        f"Saved {len(train_frames)} frames in {destination_train_file} and {len(valid_frames)} in {destination_valid_file}"
    )

    return 0
