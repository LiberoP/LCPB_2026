import ase.io
import numpy as np

traj = ase.io.read('dataset_pulito.extxyz', index=':')
print(f"Totale frame caricati: {len(traj)}")

np.random.shuffle(traj)

split = int(0.9 * len(traj))
train_frames = traj[:split]
valid_frames = traj[split:]

ase.io.write('train.extxyz', train_frames)
ase.io.write('valid.extxyz', valid_frames)

print(f"Save {len(train_frames)} frame in train.extxyz e {len(valid_frames)} in valid.extxyz")
