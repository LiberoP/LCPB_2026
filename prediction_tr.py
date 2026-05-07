import os
from ase.io import read
from mace.calculators import MACECalculator
from ase.md.langevin import Langevin
from ase import units
from ase.io.trajectory import Trajectory


DATASET_PATH = '/kaggle/input/datasets/niccoldepoli/mace-2/dataset_pulito.extxyz' 
MODEL_PATH = '/kaggle/input/datasets/niccoldepoli/mace-2/model_cristal.model'
TRAJ_OUTPUT = '/kaggle/working/traj_cristal.traj'

try:
    atoms = read(DATASET_PATH, index='-1')
except FileNotFoundError:
    print("ERROR")


calc = MACECalculator(model_paths=MODEL_PATH, device='cuda') #model
atoms.set_calculator(calc)

#Langevin Dinamic
dyn = Langevin(atoms, #input
               timestep=1.0 * units.fs, 
               temperature_K=300, 
               friction=0.01)

traj = Trajectory(TRAJ_OUTPUT, 'w', atoms)
dyn.attach(traj.write, interval=10) #1 step evry 10

print("MD:")
dyn.run(100) #just try
print(f"SSimulation ok: {TRAJ_OUTPUT}")