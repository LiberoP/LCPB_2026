# reads local OUTCAR file for analysis (see "general analysis" notebook)

from ase.io import read

try:
    atoms_list = read('./data/OUTCAR', index=':')
    print(f"Retriver {len(atoms_list)} structures from OUTCAR.")
    #usable file
    from ase.io import write
    file_path='./data/cleaned_dataset.extxyz'
    write(file_path, atoms_list)
    print(f"File saved as {file_path}")
except Exception as e:
    print(f"Error reading OUTCAR: {e}")