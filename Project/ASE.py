from ase.io import read

try:
    atoms_list = read('OUTCAR', index=':')
    print(f"Retriver {len(atoms_list)} structures from OUTCAR.")
    #usable file
    from ase.io import write
    write('dataset_pulito1.extxyz', atoms_list)
    print("File saved as dataset_pulito1.extxyz")
except Exception as e:
    print(f"Error reading OUTCAR: {e}")


#from ase.io import write
#('database.extxyz', atoms_list)

#################################################
#from dscribe.descriptors import SOAP

#atoms = read('dataset_pulito.extxyz', index=0)

#soap = SOAP(
    #species=["H", "C", "N", "Sn", "I"],  
    #periodic=True,        
    #r_cut=5.0,            
    #n_max=8,               
    #l_max=6,              
#)

#soap_vector = soap.create(atoms)
#print(f"Dimension vector SOAP: {soap_vector.shape}")
