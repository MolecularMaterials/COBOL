def smiles2dotcom(list_of_smiles,method="wB97XD",basis="CEP-31G",charge=0,mult=1,fileName="g16_input"):
    """ 
    This function will take a list (or a Pandas Series) of SMILES and generate *.com files (in the current folder) for all the smiles in the list.
    
    For single SMILES string: smiles_to_dotcom(["CC"])
    """
    import os
    import numpy as np
    import rdkit.Chem as Chem
    from rdkit.Chem import AllChem    

    def get_atoms(mol):
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        return atoms
    
    def generate_structure_from_smiles(smiles):
        # Generate a 3D structure from smiles
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        status = AllChem.EmbedMolecule(mol)
        status = AllChem.MMFFOptimizeMolecule(mol) #UFFOptimizeMolecule(mol)
        conformer = mol.GetConformer()
        coordinates = conformer.GetPositions()
        coordinates = np.array(coordinates)
        atoms = get_atoms(mol)
        return atoms, coordinates

    def mkGaussianInputScriptNeutral(comName,method,basis, fragment,atoms,coordinates,charge,mult):
        file = open(comName+".com", 'w')
        file.write("""%mem=16GB \n""")
        file.write("""%Chk="""+comName+""".chk \n""")
        file.write("""#p """+ method +"""/""" + basis+ " opt(MaxCycles=200) scf(xqc,MaxConventional=200) Freq scrf(cpcm,solvent=acetonitrile)"""+ """\n\n""")
        file.write(fragment + " " + str(charge) + " " + str(mult)+"\n\n")
        file.write(str(charge)+""" """+str(mult)+"""\n""")
        for i,atom in enumerate(atoms):
            file.write(str(atom) + "\t"+str(coordinates[i][0]) + "\t\t"+str(coordinates[i][1]) + "\t\t"+str(coordinates[i][2]) + "\n")
        file.write("\n")
        file.close()   

    for i,smilesName in enumerate(list_of_smiles):
        atoms,coordinates=generate_structure_from_smiles(smilesName)
        fileNameNeutral = fileName + "-" + method +"-"+str(i+1)
        mkGaussianInputScriptNeutral(fileNameNeutral,method,basis,smilesName, atoms, coordinates,charge, mult)
    
    print("Files generated in: ", os.getcwd())
