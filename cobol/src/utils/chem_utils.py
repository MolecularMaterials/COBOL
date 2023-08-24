from typing import Callable, List, Tuple

import pandas as pd
import rdkit as rdkit
import rdkit.Chem as Chem
import rdkit.Chem.AtomPairs.Pairs as Pairs
import rdkit.Chem.AllChem as AllChem
import rdkit.Chem.Fragments as Fragments
import rdkit.Chem.Crippen as Crippen
import rdkit.Chem.Lipinski as Lipinski
import rdkit.Chem.rdMolDescriptors as MolDescriptors
import rdkit.Chem.Descriptors as Descriptors
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
import numpy as np

# List of descriptors, populated once
_descriptor_list: List[Tuple[str, Callable]] = []

_descriptor_list.extend((name, getattr(Fragments, name)) for name in dir(Fragments) if "fr" in name)
_descriptor_list.extend((name, getattr(Lipinski, name)) for name in dir(Lipinski) if ("Count" in name and "Smarts" not in name))
_descriptor_list.extend((name, getattr(MolDescriptors, name)) for name in dir(MolDescriptors)
                        if (("CalcNum" in name or "CalcExact" in name or "CalcTPSA" in name or "CalcChi" in name or "Kappa" in name or "Labute" in name)
                            and "Stereo" not in name and "_" not in name and "ChiN" not in name))
_descriptor_list.extend((name, getattr(Descriptors, name)) for name in dir(Descriptors)
                        if ("Num" in name or "Min" in name or "Max" in name))
_descriptor_list.extend((name, getattr(Crippen, name)) for name in dir(Crippen)
                        if (("MolLogP" in name or "MolMR" in name) and "_" not in name))

# Remove duplicated descriptors
_descriptor_list = sorted(
    (name, func) for name, func in _descriptor_list
    if name not in ['CalcNumAromaticHeterocycles', 'CalcNumSaturatedHeterocycles', 'CalcNumSaturatedRings',
                    'CalcNumAromaticCarbocycles', 'CalcNumRings', 'CalcNumHeavyAtoms', 'CalcNumRotatableBonds',
                    'CalcNumAliphaticRings', 'CalcNumHeteroatoms', 'CalcNumAromaticRings', 'CalcNumAliphaticHeterocycles']
)

class FingerprintGenerator:
    ''' Generate the fingerprint for a molecule, given the fingerprint type
    Parameters: 
        mol (rdkit.Chem.rdchem.Mol) : RdKit mol object (None if invalid smile string smi)
        fp_type (string)            : Fingerprint type  (choices: AtomPair|Pharmacophore|RDK|ECFP4|ECFP6|FCFP4|FCFP6|PhysChemFeatures)  
    Returns:
        Bit vector (of size 2048 by default)
    '''

    def get_fingerprint(self, mol: Chem.rdchem.Mol, fp_type: str):
        method_name = 'get_' + fp_type
        method = getattr(self, method_name)
        if method is None:
            raise Exception(f'{fp_type} is not a supported fingerprint type.')
        return method(mol)
    
    def get_AtomPairs(self, mol: Chem.rdchem.Mol, numBits=2048):
        atom_pair_hashed = Pairs.GetHashedAtomPairFingerprint(mol, nBits=numBits)
        numpy_array = np.zeros(numBits, dtype=np.int32)
        for index, value in atom_pair_hashed.GetNonzeroElements().items():
            numpy_array[index] = value
        return np.array(['AtomPairs']), numpy_array

    def get_Pharmacophore(self, mol: Chem.rdchem.Mol):
        return np.array(['Pharmacophore']), np.array(Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory))

    def get_RDK(self, mol: Chem.rdchem.Mol):
        return np.array(['RDK']), np.array(AllChem.RDKFingerprint(mol))

    def get_ECFP4(self, mol: Chem.rdchem.Mol): 
        """ Morgan fingperint at radius equal to 2 """
        return np.array(['ECFP4']), np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2))

    def get_ECFP6(self, mol: Chem.rdchem.Mol):
        """ Morgan fingerprint at radius equal to 3 """
        return np.array(['ECFP6']), np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3))

    def get_FCFP4(self, mol: Chem.rdchem.Mol):
        return np.array(['FCFP4']), np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True))

    def get_FCFP6(self, mol: Chem.rdchem.Mol):
        return np.array(['FCFP6']), np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3, useFeatures=True))
	
    def get_PhysChemFeats(self, mol: Chem.rdchem.Mol):
        """
        Compute the set of physicochemical features used in Doan et al. Chem. Mat. 2020
        """
        descNameList = []
        descValList = []
        for name, attr in _descriptor_list:
            descNameList.append(name)
            try:
                descValList.append(attr(mol))
            except:
                descValList.append(0)   # assign 0 for non-SMILES, NaN, or None

        return np.array(descNameList), np.array(descValList)

def get_fingerprint(smiles: str, fp_type: str):

    ''' Fingerprint getter method. Fingerprint is returned after using object of 
        class 'FingerprintGenerator'
        
    Parameters: 
        mol (rdkit.Chem.rdchem.Mol) : RdKit mol object (None if invalid smile string smi)
        fp_type (string)            : Fingerprint type  (choices: AP/PHCO/BPF,BTF,PAT,ECFP4,ECFP6,FCFP4,FCFP6)  
    Returns:
        RDKit fingerprint object
        
    '''
    return FingerprintGenerator().get_fingerprint(mol=Chem.MolFromSmiles(smiles), fp_type=fp_type)

def Smiles2Fingerprint(smilesList, fp_type='PhysChemFeats') -> pd.DataFrame:
    """
    Input: List of SMILES
    Output: Dataframe of molecular descriptors
    """
    
    SmilesDescriptors = [get_fingerprint(sml,fp_type) for sml in smilesList]
    descValues = [list(l[1]) for l in SmilesDescriptors]
    if fp_type=='PhysChemFeats':
        descNames = SmilesDescriptors[0][0]
    else:
        descNames = np.arange(len(descValues[0]))
        print(descNames)
    dfFeatures = pd.DataFrame(descValues, columns=descNames)
    return dfFeatures

if __name__ == '__main__':
    smlList = ['[R]','COC','C','CO']
    desc, fp = get_fingerprint(smlList[1], 'PhysChemFeats')
    #fp = Smiles2Fingerprint(smlList,'PhysChemFeats')
    print(desc, fp)
