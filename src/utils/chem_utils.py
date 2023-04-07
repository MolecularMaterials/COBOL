import pandas as pd
import rdkit as rdkit
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint, GetBTFingerprint
import rdkit.Chem as Chem
import rdkit.Chem.AtomPairs.Pairs as Pairs
import rdkit.Chem.AllChem as AllChem
import rdkit.Chem.Fragments as Fragments
import rdkit.Chem.Crippen as Crippen
import rdkit.Chem.Lipinski as Lipinski
import rdkit.Chem.rdMolDescriptors as MolDescriptors
import rdkit.Chem.Descriptors as Descriptors
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem.rdmolops import FastFindRings
import numpy as np

class FingerprintGenerator:
    ''' Generate the fingerprint for a molecule, given the fingerprint type
    Parameters: 
        mol (rdkit.Chem.rdchem.Mol) : RdKit mol object (None if invalid smile string smi)
        fp_type (string)            : Fingerprint type  (choices: AtomPair|Pharmacophore|ECFP4|ECFP6|FCFP4|FCFP6)  
    Returns:
        Bit vector (of size 2048 by default)
    '''

    def get_fingerprint(self, mol: Chem.rdchem.Mol, fp_type: str):
        method_name = 'get_' + fp_type
        method = getattr(self, method_name)
        if method is None:
            raise Exception(f'{fp_type} is not a supported fingerprint type.')
        return method(mol)

    def get_AtomPair(self, mol: Chem.rdchem.Mol):
        return np.array(Pairs.GetAtomPairFingerprintAsBitVect(mol))

    def get_Pharmacophore(self, mol: Chem.rdchem.Mol):
        return np.array(Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory))

    def get_BPF(self, mol: Chem.rdchem.Mol):
        return np.araay(GetBPFingerprint(mol))

    def get_BTF(self, mol: Chem.rdchem.Mol):
        return np.array(GetBTFingerprint(mol))

    def get_RDK(self, mol: Chem.rdchem.Mol):
        return np.array(AllChem.RDKFingerprint(mol))

    def get_ECFP4(self, mol: Chem.rdchem.Mol):
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2))

    def get_ECFP6(self, mol: Chem.rdchem.Mol):
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3))

    def get_FCFP4(self, mol: Chem.rdchem.Mol):
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True))

    def get_FCFP6(self, mol: Chem.rdchem.Mol):
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3, useFeatures=True))

def get_fingerprint(mol: Chem.rdchem.Mol, fp_type: str):
    ''' Fingerprint getter method. Fingerprint is returned after using object of 
        class 'FingerprintGenerator'
        
    Parameters: 
        mol (rdkit.Chem.rdchem.Mol) : RdKit mol object (None if invalid smile string smi)
        fp_type (string)            : Fingerprint type  (choices: AP/PHCO/BPF,BTF,PAT,ECFP4,ECFP6,FCFP4,FCFP6)  
    Returns:
        RDKit fingerprint object
        
    '''
    return FingerprintGenerator().get_fingerprint(mol=mol, fp_type=fp_type)

def get_RDKitFeatures(smiles: str) -> np.ndarray:
    """
    Input: SMILES

    Output: An 1D array of molecular descriptors from
    'Quantum Chemistry-Informed Active Learning to Accelerate the Design and Discovery of Sustainable Energy Storage Materials'
    Doan et al.
    """
    mol = Chem.MolFromSmarts(smiles,False)
    mol.UpdatePropertyCache()
    FastFindRings(mol)
    descNameList = []
    descValList = []
    funcList = [MolDescriptors, Descriptors, Lipinski, Crippen, Fragments]
    for func in funcList:
        if func == Fragments:
            attrs = [getattr(func, name) for name in dir(func) if "fr" in name]
            funcNames = [name for name in dir(func) if "fr" in name]
        if func == Lipinski:
            attrs = [getattr(func, name) for name in dir(func) if ("Count" in name and "Smarts" not in name)]
            funcNames = [name for name in dir(func) if ("Count" in name and "Smarts" not in name)]
        if func == MolDescriptors:
            attrs = [getattr(func, name) for name in dir(func) 
            if (("CalcNum" in name or "CalcExact" in name or "CalcTPSA" in name or "CalcChi" in name or "Kappa" in name or "Labute" in name) 
            and "Stereo" not in name and "_" not in name and "ChiN" not in name)]
            funcNames = [name for name in dir(func) 
            if (("CalcNum" in name or "CalcExact" in name or "CalcTPSA" in name or "CalcChi" in name or "Kappa" in name or "Labute" in name) 
            and "Stereo" not in name and "_" not in name and "ChiN" not in name)]
        if func == Descriptors:
            attrs = [getattr(func, name) for name in dir(func) if ("Num" in name or "Min" in name or "Max" in name)]
            funcNames = [name for name in dir(func) if ("Num" in name or "Min" in name or "Max" in name)]
        if func == Crippen:
            attrs = [getattr(func, name) for name in dir(func) if (("MolLogP" in name or "MolMR" in name) and "_" not in name)]
            funcNames = [name for name in dir(func) if (("MolLogP" in name or "MolMR" in name) and "_" not in name)]
        for name, attr in zip(funcNames,attrs):
            descValList.append(attr(mol))
            descNameList.append(name)
    return np.array(descNameList), np.array(descValList)

def Smiles2Fingerprint(smilesList) -> pd.DataFrame:
    """
    Input: List of SMILES
    Output: Dataframe of molecular descriptors
    """
    
    SmilesDescriptors = [get_RDKitFeatures(sml) for sml in smilesList]
    descValues = [list(l[1]) for l in SmilesDescriptors]
    descNames = SmilesDescriptors[0][0]
    dfFeatures = pd.DataFrame(descValues, columns=descNames)

    return dfFeatures

if __name__ == '__main__':
    sml = 'CCOCCO'
    fp = get_fingerprint(Chem.MolFromSmiles(sml),'ECFP4')
    print(fp)
    print(len(fp))