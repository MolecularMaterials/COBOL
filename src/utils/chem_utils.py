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
        Features used in Doan et al. Chem. Mat. 2020
        """
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
    #sml = 'O'
    smlList = ['O','COC']
    #_, fp = get_fingerprint(sml, 'AtomPairs')
    fp = Smiles2Fingerprint(smlList,'ECFP6')
    print(fp)
    print(len(fp))