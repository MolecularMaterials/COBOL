""" Utility functions for generating inputs and analyzing results (e.g., Gaussian 16 outputs)"""
import os
from ase.io import read
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
from IPython.display import Image
import numpy as np

class GaussianInputGenerator:
    """
    Class to generate Gaussian input files from SMILES strings.

    Attributes
    ----------
    file_name : str
        The name of the Gaussian input file to be created, must end with '.com'.
    smiles : str
        The SMILES string of the molecule for which the input file is generated.
    charge : int
        The charge of the molecule.
    mult : int
        The multiplicity of the molecule.
    computational_method : str
        The computational method and basis set used in the Gaussian calculation.
    additional_keywords : str
        Additional keywords to be included in the Gaussian input file.

    Methods
    -------
    _smiles_to_3d_coords():
        Converts the SMILES string to 3D coordinates.
    generate_input_file():
        Generates the Gaussian input file.

    Usage
    -------
    input_generator = GaussianInputGenerator('ginput.com', 'C1=CC=CC=C1', 'b3lyp/6-31+G(d,p) scrf(PCM,solvent=generic,read)
    opt(calcFC,MaxCycles=100) scf(xqc,MaxConventional=200) freq', 0, 1)
    input_generator.generate_input_file()
    """

    def __init__(self, file_name, smiles, computational_method, additional_keywords='', charge=0, mult=1):
        self.file_name = file_name
        self.smiles = smiles
        self.computational_method = computational_method
        self.additional_keywords = additional_keywords
        self.charge = charge
        self.mult = mult

    def _smiles_to_3d_coords(self):
        mol = Chem.MolFromSmiles(self.smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        atoms = mol.GetAtoms()
        coordinates = [f"{atom.GetSymbol():<2} " + '  '.join(f"{coord:15.3f}" for coord in conf.GetAtomPosition(atom.GetIdx())) for atom in atoms]
        return "\n".join(coordinates)

    def generate_input_file(self):
        # Add error handling for incorrect file extension
        if not self.file_name.endswith('.com'):
            raise ValueError('Filename must end with .com extension')

        # Prepare Gaussian input file content
        coordinates = self._smiles_to_3d_coords()
        gaussian_input = f"""%mem=16GB
%chk={os.path.splitext(self.file_name)[0]}.chk
#p {self.computational_method}

{self.smiles} {self.charge} {self.mult}

{self.charge} {self.mult}
{coordinates}

{self.additional_keywords}

"""
        # Write the Gaussian input file
        with open(self.file_name, 'w') as f:
            f.write(gaussian_input)


class GaussianOutputAnalyzer:
    """
    Class to analyze Gaussian output files.

    Attributes
    ----------
    output_file : str
        Path to the Gaussian output file.

    Methods
    -------
    extract_homo_energy():
        Extracts the HOMO energy from the Gaussian output file.
    extract_lumo_energy():
        Extracts the LUMO energy from the Gaussian output file.
    extract_total_energy():
        Extracts the total energy from the Gaussian output file.
    extract_gibbs_free_energy():
        Extracts the Gibbs free energy from the Gaussian output file.
    """
    def __init__(self, output_file):
        self.output_file = output_file

    def _read_file(self):
        with open(self.output_file, 'r') as file:
            return file.readlines()

    def extract_homo_energy(self):
        lines = self._read_file()
        homo_E = None
        for line in lines:
            if 'Alpha  occ. eigenvalues' in line or 'Beta  occ. eigenvalues' in line:
                homo_E = float(line.split()[-1])
        return homo_E

    def extract_lumo_energy(self):
        lines = self._read_file()
        lumo_E = None
        homoOrb_before = False
        for line in lines:
            if 'Alpha  occ. eigenvalues' in line or 'Beta  occ. eigenvalues' in line:
                homoOrb_before = True
            if ('Alpha virt. eigenvalues' in line or 'Beta virt. eigenvalues' in line) and homoOrb_before:
                lumo_E = float(line.split()[4])
                homoOrb_before = False
        return lumo_E

    def extract_total_energy(self):
        lines = self._read_file()
        for line in lines:
            if 'SCF Done:' in line:
                return float(line.split()[4])
        return None

    def extract_gibbs_free_energy(self):
        lines = self._read_file()
        for line in lines:
            if 'Sum of electronic and thermal Free Energies=' in line:
                return float(line.split()[-1])
        return None

    def extract_charge_and_multiplicity(self):
        lines = self._read_file()
        for line in lines:
            if "Charge = " in line and "Multiplicity = " in line:
                parts = line.split()
                charge = int(parts[2])
                multiplicity = int(parts[5])
                return charge, multiplicity
        return None, None

def deprotonateSMILES(smiles):
    """
    Function to generate all possible deprotonated species given a SMILES input
    """
    # Store all unique deprotonated molecules as SMILES strings and atom index
    deprotonated_species = {}

    mol = Chem.MolFromSmiles(smiles)

    # Iterate over all atoms in the molecule
    for atom in mol.GetAtoms():
        # Check if the atom is a carbon, oxygen, or nitrogen and if it has at least one hydrogen
        if atom.GetAtomicNum() in [6, 7, 8] and atom.GetTotalNumHs() > 0:
            # Create a copy of the original molecule for deprotonation
            new_mol = Chem.Mol(mol)
            new_atom = new_mol.GetAtomWithIdx(atom.GetIdx())

            # Remove a hydrogen from the atom and adjust the formal charge
            new_atom.SetNumExplicitHs(new_atom.GetTotalNumHs() - 1)
            new_atom.SetFormalCharge(new_atom.GetFormalCharge() - 1)

            # Generate the SMILES string of the deprotonated molecule
            smiles = Chem.MolToSmiles(new_mol)

            # Check if the SMILES string is valid
            if Chem.MolFromSmiles(smiles) is not None and smiles not in deprotonated_species:
                # Add the valid SMILES string and atom index to the dictionary
                deprotonated_species[smiles] = atom.GetIdx()

    # Generate molecule objects from valid SMILES strings
    return deprotonated_species

def drawAndHighlightAtom(mol, atom_index=None):
    """
    Given a mol object and an atom index, draw the molecule and highlight that atom

    Return a drawer object that can be saved as an SVG image

    """
    # Initialize drawer
    drawer = rdMolDraw2D.MolDraw2DSVG(300,300)

    # Draw the molecule with highlighted atom
    drawer.drawOptions().addAtomIndices = False  # Optional: Display atom indices

    if atom_index!=None:
        drawer.DrawMolecule(mol, highlightAtoms=[atom_index])
    else:
        drawer.DrawMolecule(mol)

    drawer.FinishDrawing()

    # Save the SVG and display
    return drawer.GetDrawingText().replace('svg:','')

def calc_pKa(G_HA, G_A, G_proton=-273.1, T=298.15):
    """
    Calculate pKa value of acid HA using the direct deprotonation reaction:
    HA <=> H+ + A-
    The equation:
    pKa = [G(proton)+G(A-)-G(HA)]/[RTln(10)]
    where G are Gibbs free energies computed using DFT at temperature T. G(proton) is typically derived from experimental measurements.
    Units:
    G_HA, G_A: Hartree
    G_proton: kcal/mol
    """
    R = 1.987E-3 # kcal/molK
    hartree2kcalPerMol = 627.5
    G_proton = G_proton/hartree2kcalPerMol # kcal/mol to hartree

    return (G_A+G_proton-G_HA)*hartree2kcalPerMol/(R*T*np.log(10))

if __name__ == '__main__':
    smiles = 'FC1=C(F)C=CC=C1'
    ds = deprotonateSMILES(smiles)
    # Create a molecule from SMILES string
    mol = Chem.MolFromSmiles(smiles)

    # Add Hydrogens to the molecule
    mol = Chem.AddHs(mol)
    count=0
    for sml, deprotIdx in ds.items():
        new_mol = Chem.Mol(mol)
        new_atom = new_mol.GetAtomWithIdx(deprotIdx)
        # Get the index of the H atom connected to the atom with the deprotIdx
        for neighbor in new_atom.GetNeighbors():
            if neighbor.GetAtomicNum()==1: atom_index=neighbor.GetIdx()

        svg = drawAndHighlightAtom(mol,atom_index)
        with open('mol'+str(count)+'.svg', 'w') as f:
            f.write(svg)
        count+=1
