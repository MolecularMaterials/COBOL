import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import rdkit.Chem.Fragments as Fragments
from copy import copy

df_BTZ = pd.read_csv("../../BTZ_1400Mols_AllProps.csv")

#df_BTZ['WLtargetDiff'] = df_BTZ['Absorption Wavelength']-targetWL
#df_BTZ['WLtarget'] = abs(df_BTZ['Absorption Wavelength']-targetWL)
#df_BTZ['normWLtarget'] = (df_BTZ['WLtarget']-min(df_BTZ['WLtarget']))/(max(df_BTZ['WLtarget'])-min(df_BTZ['WLtarget']))
#df_BTZ['normGsol'] = (df_BTZ['Gsol']-min(df_BTZ['Gsol']))/(max(df_BTZ['Gsol'])-min(df_BTZ['Gsol']))
#df_BTZ['normEred'] = (df_BTZ['Ered']-min(df_BTZ['Ered']))/(max(df_BTZ['Ered'])-min(df_BTZ['Ered']))

import rdkit.Chem as Chem
import rdkit.Chem.Crippen as Crippen
import rdkit.Chem.Lipinski as Lipinski
import rdkit.Chem.rdMolDescriptors as MolDescriptors
import rdkit.Chem.Descriptors as Descriptors

def evaluate_chem_mol(mol):
    mol_sssr = Chem.GetSSSR(mol)
#    return Chem.GetSSSR(mol), Crippen.MolLogP(mol)
    clogp    = Crippen.MolLogP(mol)
    mr       = Crippen.MolMR(mol)
    mw       = MolDescriptors.CalcExactMolWt(mol)
    tpsa    = MolDescriptors.CalcTPSA(mol)
    Chi0n    = MolDescriptors.CalcChi0n(mol)
    Chi1n    = MolDescriptors.CalcChi1n(mol)
    Chi2n    = MolDescriptors.CalcChi2n(mol)
    Chi3n    = MolDescriptors.CalcChi3n(mol)
    Chi4n    = MolDescriptors.CalcChi4n(mol)
    Chi0v    = MolDescriptors.CalcChi0v(mol)
    Chi1v    = MolDescriptors.CalcChi1v(mol)
    Chi2v    = MolDescriptors.CalcChi2v(mol)
    Chi3v    = MolDescriptors.CalcChi3v(mol)
    Chi4v    = MolDescriptors.CalcChi4v(mol)
    fracsp3  = MolDescriptors.CalcFractionCSP3(mol)
    Hall_Kier_Alpha = MolDescriptors.CalcHallKierAlpha(mol)    
    Kappa1      = MolDescriptors.CalcKappa1(mol)
    Kappa2      = MolDescriptors.CalcKappa2(mol)
    Kappa3      = MolDescriptors.CalcKappa3(mol)    
    LabuteASA   = MolDescriptors.CalcLabuteASA(mol)  
    Number_Aliphatic_Rings = MolDescriptors.CalcNumAliphaticRings(mol)   
    Number_Aromatic_Rings = MolDescriptors.CalcNumAromaticRings(mol) 
    Number_Amide_Bonds = MolDescriptors.CalcNumAmideBonds(mol) 
    Number_Atom_Stereocenters = MolDescriptors.CalcNumAtomStereoCenters(mol)
    Number_BridgeHead_Atoms = MolDescriptors.CalcNumBridgeheadAtoms(mol)
    Number_HBA = MolDescriptors.CalcNumHBA(mol)
    Number_HBD = MolDescriptors.CalcNumHBD(mol)    
    Number_Hetero_Atoms = MolDescriptors.CalcNumHeteroatoms(mol)
    Number_Hetero_Cycles = MolDescriptors.CalcNumHeterocycles(mol)
    Number_Rings = MolDescriptors.CalcNumRings(mol)
    Number_Rotatable_Bonds = MolDescriptors.CalcNumRotatableBonds(mol)    
    Number_Spiro = MolDescriptors.CalcNumSpiroAtoms(mol)
    Number_Saturated_Rings = MolDescriptors.CalcNumSaturatedRings(mol)
    Number_Heavy_Atoms = Lipinski.HeavyAtomCount(mol)
    Number_NH_OH = Lipinski.NHOHCount(mol)
    Number_N_O = Lipinski.NOCount(mol)
    Number_Valence_Electrons = Descriptors.NumValenceElectrons(mol)
    Max_Partial_Charge = Descriptors.MaxPartialCharge(mol)
    Min_Partial_Charge = Descriptors.MinPartialCharge(mol)
    
    return mol_sssr, clogp, mr, mw, tpsa, Chi0n, Chi1n, Chi2n, Chi3n, Chi4n, Chi0v, Chi1v, Chi2v, Chi3v, Chi4v, fracsp3,\
           Hall_Kier_Alpha,Kappa1, Kappa2, Kappa3, LabuteASA, Number_Aliphatic_Rings, Number_Aromatic_Rings, \
           Number_Amide_Bonds, Number_Atom_Stereocenters, Number_BridgeHead_Atoms, Number_HBA, Number_HBD, \
           Number_Hetero_Atoms, Number_Hetero_Cycles, Number_Rings, Number_Rotatable_Bonds, Number_Spiro,\
           Number_Saturated_Rings, Number_Heavy_Atoms, Number_NH_OH, Number_N_O, Number_Valence_Electrons,\
           Max_Partial_Charge, Min_Partial_Charge
sssr=[]    
clogp=[]
mr=[]
mw=[]
tpsa=[]
chi0n=[]
chi1n=[]
chi2n=[]
chi3n=[]
chi4n=[]
chi0v=[]
chi1v=[]
chi2v=[]
chi3v=[]
chi4v=[]
fracsp3=[]
hall_kier_alpha=[]
kappa1=[]
kappa2=[]
kappa3=[]
labuteasa=[]
number_aliphatic_rings=[]
number_aromatic_rings=[]
number_amide_bonds=[]
number_atom_stereocenters=[]
number_bridgehead_atoms=[]
number_HBA=[]
number_HBD=[]
number_hetero_atoms=[]
number_hetero_cycles=[]
number_rings=[]
number_rotatable_bonds=[]
number_spiro=[]
number_saturated_rings=[]
number_heavy_atoms=[]
number_nh_oh=[]
number_n_o=[]
number_valence_electrons=[]
max_partial_charge=[]
min_partial_charge=[]

fr_C_O = []
fr_C_O_noCOO = []
fr_Al_OH = []
fr_Ar_OH = []
fr_methoxy = []
fr_oxime = []
fr_ester = []
fr_Al_COO = []
fr_Ar_COO = []
fr_COO = []
fr_COO2 = []
fr_ketone = []
fr_ether = []
fr_phenol = []
fr_aldehyde = []
fr_quatN = []
fr_NH2 = []
fr_NH1 = []
fr_NH0 = []
fr_Ar_N = []
fr_Ar_NH = []
fr_aniline = []
fr_Imine = []
fr_nitrile = []
fr_hdrzine = []
fr_hdrzone = []
fr_nitroso = []
fr_N_O = []
fr_nitro = []
fr_azo = []
fr_diazo = []
fr_azide = []
fr_amide = []
fr_priamide = []
fr_amidine = []
fr_guanido = []
fr_Nhpyrrole = []
fr_imide = []
fr_isocyan = []
fr_isothiocyan = []
fr_thiocyan = []
fr_halogen = []
fr_alkyl_halide = []
fr_sulfide = []
fr_SH = []
fr_C_S = []
fr_sulfone = []
fr_sulfonamd = []
fr_prisulfonamd = []
fr_barbitur = []
fr_urea = []
fr_term_acetylene = []
fr_imidazole = []
fr_furan = []
fr_thiophene = []
fr_thiazole = []
fr_oxazole = []
fr_pyridine = []
fr_piperdine = []
fr_piperzine = []
fr_morpholine = []
fr_lactam = []
fr_lactone = []
fr_tetrazole = []
fr_epoxide = []
fr_unbrch_alkane = []
fr_bicyclic = []
fr_benzene = []
fr_phos_acid = []
fr_phos_ester = []
fr_nitro_arom = []
fr_nitro_arom_nonortho = []
fr_dihydropyridine = []
fr_phenol_noOrthoHbond = []
fr_Al_OH_noTert = []
fr_benzodiazepine = []
fr_para_hydroxylation = []
fr_allylic_oxid = []
fr_aryl_methyl = []
fr_Ndealkylation1 = []
fr_Ndealkylation2 = []
fr_alkyl_carbamate = []
fr_ketone_Topliss = []
fr_ArN = []
fr_HOCCN = []

for f in df_BTZ['SMILES']:
    f1=Chem.MolFromSmiles(f)
    mol_sssr, mol_clogp, mol_mr, mol_mw, mol_tpsa, mol_chi0n, mol_chi1n, mol_chi2n, mol_chi3n, mol_chi4n, mol_chi0v,\
    mol_chi1v, mol_chi2v, mol_chi3v, mol_chi4v, mol_fracsp3, mol_hall_kier_alpha, mol_kappa1, mol_kappa2,\
    mol_kappa3, mol_labuteasa, mol_number_aliphatic_rings, mol_number_aromatic_rings, mol_number_amide_bonds,\
    mol_number_atom_stereocenters, mol_number_bridgehead_atoms, mol_number_HBA, mol_number_HBD, \
    mol_number_hetero_atoms, mol_number_hetero_cycles, mol_number_rings, mol_number_rotatable_bonds, \
    mol_number_spiro, mol_number_saturated_rings, mol_number_heavy_atoms, mol_number_nh_oh, \
    mol_number_n_o, mol_number_valence_electrons, mol_max_partial_charge, mol_min_partial_charge= evaluate_chem_mol(f1)
    
    sssr.append(mol_sssr)
    clogp.append(mol_clogp) 
    mr.append(mol_mr)
    mw.append(mol_mw)
    tpsa.append(mol_tpsa)
    chi0n.append(mol_chi0n)
    chi1n.append(mol_chi1n)
    chi2n.append(mol_chi2n)
    chi3n.append(mol_chi3n)
    chi4n.append(mol_chi4n)
    chi0v.append(mol_chi0v)
    chi1v.append(mol_chi1v)
    chi2v.append(mol_chi2v)
    chi3v.append(mol_chi3v)
    chi4v.append(mol_chi4v)
    fracsp3.append(mol_fracsp3)
    hall_kier_alpha.append(mol_hall_kier_alpha)
    kappa1.append(mol_kappa1)
    kappa2.append(mol_kappa2)
    kappa3.append(mol_kappa3)
    labuteasa.append(mol_labuteasa)
    number_aliphatic_rings.append(mol_number_aliphatic_rings)
    number_aromatic_rings.append(mol_number_aromatic_rings)
    number_amide_bonds.append(mol_number_amide_bonds)
    number_atom_stereocenters.append(mol_number_atom_stereocenters)
    number_bridgehead_atoms.append(mol_number_bridgehead_atoms)
    number_HBA.append(mol_number_HBA)
    number_HBD.append(mol_number_HBD)
    number_hetero_atoms.append(mol_number_hetero_atoms)
    number_hetero_cycles.append(mol_number_hetero_cycles)
    number_rings.append(mol_number_rings)
    number_rotatable_bonds.append(mol_number_rotatable_bonds)
    number_spiro.append(mol_number_spiro)
    number_saturated_rings.append(mol_number_saturated_rings)
    number_heavy_atoms.append(mol_number_heavy_atoms)
    number_nh_oh.append(mol_number_nh_oh)
    number_n_o.append(mol_number_n_o)
    number_valence_electrons.append(mol_number_valence_electrons)
    max_partial_charge.append(mol_max_partial_charge)
    min_partial_charge.append(mol_min_partial_charge)
    
    fr_C_O.append(Fragments.fr_C_O(f1))
    fr_C_O_noCOO.append(Fragments.fr_C_O_noCOO(f1))
    fr_Al_OH.append(Fragments.fr_Al_OH(f1))
    fr_Ar_OH.append(Fragments.fr_Ar_OH(f1))
    fr_methoxy.append(Fragments.fr_methoxy(f1))
    fr_oxime.append(Fragments.fr_oxime(f1))
    fr_ester.append(Fragments.fr_ester(f1))
    fr_Al_COO.append(Fragments.fr_Al_COO(f1))
    fr_Ar_COO.append(Fragments.fr_Ar_COO(f1))
    fr_COO.append(Fragments.fr_COO(f1))
    fr_COO2.append(Fragments.fr_COO2(f1))
    fr_ketone.append(Fragments.fr_ketone(f1))
    fr_ether.append(Fragments.fr_ether(f1))
    fr_phenol.append(Fragments.fr_phenol(f1))
    fr_aldehyde.append(Fragments.fr_aldehyde(f1))
    fr_quatN.append(Fragments.fr_quatN(f1))
    fr_NH2.append(Fragments.fr_NH2(f1))
    fr_NH1.append(Fragments.fr_NH1(f1))
    fr_NH0.append(Fragments.fr_NH0(f1))
    fr_Ar_N.append(Fragments.fr_Ar_N(f1))
    fr_Ar_NH.append(Fragments.fr_Ar_NH(f1))
    fr_aniline.append(Fragments.fr_aniline(f1))
    fr_Imine.append(Fragments.fr_Imine(f1))
    fr_nitrile.append(Fragments.fr_nitrile(f1))
    fr_hdrzine.append(Fragments.fr_hdrzine(f1))
    fr_hdrzone.append(Fragments.fr_hdrzone(f1))
    fr_nitroso.append(Fragments.fr_nitroso(f1))
    fr_N_O.append(Fragments.fr_N_O(f1))
    fr_nitro.append(Fragments.fr_nitro(f1))
    fr_azo.append(Fragments.fr_azo(f1))
    fr_diazo.append(Fragments.fr_diazo(f1))
    fr_azide.append(Fragments.fr_azide(f1))
    fr_amide.append(Fragments.fr_amide(f1))
    fr_priamide.append(Fragments.fr_priamide(f1))
    fr_amidine.append(Fragments.fr_amidine(f1))
    fr_guanido.append(Fragments.fr_guanido(f1))
    fr_Nhpyrrole.append(Fragments.fr_Nhpyrrole(f1))
    fr_imide.append(Fragments.fr_imide(f1))
    fr_isocyan.append(Fragments.fr_isocyan(f1))
    fr_isothiocyan.append(Fragments.fr_isothiocyan(f1))
    fr_thiocyan.append(Fragments.fr_thiocyan(f1))
    fr_halogen.append(Fragments.fr_halogen(f1))
    fr_alkyl_halide.append(Fragments.fr_alkyl_halide(f1))
    fr_sulfide.append(Fragments.fr_sulfide(f1))
    fr_SH.append(Fragments.fr_SH(f1))
    fr_C_S.append(Fragments.fr_C_S(f1))
    fr_sulfone.append(Fragments.fr_sulfone(f1))
    fr_sulfonamd.append(Fragments.fr_sulfonamd(f1))
    fr_prisulfonamd.append(Fragments.fr_prisulfonamd(f1))
    fr_barbitur.append(Fragments.fr_barbitur(f1))
    fr_urea.append(Fragments.fr_urea(f1))
    fr_term_acetylene.append(Fragments.fr_term_acetylene(f1))
    fr_imidazole.append(Fragments.fr_imidazole(f1))
    fr_furan.append(Fragments.fr_furan(f1))
    fr_thiophene.append(Fragments.fr_thiophene(f1))
    fr_thiazole.append(Fragments.fr_thiazole(f1))
    fr_oxazole.append(Fragments.fr_oxazole(f1))
    fr_pyridine.append(Fragments.fr_pyridine(f1))
    fr_piperdine.append(Fragments.fr_piperdine(f1))
    fr_piperzine.append(Fragments.fr_piperzine(f1))
    fr_morpholine.append(Fragments.fr_morpholine(f1))
    fr_lactam.append(Fragments.fr_lactam(f1))
    fr_lactone.append(Fragments.fr_lactone(f1))
    fr_tetrazole.append(Fragments.fr_tetrazole(f1))
    fr_epoxide.append(Fragments.fr_epoxide(f1))
    fr_unbrch_alkane.append(Fragments.fr_unbrch_alkane(f1))
    fr_bicyclic.append(Fragments.fr_bicyclic(f1))
    fr_benzene.append(Fragments.fr_benzene(f1))
    fr_phos_acid.append(Fragments.fr_phos_acid(f1))
    fr_phos_ester.append(Fragments.fr_phos_ester(f1))
    fr_nitro_arom.append(Fragments.fr_nitro_arom(f1))
    fr_nitro_arom_nonortho.append(Fragments.fr_nitro_arom_nonortho(f1))
    fr_dihydropyridine.append(Fragments.fr_dihydropyridine(f1))
    fr_phenol_noOrthoHbond.append(Fragments.fr_phenol_noOrthoHbond(f1))
    fr_Al_OH_noTert.append(Fragments.fr_Al_OH_noTert(f1))
    fr_benzodiazepine.append(Fragments.fr_benzodiazepine(f1))
    fr_para_hydroxylation.append(Fragments.fr_para_hydroxylation(f1))
    fr_allylic_oxid.append(Fragments.fr_allylic_oxid(f1))
    fr_aryl_methyl.append(Fragments.fr_aryl_methyl(f1))
    fr_Ndealkylation1.append(Fragments.fr_Ndealkylation1(f1))
    fr_Ndealkylation2.append(Fragments.fr_Ndealkylation2(f1))
    fr_alkyl_carbamate.append(Fragments.fr_alkyl_carbamate(f1))
    fr_ketone_Topliss.append(Fragments.fr_ketone_Topliss(f1))
    fr_ArN.append(Fragments.fr_ArN(f1))
    fr_HOCCN.append(Fragments.fr_HOCCN(f1))

df_Solvent_Features=pd.DataFrame(
   {'sssr':sssr,
    'clogp':clogp,
    'mr':mr,
    'mw':mw,
    'tpsa': tpsa,
    'chi0n':chi0n,
    'chi1n':chi1n,
    'chi2n':chi2n,
    'chi3n':chi3n,
    'chi4n':chi4n,
    'chi0v':chi0v,
    'chi1v':chi1v,
    'chi2v':chi2v,
    'chi3v':chi3v,
    'chi4v':chi4v,
    'fracsp3':fracsp3,
    'hall_kier_alpha':hall_kier_alpha,
    'kappa1':kappa1,
    'kappa2':kappa2,
    'kappa3':kappa3,
    'labuteasa':labuteasa,
    'number_aliphatic_rings':number_aliphatic_rings,
    'number_aromatic_rings':number_aromatic_rings,
    'number_amide_bonds':number_amide_bonds,
    'number_atom_stereocenters':number_atom_stereocenters,
    'number_bridgehead_atoms':number_bridgehead_atoms,
    'number_HBA':number_HBA,
    'number_HBD':number_HBD,
    'number_hetero_atoms':number_hetero_atoms,
    'number_hetero_cycles':number_hetero_cycles,
    'number_rings':number_rings,
    'number_rotatable_bonds':number_rotatable_bonds,
    'number_spiro':number_spiro,
    'number_saturated_rings':number_saturated_rings,
    'number_heavy_atoms':number_heavy_atoms,
    'number_nh_oh':number_nh_oh,
    'number_n_o':number_n_o,
    'number_valence_electrons':number_valence_electrons,
    'max_partial_charge':max_partial_charge,
    'min_partial_charge':min_partial_charge
   })

df_Solvent_Features_Frags=pd.DataFrame(
    {'fr_C_O':fr_C_O,
    'fr_C_O_noCOO':fr_C_O_noCOO,
    'fr_Al_OH':fr_Al_OH,
    'fr_Ar_OH':fr_Ar_OH,
    'fr_methoxy':fr_methoxy,
    'fr_oxime':fr_oxime,
    'fr_ester':fr_ester,
    'fr_Al_COO':fr_Al_COO,
    'fr_Ar_COO':fr_Ar_COO,
    'fr_COO':fr_COO,
    'fr_COO2':fr_COO2,
    'fr_ketone':fr_ketone,
    'fr_ether':fr_ether,
    'fr_phenol':fr_phenol,
    'fr_aldehyde':fr_aldehyde,
    'fr_quatN':fr_quatN,
    'fr_NH2':fr_NH2,
    'fr_NH1':fr_NH1,
    'fr_NH0':fr_NH0,
    'fr_Ar_N':fr_Ar_N,
    'fr_Ar_NH':fr_Ar_NH,
    'fr_aniline':fr_aniline,
    'fr_Imine':fr_Imine,
    'fr_nitrile':fr_nitrile,
    'fr_hdrzine':fr_hdrzine,
    'fr_hdrzone':fr_hdrzone,
    'fr_nitroso':fr_nitroso,
    'fr_N_O':fr_N_O,
    'fr_nitro':fr_nitro,
    'fr_azo':fr_azo,
    'fr_diazo':fr_diazo,
    'fr_azide':fr_azide,
    'fr_amide':fr_amide,
    'fr_priamide':fr_priamide,
    'fr_amidine':fr_amidine,
    'fr_guanido':fr_guanido,
    'fr_Nhpyrrole':fr_Nhpyrrole,
    'fr_imide':fr_imide,
    'fr_isocyan':fr_isocyan,
    'fr_isothiocyan':fr_isothiocyan,
    'fr_thiocyan':fr_thiocyan,
    'fr_halogen':fr_halogen,
    'fr_alkyl_halide':fr_alkyl_halide,
    'fr_sulfide':fr_sulfide,
    'fr_SH':fr_SH,
    'fr_C_S':fr_C_S,
    'fr_sulfone':fr_sulfone,
    'fr_sulfonamd':fr_sulfonamd,
    'fr_prisulfonamd':fr_prisulfonamd,
    'fr_barbitur':fr_barbitur,
    'fr_urea':fr_urea,
    'fr_term_acetylene':fr_term_acetylene,
    'fr_imidazole':fr_imidazole,
    'fr_furan':fr_furan,
    'fr_thiophene':fr_thiophene,
    'fr_thiazole':fr_thiazole,
    'fr_oxazole':fr_oxazole,
    'fr_pyridine':fr_pyridine,
    'fr_piperdine':fr_piperdine,
    'fr_piperzine':fr_piperzine,
    'fr_morpholine':fr_morpholine,
    'fr_lactam':fr_lactam,
    'fr_lactone':fr_lactone,
    'fr_tetrazole':fr_tetrazole,
    'fr_epoxide':fr_epoxide,
    'fr_unbrch_alkane':fr_unbrch_alkane,
    'fr_bicyclic':fr_bicyclic,
    'fr_benzene':fr_benzene,
    'fr_phos_acid':fr_phos_acid,
    'fr_phos_ester':fr_phos_ester,
    'fr_nitro_arom':fr_nitro_arom,
    'fr_nitro_arom_nonortho':fr_nitro_arom_nonortho,
    'fr_dihydropyridine':fr_dihydropyridine,
    'fr_phenol_noOrthoHbond':fr_phenol_noOrthoHbond,
    'fr_Al_OH_noTert':fr_Al_OH_noTert,
    'fr_benzodiazepine':fr_benzodiazepine,
    'fr_para_hydroxylation':fr_para_hydroxylation,
    'fr_allylic_oxid':fr_allylic_oxid,
    'fr_aryl_methyl':fr_aryl_methyl,
    'fr_Ndealkylation1':fr_Ndealkylation1,
    'fr_Ndealkylation2':fr_Ndealkylation2,
    'fr_alkyl_carbamate':fr_alkyl_carbamate,
    'fr_ketone_Topliss':fr_ketone_Topliss,
    'fr_ArN':fr_ArN,
    'fr_HOCCN':fr_HOCCN})

df_Solvent_Features_All = pd.concat([df_Solvent_Features,df_Solvent_Features_Frags], axis=1)

X = df_Solvent_Features_All

X = X.loc[:, (X != 0).any(axis=0)]

Y_Ered=df_BTZ['Ered']
Y_HOMO_Opt=df_BTZ['HOMO']
Y_GSol=df_BTZ['Gsol']
Y_TDDFT=df_BTZ['Absorption Wavelength']

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.decomposition import PCA

st = StandardScaler()
Xdata = st.fit_transform(X)

pca = PCA(n_components=22)
Xdata = pca.fit_transform(Xdata)

natom_layer = Xdata.shape[1]

from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C ,WhiteKernel as Wht,Matern as matk
#from celluloid import Camera
#from IPython.display import HTML

def gpregression(Xtrain,Ytrain,Nfeature):    
    cmean=[1.0]*22
    cbound=[[1e-3, 1e3]]*22
#   kernel = C(1.0, (1e-3,1e3)) * RBF(cmean, cbound) + Wht(1.0, (1e-3, 1e3))
    kernel = C(1.0, (1e-3,1e3)) * matk(cmean,cbound,1.5) + Wht(1.0, (1e-3, 1e3))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=40, normalize_y=False)
    gp.fit(Xtrain, Ytrain)
    return gp

#predict result using GP regression model
def gprediction(gpnetwork,xtest):
    y_pred, sigma = gpnetwork.predict(xtest, return_std=True)
    return y_pred, sigma

#compute expected improvement
def expectedimprovement(xdata,gpnetwork,ybest,itag,epsilon):
    ye_pred, esigma = gprediction(gpnetwork, xdata)
    expI = np.empty(ye_pred.size, dtype=float)
    for ii in range(0,ye_pred.size):
        if esigma[ii] > 0:
            zzval=itag*(ye_pred[ii]-ybest)/float(esigma[ii])
            expI[ii]=itag*(ye_pred[ii]-ybest-epsilon)*norm.cdf(zzval)+esigma[ii]*norm.pdf(zzval)
        else:
            expI[ii]=0.0
    return expI

def paretoSearch(capP,search='min'):
    # Non-dominated sorting
    paretoIdx=[]
    F0 = []
    for i,p in enumerate(capP):
        Sp = []
        nps = 0
        for j,q in enumerate(capP):
            if i!=j:
                if search=='min':
                    compare = p < q
                elif search=='max':
                    compare = p > q
                if any(compare):
                    Sp.append(q)
                else: 
                    nps+=1
        if nps==0:
            paretoIdx.append(i)
            prank = 1
            F0.append(p.tolist())
    F0 = np.array(F0)
    return F0, paretoIdx

def paretoOpt(capP, metric='crowdingDistance',opt='min'):
    if capP.shape[0]<=1000:
        F0, paretoIdx = paretoSearch(capP, search=opt)
    else:
        n_parts = int(capP.shape[0]//1000.)
        rem = capP.shape[0] % 1000.  
        FList = [] 
        paretoIdxList = []
        for i in range(n_parts):
            Fi, paretoIdxi = paretoSearch(capP[1000*i:1000*(i+1)], search=opt)
            FList.append(Fi)
            ar_paretoIdxi = np.array(paretoIdxi)+1000*i
            paretoIdxList.append(ar_paretoIdxi.tolist())  
        if rem>0:
            Fi, paretoIdxi = paretoSearch(capP[1000*n_parts-1:-1], search=opt)
            FList.append(Fi)
            ar_paretoIdxi = np.array(paretoIdxi)+1000*n_parts
            paretoIdxList.append(ar_paretoIdxi.tolist())  
            
        F1 = np.concatenate(FList)
        paretoIdx1=np.concatenate(paretoIdxList)
        F0, paretoIdxTemp = paretoSearch(F1, search=opt)
        paretoIdx=[]
        for a in paretoIdxTemp:
            matchingArr = np.where(capP==F1[a])[0]
            counts = np.bincount(matchingArr)
            pt = np.argmax(counts)
            paretoIdx.append(pt)
   
    m=F0.shape[-1]
    l = len(F0)
    ods = np.zeros(np.max(paretoIdx)+1)
    if metric == 'crowdingDistance':
        infi = 1E6
        for i in range(m):
            order = []
            sortedF0 = sorted(F0, key=lambda x: x[i])
            for a in sortedF0: 
                matchingArr = np.where(capP==a)[0]
                counts = np.bincount(matchingArr)
                o = np.argmax(counts)
                order.append(o)
            ods[order[0]]=infi
            ods[order[-1]]=infi
            fmin = sortedF0[0][i]
            fmax = sortedF0[-1][i]
            for j in range(1,l-1):
                ods[order[j]]+=(capP[order[j+1]][i]-capP[order[j-1]][i])/(fmax-fmin)
        # Impose criteria on selecting pareto points
        if min(ods[np.nonzero(ods)])>=infi:
            bestIdx = np.argmax(ods)
        else:
            if l>2: # if there are more than 2 pareto points, pick inner points with largest crowding distance (i.e most isolated)
                tempOds=copy(ods)
                for i,a in enumerate(tempOds):
                    if a>=infi: tempOds[i]=0.
                bestIdx = np.argmax(tempOds)
            else: #pick pareto point with lower index
                bestIdx = np.argmax(ods)
    elif metric == 'euclideanDistance':  # To the hypothetical point of the current data
        for i in range(m):
            order = []
            sortedF0 = sorted(F0, key=lambda x: x[i])
            for a in sortedF0:
                matchingArr = np.where(capP==a)[0]
                counts = np.bincount(matchingArr)
                o = np.argmax(counts)
                order.append(o)          
            fmin = sortedF0[0][i]
            fmax = sortedF0[-1][i]
            for j in range(0,l):
                ods[order[j]]+=((capP[order[j]][i]-fmax)/(fmax-fmin))**2
        ods = np.sqrt(ods)
        for i,a in enumerate(ods):
            if a!=0: print(i,a)
        bestIdx = np.where(ods==np.min(ods[np.nonzero(ods)]))[0][0]
    return paretoIdx,bestIdx

import matplotlib.pyplot as plt
optimalValue = 375

Ydata1 = -1*df_BTZ['Ered'].values # Min
Ydata2 = -1*df_BTZ['Gsol'].values
Ydata3 = -1*abs(df_BTZ['Absorption Wavelength']-optimalValue)

Ydata = np.vstack((Ydata1, Ydata2, Ydata3)).T
nobj = Ydata.shape[1]
Xinfo = df_BTZ['SMILES']
ndata = len(Ydata)

#Bayesian optimization run
def numberofopt(Xdata,Ydata,Xinfo,ndata,natom_layer,BOmetric='crowdingDistance'):
    itag = 1
    epsilon = 0.01
    ntrain = 5 # int(train_test_split * ndata)
    nremain = ndata - ntrain
    dataset = np.random.permutation(ndata)
    a1data = np.empty(ntrain, dtype=int)
    a2data = np.empty(nremain, dtype=int)
    a1data[:] = dataset[0:ntrain]
    a2data[:] = dataset[ntrain:ndata]
    # info for the initial training set
    Xtrain = np.ndarray(shape=(ntrain, natom_layer), dtype=float)
    Xtraininfo = np.chararray(ntrain, itemsize=100)
    Ytrain = np.empty((ntrain,nobj), dtype=float)
    Xtrain[0:ntrain, :] = Xdata[a1data, :]
    Xtraininfo[0:ntrain] = Xinfo[a1data]
    Ytrain[0:ntrain, :] = Ydata[a1data, :]
    
    XtraininfoIni = Xtraininfo
    XtraininfoIni=np.array([x.decode() for x in XtraininfoIni])
    XtraininfoIniList = XtraininfoIni.tolist()

    _,yoptLoc = paretoOpt(Ytrain,metric=BOmetric,opt='max')
    
    yopttval = Ytrain[yoptLoc]
    xoptval = Xtraininfo[yoptLoc]
    yoptstep=0
    yopinit = yopttval
    xoptint = xoptval
    # info for the remaining data set
    Xremain = np.ndarray(shape=(nremain, natom_layer), dtype=float)
    Xremaininfo = np.chararray(nremain, itemsize=100)
    Yremain = np.empty((nremain,nobj), dtype=float)
    Xremain[0:nremain, :] = Xdata[a2data, :]
    Xremaininfo[0:nremain] = Xinfo[a2data]
    Yremain[0:nremain] = Ydata[a2data]
#    print("Xremain: ", Xremain.shape)
#    print("Yremain:", Yremain.shape)
#    print("Xremaininfo: ", Xremaininfo.shape)
#    print("Initial best value 0th run : ", xoptval.decode(), yopttval)
#    print("Total number of inital training points: ", ntrain)
    # print("Xtrain: ",Xtrain)
    targetRM = []
#    print("starting epsilon: ", epsilon)
    for ii in range(0, Niteration):
        if ii > int(0.5*Niteration):
            epsilon=0.01
#            print("updated epsilon: ",epsilon)
        #print('Ytrain before GPR',Ytrain)
        gpnetworkList = []
        yt_predList = []
        for i in range(nobj):
            gpnetwork = gpregression(Xtrain, Ytrain[:,i], natom_layer)
            yt_pred, tsigma = gprediction(gpnetwork, Xtrain)
            yt_predList.append(yt_pred)
            gpnetworkList.append(gpnetwork)
            
        yt_pred=np.vstack((yt_predList)).T

        _, ybestloc = paretoOpt(yt_pred,metric=BOmetric,opt='max') 
        
        ybest = yt_pred[ybestloc]
        ytrue = Ytrain[ybestloc]
        #print('ytrue',ytrue)
        
#        print('------------')
#        print("current Best in iteration ", ii + 1, " has the predicted/true value of ", 
#              ybest, '/', ytrue, "for the structure: ", Xtraininfo[ybestloc].decode())
        
        #print(Ytrain)
#        currentPareto, currentBest = NSGAII_F0(Ytrain,opt='max')
        currentPareto, currentBest = paretoOpt(Ytrain,metric=BOmetric,opt='max')

        #print(currentBest,Ytrain[currentBest])
        
#        print(Ytrain[currentBest],ytrue,yopttval)
        if any(Ytrain[currentBest]!=yopttval):
            yopttval = Ytrain[currentBest]
            xoptval = Xtraininfo[currentBest]
            yoptstep=ii
            
#        print("Best Strucutre so far", yopttval, "for the structure: ", xoptval.decode())

        expIList = []
        for i in range(nobj):
            expI = expectedimprovement(Xremain, gpnetworkList[i], ybest[i], itag, epsilon)
            expIList.append(expI)
        
        expI = np.vstack((expIList)).T

        
#        expI1 = expectedimprovement(Xremain, gpnetwork1, ybest[0], itag, epsilon)
#        expI2 = expectedimprovement(Xremain, gpnetwork2, ybest[1], itag, epsilon)
        
#        expI = np.vstack((expI1, expI2)).T
        
        _, expimaxloc = paretoOpt(expI,metric=BOmetric,opt='max')
        expImax = expI[expimaxloc] 

#        print("Next Structure to evaluate has expI ", expImax, "for the structure: ", Xremaininfo[expimaxloc].decode(),
#              "has Y: ", Yremain[expimaxloc])
        
        # Live plotting
        #plt.scatter(-Ydata[:,0],-Ydata[:,1],color='k',s=60)
        #for i in currentPareto:
        #    plt.scatter(-Ytrain[i][0],-Ytrain[i][1],color='c',s=80)
        #plt.scatter(-Ytrain[currentBest][0],-Ytrain[currentBest][1],color='r')
        #plt.scatter(-Ytrain[:,0],-Ytrain[:,1],facecolors='none', edgecolors='r',s=80)
        #display.clear_output(wait=True)
        #display.display(pl.gcf())
        #time.sleep(1.0)
        
        
        xnew = np.append(Xtrain, Xremain[expimaxloc]).reshape(-1, natom_layer)
        xnewinfo = np.append(Xtraininfo, Xremaininfo[expimaxloc])
        #print('Ytrain',Ytrain)
        #print('Yremain[expimaxloc]', Yremain[expimaxloc])
        ynew = np.concatenate((Ytrain, Yremain[expimaxloc].reshape(1,-1)))
        #ynew = np.append(Ytrain, Yremain[expimaxloc])
        #print('ynew',ynew,'end')
        xrnew = np.delete(Xremain, expimaxloc, 0)
        xrnewinfo = np.delete(Xremaininfo, expimaxloc)
        yrnew = np.delete(Yremain, expimaxloc,0)
        #print('yrnew',yrnew)
        if ii==0:
            Xexplored=Xremaininfo[expimaxloc]
            Yexplored=Yremain[expimaxloc]
        else:
            Xexploredtemp=np.append(Xexplored, Xremaininfo[expimaxloc])
            Yexploredtemp=np.append(Yexplored, Yremain[expimaxloc])
            del Xexplored,Yexplored
            Xexplored=Xexploredtemp
            Yexplored=Yexploredtemp
        #    print("Xremain info: ",xrnew.shape,yrnew.shape,xrnewinfo.shape)
        del Xtrain, Ytrain, Xremaininfo, gpnetwork
        Xtrain = xnew
        Xtraininfo = xnewinfo
        Ytrain = ynew
        Xremain = xrnew
        Xremaininfo = xrnewinfo
        Yremain = yrnew
        del xnew, xnewinfo, ynew, xrnew, xrnewinfo, yrnew
        
    Yexplored = Yexplored.reshape(-1,nobj)

#    print(Xexplored)
    Xexplored=np.array([x.decode() for x in Xexplored])

    XexploredList = Xexplored.tolist()
    result = XtraininfoIniList+XexploredList

#    if not all(yopinit==yopttval):
#        _, bestIdx = NSGAII_F0(Yexplored,opt='max')
#        yoptstep = bestIdx + 1
#        yoptstep +=1
#        #yoptstep = np.argmax(Yexplored) + 1       
#    else:
#        yoptstep=0
    
#    print("Explored Y values:",Yexplored,type(Yexplored))
#    dataorder = np.argsort(Yexplored)
#    print('dataorder',dataorder)
#    Yexploredtemp=Yexplored[dataorder]
#    Xexploredtemp = Xexplored[dataorder]
#    print('Yexplored',Yexplored)
#    Xbest=Xexploredtemp[Niteration-3:Niteration]
#    print('Xbest',Xbest)
#    Ybest=Yexploredtemp[Niteration - 3:Niteration]
#    print("\n")
#    print("Initial Best Strucuture: ", xoptint.decode(), "has value: ", yopinit)
#    print("Final Optimal Strucuture: ", xoptval.decode(), "has value: ", yopttval,"in step: ",yoptstep)
#    print("Final Best Structure 1st: ",Xbest[2].decode(),"has value: ",  Ybest[2])
#    print("Final Best Structure 2st: ", Xbest[1].decode(),"has value: ", Ybest[1])
#    print("Final Best Structure 2st: ", Xbest[0].decode(),"has value: ", Ybest[0])
    return xoptint,yopinit,xoptval,yopttval,Xexplored,Yexplored,result

#------- Program Starts from here -------------
#print("Original Training X and Y :",np.shape(Xdata),np.shape(Xdata))
fileName = 'BayesOptRunProgress'
pfile = open(fileName+'.txt','a+')
pfile.write("Original Training X and Y : \n")
pfile.write(str(np.shape(Xdata)))
pfile.write(str(np.shape(Ydata)))
pfile.write("\n")
pfile.close()
Nruns = 10
Niteration = 100     # number of iteration in a given Bayesian  Optimization
#input parameters
#train_test_split=0.001                  # initial sampled data in a given Bayesian  Optimization run
Xinitguess = np.chararray(Nruns, itemsize=100)
Yinitguess = np.empty((Nruns,nobj), dtype=float)
Xoptimal = np.chararray(Nruns, itemsize=100)
Yoptimal = np.empty((Nruns,nobj), dtype=float)
Yexplored = []
Xexplored = []

res = []
metr = 'crowdingDistance'
for ii in range(0,Nruns):
    Xinitguess[ii], Yinitguess[ii], Xoptimal[ii], Yoptimal[ii], xexploredii, yexploredii, resultRow =numberofopt(Xdata, Ydata, Xinfo, ndata, natom_layer,BOmetric=metr)
    pfile = open(fileName+'.txt','a+')
    pfile.write("Run #"+str(ii+1)+"\n")
    pfile.write('---------------------------\n')
    pfile.close()
    resFile = open('Result-temp.csv','a+')
    for i,a in enumerate(resultRow):
        if i != len(resultRow)-1:
            resFile.write(a+',')
        else:
            resFile.write(a+'\n')
    resFile.close()
    xFile = open('Xexplored-temp.csv','a+')
    for i,a in enumerate(xexploredii):
        if i != len(xexploredii)-1:
            xFile.write(a+',')
        else:
            xFile.write(a+'\n')
    xFile.close()
    res.append(resultRow)
    Xexplored.append(xexploredii)
df_Xexplored = pd.DataFrame.from_records(Xexplored)
df = pd.DataFrame.from_records(res)

df_Xexplored.to_csv('BayesOptResult-ini05-PointZeroOne-Xexplored-'+metr+'.csv',index=False)
df.to_csv('BayesOptResult-ini05-PointZeroOne-'+metr+'.csv',index=False)
