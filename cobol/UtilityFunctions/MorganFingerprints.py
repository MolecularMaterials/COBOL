def smi_to_morgan_fingerprints(list_of_smiles,radius=2,length=128):
    """
    Usage: smi_to_morgan_fingerprints(list_of_smiles,radius=2,length=128)
    Input: A pandas series of smiles 
    Output: A pandas dataframe with SMILES and Morgan Fingerprints (default=128 bits) with radius 2 (~ ECFP4)
    """
    import numpy as np
    import pandas as pd
    import rdkit.Chem as Chem
    from rdkit.Chem import AllChem
    
    morgan_matrix=np.zeros((1,length))
    l=len(list_of_smiles)

    for i in range(l):
        """
        For each compound get the strucuture and convert to morgan fingerprint 
        and add to data matrix
        """
        try:
            compound = Chem.MolFromSmiles(list_of_smiles.values[i])
            fp=Chem.AllChem.GetMorganFingerprintAsBitVect(compound,radius,nBits=length)
            fp=fp.ToBitString()
            matrix_row=np.array([int(x) for x in list(fp)])
            morgan_matrix=np.row_stack((morgan_matrix,matrix_row))

            #Progress checker
            if i%1000==0:
                percentage=np.round(100*(i/l),1)
                print(f'{percentage}% done')
        except:
            print(f'problem index:{i}')

    #Deleting first row of zeros
    morgan_matrix = np.delete(morgan_matrix,0,axis=0)
    X=pd.DataFrame(morgan_matrix)
    X["SMILES"]=list_of_smiles.values
    # print("\n")
    print(f'Morgan Matrix dimensions:{morgan_matrix.shape}')
    return X
