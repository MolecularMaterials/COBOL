# One step of Bayesian optimization 
def next_smiles_gpr(Xtrain,Xtrain_smi, Ytrain,Xremain,Xremain_smi,BOmetric='crowdingDistance'):
    """
    Use: next_smi_from_gpr(Xtrain,Xtrain_smi, Ytrain,Xremain,Xremain_smi,BOmetric='crowdingDistance',epsilon=0.01)
    Xtrain, Ytrain, Xremain are numpy array
    Xtrain_smi, Xremain_smi are python list
    natom_layer: number of PCs used in PCA

    Output: next_mol_smi,Xtrain_new,Xtrain_new_smi,Xremain_new,Xremain_new_smi
    
    SMILES for next DFT calculation and updated Xtrain and Xremain. 
    Caution: Update Ytrain after DFT calculations.
    """

    from copy import copy
    import numpy as np
    from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    from sklearn.gaussian_process import GaussianProcessRegressor
    from scipy.stats import norm
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C ,WhiteKernel as Wht,Matern as matk

    def gpregression(Xtrain,Ytrain,Nfeature):    
        cmean=[1.0]*Nfeature
        cbound=[[1e-3, 1e3]]*Nfeature
        kernel = C(1.0, (1e-3,1e3)) * matk(cmean,cbound,1.5) + Wht(1.0, (1e-3, 1e3))  # Matern kernel
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=40, normalize_y=False)
        gpr.fit(Xtrain, Ytrain)
        return gpr

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

    nobj=Ytrain.shape[1]
    natom_layer = Xtrain.shape[1]
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

    expIList = []
    for i in range(nobj):
        expI = expectedimprovement(Xremain, gpnetworkList[i], ybest[i], itag=1, epsilon=0.01)
        expIList.append(expI)
    
    expI = np.vstack((expIList)).T
            
    _, expimaxloc = paretoOpt(expI,metric=BOmetric,opt='max')
    expImax = expI[expimaxloc] 
    next_mol_smi=Xremain_smi[expimaxloc]    

    print("Next molecule: ",next_mol_smi)  
    Xtrain_new = np.append(Xtrain, Xremain[expimaxloc]).reshape(-1, natom_layer)
    Xtrain_new_smi = np.append(Xtrain_smi, Xremain_smi[expimaxloc])
    Xremain_new=np.delete(Xremain, expimaxloc, 0)
    Xremain_new_smi=np.delete(Xremain_smi, expimaxloc)

    return next_mol_smi,Xtrain_new,Xtrain_new_smi,Xremain_new,Xremain_new_smi
