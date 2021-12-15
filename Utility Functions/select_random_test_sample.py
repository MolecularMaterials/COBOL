def get_samples(ndata,ntrainInit):
    """
    Input: 
    ndata = df.shape[0], total number of samples
    ntrainInit = number of points for initial training
    Output:
    Two arrays of indexes of ndata: sample_idx, and remaining_idx 
    Use: 
    test_data_idx,remaing_data_idx = get_samples(ndata,ntrainInit)
    """
    import numpy as np
    nremain = ndata - ntrainInit
    dataset = np.random.permutation(ndata)
    a1data = np.empty(ntrainInit, dtype=int) # randomly chosen data points
    a2data = np.empty(nremain, dtype=int) # remaining data points
    a1data[:] = dataset[0:ntrainInit]
    a2data[:] = dataset[ntrainInit:ndata]
    return a1data,a2data
