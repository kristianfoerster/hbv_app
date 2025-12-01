import numpy as np
import scipy.stats

def model_performance(y_b, y_s):
    # check for nan values
    i_nan = y_b[np.isnan(y_b)]

    nnan = len(i_nan)
    if nnan > 0:
        np.delete(y_b[i_nan])
        np.delete(y_s[i_nan])


    if nnan == len(y_b):
        print('Error: No valid data in observed time series!')
        return


    if len(y_s) != len(y_b):
        print('Error: dimension of observed and modelled time series do not agree!');
        return



    #Nash-Sutcliffe efficiency (NSE) Optimum: 1
    NSE=1-(np.sum((y_b-y_s)**2)/np.sum((y_b-np.mean(y_b))**2))
    #Percent bias (PBIAS) Optimum: 0, + Unterschaetzung [percent], - Ueberschaetzung [percent]
    PBIAS=((np.sum(y_b-y_s))*100)/np.sum(y_b)
    #Root-Mean-Square-Error (RMSE) Optimum: moeglichst klein
    RMSE=np.sqrt((np.sum((y_s-y_b)**2))/len(y_b))
    #RMSE-observations standard deviation ratio (RSR) Optimum: 0, (0 bis +inf)
    RSR=np.sqrt((sum((y_b-y_s)**2)))/(np.sqrt(np.sum((y_b-np.mean(y_b))**2)))
    #Lineare Regression
    #a1, a0, r_value, p_value, stderr = scipy.stats.linregress(df.obs, df.sim)
    #[r,m,b] = regression(y_b',y_s');
    # a1, a0, r, p_value, stderr = scipy.stats.linregress(y_b, y_s)
    # r = np.corrcoef(y_b,y_s)
    r = np.sum( (y_b - np.mean(y_b)) * (y_s - np.mean(y_s)) ) / np.sqrt( np.sum((y_b - np.mean(y_b))**2 * np.sum((y_s - np.mean(y_s))**2)) )

    #r = scipy.stats.pearsonr(y_s, y_b)
    alpha = np.std(y_s)/np.std(y_b)
    beta = np.mean(y_s)/np.mean(y_b)
    KGE = 1 - (np.sqrt((r-1)**2+(alpha-1)**2+(beta-1)**2))
    
    return NSE, KGE, PBIAS, RMSE, RSR, r

