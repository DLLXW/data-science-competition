from   sklearn.metrics import mean_squared_error
import numpy as np
import torch
def rmse(y_true, y_preds):
    return np.sqrt(mean_squared_error(y_pred = y_preds, y_true = y_true))

def score(y_true, y_preds):
    accskill_score = 0
    rmse_scores    = 0
    a = [1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6
    print(y_true.shape,y_preds.shape)
    y_true_mean = np.mean(y_true,axis=0) 
    y_pred_mean = np.mean(y_preds,axis=0) 
    print(y_true_mean.shape, y_pred_mean.shape)

    for i in range(24): 
        fenzi = np.sum((y_true[:,i] -  y_true_mean[i]) *(y_preds[:,i] -  y_pred_mean[i]) ) 
        fenmu = np.sqrt(np.sum((y_true[:,i] -  y_true_mean[i])**2) * np.sum((y_preds[:,i] -  y_pred_mean[i])**2) ) 
        cor_i = fenzi / fenmu
    
        accskill_score += a[i] * np.log(i+1) * cor_i
        rmse_score   = rmse(y_true[:,i], y_preds[:,i])
#         print(cor_i,  2 / 3.0 * a[i] * np.log(i+1) * cor_i - rmse_score)
        rmse_scores += rmse_score 
    
    return  2 / 3.0 * accskill_score - rmse_scores
if __name__=='__main__':
    a=torch.randn(20,24)
    b=torch.randn(20,24)
    a=a.numpy()
    b=b.numpy()
    print(score(a,b))
    print(rmse(a,b))