import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


#forward Selection Algorithm
def forward_sel(clf,X_train,y_train,X_val,y_val,num_sel,verbose=True):
    feats_idx=list(range(X_train.shape[1]))
    selected_feats_idx=[]
    CCR_per_iter=[]
    for i in range(num_sel):
        if verbose:print(f"iter {i} started")
        scores=[]
        for fid in feats_idx:
            selected_feats_idx.append(fid)
            scores.append([fid,(clf.fit(X_train[:,selected_feats_idx],y_train)).score(X_test[:,selected_feats_idx],y_test)])
            selected_feats_idx.remove(fid)

        sel_score=scores[np.argmax([s[1] for s in scores])]

        selected_feats_idx.append(sel_score[0])
        feats_idx.remove(sel_score[0])
        CCR_per_iter.append(sel_score[1])

        if verbose:print(f"iter {i} finished")
    return selected_feats_idx,CCR_per_iter
#backward Selection Algorithm
def backward_sel(clf,X_train,y_train,X_val,y_val,num_sel,verbose=True):
    feats_idx=list(range(X_train.shape[1]))
    selected_feats_idx=list(range(X_train.shape[1]))
    CCR_per_iter=[]
    for i in range(len(feats_idx)-num_sel):
        if verbose:print(f"iter {i} started")
        scores=[]
        for fid in feats_idx:
            selected_feats_idx.remove(fid)
            scores.append([fid,(clf.fit(X_train[:,selected_feats_idx],y_train)).score(X_test[:,selected_feats_idx],y_test)])
            selected_feats_idx.append(fid)


        sel_score=scores[np.argmin([s[1] for s in scores])]

        selected_feats_idx.remove(sel_score[0])
        feats_idx.remove(sel_score[0])
        CCR_per_iter.append(sel_score[1])

        if verbose:print(f"iter {i} finished")
    return selected_feats_idx,CCR_per_iter



#splitting the dataset
X_train = np.array(pd.read_csv('trainData.csv'  ,header=None))
X_test  = np.array(pd.read_csv('testData.csv'   ,header=None))
y_train = np.array(pd.read_csv('trainLabels.csv',header=None))
y_test  = np.array(pd.read_csv('testLabels.csv' ,header=None))


#program's paramters for choosing the algorithm, # of iterations and # of samples
type=['forward','backward'][0]# 0: means forward , 1:means backward
num_sel=X_train.shape[1]#X_train.shape[1]: means forward , # 1: means backward
num_row=X_train.shape[0]

#calling the function for the selected algorithm and capturing the CCR values of each iteration
#and the number of the selected features
if type=='backward':
    selected_feats_idx,CCR_per_iter=backward_sel(GaussianNB(),X_train[:num_row,:],y_train[:num_row],X_test[:num_row,:],y_test[:num_row],num_sel,True)
else:
    selected_feats_idx, CCR_per_iter = forward_sel(GaussianNB(), X_train[:num_row, :], y_train[:num_row], X_test[:num_row, :],y_test[:num_row], num_sel, True)

#plotting CCR vs. # of features diagram
print(selected_feats_idx)
print(CCR_per_iter)
plt.figure(figsize=(8,5))
plt.plot(range(len(CCR_per_iter)),CCR_per_iter,c='g')
plt.xlabel('#feats',fontsize=12)
plt.ylabel('CCR',fontsize=12)
plt.show()

#outputting the maximum CCR value and # of features associated with that CCR
if type=='backward':
 idx=np.argmax(CCR_per_iter)
 print(CCR_per_iter[idx],X_train.shape[1]-idx)
else:
    idx = np.argmax(CCR_per_iter)
    print(CCR_per_iter[idx], idx+1)



