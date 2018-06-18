from __future__ import division
import numpy as np

from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def blendFun(X, y, X_submission, seed = 0,n_folds = 10,shuffle = False, verbose = False, clfs = None,clf = RandomForestClassifier(n_estimators=400,random_state=33)):
    np.random.seed(seed)
      
    #X,y,X_submission
    
    #数据重排 增加随机概率 减少泛化误差
    if shuffle:
        idx = np.random.permuattion(y.size)
        X = X[idx]
        y = y[idx]
        
    
    #StratifiedKFold 是 k-fold 的变种，会返回 stratified（分层） 的折叠：每个小集合中， 各个类别的样例比例大致和完整数据集中相同。
    skf = list(StratifiedKFold(y,n_folds))
    
    if not clfs:
        clfs =  [RandomForestClassifier(n_estimators=300, n_jobs=-1, criterion='gini'),
                RandomForestClassifier(n_estimators=300, n_jobs=-1, criterion='entropy'),
                ExtraTreesClassifier(n_estimators=300, n_jobs=-1, criterion='gini'),
                ExtraTreesClassifier(n_estimators=300, n_jobs=-1, criterion='entropy'),
                GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

        
    dataset_blend_train = np.zeros((X.shape[0],len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0],len(clfs)))
    
    for j,clf in enumerate(clfs):
    
        dataset_blend_test_j = np.zeros((X_submission.shape[0],len(skf)))
        
        for i,(train,test) in enumerate(skf):
          
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            
            clf.fit(X_train, y_train)
            dataset_blend_train[test,j] = clf.predict_proba(X_test)[:,1]
            dataset_blend_test_j[:,i] = clf.predict_proba(X_submission)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
        
    print('blend...')
    
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:,1]
    
    return y_submission