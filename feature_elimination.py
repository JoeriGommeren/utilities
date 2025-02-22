# Feature elimination
import pandas as pd
import numpy as np


# methods: Variance Treshold, K best Features, Recursive Feature Elimination


# Variance Treshold
from sklearn.preprocessing import MinMaxScaler

def variance_treshold(X_train,X_test,treshold):
    '''
    Description:
    Eliminate features from dataframe when feature variance under treshold. 

    Input: 
    X_train: unscaled numerical train dataframe, excluding target variable. 
    X_test: unscaled numerical test dataframe, excluding target variable. 
    treshold: variance treshold variable. All features with lower variance than treshold will be excluded. 

    Output: 
    X_train: input X_train df, excluding the features with low variance. 
    X_test: input X_test df, excluding the features with low variance. 
    '''

    # scaling is required for good comparison between features. 
    scaler = MinMaxScaler()
    scaled_X_train_v1 = scaler.fit_transform(X_train)
    # calculate variance of each feature. 
    df_variance = pd.DataFrame(scaled_X_train_v1,columns = X_train.columns).var(axis=0).reset_index()
    df_variance.columns = ['feature','variance']
    # define worst features
    worst_features = df_variance[df_variance['variance']<treshold]['feature'].to_list()
    X_train.drop(columns=worst_features)
    X_test.drop(columns=worst_features)
    return X_train,X_test




# K best Features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score

def classifier_k_best_features(X_train,y_train,X_test,y_test,model):
    '''
    Description:
    Selects the optimal number of features using SelectKBest and mutual_info_classif.
    For binary classification usecase using F1 scoring. 

    Input:
    X_train: unscaled numerical train dataframe, excluding target variable. 
    y_train: numerical target variable of train set (result of sklearn Test_Train_Split).
    X_test: unscaled numerical test dataframe, excluding target variable. 
    y_test: numerical target variable of test set (result of sklearn Test_Train_Split).
    model: classification model. For example: GradientBoostingClassifier() or LogisticRegression().

    Output:
    X_train: input X_train df, only including required features.
    X_test: input X_test df, only including required features.
    '''

    f1_score_list = []
    for k in range(1, len(X_train.columns.tolist())):
        # SelectKBest: Select features according to the k highest scores.
        # mutual_info_classif: Estimate mutual information for a discrete target variable.
        selector = SelectKBest(mutual_info_classif, k=k) 
        selector.fit(X_train, y_train)
        
        sel_X_train = selector.transform(X_train)
        sel_X_test = selector.transform(X_test)
        # test 
        gbc = model
        gbc.fit(sel_X_train, y_train)
        kbest_preds = gbc.predict(sel_X_test)
        
        # the amount of features for the model with the best f1 score is selected. 
        f1_score_kbest = round(f1_score(y_test, kbest_preds, average='weighted'), 3)
        f1_score_list.append(f1_score_kbest)
        if f1_score_kbest>f1_score_list[-2]:
            k_best = k
    
    selector = SelectKBest(mutual_info_classif, k=k_best)
    selector.fit(X_train, y_train)
    selected_feature_mask = selector.get_support()
    X_train = X_train[X_train.columns[selected_feature_mask]]
    X_test = X_test[X_test.columns[selected_feature_mask]]
    return X_train,X_test




# Recursive Feature Elimination RFE
from sklearn.feature_selection import RFE

def classifier_k_best_features(X_train,y_train,X_test,y_test,model):
    '''
    Description:
    Selects the optimal number of features using RFE.
    For binary classification usecase using F1 scoring. 
    
    Input:
    X_train: unscaled numerical train dataframe, excluding target variable. 
    y_train: numerical target variable of train set (result of sklearn Test_Train_Split).
    X_test: unscaled numerical test dataframe, excluding target variable. 
    y_test: numerical target variable of test set (result of sklearn Test_Train_Split).
    model: classification model. For example: GradientBoostingClassifier() or LogisticRegression().
    
    Output:
    X_train: input X_train df, only including required features. 
    X_test: input X_test df, only including required features.
    '''

    rfe_f1_score_list = []
    for k in range(1, len(X_train.columns.tolist())):
        RFE_selector = RFE(estimator=model, n_features_to_select=k, step=1) #it is possible to include a step size which can be required for big datasets. 
        RFE_selector.fit(X_train, y_train)
        
        sel_X_train = RFE_selector.transform(X_train)
        sel_X_test = RFE_selector.transform(X_test)
        
        model.fit(sel_X_train, y_train)
        RFE_preds = model.predict(sel_X_test)
        
        f1_score_rfe = round(f1_score(y_test, RFE_preds, average='weighted'), 3)
        rfe_f1_score_list.append(f1_score_rfe)
        if f1_score_rfe>rfe_f1_score_list[-2]:
            k_best = k
    
    RFE_selector = RFE(estimator=model, n_features_to_select=k_best, step=10)
    RFE_selector.fit(X_train, y_train)
    selected_feature_mask = RFE_selector.get_support()
    X_train = X_train[X_train.columns[selected_feature_mask]]
    X_test = X_test[X_test.columns[selected_feature_mask]]
    return X_train,X_test 

