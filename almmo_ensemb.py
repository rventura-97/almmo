import numpy as np
from almmo_core import ALMMo_0, ALMMo0_WCC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from timeit import default_timer as timer 
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, recall_score, precision_score,\
    cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score
from aux_functions import specificity, geometric_mean

class ALMMo0_WCCE:
    def __init__(self, costFunction, validationType="cross_valid"):
        
        self.ValidationType = validationType
        self.base_model = [ALMMo0_WCC(costFunction=costFunction) for i in range(0,5)] 

        # Dataset variables:
        self.X_train = []
        self.X_valid = []
        self.y_train = []
        self.y_valid = []
        
        self.Labels = []


        # Performance variables:
        self.TrainingTime = -1.0

        # Weight variables:
        self.LambdaValid = []
        self.OptReport = []
        self.DefClass = -1
        self.MinClass = -1
        self.Weights = []
        self.CostFunctionID = costFunction
        
        
        # Dataset variables:
        self.X_train = []
        self.X_valid = []
        self.y_train = []
        self.y_valid = []
        
        self.x_idxs = []
        self.y_idxs = []
        self.Labels = []
        
        
        # Performance variables:
        self.TrainingTime = -1.0
        
        # Weight variables:
        self.LambdaValid = []
        self.OptReport = []
        self.DefClass = -1
        self.MinClass = -1
        self.Weights = []
        self.CostFunctionID = costFunction
        
        
    def fit(self, x_train, y_train):
        
        tStart = timer()



        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
        for train_index, test_index in skf.split(x_train, y_train):
            self.X_train.append(x_train[train_index])
            self.X_valid.append(x_train[test_index])
            self.y_train.append(y_train[train_index])
            self.y_valid.append(y_train[test_index])
            
        for i in range(0,len(self.base_model)):
            self.base_model[i].fit(self.X_train[i], self.y_train[i], self.X_valid[i], self.y_valid[i])
            self.Weights.append(self.base_model[i].Weights)
        
        tEnd = timer()
        self.TrainingTime = tEnd - tStart
        self.X_train = []
        self.X_valid = []
        self.y_train = []
        self.y_valid = []
            
            
    def predict(self, x_test):
        lambda_weight = [self.base_model[i].predict_proba(x_test) for i in range(0,len(self.base_model))]
        lambda_weight_norm = [np.divide(lambda_weight[i], np.sum(lambda_weight[i],axis=0)) for i in range(0,len(self.base_model))]
        ensemble_vote = np.sum(lambda_weight_norm,axis=0) / np.sum(np.sum(lambda_weight_norm,axis=0),axis=0)
        y_pred = np.argmax(ensemble_vote, axis=0)

        return y_pred
            
    
class ALMMo0_STACK:
    def __init__(self, costFunction, validationType="cross_valid"):
        
        self.ValidationType = validationType
        self.base_model = [ALMMo_0() for i in range(0,5)] 
        self.meta_model = ALMMo_0()

        # Dataset variables:
        self.X_train = []
        self.X_valid = []
        self.y_train = []
        self.y_valid = []
        
        self.Labels = []


        # Performance variables:
        self.TrainingTime = -1.0

        # Weight variables:
        self.LambdaValid = []
        self.OptReport = []
        self.DefClass = -1
        self.MinClass = -1
        self.Weights = []
        self.CostFunctionID = costFunction
        self.BaseModelScores = []
        
        
        # Dataset variables:
        self.X_train = []
        self.X_valid = []
        self.y_train = []
        self.y_valid = []
        
        self.x_idxs = []
        self.y_idxs = []
        self.Labels = []
        
        
        # Performance variables:
        self.TrainingTime = -1.0
        
        # Weight variables:
        self.LambdaValid = []
        self.OptReport = []
        self.DefClass = -1
        self.MinClass = -1
        self.Weights = []
        self.CostFunctionID = costFunction
        
        
    def fit(self, x_train, y_train):
        
        tStart = timer()

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
        for train_index, test_index in skf.split(x_train, y_train):
            self.X_train.append(x_train[train_index])
            self.X_valid.append(x_train[test_index])
            self.y_train.append(y_train[train_index])
            self.y_valid.append(y_train[test_index])
            
        for i in range(0,len(self.base_model)):
            self.base_model[i].fit(self.X_train[i], self.y_train[i])
            
        # Evaluate base model scores and assign weights
        base_model_score = np.zeros((len(self.base_model),1))
        for i in range(0,len(self.base_model)):
            base_model_score[i] = self.computeBaseModelScore(self.y_valid[i],\
                                  self.base_model[i].predict(self.X_valid[i]))
            
            if base_model_score[i] < 0:
                base_model_score[i] = 0
                
        self.BaseModelScores = base_model_score


        if not np.all(base_model_score == 0):
            base_model_weights = np.zeros((len(self.base_model),1))
            base_model_weights[base_model_score != 0.0] = base_model_score[base_model_score != 0.0] / \
                                                          np.sum(base_model_score[base_model_score != 0.0])
            self.BaseModelWeights = base_model_weights
        else:
            self.BaseModelWeights = np.ones((len(self.base_model),1))/5
                
    
        # Train meta model:
        self.X_valid = np.vstack(self.X_valid)    
        self.y_valid = np.hstack(self.y_valid) 
            
        base_model_activs_score = []
        for i in range(0,len(self.base_model)):
            base_model_activs_score.append(self.base_model[i].predict_proba(self.X_valid))
        base_model_activs_score = np.vstack(base_model_activs_score)   
        
        
        base_model_activs_score = np.vstack((base_model_activs_score,\
                                  np.repeat(self.BaseModelWeights,base_model_activs_score.shape[1],axis=1)))

        self.meta_model.fit(np.transpose(base_model_activs_score), self.y_valid)

        
        tEnd = timer()
        self.TrainingTime = tEnd - tStart
        self.X_train = []
        self.X_valid = []
        self.y_train = []
        self.y_valid = []
            
            
    def predict(self, x_test):
        base_model_activs = []
        for i in range(0,len(self.base_model)):
            base_model_activs.append(self.base_model[i].predict_proba(x_test))
         
        base_model_activs_score = np.vstack(base_model_activs)
            
        base_model_activs_score = np.vstack((base_model_activs_score,\
                                  np.repeat(self.BaseModelWeights,base_model_activs_score.shape[1],axis=1)))   
         
        y_pred = self.meta_model.predict(np.transpose(base_model_activs_score))

        return y_pred
    
    def computeBaseModelScore(self, y_true, y_pred):
        if self.CostFunctionID == 1: # Geometric Mean
            metric_score = geometric_mean(y_true, y_pred)
        elif self.CostFunctionID == 2: # F1-Score
            metric_score = f1_score(y_true, y_pred)
        elif self.CostFunctionID == 3:  # Cohen's Kappa Coefficient
            metric_score = cohen_kappa_score(y_true, y_pred)
        elif self.CostFunctionID == 4: # Mathew's Correlation Coefficient
            metric_score = matthews_corrcoef(y_true,y_pred)
        return metric_score
    
# class ALMMo0_STACK:
#     def __init__(self, costFunction, validationType="cross_valid", train_mode="normal"):
        
#         self.ValidationType = validationType
        
#         self.TrainMode = train_mode
        
#         self.meta_model = ALMMo_0()
        
#         if self.ValidationType == "single_split":
#             self.base_model = ALMMo_0()
        
#         elif self.ValidationType == "cross_valid":
#             self.base_model = [ALMMo_0() for i in range(0,5)] 
        
        
#         # Dataset variables:
#         self.X_train = []
#         self.X_valid = []
#         self.y_train = []
#         self.y_valid = []
        
#         self.x_idxs = []
#         self.y_idxs = []
#         self.Labels = []


#         # Performance variables:
#         self.TrainingTime = -1.0

#         self.BaseModelWeights = []

#         # Weight variables:
#         self.LambdaValid = []
#         self.OptReport = []
#         self.DefClass = -1
#         self.MinClass = -1
#         self.Weights = []
#         self.CostFunctionID = costFunction
        

#     def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        
#         tStart = timer()

#         # Set default class to be the majority class:
#         self.MajClass = 0#np.argmax(valid_labelCount)
#         self.MinClass = 1#np.argmin(valid_labelCount)
#         if self.MajClass == self.MinClass:
#             self.MajClass = 0
#             self.MinClass = 1

        
#         if x_valid is None or y_valid is None:
#             if self.ValidationType == "single_split":
#                 self.X_train, self.X_valid, self.y_train, self.y_valid =\
#                     train_test_split(x_train, y_train, test_size=0.20, random_state=42)
                    
#                 self.base_model.fit(self.X_train, self.y_train)
                    
#                 self.base_model.fit(self.X_train, self.y_train)
#             elif self.ValidationType == "cross_valid":
#                 skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    
#                 for train_index, test_index in skf.split(x_train, y_train):
#                     self.X_train.append(x_train[train_index])
#                     self.X_valid.append(x_train[test_index])
#                     self.y_train.append(y_train[train_index])
#                     self.y_valid.append(y_train[test_index])
#                 ###
#                 #self.X_valid = [np.vstack(self.X_valid)] * 5   
#                 #self.y_valid = [np.hstack(self.y_valid)] * 5       
#                 ###
        
#         # Train base models
#         if self.ValidationType == "cross_valid":
#             if self.TrainMode == "normal":
#                 for i in range(0,len(self.base_model)):
#                     self.base_model[i].fit(self.X_train[i], self.y_train[i])
                    
#             elif self.TrainMode == "ray":
                
#                 simulations = [train_ray.remote(self.X_train[i], self.y_train[i]) for i in range(0, len(self.X_train))]
#                 base_models = [ray.get(s) for s in simulations]
#                 self.base_model = base_models
                
#         print("Base models trained [{}]".format(timer()-tStart))

#         # Evaluate base model scores and assign weights
#         base_model_score = np.zeros((len(self.base_model),1))
#         if self.ValidationType == "cross_valid":
#             for i in range(0,len(self.base_model)):
#                 base_model_score[i] = self.computeBaseModelScore(self.y_valid[i],\
#                                       self.base_model[i].predict(self.X_valid[i]))

#         print("Base models evaluated [{}]".format(timer()-tStart))

#         if np.all(base_model_score > 0):
#             base_model_weights = np.zeros((len(self.base_model),1))
#             base_model_weights[base_model_score != 0.0] = base_model_score[base_model_score != 0.0] / \
#                                                           np.sum(base_model_score[base_model_score != 0.0])
#             self.BaseModelWeights = base_model_weights
#         else:
#             self.BaseModelWeights = np.ones((len(self.base_model),1))/5
        
        
#         print("Base models weights assigned [{}]".format(timer()-tStart))
        
#         # Train meta model:
#         self.X_valid = np.vstack(self.X_valid)    
#         self.y_valid = np.hstack(self.y_valid) 
            
#         base_model_activs_score = []
#         for i in range(0,len(self.base_model)):
#             base_model_activs_score.append(self.base_model[i].predict_proba(self.X_valid))
#         base_model_activs_score = np.vstack(base_model_activs_score)
        
        
#         base_model_activs_score = np.vstack((base_model_activs_score,\
#                                   np.repeat(self.BaseModelWeights,base_model_activs_score.shape[1],axis=1)))

#         self.meta_model.fit(np.transpose(base_model_activs_score), self.y_valid)

#         print("Meta model trained [{}]".format(timer()-tStart))

#         tStop = timer()
        
#         self.TrainingTime = tStop - tStart

     
        
#     def computeActivations(self, X):
#         # Initialize activation matrices:
#         Lambda_mtx = [None]*2
#         Lambda_mtx[self.MajClass] = zeros((2,X[self.MajClass].shape[1]))
#         Lambda_mtx[self.MinClass] = zeros((2,X[self.MinClass].shape[1]))
#         # Compute minority class cloud max activations for all training samples:
#         for i in range(len(Lambda_mtx)):
#             for k in range(X[i].shape[1]):
#                 # Normalize sample:
#                 x = transpose(mat(normSample(X[i][self.x_idxs,k])))
#                 for c in range(len(self.F)):
#                     # Compute class cloud activations:
#                     Lambda_mtx[i][c,k] = np.max(computeLambda(computeDistance(x,self.F[c])))
#         return Lambda_mtx

#     def predict(self, x_test):
#         base_model_activs = []
#         for i in range(0,len(self.base_model)):
#             base_model_activs.append(self.base_model[i].predict_proba(x_test))
         
#         base_model_activs_score = np.vstack(base_model_activs)
            
#         base_model_activs_score = np.vstack((base_model_activs_score,\
#                                   np.repeat(self.BaseModelWeights,base_model_activs_score.shape[1],axis=1)))   
         
#         y_pred = self.meta_model.predict(np.transpose(base_model_activs_score))


#         return y_pred
    
#     def computeBaseModelScore(self, y_true, y_pred):
#         if self.CostFunctionID == 1: # Geometric Mean
#             metric_score = geometric_mean(y_true, y_pred)
#         elif self.CostFunctionID == 2: # F1-Score
#             metric_score = f1_score(y_true, y_pred)
#         elif self.CostFunctionID == 3:  # Cohen's Kappa Coefficient
#             metric_score = cohen_kappa_score(y_true, y_pred)
#         elif self.CostFunctionID == 4: # Mathew's Correlation Coefficient
#             metric_score = matthews_corrcoef(y_true,y_pred)
#         return metric_score