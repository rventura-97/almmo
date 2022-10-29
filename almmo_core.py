from numpy import transpose, squeeze, hstack, mat, argsort, array, unique, split, zeros, \
    uint64, float64, power, min, max, append, argmin, asscalar, sqrt, argmax, asarray, \
    zeros_like, concatenate, ones, meshgrid, linspace, row_stack
from numpy.linalg import norm
from timeit import default_timer as timer 
from aux_functions import normSample, computeDensity, computeDistance, computeLambda, r0
from matplotlib.pyplot import scatter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, cohen_kappa_score, matthews_corrcoef
from aux_functions import specificity, geometric_mean
from skopt.space import Real
from skopt import Optimizer

class ALMMo_0():
    def __init__(self, verbose=False, performanceMode=True, normalization="NORM"):
        self.Verbose = verbose
        # Dataset variables:
        self.X_train = []
        self.X_test = []
        self.x_idxs = []
        self.y_idxs = []
        self.Labels = []

        # Sub-model variables:
        self.K = []
        self.Mu = []
        self.X = []

        # Cloud variables:
        self.F = []
        self.XF = []
        self.M = []
        self.r = []

        # Performance variables:
        self.TrainingTime = -1.0
        self.CloudsPerClass = []
        self.FRDP = -1
        self.FRDPperClass = []
        self.NumOfClouds = -1
        self.CloudHistory = []
        self.Normalization = normalization
        self.NormalizationTransform = []


    def fit(self, x_train, y_train):
        
        if self.Normalization == "STD-MINMAX":
            self.NormalizationTransform = Pipeline([('std_scaler',StandardScaler()), ('minmax_scalar', MinMaxScaler())])
            self.X_train = transpose(hstack((self.NormalizationTransform.fit_transform(x_train),transpose(mat(y_train)))))
        elif self.Normalization == "MINMAX":
            self.NormalizationTransform = Pipeline([('minmax_scalar', MinMaxScaler())])
            self.X_train = transpose(hstack((self.NormalizationTransform.fit_transform(x_train),transpose(mat(y_train)))))
            
        else:
            self.X_train = transpose(hstack((x_train,transpose(mat(y_train)))))

        
        # Dataset variables:
        self.x_idxs = list(range(0,self.X_train.shape[0]-1,1))
        self.y_idxs = self.X_train.shape[0]-1

        # Processed dataset variables:
        sort_label_idxs = squeeze(array(argsort(self.X_train[self.y_idxs,:])))
        labels, part_idxs = unique(self.X_train[self.y_idxs,sort_label_idxs],axis=1,return_index=True)
        self.Labels = mat(labels)
        self.X_train = split(self.X_train[:,sort_label_idxs],part_idxs[1:],axis=1)

        # Sub-model variables:
        self.K = mat(zeros(self.Labels.size, dtype=uint64))
        self.Mu = mat(zeros((len(self.x_idxs),self.Labels.size)))
        self.X = mat(zeros(self.Labels.size, dtype=float64))

        # Cloud variables:
        self.F = [None]*self.Labels.size
        self.M = [None]*self.Labels.size
        self.r = [None]*self.Labels.size
        #
        self.XF = [None]*self.Labels.size

        # Performance variables:
        self.TrainingTime = -1.0
        self.CloudsPerClass = mat(zeros(self.Labels.size, dtype=int))
        self.FRDPperClass = mat(zeros(self.Labels.size))
        self.FRDP = -1
        self.NumOfClouds = -1
        self.CloudHistory = [zeros(1)]*len(self.X_train)


        
        tStart = timer()
        # For each class:
        for i in range(len(self.X_train)):
            # For each class sample:
            for k in range(self.X_train[i].shape[1]):
                #print(k)
                
                if self.Normalization == "NORM":
                    # Normalize sample:
                    x = normSample(self.X_train[i][self.x_idxs,k])
                else:
                    x = self.X_train[i][self.x_idxs,k]
                    
                    
                if k == 0:
                    # Initialize sub-model global parameters:
                    self.K[0,i] += 1
                    self.Mu[:,i] = x
                    self.X[0,i] = mat(power(norm(x),2))
                    
                    # Initialize sub-model cloud structure:
                    self.F[i] = x
                    self.XF[i] = mat(power(norm(x),2))
                    self.M[i] = mat(1,dtype=uint64)
                    self.r[i] = mat(r0)

                    
                    
                else:
                    self.K[0,i] += 1
                    self.Mu[:,i] = ((self.K[0,i]-1)/self.K[0,i])*self.Mu[:,i] + (1/self.K[0,i])*x
                    self.X[0,i] = ((self.K[0,i]-1)/self.K[0,i])*self.X[0,i] + (1/self.K[0,i])*mat(power(norm(x),2))
                    
                    x_density = computeDensity(x,self.Mu[:,i],self.X[0,i])
                    F_densities = computeDensity(self.F[i],self.Mu[:,i],self.X[0,i])
                    if (x_density<min(F_densities)) or (x_density>max(F_densities)):
                        self.F[i] = append(self.F[i],x,axis=1)
                        #
                        self.XF[i] = append(self.XF[i],mat(power(norm(x),2)),axis=1)
                        #
                        self.M[i] = append(self.M[i],mat(1),axis=1)
                        self.r[i]= append(self.r[i],mat(r0),axis=1)
                    else:
                        F_dists = computeDistance(x,self.F[i])
                        nearest_idx = argmin(F_dists)
                        nearest_dist = F_dists[nearest_idx]
                        if nearest_dist < self.r[i][:,nearest_idx]:
                            self.F[i][:,nearest_idx] = asscalar((self.M[i][0,nearest_idx]/(self.M[i][0,nearest_idx]+1)))*self.F[i][:,nearest_idx]+asscalar((1/(self.M[i][0,nearest_idx]+1)))*x
                            self.XF[i][:,nearest_idx] = asscalar((self.M[i][0,nearest_idx]/(self.M[i][0,nearest_idx]+1)))*self.XF[i][:,nearest_idx]+asscalar((1/(self.M[i][0,nearest_idx]+1)))*power(norm(x),2)
                            self.M[i][0,nearest_idx] += 1
                            a = self.XF[i][:,nearest_idx]-power(norm(self.F[i][:,nearest_idx]),2)
                            if a>=0:
                                self.r[i][0,nearest_idx] = sqrt(0.5*(power(self.r[i][0,nearest_idx],2)+a))
                            else:
                                self.r[i][0,nearest_idx] = sqrt(0.5*(power(self.r[i][0,nearest_idx],2)))

                        else:
                            self.F[i] = append(self.F[i],x,axis=1)
                            #
                            self.XF[i] = append(self.XF[i],mat(power(norm(x),2)),axis=1)
                            #
                            self.M[i] = append(self.M[i],mat(1),axis=1)
                            self.r[i] = append(self.r[i],mat(r0),axis=1)
            self.CloudsPerClass[0,i] = self.F[i].shape[1]
            self.FRDPperClass[0,i] = self.CloudsPerClass[0,i] / self.K[0,i]

        tEnd = timer()
        self.TrainingTime = tEnd-tStart
        self.NumOfClouds = sum(self.CloudsPerClass)
        self.FRDP = self.NumOfClouds/sum(self.K)
        self.X_train = []
        self.y_train = []

        return self
        
    def predict(self, x_test):
        
        if self.Normalization != "NORM":
            x_test = self.NormalizationTransform.transform(x_test)
            
            
        x_test = transpose(x_test)
        y_pred = zeros(x_test.shape[1])
        scores = zeros((len(self.F),x_test.shape[1]))
        for k in range(x_test.shape[1]):
            # For each class model:
            for i in range(len(self.F)):
                if self.Normalization == "NORM":
                    # Normalize sample:
                    x = transpose(mat(normSample(x_test[:,k])))
                else:
                    x = transpose(mat(x_test[:,k]))
                    
                # Compute class cloud distances:
                F_dists = computeDistance(x,self.F[i])
                # Compute class cloud activations:
                scores[i,k] = max(computeLambda(F_dists))
            # Predict sample class:
            y_pred[k] = self.Labels[0,argmax(scores[:,k])]
        return y_pred #scores

    def predict_proba(self, X):
        if self.Normalization != "NORM":
            x_test = self.NormalizationTransform.transform(X)
            
            
        x_test = transpose(X)
        y_pred = zeros(x_test.shape[1])
        scores = zeros((len(self.F),x_test.shape[1]))
        for k in range(x_test.shape[1]):
            # For each class model:
            for i in range(len(self.F)):
                if self.Normalization == "NORM":
                    # Normalize sample:
                    x = transpose(mat(normSample(x_test[:,k])))
                else:
                    x = transpose(mat(x_test[:,k]))
                    
                # Compute class cloud distances:
                F_dists = computeDistance(x,self.F[i])
                # Compute class cloud activations:
                scores[i,k] = max(computeLambda(F_dists))
            
        return scores

    def predict_activ(self, X):
        if self.Normalization != "NORM":
            x_test = self.NormalizationTransform.transform(X)
            
            
        x_test = transpose(X)
        y_pred = zeros(x_test.shape[1])
        scores = zeros((len(self.F),x_test.shape[1]))
        for k in range(x_test.shape[1]):
            # For each class model:
            for i in range(len(self.F)):
                if self.Normalization == "NORM":
                    # Normalize sample:
                    x = transpose(mat(normSample(x_test[:,k])))
                else:
                    x = transpose(mat(x_test[:,k]))
                    
                # Compute class cloud distances:
                F_dists = computeDistance(x,self.F[i])
                # Compute class cloud activations:
                scores[i,k] = max(computeLambda(F_dists))
            
        return scores


    def plotCloudSpace(self, x_dim=0, y_dim=1, subModel=-1, mesh_resol=1000):
        mesh_lims = [concatenate([self.F[i].min(axis=1) for i in range(0,len(self.F))],axis=1).min(axis=1)[[x_dim,y_dim],0],\
                     concatenate([self.F[i].max(axis=1) for i in range(0,len(self.F))],axis=1).max(axis=1)[[x_dim,y_dim],0]]

        mesh_grid = meshgrid(linspace(mesh_lims[0][0,0],mesh_lims[1][0,0],mesh_resol),\
                             linspace(mesh_lims[0][1,0],mesh_lims[1][1,0],mesh_resol))
            
        D_mesh = [zeros((self.F[i].shape[1],mesh_resol,mesh_resol)) for i in range(0,len(self.F))]
            
        for i in range(0,len(self.F)):
            for j in range (0,self.F[i].shape[1]):
                Fij = self.F[i][[x_dim,y_dim],j]
                Xij = self.XF[i][0,j]
                Dij = computeDensity(row_stack((mesh_grid[0].flatten(),mesh_grid[1].flatten())),\
                      Fij, Xij).reshape(mesh_resol,mesh_resol)
                D_mesh[i][j,:,:] = Dij

        D_mesh_max = [D_mesh[i].max(axis=0) for i in range(0,len(self.F))]
        
        
        fig, ax = plt.subplots()
        ax.pcolormesh(mesh_grid[0][0,:], mesh_grid[1][:,0], D_mesh_max[2])
        
        return 0
    
    
class ALMMo0_WCC:
    def __init__(self, costFunction):
        
        self.base_model = ALMMo_0()

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
        
    def fit(self, X_train, y_train, X_valid, y_valid):
        tStart = timer()

        # Set default class to be the majority class:
        self.MajClass = 0#np.argmax(valid_labelCount)
        self.MinClass = 1#np.argmin(valid_labelCount)
        if self.MajClass == self.MinClass:
            self.MajClass = 0
            self.MinClass = 1

        
        # Train unweighted model:
        self.base_model.fit(X_train, y_train)
        
        # Compute minority class cloud max activations for all training samples:
        self.LambdaValid = self.base_model.predict_proba(X_valid)

        FN_Lambda = self.LambdaValid[:,np.logical_and(y_valid ==\
                    self.MinClass, np.argmax(self.LambdaValid, axis=0) == self.MajClass)]

        # Find candidate weights using the minority class samples activations:
        cand_weights = np.hstack((np.array([0.5]),\
                       np.sort(np.divide(FN_Lambda,\
                       np.sum(FN_Lambda,axis=0))[self.MajClass,:])))
            
        if cand_weights.size > 1 and max(cand_weights) > 0.5:
            
            opt = Optimizer(dimensions= [Real(0.500,max(cand_weights))],\
                            acq_func='LCB')   
    
            x0 = [0.5]
            f0 = self.CostFunction(x0, y_valid)
            opt.tell(x0,f0)
            #print('Initial Cost: {}'.format(f0))
    
            max_iters = 50
            iter_count = 1
            stop_criterion = False
            n_best = 5
    
            # Optimization loop:
            while stop_criterion==False:
                x_suggested = opt.ask()
                y_suggested = self.CostFunction(x_suggested, y_valid)
    
                opt.tell(x_suggested, y_suggested)
                #print('Iteration: {}, x:{}, y:{}'.format(iter_count,x_suggested,y_suggested))
    
                if iter_count >= n_best:
                    best_idxs = np.argsort(opt.yi)[:n_best]
                    f_best = [opt.yi[best_idx] for best_idx in best_idxs]
                    x_best = [opt.Xi[best_idx][0] for best_idx in best_idxs]
                    if max(f_best)-min(f_best) < 0.000001 or max(x_best)-min(x_best) < 0.000001:
                        stop_criterion = True
                        #print('Converged...')
                    elif iter_count >= max_iters:
                        stop_criterion = True
                        #print('Reached max iterations...')
                iter_count += 1
    
            self.OptReport = opt.get_result()
            self.Weights = np.array([1.0-self.OptReport.x[0],self.OptReport.x[0]])
            
        else:
            self.OptReport.append(None)
            self.Weights.append(np.array([0.5, 0.5]))
            print("No minority class candidates!")
            
        tEnd = timer()
        self.TrainingTime = tEnd-tStart
        
        
    def predict_proba(self, X):
        lambda_base = self.base_model.predict_proba(X)
        lambda_weight = np.multiply(lambda_base, np.reshape(self.Weights,(2,1))) 
        return lambda_weight
                


    def CostFunction(self, weight, y_true):
        weight = weight[0]
        weight_mask = zeros((2,1))
        weight_mask[self.MinClass] = weight
        weight_mask[self.MajClass] = 1.0 - weight

        y_pred = np.argmax(np.multiply(weight_mask, self.LambdaValid), axis=0)

        metric_score = np.float64(0)
        
        
        if self.CostFunctionID == 1: # Geometric Mean
            metric_score = geometric_mean(y_true, y_pred)
        elif self.CostFunctionID == 2: # F1-Score
            metric_score = f1_score(y_true, y_pred)
        elif self.CostFunctionID == 3:  # Cohen's Kappa Coefficient
            metric_score = cohen_kappa_score(y_true, y_pred)
        elif self.CostFunctionID == 4: # Mathew's Correlation Coefficient
            metric_score = matthews_corrcoef(y_true,y_pred)
            
        return 1.0 - metric_score  
    
    
# class ALMMo_sys():
#     def __init__(self, problemType, p=1, epochs=1, deleteStale=False):
#         # Dataset variables:
#         self.X_train = []
#         self.X_test = []
#         self.x_idxs = []
#         self.y_idxs = []
        
#         self.label_encoder = LabelBinarizer()

#         #self.Type = type # Problem type

#         # System variables:
#         self.ProblemType = problemType
#         self.m = -1 # Input size
#         self.n = -1 # Output size
#         self.p = p # Consequents order
#         self.Epochs = epochs # Number of training epochs
#         self.EXP = []


#         # Global variables:
#         self.K = -1
#         self.Mu = []
#         self.X = -1.0

#         # Cloud variables:
#         self.M = array([-1])
#         self.F = []
#         self.XF = array([-1.0])
#         self.C = []
#         self.A = []
#         self.B = array([-1])
#         self.P = array([-1.0])

#         # Performance variables:
#         # self.y_pred = zeros(self.X_test.shape[1])
#         # self.y_true = zeros(self.X_test.shape[1])

#         #self.ParamRecord = paramRecord
#         self.DeleteStale = deleteStale
#         self.NumOfClouds = -1
#         self.FRDP = -1.0
#         self.TrainingTime = -1.0
#         self.CloudHistory = array([-1])
#         self.StaleCloudHistory = array([-1])
#         self.MeanParamVar = array([-1.0])
#         self.Report = []

#     def fit(self, x_train, y_train):
#         # Initialize data variables:
            
#         if self.ProblemType == 'MultiClass':
#             y_train = self.label_encoder.fit_transform(y_train)
            
            
#         self.X_train = mat(transpose(hstack((x_train,y_train))))
#         self.x_idxs = array(list(range(0,x_train.shape[1],1))) #array(list(range(0,self.X_train.shape[0]-1,1)))
#         self.y_idxs = array(list(range(x_train.shape[1],x_train.shape[1]+y_train.shape[1],1))) #array(self.X_train.shape[0]-1)
#         self.m = self.x_idxs.size # Input size
#         self.n = self.y_idxs.size # Output size
#         self.EXP, self.m = computeConseqExpMatrix(self.m,self.p)
        
#         # Initialize global parameters:
#         x1 = self.X_train[self.x_idxs,0]
#         self.Mu = zeros_like(self.X_train[self.x_idxs,0])
#         self.Mu[:,0] = x1
#         self.K = 1
#         self.X = sum(power(x1,2))


#         # Initialize cloud variables:
#         self.F = zeros_like(self.Mu)
#         self.C = zeros((self.m,self.m,1))
#         self.A = zeros((self.m,self.n,1))



#         tStart = timer()

#         # Create first cloud:
#         self.M[0] = 1
#         self.F[:,0] = x1
#         self.XF[0] = sum(power(x1,2))
#         self.C[:,:,0] = omega0*eye(self.m)
#         self.B[0] = 1
#         self.P[0] = 1.0 #0.0
#         self.CloudHistory[0] = 1
#         self.StaleCloudHistory[0] = 0
#         self.MeanParamVar[0] = 0.0
        
#         np.random.seed(42)

#         # For each epoch:
#         for e in range(self.Epochs):
#             #print(e)
#             # Shuffle X_train
#             if e > 0:
#                 self.X_train = transpose(self.X_train)
#                 np.random.shuffle(self.X_train)
#                 self.X_train = transpose(self.X_train)

#             # For each training sample:
#             for k in range(1,self.X_train.shape[1]):
#                 print(self.X_train.shape[1]-k)
#                 x = self.X_train[self.x_idxs,k]
#                 u = np.prod(np.power(np.transpose(x),self.EXP),axis=1)
#                 y = np.transpose(np.array(self.X_train[self.y_idxs,k],ndmin=2))
                
#                 # Update global parameters:
#                 self.K += 1
#                 self.Mu = ((self.K-1)/self.K)*self.Mu + (1/self.K)*x
#                 self.X = ((self.K-1)/self.K)*self.X + (1/self.K)*sum(power(x,2))

#                 # Compute sample and clouds global densities:
#                 Dx = computeDensity(x,self.Mu,self.X)
#                 D = computeDensity(self.F,self.Mu,self.X)

#                 # Compute cloud activations:
#                 DF = computeAdjustedDensity(x,self.F,self.XF,self.M)
#                 lambdas = DF/sum(DF)

#                 # if self.ParamRecord == True:
#                 #     # Compute output:
#                 #     y_pred = einsum('i,jki->jk',lambdas,einsum('ij,jkl->ikl',np.transpose(u),self.A))
#                 #     error = sum(np.absolute(y-y_pred))
#                 #     #print(error)

#                 # Condition 1:
#                 if Dx<min(D) or Dx>max(D):
#                     # Detect overlapping cloud:S
#                     F_over = argmax(DF)
#                     # Condition 2:
#                     if DF[F_over] > lambda0:
#                         # Update overlapping cloud:
#                         self.F[:,F_over] = (x + self.F[:,F_over])/2
#                         self.XF[F_over] = (sum(power(x,2)) + self.XF[F_over])/2
#                         self.B[F_over] = self.K
#                         self.P[F_over] = 0.0
#                         self.M[F_over] = ceil((1+self.M[F_over])/2)
#                     else:
#                         # Create new cloud:
#                         self.F = hstack((self.F,x))
#                         self.M = append(self.M,array([1]))
#                         self.XF = append(self.XF,sum(power(x,2)))
#                         self.B = append(self.B,array([self.K]))
#                         self.P = append(self.P,array([0.0]))
#                         self.C = dstack((self.C,omega0*eye(self.m)))
#                         self.A = dstack((self.A,sum(self.A,axis=2)/self.A.shape[2]))
#                 else:
#                     # Update nearest cloud:
#                     F_dists = computeDistance(x,self.F)
#                     F_near = argmin(F_dists)
#                     self.M[F_near] += 1
#                     self.F[:,F_near] = ((self.M[F_near]-1)/self.M[F_near])*self.F[:,F_near] + (1/self.M[F_near])*x
#                     self.XF[F_near] = ((self.M[F_near]-1)/self.M[F_near])*self.XF[F_near] + (1/self.M[F_near])*sum(power(x,2))

                

#                 # Compute cloud activations:
#                 DF = computeAdjustedDensity(x,self.F,self.XF,self.M)
#                 lambdas = DF/sum(DF)  

#                 if self.DeleteStale == True:
#                     # Compute cloud utilities:
#                     self.P += lambdas
#                     etas = zeros_like(lambdas)
#                     cloud_idxs = (self.B!=self.K) 
#                     etas[cloud_idxs]  = self.P[cloud_idxs] /(self.K-self.B[cloud_idxs] +1)
#                     etas[np.invert(cloud_idxs)] = 1.0
#                     # Remove stale clouds:
#                     F_non_stale = (etas>eta0)
#                     self.F = self.F[:,F_non_stale]
#                     self.M = self.M[F_non_stale]
#                     self.XF = self.XF[F_non_stale]
#                     self.B = self.B[F_non_stale]
#                     self.P = self.P[F_non_stale]
#                     self.C = self.C[:,:,F_non_stale]
#                     self.A = self.A[:,:,F_non_stale]
#                     # Update cloud activations:
#                     lambdas = DF[F_non_stale]/sum(DF[F_non_stale]) 


#                 # if self.ParamRecord == True:
#                 #     self.CloudHistory = np.append(self.CloudHistory,self.F.shape[1])
#                 #     if self.DeleteStale == True:
#                 #         self.StaleCloudHistory = np.append(self.StaleCloudHistory,sum(np.invert(F_non_stale)))

#                 # Update consequent parameters:
#                 # if self.ParamRecord == False:

#                 self.C, self.A = computeWeightedRLS(self.C,self.A,u,lambdas,y,mode='matrix')

#                 # else:
#                 #     Ao = np.copy(self.A)
#                 #     self.C, self.A = computeWeightedRLS(self.C,self.A,u,lambdas,y,mode='matrix')
#                 #     A_var = np.nan_to_num(np.absolute((self.A-Ao)/Ao),posinf=0.0)
#                 #     A_mean_var = np.mean(A_var)
#                 #     self.MeanParamVar = np.append(self.MeanParamVar,A_mean_var)


#         tStop = timer()
#         self.TrainingTime = tStop-tStart
#         self.NumOfClouds = self.C.shape[2]
#         self.FRDP = self.NumOfClouds/self.K


#     def predict(self, x_test):
        
#         x_test = transpose(x_test)
#         y_pred = np.zeros(x_test.shape[1])
#         probs = np.zeros(x_test.shape[1])
#         scores = np.zeros((self.F.shape[1],x_test.shape[1]))

#         for k in range(0, x_test.shape[1]):
#             # Read sample:
#             x = transpose(mat(x_test[self.x_idxs,k]))
#             u = np.prod(np.power(np.transpose(x),self.EXP),axis=1)
#             #label = np.transpose(np.array(self.X_test[self.y_idxs,k],ndmin=2))
#             # Compute cloud activations:
#             DF = computeAdjustedDensity(x,self.F,self.XF,self.M)
#             lambdas = DF/sum(DF)
#             scores[:,k] = lambdas
#             # Compute output:
#             pred = einsum('i,jki->jk',lambdas,einsum('ij,jkl->ikl',np.transpose(u),self.A))


#             if self.ProblemType== 'Regression':
#                 y_pred[k] = pred

#             elif self.ProblemType == 'BinClass':
#                 probs[k] = pred
#                 if pred > 0.5:
#                     y_pred[k] = 1
#                 else:
#                     y_pred[k] = 0

#             elif self.ProblemType == 'MultiClass':
#                 #self.Labels = []
#                 pred_temp = zeros(self.n)
#                 pred_temp[np.argmax(pred)] = 1
#                 y_pred[k] = self.label_encoder.inverse_transform(np.mat(pred_temp))

#         return y_pred #, probs, scores


#     def plotCloudHistory(self):
#         data = np.hstack((np.transpose(np.mat(np.linspace(1,self.K,num=self.K))),np.transpose(np.mat(self.CloudHistory))))
#         dataTable = pd.DataFrame(data=data,columns=["Iteration","Number Of Clouds"])
#         sb.lineplot(data=dataTable,x='Iteration',y='Number Of Clouds')

#     def plotParamHistory(self):
#         data = np.hstack((np.transpose(np.mat(np.linspace(1,self.K,num=self.K))),np.transpose(np.mat(self.MeanParamVar))))
#         dataTable = pd.DataFrame(data=data,columns=["Iteration","Mean Parameter Variation"])
#         sb.lineplot(data=dataTable,x='Iteration',y='Mean Parameter Variation')

#     def plotSISOoutput(self):
#         x_vals = np.linspace(0,1.0,1000)
#         y_pred = zeros(x_vals.shape[0])
#         for i in range(x_vals.shape[0]):
#             x = x_vals[i]
#             u = np.transpose(np.mat(np.prod(np.power(np.transpose(x),self.EXP),axis=1)))
#             # Compute cloud activations:
#             M_temp = self.M + 1
#             F_temp = self.F*diag((M_temp-1)/M_temp) + x*(1/M_temp)
#             XF_temp = self.XF*((M_temp-1)/M_temp)+ sum(power(x,2))*(1/M_temp)
#             DF = computeDensity(x,F_temp,XF_temp,mode='local')
#             lambdas = DF/sum(DF)
#             # Compute output:
#             y_pred[i] = einsum('i,jki->jk',lambdas,einsum('ij,jkl->ikl',np.transpose(u),self.A))
#         # Plot outputs:
#         data = np.hstack((np.transpose(np.mat(x_vals)),np.transpose(np.mat(y_pred))))
#         dataTable = pd.DataFrame(data=data,columns=["x","y"])
#         sb.lineplot(data=dataTable,x='x',y='y')
    
    