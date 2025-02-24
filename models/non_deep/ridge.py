# Angus Dempster, Chang Wei Tan, Lynn Miller
# Navid Mohammadi Foumani, Daniel F Schmidt, and Geoffrey I Webb
# Highly Scalable Time Series Classification for Very Large Datasets
# AALTD 2024 (ECML PKDD 2024)

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from utils import stratified_split

def binarize(Y, n):
    return -torch.ones(Y.shape[0], n).scatter(-1, torch.tensor(Y[:, None]).long(), -1)

class Scaler(nn.Module):

    def __init__(self, **kwargs):

        super().__init__()

        self.register_buffer("_mean", torch.tensor(0))
        self.register_buffer("_std", torch.tensor(0))
        self.register_buffer("_count", torch.tensor(0))
        self.register_buffer("_eps", torch.tensor(kwargs.get("eps", np.finfo(np.float32).eps * 10)))

        self._with_std = kwargs.get("with_std", True)

    def partial_fit(self, X):

        batch_size = X.shape[0]
        new_count = self._count + batch_size

        batch_mean = X.mean(0)
        batch_std = X.std(0) if batch_size > 1 else 0

        self._mean = self._mean + ((batch_mean - self._mean) * (batch_size / new_count))
        self._std = self._std + ((batch_std - self._std) * (batch_size / new_count))

        self._count = new_count

    def fit(self, X):

        self._mean = X.mean(0)
        self._std = X.std(0)

    def scale(self, X):

        if self._with_std:
            return (X - self._mean) / (self._std + self._eps)
        else:
            return (X - self._mean)

class RidgeClassifier():

    def __init__(self, transform, device = "cpu", **kwargs):

        self.transform = transform

        self.device = device
        
        self.X_scaler = kwargs.get("X_scaler", Scaler())
        self.Y_scaler = kwargs.get("Y_scaler", Scaler(with_std = False))

        self.lambdas = kwargs.get("lambdas", torch.logspace(-6, 6, 21))
        
        self.verbose = kwargs.get("verbose", False)

    def fit(self, training_data, **kwargs):

        n = training_data.shape[0]
        p = self.transform.num_features

        k = kwargs.get("num_classes", len(training_data.classes))
        self.k = k

        max_val_size = kwargs.get("max_val_size", 8_192)
        val_size = min(int(training_data.shape[0] * 0.2), max_val_size)

        eps = np.finfo(np.float32).eps

        # ======================================================================
        # == n < p =============================================================
        # ======================================================================

        if n < p:

            X0 = torch.zeros((n, p), device = self.device)
            Y0 = torch.zeros((n, k), device = self.device)

            batch_count = np.ceil(training_data.shape[0] / training_data.batch_size)

            i = 0
            
            for X, Y in tqdm(training_data, total = batch_count, disable = not self.verbose):

                j = i + X.shape[0]
    
                _X = self.transform(torch.tensor(X.astype(np.float32, copy = False)).to(self.device))
    
                _Y = binarize(Y, k)

                X0[i:j] = _X
                Y0[i:j] = _Y

                i = j

            self.X_scaler.fit(X0)
            X0 = self.X_scaler.scale(X0)

            self.Y_scaler.fit(Y0)
            self.B0 = self.Y_scaler._mean.to(self.device)
            Y0 = self.Y_scaler.scale(Y0)
            
            S2, U = torch.linalg.eigh((X0 @ X0.T))
            S2 = S2.clip(eps)
            S = S2.sqrt()
            V = (X0.T @ U) * (1 / S)

            R = U * S
            R2 = R ** 2

            RTY = R.T @ Y0
                
            best_alpha_hat = None
            best_error = np.inf

            Y_TRUE = Y0.argmax(-1)
    
            for lambda_ in self.lambdas * np.sqrt(n):
    
                alpha_hat = (1 / (S2[:, None] + lambda_)) * RTY
            
                Y_hat = R @ alpha_hat
    
                E = Y0 - Y_hat
    
                diag_H = (R2 / (S2 + lambda_)).sum(1)
                
                E_loocv = E / (1 - diag_H[:, None]).clip(eps)
                
                err_lambda = (E_loocv ** 2).mean()
    
                if err_lambda < best_error:
    
                    best_error = err_lambda
                    best_alpha_hat = alpha_hat
                
                delta = E_loocv - E
                Y_loocv = Y_hat - delta

                YP = Y_loocv.argmax(-1)

            self.B = V @ best_alpha_hat

        # ======================================================================
        # == n >= p ============================================================
        # ======================================================================

        else:

            TR, VA = stratified_split(training_data.Y, val_size)

            TR = np.sort(TR)
            VA = np.sort(VA)

            training_data_1 = training_data[TR]
            validation_data = training_data[VA]

            n1 = training_data_1.shape[0]

            XTX = torch.zeros((p, p), device = self.device)
            XTY = torch.zeros((p, k), device = self.device)
    
            batch_count = np.ceil(training_data_1.shape[0] / training_data_1.batch_size)

            for X, Y in tqdm(training_data_1, total = batch_count, disable = not self.verbose):
    
                _X = self.transform(torch.tensor(X.astype(np.float32, copy = False)).to(self.device))
    
                _Y = binarize(Y, k).to(self.device)
    
                self.X_scaler.partial_fit(_X)
                self.Y_scaler.partial_fit(_Y)
                
                _XT = _X.T

                XTX = XTX + (_XT @ _X)
                XTY = XTY + (_XT @ _Y)

            mX = self.X_scaler._mean
            sX = self.X_scaler._std + np.finfo(np.float32).eps * 10

            self.B0 = self.Y_scaler._mean.to(self.device)
            mY = self.B0
    
            mXX = mX[:, None] @ mX[None, :] * np.float32(n1)
            sXX = sX[:, None] @ sX[None, :]
    
            mXY = mX[:, None] @ mY[None, :] * np.float32(n1)

            XTX = (XTX - mXX) / sXX
            XTY = (XTY - mXY) / sX[:, None]
            
            S2, V = torch.linalg.eigh(XTX.to(self.device))
            S2 = S2.clip(eps)

            XV = torch.zeros((validation_data.shape[0], p), device = self.device)
            YV = torch.zeros(validation_data.shape[0], dtype = torch.int64, device = self.device)
    
            i = 0
    
            for X, Y in validation_data:
    
                j = i + X.shape[0]

                _XV = self.transform(torch.tensor(X.astype(np.float32)).to(self.device))
                _XV = self.X_scaler.scale(_XV)
                
                XV[i:j] = _XV
                YV[i:j] = torch.tensor(Y, dtype = torch.int64)
    
                i = j
        
            best_error = np.inf

            self.YV = YV.clone()
            self.XV = XV.clone()

            for lambda_ in self.lambdas * np.sqrt(n1):
    
                _XTXi = (V * (1 / (S2 + lambda_))) @ V.T
    
                _B = _XTXi @ XTY
                
                err_lambda = (YV != ((XV @ _B) + self.B0).argmax(-1)).float().mean()
        
                if err_lambda < best_error:
    
                    best_error = err_lambda
                    self.B = _B.clone()
    
            validation_data.close()
            training_data_1.close()

    def _predict(self, X):

        _X = self.transform(torch.tensor(X.astype(np.float32, copy = False)).to(self.device))
        _X = self.X_scaler.to(_X.device).scale(_X)
        
        return _X @ self.B + self.B0

    def score(self, data):

        incorrect = 0
        count = 0

        for X, Y in tqdm(data, total = np.ceil(data.shape[0] / data.batch_size), disable = not self.verbose):

            incorrect += (torch.tensor(Y).to(self.device) != self._predict(X).argmax(-1)).sum()
            count += X.shape[0]

        return incorrect / count

    def score_logloss(self, data):

        eps = torch.finfo(torch.float32).eps

        incorrect = 0
        loss = 0

        count = 0

        for X, Y in tqdm(data, total = np.ceil(data.shape[0] / data.batch_size), disable = not self.verbose):

            Z = self._predict(X)

            _Y = torch.tensor(Y, dtype = torch.int64).to(self.device)

            incorrect += (_Y != Z.argmax(-1)).sum()
            loss += F.nll_loss(F.softmax(Z, -1).clip(eps, 1 - eps).log(), _Y, reduction = "sum")

            count += X.shape[0]

        return incorrect / count, loss / count