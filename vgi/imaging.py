import numpy as np
import torch
import torch.nn.functional as F
import time
import json
import cv2 
import matplotlib.pyplot as plt
import os
import sys
import vgi
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from vgi.ssim import SSIM
from vgi.ssim import MSSSIM
from torchvision import transforms

img2Tensor = transforms.ToTensor()
# SSIM from
# https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py
          
ARGN = 9  # mu_x, mu_y, sigma_x, sigma_y, theta, alpha, beta_r, beta_g, beta_b
          # 0     1     2        3         4     5      6       7       8
# Samples are uniformly distributed over the half-open interval [low, high)
# return [n, ARGN]
def randArg(argMin, argMax, n = 1):
    return np.random.uniform(argMin, argMax, [n, ARGN]) 

def clone(_tensor):
    return _tensor.detach().clone()

def toNumpy(_tensor):
    return _tensor.detach().cpu().numpy()  

def toNumpyImage(_tensor, normalize = False):
    img = _tensor.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
    if normalize:
        img = vgi.normalize(img)
    return img
# 4 bytes * 3 channels * 9 dimentions * 10 batch_size * 7 * 7 patch_size * 256 * 256 pixels
# 108 * 10 batch_size * 7 * 7 patch_size * 256 * 256 pixels
# 1080 * 7 * 7 patch_size * 256 * 256 pixels
# 1080 * 7 * 7 patch_size * 256 * 256 pixels
# 52920 * 256 * 256 pixels
# 3,468,165,120 ==> 3.23 GB
class GaussImaging:
    # All torch.tensor of images must be packed as (batch_size, channels, height, width)
    #lobe = 4.0
    #lobeq = 16
    def tensor(self, data):
        return torch.tensor(data, dtype = torch.float, device = self.device)
    def zeros(self, shape):
        return torch.zeros(shape, dtype = torch.float, device = self.device)
    def ones(self, shape):
        return torch.ones(shape, dtype = torch.float, device = self.device)   
    def full(self, shape, value):
        return torch.full(shape, value, dtype = torch.float, device = self.device)                
    
    def __init__(self, target = None, arg = None, shape = (1, 3, 256, 256), 
                    bg = [0.0, 0.0, 0.0], _prev = None, gpu = True, window_size = 7, 
                    random_type = 0, min_size = 1.0, ssim_alpha = 0.84):
        self.debug = False
        self._debugData = None
        self.gamma = 80.0
        if gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.gpu = True
        else:
            self.device = torch.device("cpu")
            self.gpu = False

        if target is None:   
            self._target = None
            self.shape = shape
            self.images, self.channels, self.height, self.width = self.shape
        else:
            self._target = self.tensor(target).permute(2, 0, 1).unsqueeze(0)
            self.shape = self._target.shape 
            self.images, self.channels, self.height, self.width = self.shape
            

        self.pixels = self.height * self.width
        self.boundary = vgi.imageBoundary((self.height, self.width))  # (ymin, ymax, xmin, xmax) in Cartesian  

        self._Aini = None # [nQ, ARGN], argument set
        self._A = None    # [nQ, ARGN], argument set

        if not (arg is None):
            self.setInitalArg(arg)

        # self._X: [H, W], X coordinates
        # self._Y: [H, W], Y coordinates 
        self._Y, self._X = torch.meshgrid(torch.arange(self.boundary[0], self.boundary[1], device = self.device), 
                                          torch.arange(self.boundary[2], self.boundary[3], device = self.device))
        self._Y = self._Y.flatten() # (pixels)
        self._X = self._X.flatten() # (pixels)
        self._bg = self.tensor(bg)
        if _prev is None:
            self._I = self._bg.repeat((self.height, self.width, 1)).permute(2, 0, 1).unsqueeze(0)
        else:
            self._I = _prev

        self._err = None

        #self._step_size = torch.ones([ARGN, 1], dtype = torch.float, device = self.device)
        self._step_size = torch.tensor( [0.5, 0.5, 0.5, 0.5, np.pi / 180., 
                                          1/256.0, 1/256.0, 1/256., 1/256], device = self.device).unsqueeze(dim = -1)
        self.argMin = np.array([-self.width / 2.0, -self.height / 2.0, min_size, min_size, -np.pi, 0.01, 0.0, 0.0, 0.0])
        self.argMax = np.array([ self.width / 2.0, self.height / 2.0, self.width , self.height , np.pi, 0.9, 1.0, 1.0, 1.0])
        self._expThres = self.tensor(-15.0)
        self._one = self.tensor(1.0)    
        self._zero = self.tensor(0.0) 
        self._sigmoidThres = self.tensor(10.0) 
        self._sigmoidThresN = -self._sigmoidThres           

        self.ssim_alpha = ssim_alpha
        self._weight = None
        self._mse = torch.nn.MSELoss()
        self._L1 = torch.nn.L1Loss()
        if target is None:
            self._SSIM = SSIM(val_range = 1.0, window_size = window_size, mean_axis = (1, 2, 3))
        else:
            self._SSIM = SSIM(img2 = self._target, window_size = window_size, mean_axis = (1, 2, 3), val_range = 1.0)

        self._MSSSIM = MSSSIM(val_range = 1.0)   

        self._sigmoid = torch.nn.Sigmoid()

        # load brush1
        #brushDir = 'brush/'
        brushDir = 'vgi/brush/'
        #brushDir = os.path.dirname(vgi.__file__) + '/brush/'
        #print('brush path', brushDir)

        imgBrush = 1.0 - vgi.loadImage(brushDir + 'brush1.png', gray = True)
        #print('imgBrush', imgBrush.shape)
        imgBrush = imgBrush.reshape(imgBrush.shape[0:2])
        #imgBrush = cv2.GaussianBlur(imgBrush,(5, 5), 1.0)
        self._brush1 = self.tensor(imgBrush)
        
        imgBrush = 1.0 - vgi.loadImage(brushDir + 'brush2.png', gray = True)
        imgBrush = imgBrush.reshape(imgBrush.shape[0:2])
        #imgBrush = cv2.GaussianBlur(imgBrush,(5, 5), 1.0)
        self._brush2 = self.tensor(imgBrush) 

        imgBrush = 1.0 - vgi.loadImage(brushDir + 'brush3.png', gray = True)
        #imgBrush = vgi.normalize((1.0 - vgi.loadImage('brush3.png', gray = True)) * 2.0 + 0.5)
        imgBrush = imgBrush.reshape(imgBrush.shape[0:2])
        #imgBrush = cv2.GaussianBlur(imgBrush,(5, 5), 1.0)
        self._brush3 = self.tensor(imgBrush)   

        imgBrush = 1.0 - vgi.loadImage(brushDir + 'brush4.png', gray = True)        
        imgBrush = imgBrush.reshape(imgBrush.shape[0:2])        
        self._brush4 = self.tensor(imgBrush)                 

        imgBrush = 1.0 - vgi.loadImage(brushDir + 'brush5.png', gray = True)        
        imgBrush = np.clip(imgBrush* 2.2, 0.1, 1.0) 
        imgBrush = vgi.normalize(imgBrush)
        #vgi.showImage(1.0 - imgBrush)  
        #vgi.saveImage('brush5_en.png', 1.0 - imgBrush)
        imgBrush = imgBrush.reshape(imgBrush.shape[0:2])        
        self._brush5 = self.tensor(imgBrush)                 

        imgBrush = 1.0 - vgi.loadImage(brushDir + 'brush6.png', gray = True)        
        imgBrush = imgBrush.reshape(imgBrush.shape[0:2])        
        self._brush6 = self.tensor(imgBrush)      

        imgBrush = 1.0 - vgi.loadImage(brushDir + 'brush7.png', gray = True)        
        imgBrush = imgBrush.reshape(imgBrush.shape[0:2])        
        self._brush7 = self.tensor(imgBrush)                     

        self.random_type = random_type
        # random selection with high-entropy features
        if self.random_type & 1:
            target_blur = cv2.GaussianBlur(target, (window_size, window_size), 1.0)
            target_gray = np.mean(target_blur, axis = 2)

            target_ub = np.array(target_gray * 256, dtype = np.uint8)
            target_enp = vgi.normalize(entropy(target_ub, disk(window_size)))
            target_enp = np.expand_dims(target_enp, axis = -1)
            fiels = [('enp', float), ('x', float), ('y', float), ('r', float), ('g', float), ('b', float)]
            pxm = np.zeros(self.pixels, dtype = fiels)
            pxm['enp'] = target_enp.flatten()
            pxm['x'] = toNumpy(self._X)
            pxm['y'] = toNumpy(self._Y)
            pxm['r'] = target_blur[:,:,0].flatten()
            pxm['g'] = target_blur[:,:,1].flatten()
            pxm['b'] = target_blur[:,:,2].flatten()
            self.pxm = np.sort(pxm, order='enp') 

    # GaussImaging::__init__

    def image(self, BHWC = True):
        if self._I is None:
            return None
        else:    
            _I = self._I 
            if BHWC:
                # self._I is (batch_size, channels, height, width)
                _I = _I.permute(0, 2, 3, 1)                 
                if(_I.shape[0] == 1):
                    return toNumpy(_I[0])
                else:
                    return toNumpy(_I)
    def setWeight(self, weight):
        height = weight.shape[0]
        width = weight.shape[1]
        self._weight = self.tensor(weight).reshape((1, height, width))

        fiels = [('enp', float), ('x', float), ('y', float), ('r', float), ('g', float), ('b', float)]
        pxm = np.zeros(self.pixels, dtype = fiels)
        pxm['enp'] = weight.flatten()
        pxm['x'] = toNumpy(self._X)
        pxm['y'] = toNumpy(self._Y)
        pxm = np.sort(pxm, order='enp')
        ID = np.argwhere(pxm['enp'] <= 0.0001)
        pxm = np.delete(pxm, ID, axis = 0)    
        ID = np.argwhere(pxm['enp'] < 1.0)   
        self.pxm_h = np.delete(pxm, ID, axis = 0) 
        ID = np.argwhere(pxm['enp'] >= 1.0)  
        self.pxm_l = np.delete(pxm, ID, axis = 0)     
        self.pxm = pxm


    def X(self):
        return toNumpy(self._X)
    def Y(self):
        return toNumpy(self._Y)

    def setBackground(self, bg):
        self._bg = self.tensor(bg)
        self._I = self._bg.repeat((self.height, self.width, 1)).permute(2, 0, 1).unsqueeze(0)

    def background(self):
        if self._bg is None:
            return np.array([0.0, 0.0, 0.0])
        else:
            return toNumpy(self._bg)

    def argument(self):
        if self._A is None:
            return None
        else:                    
            arg = toNumpy(self._A)
            nQ, nArg, _ = arg.shape
            arg = arg.reshape([nQ, nArg])
            return arg

    def error(self):
        if self._err is None:
            return None
        else:
            return toNumpy(self._err)


    def setInitialArg(self, arg):
        nQ = arg.shape[0]
        self._Aini = self.tensor(arg.reshape([nQ, ARGN, 1]))
        #self.updateArg()
    # GaussImaging::setInitialArg

    def setArg(self, arg):
        nQ = arg.shape[0]
        self._A = self.tensor(arg.reshape([nQ, ARGN, 1]))
        #self.updateArg()
    # GaussImaging::setArg    
    
    def appendArg(self, _arg):
        if self._A is None:
            self._A = _arg.reshape([1, ARGN, 1])
        else:
            self._A = torch.cat( (self._A, _arg.reshape([1, ARGN, 1])) )
    # GaussImaging::appendArg
    
    def lossL1(self, _I, mean_axis = (1, 2, 3)):
        _loss = (_I -  self._target).abs()
        if not(self._weight is None):
            _loss *= self._weight
        if mean_axis == 0:
            _loss = _loss.mean()    
        elif not(mean_axis is None):
            _loss = _loss.mean(dim = mean_axis) 
        return _loss                 
    
    def lossMSE(self, _I, mean_axis = (1, 2, 3) ):
        _loss = ((_I -  self._target) ** 2.0)
        if not(self._weight is None):
            _loss *= self._weight
        if mean_axis == 0:
            _loss = _loss.mean()    
        elif not(mean_axis is None):
            _loss = _loss.mean(dim = mean_axis) 
        _loss = _loss ** 0.5
        return _loss                 

        
    def lossSSIM(self, _I, mean_axis = (1, 2, 3) ):
        #n_dim = len(_I.shape)
        #if n_dim == 3:
        #    _I = _I.permute(2, 0, 1).unsqueeze(0)
        #elif n_dim == 4:
        #    _I = _I.permute(0, 3, 1, 2)
        if self._weight is None:
            self._SSIM.mean_axis = mean_axis
            _loss = self._one - self._SSIM.forward(_I)  

        else:
            self._SSIM.mean_axis = None
            _loss = (self._one - self._SSIM.forward(_I)) * self._weight  
            if mean_axis == 0:
                _loss = _loss.mean()
            elif not(mean_axis is None):
                _loss = _loss.mean(dim = mean_axis)                  
        return _loss       
    
    def lossMSSSIM(self, _I, size_average=True):
        #_I = _I.permute(2, 0, 1).unsqueeze(0)
        return self._one - self._MSSSIM.forward(_I, self._targetCHW)
    
    def lossSSIML1(self, _I, mean_axis = (1, 2, 3)):
        self._SSIM.L1 = True
        self._SSIM.alpha = self.ssim_alpha
        if self._weight is None:
            self._SSIM.mean_axis = mean_axis
            _loss = self._SSIM.alpha - self._SSIM.forward(_I)  

        else:
            self._SSIM.mean_axis = None
            _loss = (self._SSIM.alpha - self._SSIM.forward(_I)) * self._weight  
            if mean_axis == 0:
                _loss = _loss.mean()
            elif not(mean_axis is None):
                _loss = _loss.mean(dim = mean_axis)                  
        self._SSIM.L1 = False
        return _loss


    def lossMSSSIML1(self, _I, size_average=True):
        self._MSSSIM.L1 = True
        self._MSSSIM.alpha = 0.9
        err =  self._MSSSIM.alpha - self._MSSSIM.forward(_I, self._targetCHW)
        self._MSSSIM.L1 = False
        return err  
  
    def scale(self, x, y = None, shape = None):
        if y is None:
            y = x
        if not(shape is None):
            self.shape = shape         
        else:
            self.shape = (1, self.channels, self.height * y, self.width * x)
        self.images, self.channels, self.height, self.width = self.shape
        self.pixels = self.height * self.width

        if not(self._Aini is None) and (self._Aini.shape[0] > 0):
            self._Aini[:, 0, 0] *= x
            self._Aini[:, 1, 0] *= y
            self._Aini[:, 2, 0] *= x
            self._Aini[:, 3, 0] *= y
        if not(self._A is None) and (self._A.shape[0] > 0):
            self._A[:, 0, 0] *= x
            self._A[:, 1, 0] *= y
            self._A[:, 2, 0] *= x
            self._A[:, 3, 0] *= y            
    def generateInitialArg(self, n, factor_u = 0.5, factor_s = 0.125, shrink_rate = 0.9, shrink_min = 0.1, shrink_n = None, loaded = True, min_size = 1.0):
        x_max = self.width * factor_u
        y_max = self.height * factor_u
        x_min = -x_max
        y_min = -y_max
        witdh_max = self.width * factor_s
        height_max = self.height * factor_s
        witdh_min = min_size
        height_min = min_size
        theta_max = np.pi / 2.0
        theta_min = -theta_max
        alpha_max = 0.8
        alpha_min = 0.8
        beta_max = 1.0
        beta_min = 0.0

        argRandMin = np.array([x_min, y_min, witdh_min, height_min, theta_min, alpha_min, beta_min, beta_min, beta_min])
        argRandMax = np.array([x_max, y_max, witdh_max, height_max, theta_max, alpha_max, beta_max, beta_max, beta_max])
        remain = n
        arg = None
        shrink = True
        if shrink_n is None:
            shrink_n = n
            shrink = False
        while remain > 0:
            if remain < shrink_n:
                shrink_n = remain
            argB = randArg(argRandMin, argRandMax, shrink_n) 

            # random selection with high-entropy features
            if self.random_type & 1:
                if self._weight is None:
                    rem = shrink_n
                    n0 = min(int(0.2 * shrink_n + 0.5), rem)
                    rem = max(0, rem - n0)
                    n1 = min(int(0.3 * shrink_n + 0.5), rem)
                    rem = max(0, rem - n1)
                    n2 = min(int(0.5 * shrink_n + 0.5), rem)
                    selector = [(0.5,  n0), (0.25, n1), (0.25, n2)]
                    u = 0
                    v = 0
                    i = 0
                    j = 0
                    avalible_pixels = self.pxm.size
                    for s in selector:
                        sel_range = int(avalible_pixels * s[0] + 0.5)
                        v = min(u + sel_range, avalible_pixels)
                        j = min(i + s[1], shrink_n)
                        sel_pxi = np.random.randint(u, v, s[1])
                        sel_px = np.array(self.pxm[sel_pxi])
                        sel_px = sel_px.view( (np.float, len(sel_px.dtype.names)))
                        # fiels = [('enp', float), ('x', float), ('y', float), ('r', float), ('g', float), ('b', float)]
                        argB[i:j, 0] = sel_px[:, 1]
                        argB[i:j, 1] = sel_px[:, 2]
                        #argB[i:j, 6] = sel_px[:, 3]
                        #argB[i:j, 7] = sel_px[:, 4]
                        #argB[i:j, 8] = sel_px[:, 5]
                        #print(u, v, sel_px)
                        u = v
                        i = j
                else:                    
                    n_sel_h =  int(1.0 * shrink_n + 0.5) 
                    n_sel_l = shrink_n - n_sel_h
                    n_h = self.pxm_h.size
                    n_l = self.pxm_l.size                    
                    sel_pxi_h = np.random.randint(0, n_h, n_sel_h)
                    sel_px_h = np.array(self.pxm_h[sel_pxi_h])
                    sel_px_h = sel_px_h.view( (np.float, len(sel_px_h.dtype.names)))                    
                    sel_px = sel_px_h
                    if n_l > 0:
                        sel_pxi_l = np.random.randint(0, n_l, n_sel_l)
                        sel_px_l = np.array(self.pxm_l[sel_pxi_l])
                        sel_px_l = sel_px_l.view( (np.float, len(sel_px_l.dtype.names)))
                        sel_px = np.concatenate( (sel_px, sel_px_l), axis = 0)
                    argB[:, 0] = sel_px[:, 1]
                    argB[:, 1] = sel_px[:, 2]
            # if self.random_type & 1
            
            argB = argB.reshape([shrink_n, ARGN, 1])
            if arg is None:
                arg = argB
            else:
                arg = np.concatenate((arg, argB))
            remain -= shrink_n
            if shrink:
                factor_s  *= shrink_rate
                if factor_s < shrink_min:
                    factor_s = shrink_min
                    shrink = False
                argRandMax[2] = self.width * factor_s
                argRandMax[3] = self.height * factor_s
                if argRandMax[2] <= argRandMin[2]:
                    argRandMax[2] = argRandMin[2]
                if argRandMax[3] <= argRandMin[3]:
                    argRandMax[3] = argRandMin[3]
        if loaded:
            self.setInitialArg(arg)
            return self._Aini
        else:
            return self.tensor(arg)                                                         
    # GaussImaging::generateInitialArg

    def json(self):
        argSet = self.argument()
        bg = self.background()
        jsonOut = {'shape':self.shape, 
                   'bg': None if bg is None else bg.tolist(), 
                   'arg': None if argSet is None else argSet.tolist()}           
        return jsonOut
    # GaussImaging::json

    def save(self, image_path, json_path = None): 
        I = None       
        if not(image_path is None):
            I = self.image()
            vgi.saveImage(image_path, I, revChannel = True)

        json_out = None
        if not(json_path is None):
            json_out = self.json()
            with open(json_path, 'w') as jfile:
                json.dump(json_out, jfile) 
        if json_out is None:
            return I
        else:
            return I, json_out
    # GaussImaging::save

    # _A: a set of argument vectors of primitives, (batch_size, dimensions, 1)    
    def primitiveMeta(self, _A):
        if _A is None:
            _A = self._A 
        images, channels, height, width = self.shape
        pixels = self.pixels
        primitives = _A.shape[0]

        # (primitives, 1)
        _ux = _A[:, 0, :] 
        _uy = _A[:, 1, :] 
        _sx = _A[:, 2, :]
        _sy = _A[:, 3, :]
        _sxsx = torch.square(_sx) 
        _sysy = torch.square(_sy) 
        _A_theta = _A[:, 4, :]
        _cosT = torch.cos(_A_theta) 
        _sinT = torch.sin(_A_theta) 
        _alpha = _A[:, 5, :] 
        _betaR = _A[:, 6, :] 
        _betaG = _A[:, 7, :] 
        _betaB = _A[:, 8, :] 
        
        _Xu = self._X - _ux # (primitives, pixels)
        _Yu = self._Y - _uy

        #_Xt = _Xu * _cosT + _Yu * _sinT
        #_Yt = -_Xu * _sinT + _Yu * _cosT
        _Xt = _Xu * _cosT - _Yu * _sinT
        _Yt = _Xu * _sinT + _Yu * _cosT

        return _Xt, _Yt, _sx, _sy, _sxsx, _sysy, _sinT, _cosT, _alpha, _betaR, _betaG, _betaB


    # returns: 
    #  _I_Ga: (primitives, channels, pixels), the images of all Gaussians, alpha * G
    #  _f: (primitives, pixels), 1 - alpha * Q    
    #  _G: (primitives, pixels)
    def drawGaussian(self, _A = None, weight_only = False, f_only = False, gradient = False):        
        _Xt, _Yt, _sx, _sy, _sxsx, _sysy, _sinT, _cosT, _alpha, _betaR, _betaG, _betaB = self.primitiveMeta(_A)
        primitives = _sx.shape[0]
        _2sxsx = _sxsx + _sxsx
        _2sysy = _sysy + _sysy
        _XtXt = _Xt * _Xt
        _YtYt = _Yt * _Yt        
        _Z = -(_XtXt / _2sxsx + _YtYt / _2sysy)
        _G = torch.where(_Z > self._expThres, torch.exp(_Z), self._zero) # (primitives, pixels)

        _gd = None
        if gradient:
            #_gd = self.ones((batch_size, 5, 1, self.height, self.width))
            _XtYt = _Xt * _Yt
            _Xs = _Xt / _sxsx
            _Ys = _Yt / _sysy
            shape_gd = (primitives, 1, 1, self.height, self.width)
            _dG_ux = ((_Xs * _cosT + _Ys * _sinT) * _G).reshape(shape_gd)
            _dG_uy = ((-_Xs * _sinT + _Ys * _cosT) * _G).reshape(shape_gd)
            _dG_sx = (_XtXt / _sxsx / _sx * _G).reshape(shape_gd)
            _dG_sy = (_YtYt / _sysy / _sy * _G).reshape(shape_gd)
            _dG_theta = ((_XtYt / _sxsx - _XtYt / _sysy) * _G).reshape(shape_gd)

            _gd = torch.cat((_dG_ux, _dG_uy, _dG_sx, _dG_sy, _dG_theta), dim = 1)

        ret = None
        if weight_only:
            ret = _G    
        else:
            _Ga = _G * _alpha
            _f = self._one - _Ga # (primitives, pixels)
            if f_only:                
                ret = _f                     
            else:            
                _I_Ga = torch.stack((_Ga * _betaR, _Ga * _betaG, _Ga * _betaB), dim = 1) # (primitives, channels, pixels)
                ret = _I_Ga, _f, _G
        if not(_gd is None):
            ret += (_gd,)
        return ret 
    # GaussImaging::drawGaussian
    
    # returns: 
    #  _I_Ga: [primitives, channels, pixels], the images of all Gaussians, alpha * Q
    #  _f: [primitives, pixels], 1 - alpha * Q    
    def drawEllipse(self, _A = None, weight_only = False, f_only = False):
        _Xt, _Yt, _sx, _sy, _sxsx, _sysy, _sinT, _cosT, _alpha, _betaR, _betaG, _betaB = self.primitiveMeta(_A)
        primitives = _sx.shape[0]
        _2sxsx = _sxsx + _sxsx
        _2sysy = _sysy + _sysy
        _gamma = self.full(_alpha.shape, self.gamma)
        scalef = 1.414
        _Z = (self._one -(torch.square(_Xt) / _2sxsx + torch.square(_Yt) / _2sysy)) * _gamma
             
        _Q = torch.where(torch.logical_and(_Z < self._sigmoidThres, _Z > self._sigmoidThresN), self._sigmoid(_Z), _Z)
        _Q = torch.where(_Q >= self._sigmoidThres, self._one, _Q)
        _Q = torch.where(_Q <= self._sigmoidThresN, self._zero, _Q)
        
        if weight_only:
            return _Q   
        else:
            _Qa = _Q * _alpha
            _f = self._one - _Qa
            if f_only:                
                return _f                     
            else:            
                _IQ = torch.stack((_Qa * _betaR, _Qa * _betaG, _Qa * _betaB), dim = 1)  # [primitives, channels, pixels] 
                return _IQ, _f, _Q   
    # GaussImaging::drawEllipse    

    def drawRect(self, _A = None, weight_only = False, f_only = False):
        _Xt, _Yt, _sx, _sy, _sxsx, _sysy, _sinT, _cosT, _alpha, _betaR, _betaG, _betaB = self.primitiveMeta(_A)
        primitives = _sx.shape[0]
        _gamma = self.full(_alpha.shape, self.gamma)
        scalef = 1.414
        _Zx = (self._one - torch.abs( _Xt) / (_sx * scalef)) * _gamma
        _Zy = (self._one - torch.abs(_Yt) / (_sy * scalef)) * _gamma

        _Qx = torch.where(torch.logical_and(_Zx < self._sigmoidThres, _Zx > self._sigmoidThresN), self._sigmoid(_Zx), _Zx)
        _Qx = torch.where(_Qx >= self._sigmoidThres, self._one, _Qx)
        _Qx = torch.where(_Qx <= self._sigmoidThresN, self._zero, _Qx)

        _Qy = torch.where(torch.logical_and(_Zy < self._sigmoidThres, _Zy > self._sigmoidThresN), self._sigmoid(_Zy), _Zy)
        _Qy = torch.where(_Qy >= self._sigmoidThres, self._one, _Qy)
        _Qy = torch.where(_Qy <= self._sigmoidThresN, self._zero, _Qy)   
        _Q = _Qx * _Qy   

        if weight_only:
            return _Q   
        else:
            _Qa = _Q * _alpha
            _f = self._one - _Qa
            if f_only:                
                return _f                     
            else:            
                _IQ = torch.stack((_Qa * _betaR, _Qa * _betaG, _Qa * _betaB), dim = 1)  # [primitives, chennels, pixels] 
                return _IQ, _f, _Q           
    # GaussImaging::drawRect  

    def drawBrush(self, _brush, _A = None, weight_only = False, f_only = False):
        _Xt, _Yt, _sx, _sy, _sxsx, _sysy, _sinT, _cosT, _alpha, _betaR, _betaG, _betaB = self.primitiveMeta(_A)
        primitives = _sx.shape[0]
        _2sx = _sx + _sx
        _2sy = _sy + _sy
        _gamma = self.full(_alpha.shape, self.gamma)

        #_Xu = self._X - _ux # [primitives, px]
        #_Yu = self._Y - _uy
        
        _Zx = _Xt / _2sx
        _Zy = _Yt / _2sy
        bH, bW = _brush.shape[0:2]  

        longTensor = torch.cuda.LongTensor if self.gpu else torch.LongTensor
        _Zxi = torch.where(torch.logical_and(_Zx <= 1.0, _Zx >= -1.0), 
                          ((_Zx + 1.0) / 2.0 * (bW - 1)).type(longTensor), 0)
        _Zyi = torch.where(torch.logical_and(_Zy <= 1.0, _Zy >= -1.0), 
                          ((1.0 - _Zy) / 2.0 * (bH - 1)).type(longTensor), 0)        

        _Q = torch.where(torch.logical_and(
                            torch.logical_and(_Zx <= 1.0, _Zx >= -1.0), 
                            torch.logical_and(_Zy <= 1.0, _Zy >= -1.0)), 
                          _brush[_Zyi, _Zxi], self._zero)

        if weight_only:
            return _Q   
        else:
            _Qa = _Q * _alpha
            _f = self._one - _Qa
            if f_only:                
                return _f                     
            else:            
                _IQ = torch.stack((_Qa * _betaR, _Qa * _betaG, _Qa * _betaB), dim = 1)  # [primitives, channels, pixels] 
                return _IQ, _f, _Q         
    # GaussImaging::drawBrush  

    def drawBrush1(self, _A = None, weight_only = False, f_only = False):      
        return self.drawBrush(_brush = self._brush1, _A = _A, weight_only = weight_only, f_only = f_only)
    def drawBrush2(self, _A = None, weight_only = False, f_only = False):      
        return self.drawBrush(_brush = self._brush2, _A = _A, weight_only = weight_only, f_only = f_only)        
    def drawBrush3(self, _A = None, weight_only = False, f_only = False):      
        return self.drawBrush(_brush = self._brush3, _A = _A, weight_only = weight_only, f_only = f_only)        
    def drawBrush4(self, _A = None, weight_only = False, f_only = False):      
        return self.drawBrush(_brush = self._brush4, _A = _A, weight_only = weight_only, f_only = f_only)        
    def drawBrush5(self, _A = None, weight_only = False, f_only = False):      
        return self.drawBrush(_brush = self._brush5, _A = _A, weight_only = weight_only, f_only = f_only)        
    def drawBrush6(self, _A = None, weight_only = False, f_only = False):      
        return self.drawBrush(_brush = self._brush6, _A = _A, weight_only = weight_only, f_only = f_only)        
    def drawBrush7(self, _A = None, weight_only = False, f_only = False):      
        return self.drawBrush(_brush = self._brush7, _A = _A, weight_only = weight_only, f_only = f_only)        
                                   

    def composite(self, _A = None, primitive = 'Gaussian'):
        if primitive == 'Gaussian':
            _IG, _f, _ = self.drawGaussian(_A)
        elif primitive == 'ellipse':
            _IG, _f, _ = self.drawEllipse(_A)
        elif primitive == 'rect':
            _IG, _f, _ = self.drawRect(_A)
        elif primitive == 'brush1':
            _IG, _f, _ = self.drawBrush1(_A)                
        elif primitive == 'brush2':
            _IG, _f, _ = self.drawBrush2(_A)     
        elif primitive == 'brush3':
            _IG, _f, _ = self.drawBrush3(_A)                         
        elif primitive == 'brush4':
            _IG, _f, _ = self.drawBrush4(_A)                
        elif primitive == 'brush5':
            _IG, _f, _ = self.drawBrush5(_A)     
        elif primitive == 'brush6':
            _IG, _f, _ = self.drawBrush6(_A)   
        elif primitive == 'brush7':
            _IG, _f, _ = self.drawBrush7(_A)               
        else: 
            _IG, _f, _ = self.drawGaussian(_A)                   
        primitives = _IG.shape[0]

        _I_px = self._I.flatten(start_dim = 2) #reshape( (self.images, self.pixels, self.channels) ) 
        for i in range(primitives):
            _I_px =  _IG[i, :, :] + _f[i, :] * _I_px
        self._I = _I_px.reshape((self.images, self.channels, self.height, self.width))    
        #if not(self._target is None):
        #    self._err = -vgi.imageSSIM(self._target, self._I)
    # GaussImaging::composite

    def compositeWeightOnly(self, _A = None, primitive = 'Gaussian'):
        if primitive == 'Gaussian':
            _G = self.drawGaussian(_A, weight_only = True)
        elif primitive == 'ellipse':
            _G = self.drawEllipse(_A, weight_only = True)
        elif primitive == 'rect':
            _G = self.drawRect(_A, weight_only = True)
        elif primitive == 'brush1':
            _G = self.drawBrush1(_A, weight_only = True)                
        elif primitive == 'brush2':
            _G = self.drawBrush2(_A, weight_only = True)     
        elif primitive == 'brush3':
            _G = self.drawBrush3(_A, weight_only = True)   
        # _G: [primitives, pixels]                      
        primitives = _IG.shape[0]
        _Gsum = _G.sum(dim = 0)
        _I_px = self._I.flatten(start_dim = 2) #reshape( (self.images, self.pixels, self.channels) ) 
        _I_px += _Gsum
        self._I = _I_px.reshape((self.images, self.channels, self.height, self.width))  
    # GaussImaging::compositeWeightOnly   

    def getDrawMethod(self, primitive = 'Gaussian') :
        drawShape = None
        if primitive == 'Gaussian':
            drawShape = self.drawGaussian
        elif primitive == 'ellipse':
            drawShape = self.drawEllipse
        elif primitive == 'ellipse':
            drawShape = self.drawRect    
        elif primitive == 'brush1':
            drawShape = self.drawBrush1                       
        elif primitive == 'brush2':
            drawShape = self.drawBrush2
        elif primitive == 'brush3':
            drawShape = self.drawBrush3
        return drawShape   
    def getLoss(self, loss = 'ssim'):
        lf = None
        if loss == 'ssim':
            lf = self.lossSSIM
        elif loss == 'mse':   
            lf = self.lossMSE
        elif loss == 'L1':   
            lf = self.lossL1
        elif loss == 'msssim':
            lf = self.lossMSSSIM          
        elif loss == 'ssimL1':
            lf = self.lossSSIML1
        elif loss == 'msssimL1':
            lf = self.lossMSSSIML1   
        return lf         

    def bestSearch(self, _A, _I, primitive = 'Gaussian', loss = 'ssim', _foreground = None):
        images, channels, height, width = _I.shape
        primitives = _A.shape[0] 
        draw = self.getDrawMethod(primitive)
        lf = self.getLoss(loss)
        _I_Ga, _f_each, _G = draw(_A)        
        #  _I_Ga: (primitives, channels, pixels), the images of all Gaussians, alpha * Q
        #  _f_each: (primitives, pixels), 1 - alpha * Q
        #  _G: (primitives, pixels)          
        i_best = primitives
        _I_best = None
        _min_err = None
        _Ibg_px = _I.flatten(start_dim = 2) #reshape( (images, channels, pixels) ) 
        
        #print('bestSearch::_IG', _IG.shape)
        #print('bestSearch::_f', _f.shape)
        #print('bestSearch::_I', self._I.shape)
        #print('bestSearch::_Ibg', _Ibg.shape)
        #_Ii =  (_f * _Ibg)
        #_Ii =  (_IG + _Ii)
        #_Ii = _Ii.reshape((nG, self.channels, self.height, self.width))
        _f_each = _f_each.unsqueeze(1) # (primitives, 1, pixels)
            # Backward alpha compositing: I_k  = \beta_k \alpha_k G_k + (1 - \alpha_k G_k)I_{k-1}
            # Forward alpha compositing: J_k = J_{k - 1} + \beta_k \alpha_k G_k * f_{k - 1}
        _I_all_px = (_I_Ga + _f_each * _Ibg_px)

        if not(_foreground is None):
            #print('bestSearch::_I_all_px bg', torch.min(_I_all_px), torch.max(_I_all_px))
            _Ifg_px = _foreground[0]
            _f_kp = _foreground[1] 
            _I_all_px = _Ifg_px + _f_kp * _I_all_px
            #print('bestSearch::_I_all_px fg', torch.min(_I_all_px), torch.max(_I_all_px))

        _I_all = _I_all_px.reshape((primitives, channels, height, width))    
        #print('bestSearch::_I_all', _I_all.shape)
        _err = lf(_I_all)
        i_best = torch.argmin(_err)
        _min_err = _err[i_best]
        _I_best = clone(_I_all[i_best].unsqueeze(0))

        #if not(self._weight is None):
        #    _I_best *= self._weight


        return i_best, _I_best, _min_err
    # GaussImaging::bestSearch


    # ***************
    # _arg [dimensions, 1],  _current_err [1] must be torch.tensor
    # updating by stepSize * (+/-)acceleration and stepSize * (+/-)(1 / acceleration)
    # shrinking when updating failed
    def hillClimb(self, _arg, _I, _arg_min, _arg_max, _current_err = None, 
                  step_size = None, acceleration = 1.2, min_decline = 0.0001, 
                  gray = False,
                  loss = 'ssim', rounds = 100, primitive = 'Gaussian', _foreground = None):
        
        dimensions = _arg.shape[0]
        if gray:
            dimensions -= 2
        if step_size is None:
            _step_size = clone(self._step_size)
        else:
            _step_size = self.tensor(step_size) # [ARGN, 1]
        _min_decline = self.tensor(min_decline)
        _arg_cur = clone(_arg)
        _shrink_rate = self.tensor(acceleration)
        _candidate = self.tensor([[-acceleration], [-1.0/acceleration], [1.0/acceleration], [acceleration]]) # [4, 1]
        #_candidate = self.tensor([[-accL], [-accS], [accS], [accL]]) # [4, 1]
        _min_err = _current_err          
        i_r = 0
        if self.debug:
            self._debugData = self.tensor([])
        while i_r < rounds:
            if not(_min_err is None):
                _before_err = clone(_min_err)
            else:
                _before_err = None

            for i in range(dimensions):           
                _step = _step_size[i] * _candidate # [4, 1]
                _arg_candidate = _arg_cur.repeat((4, 1, 1))
                _arg_candidate[:, i, :] += _step
                _arg_candidate[:, i, :] = torch.clip(_arg_candidate[:, i, :], _arg_min[i], _arg_max[i])
                #print('hillClimb::_argCur', _argCur)
                #print('hillClimb::_step', _step)
                #print('hillClimb::_argCandidate', _argCandidate)
                
                i_best, _I_best, _err = self.bestSearch(_A = _arg_candidate, _I = _I, 
                                                        primitive = primitive, loss = loss, 
                                                        _foreground = _foreground)
                #print('_err', _err)
                if i_best < 4 and ((_min_err is None) or (_err < _min_err)) :
                    #print('_min_err', _min_err)
                    _min_err = _err
                    _arg_cur = _arg_candidate[i_best, :, :]
                    _step_size[i] = _step[i_best] #bestStep                 
                else:  # updating failed then shrinking            
                    _step_size[i] /= _shrink_rate                   
            if gray:
                _step_size[7] = _step_size[6]
                _step_size[8] = _step_size[6]
                _arg_candidate[:, 7, :] = _arg_candidate[:, 6, :]
                _arg_candidate[:, 8, :] = _arg_candidate[:, 6, :]

            if self.debug:
                self._debugData = torch.cat([self._debugData, _min_err.reshape([1])])
                
            if not (_before_err is None) and ((_before_err - _min_err) < _min_decline):
                i_r = rounds
            else:
                i_r += 1
        return _arg_cur, _I_best, _min_err
    # GaussImaging::hillClimb 

    # .............................................................
    # Derivative methods
    # .............................................................

    # Derivative of the output image I_i with respect to a primitive image G_i.
    #   _IG_k: \alpha_k \G_k, (1, channels, pixels)
    #   _f_k: 1 - \alpha_k \G_k, (1, pixels)
    #   _G_k: (1, pixels)
    #   _Ig: the background image, (batch_size, channels, height, width)
    #   _f_kp: the (1 - alpha_i * G_i) for i = 0 to k - 1, (1, pixels)
    # returns:
    #   _dI_G: (batch_size, geo_dimensions, channels, height, width)
    #   _dI_alpha: (batch_size, 1, channels, height, width)
    #   _dI_beta: (batch_size, 1, 1, height, width)
    # Note that the batch_size of _arg and _Ig must be the same because of one-to-one mapping. 
    def gradientImageGaussian(self, _arg, _IG_k, _f_k, _G_k, _Ig, _f_kp, verbose = 0, log =  None):
        batch_size, channels, height, width = _Ig.shape

        _alpha = _arg[:, 5] # (batch_size, 1)
        _beta = _arg[:, 6:] # (batch_size, channels, 1)
        _dI_G = self.ones((batch_size, 1, channels, height, width))
        #_dI_alpha = self.ones((batch_size, 1, channels, height, width))
        #_dI_beta = self.ones((batch_size, 1, 1, height, width))
        
        _beta_Ig = _beta - _Ig.flatten(start_dim = 2) # (batch_size, channels, pixels)
        _fkbI = _f_kp * _beta_Ig
        _dI_G = (_alpha * _fkbI).reshape((batch_size, 1, channels, height, width)) 
        _dI_alpha = (_G_k * _fkbI).reshape((batch_size, 1, channels, height, width)) 
        _dI_beta = (_alpha * _G_k * _f_k).reshape((batch_size, 1, 1, height, width))

        if verbose & 2048:
            print('gradientImageGaussian::_alpha', _alpha.shape)
            print('gradientImageGaussian::_beta_Ig', _beta_Ig.shape)
            print('gradientImageGaussian::_IG_k', _IG_k.shape)
            print('gradientImageGaussian::_f_k', _f_k.shape)
            print('gradientImageGaussian::_G_k', _G_k.shape)
            print('gradientImageGaussian::_Ig', _Ig.shape)
            print('gradientImageGaussian::_f_kp', _f_kp.shape)
            print('gradientImageGaussian::_dI_G', _dI_G.shape)
            print('gradientImageGaussian::_dI_alpha', _dI_alpha.shape)
            print('gradientImageGaussian::_dI_beta', _dI_beta.shape)
        
        return _dI_G, _dI_alpha, _dI_beta

    # _dG_h: (batch_size, geo_dimensions, 1, height, width)
    # returns:
    #   _dI_h: (batch_size, h_dimensions, channels, height, width)
    #  _dI_beta: (batch_size, 1, channels, height, width)
    # h_dimensions = 7 for h = [ux, uy, sx, sy, theta, alpha, beta], where each dimension deals with multiple channels.
    def gradientImageArg(self, _arg, _IG_k, _f_k, _G_k, _dG_hk, _Ig, _f_kp, verbose = 0, log =  None):
        batch_size, channels, height, width = _Ig.shape
        _, dimensions, _, = _arg.shape
        
        _dI_G, _dI_alpha, _dI_beta = self.gradientImageGaussian(_arg, _IG_k, _f_k, _G_k, _Ig, _f_kp, verbose = verbose, log = log) 
                #  _dI_G:     (batch_size, 1, channels, height, width)
                #  _dI_alpha: (batch_size, 1, channels, height, width)
                #  _dI_beta:  (batch_size, 1, 1, height, width)

        _dI_h = _dI_G * _dG_hk #(batch_size, geo_dimensions, channels, height, width)
        _dI_h = torch.cat((_dI_h, _dI_alpha), dim = 1) #(batch_size, h_dimensions, channels, height, width)

        if verbose & 8192:
            print('gradientImageArg::_dI_h', _dI_h.shape)
            for i_h in range(6):
                img = toNumpyImage(_dI_h[0, i_h].unsqueeze(0))
                print('gradientImageArg::_dI_h[0,', i_h, ']', vgi.metric(img))
                vgi.showImage(vgi.normalize(img))        
            img = toNumpyImage(_dI_beta[0])
            print('gradientImageArg::_dI_beta', vgi.metric(img))
            vgi.showImage(vgi.normalize(img)) 
        return _dI_h, _dI_beta        

    # Derivative of the SSIM loss function with respect to the output image.
    # returns (batch_size, channels, patches, patch_size);
    #   patches = pixels if the image is padded
    #   patch_size = window_size^2 
    def gradientLSSIMImage(self, _I, verbose = 0, log =  None):
        window_size = self._SSIM.window_size
        patch_size = self._SSIM.patch_size
        patches = self.pixels
        batch_size, channels, height, width = _I.shape
        #_gd = self.ones((batch_size, channels, patches, patch_size))
        _gd = -self._SSIM.backward(_I) # (batch_size, channels, patches, patch_size)

        if verbose & 512:            
            _I_gd = _gd[0].sum(dim = -1).reshape((1, channels, height, width))
            img = toNumpyImage(_I_gd)
            print('gradientLSSIMImage::_gd[0]', vgi.metric(img))
            vgi.showImage(vgi.normalize(img))        
        return _gd

    def gradientLSSIMArg(self, _arg, _I, _IG_k, _f_k, _G_k, _dG_hk, _Ig, _f_kp, verbose = 0, log =  None):
        batch_size, channels, height, width = _Ig.shape
        _, dimensions, _ = _arg.shape # (batch_size, dimensions, 1)
        _gd = self.ones((batch_size, dimensions, 1))  

        _dI_h, _dI_beta = self.gradientImageArg(_arg, _IG_k, _f_k, _G_k, _dG_hk, _Ig, _f_kp, verbose = verbose, log = log) # (batch_size, h_dimensions, channels, height, width)
        _dSSIM_I = self.gradientLSSIMImage(_I, verbose = verbose, log = log) # (batch_size, channels, patches, patch_size)    
        _, _, patches, patch_size = _dSSIM_I.shape    
        window_size = self._SSIM.window_size
        pad_size = self._SSIM.padd
        for i_b in range(batch_size):
            _dI_h_p = vgi.ssim.padding(_dI_h[i_b], pad_size) # (h_dimensions, channels, height, width)
            _dI_h_uf = vgi.unfoldImage(_dI_h_p, window_size) # (h_dimensions, channels, patches, patch_size)
            _dE_h = _dSSIM_I[i_b] * _dI_h_uf # (h_dimensions, channels, patches, patch_size)
            _cov_h = torch.sum(_dE_h, dim = 3) # (h_dimensions, channels, patches)
            _gd_h = torch.mean(_cov_h, dim = (2, 1)).unsqueeze(-1) # (h_dimensions, 1)
            _gd[i_b][:6] = _gd_h           

            _dI_beta_p = vgi.ssim.padding(_dI_beta[i_b], pad_size) # (1, 1, height, width)
            _dI_beta_uf = vgi.unfoldImage(_dI_beta_p, window_size) # (1, 1, patches, patch_size)
            _dE_beta = _dSSIM_I[i_b] * _dI_beta_uf # (1, channels, patches, patch_size)
            _cov_beta = torch.sum(_dE_beta, dim = 3) # (1, channels, patches)
            _gd_beta = torch.mean(_cov_beta, dim = 2).unsqueeze(-1)[0] # (1, channels, 1)
            _gd[i_b][6:] = _gd_beta   

            if verbose & 1024:
                print('gradientImageArg::_dI_h', _dI_h.shape)
                print('gradientImageArg::_dI_beta', _dI_beta.shape)
                print('gradientImageArg::_dSSIM_I', _dSSIM_I.shape)
                print('gradientImageArg::_gd_h', _gd_h.shape)
                print('gradientImageArg::_gd_beta', _gd_beta.shape)
                print('gradientImageArg::_gd[i_b]', _gd[i_b])
            if verbose & 16384:
                img_shape = (1, channels, height, width)
                for i_h in range(6):
                    print('gradientImageArg::_cov_h[', i_h, ']', i_b)
                    img = toNumpyImage(_cov_h[i_h].reshape(img_shape))
                    vgi.showImage(vgi.normalize(img))        
                print('gradientImageArg::_cov_beta', i_b)
                img = toNumpyImage(_cov_beta.reshape(img_shape))
                vgi.showImage(vgi.normalize(img))              
            #_gd[i_b] = _gdx

            #_gd[i_b] = 
            #print('_dEdI', _dEdI.shape)
            #print('_cov', _cov.shape)
            #print('_gdx', _gdx.shape)
            
        # batch loop
        return _gd           
    
    # Updating rule:
    # -gradient * _step_size
    # shrinking _step_size while 
    def updateGaussianBatch(self, _arg, _arg_min, _arg_max, _arg_range, 
            rounds = 100, _step_size = None, _min_norm = None, loss = 'ssim',
            verbose = 1, verbose_rounds = 10, log =  None):
        
        batch_size, dimensions, _ = _arg.shape # (batch_size, dimensions, 1)
        images, channels, height, width = self.shape 
        _arg_out = clone(_arg)
        _arg_norm = self.zeros(batch_size)              
        if _step_size is None:
            _step_size = self._step_size.unsqueeze(dim = -1)
        _step_size = clone(_step_size)    

        if _min_norm is None:
           _min_norm = self.tensor([0.00001])        
        

        time_s = time.time()
        k_last = batch_size - 1
        
        # Rounds
        for i_r in range(rounds):
            #  Forward composition:  
            # _I: the composition results
            # _If, the compostion from top to Gk
            # _f_kp, the f_0^k-1
            if i_r == 0:
                _IG, _f_each, _G, _dG_h = self.drawGaussian(_arg_out, gradient = True) # (batch_size_, pixels)
                _I = self._I
            #print('updateGaussianBatch::_dG_h', _dG_h.shape)
            _Ig = clone(_I) 
            _I_px = self.zeros((images, channels, self.pixels)) #reshape( (images, channels, self.pixels) ) 
            _If = None
            _f = self.ones((1, self.pixels))
            i = batch_size - 1
            while i >= 0: # i = 0 is the bottom; i = batch-1 is the top 
                _I_px += _IG[i, :, :] * _f
                _f *= _f_each[i, :]
                if i == 1:
                    _If = clone(_I_px.reshape((images, channels, height, width)))
                    _f_kp = clone(_f)
                i -= 1
            _I_px += _f * _Ig.flatten(start_dim = 2) 
            _I = _I_px.reshape((images, channels, height, width))  

            if verbose & 256:
                print('updateGaussianBatch::_I')
                vgi.showImage(toNumpyImage(_I))
                #print('updateGaussianBatch::_If')
                #vgi.showImage(toNumpyImage(_If))                    
                    
            # Batch loop
            time_sb = time.time()
            k = 0
            while k < batch_size:
                _arg_k = _arg[k].unsqueeze(0) # (1, dimensions, 1)
                _IG_k = _IG[k].unsqueeze(0) # (1, channels, pixels)
                _f_k = _f_each[k].unsqueeze(0) # (1, pixels)
                _G_k = _G[k].unsqueeze(0) # (1, pixels)
                _dG_hk = _dG_h[k].unsqueeze(0) # (1, geo_dimensions, 1, height, width, )
                _d_E_arg_k = self.gradientLSSIMArg(_arg_k, _I, _IG_k, _f_k, _G_k, _dG_hk, _Ig, _f_kp, 
                                                   verbose = verbose, log = log)[0] # (dimensions, 1)

                _arg_upd_k = -_d_E_arg_k * _step_size
                _arg_out_k = _arg_out[k] + _arg_upd_k
                _arg_out_k = torch.clamp(_arg_out_k, _arg_min, _arg_max) # >= torch 1.9 
                _arg_d_k = _arg_out_k - _arg_out[k]
                _arg_upd_norm = torch.abs(_arg_d_k / _arg_range) # (dimensions, 1)
                _arg_out[k] = _arg_out_k 

                #print('_arg_upd_norm', _arg_upd_norm)

                _arg_norm[k] = torch.max(_arg_upd_norm)

                # updating 
                #   _Ig = \alpha_k \beta_k G_k +  (1 - \alpha_k G_k) _Ig
                #   _f_kp = _f_kp / _f_each[k - 1]
                _IG_k, _f_k, _G_k, _dG_hk = self.drawGaussian(_arg_out[k].unsqueeze(0), gradient = True) 
                _Ig_px =  _IG_k[0] + _f_k[0] * _Ig.flatten(start_dim = 2)
                _Ig = _Ig_px.reshape((images, channels, height, width))
                _IG[k] = _IG_k[0]
                _f_each[k] = _f_k[0]
                _G[k] = _G_k[0]
                _dG_h[k] = _dG_hk[0]

                if verbose & 4096:
                    print('updateGaussianBatch::_IG_k')
                    vgi.showImage(toNumpyImage(_IG_k.reshape(1, channels, height, width)))

                    for i_h in range(5):
                        img = toNumpyImage(_dG_hk[0, i_h].unsqueeze(0))
                        print('updateGaussianBatch::_dG_hk[0,', i_h, ']', vgi.metric(img))
                        vgi.showImage(vgi.normalize(img))

                    
                if k < k_last:
                    _f_kp_I = _f_kp.reshape((1, height, width))
                    _I = _If + _f_kp_I * _Ig
                    _f_kp = _f_kp / _f_each[k + 1]
                    _aIk = (_IG[k + 1] * _f_kp).reshape((1, channels, height, width))
                    _If = _If - _aIk
                        # Notice that it is subtracted by the original G_k
                else:
                    _I = clone(_Ig)

                if verbose & 256:
                    #print('updateGaussianBatch::_Ig', k)
                    #vgi.showImage(toNumpyImage(_Ig))  
                    #print('updateGaussianBatch::_If', k)
                    #vgi.showImage(toNumpyImage(_If))  
                    print('updateGaussianBatch::_I, round:', i_r, ', primitive:', k)
                    #vgi.showImage(toNumpyImage(_I))  
                    #print('updateGaussianBatch::_d_E_arg_k', _d_E_arg_k.squeeze())    

                    #print('updateGaussianBatch::_arg_upd_k', toNumpy(_arg_upd_k.squeeze()))
                    #print('updateGaussianBatch::_arg_out[k]', toNumpy(_arg_out[k].squeeze()))
                    #print('updateGaussianBatch::_arg_norm[k]', toNumpy(_arg_norm[k]))

                    print('updateGaussianBatch::dssim', toNumpy(_d_E_arg_k.sum()))
                    print('updateGaussianBatch::ssim', toNumpy(self.lossSSIM(_I)))
                    


                k += 1
                #print('_arg_upd time:', time.time() - time_s)
            # Batch loop

            #_I = _Ig
            _norm_r = torch.max(_arg_norm) 
            n_r = i_r + 1

            #if (n_r % verbose_rounds) == 0:
            #    torch.cuda.empty_cache()

            if verbose & 1 and (n_r % verbose_rounds) == 0:
            #if False:
                #torch.cuda.empty_cache() 
                norm_r = toNumpy(_norm_r)
                print('updateBatch::_norm_r[', n_r, ']:', norm_r)
                if not(log is None):
                    if 'norm_r' in log:
                        log['norm_r'] = np.concatenate((log['norm_r'], [norm_r]))
                    else:
                        log['norm_r'] = np.array([norm_r])
            #print('round time:', time.time() - time_s)          
            if _norm_r < _min_norm:
                break

            # Shrinking the step size
            shrink_rate = 0.9
            shrink_round = 5
            if n_r % shrink_round == 0:
                _step_size *= shrink_rate
            #print('round', i_r, ', batch time:', time.time() - time_sb)
        # round loop
        print('updateGaussianBatch time:', time.time() - time_s)
        return _arg_out, _I
# @GaussImaging

# ---------------------------------------------------------------
# Global functions

# Image decomposition by primitives (one-by-one decomposition)
# Backward alpha compositing with hill climb
def decomposite(image, n, loss = 'ssim', primitive = 'Gaussian', 
                bg = None, _prev = None, optimizer = 'hill', 
                arg = None, random_type = 0, random_n = 50, seed = 249, best_random = True, 
                shrink_n = 10, ms_n = 20, 
                rounds = 100, min_decline = 0.000001, window_size = 7,
                factor_u = 0.5, factor_s = 0.5, reopt = 0, reopt_type = 2, reopt_n = 3, 
                size_limit = False, min_size = 1.0,
                ssim_alpha = 0.84,
                weight = None, gray = False,
                verbose = 1, verbose_rounds = (10, 50), log =  None):
    if not(seed is None):
        np.random.seed(seed)
    #accL = 18.
    acc = 1.2
    _min_err = None

    log_t = None
    if not(log is None):
        log_t = [] 

    img_shape = image.shape
    height, width, channels = img_shape
    if bg is None:
        bg_bins = 256
        if gray:
            bgi = vgi.mostIntensity(image[:, :, 0], bins = bg_bins)
            bg = [bgi, bgi, bgi]
        else:
            bg = vgi.mostRGB(image, bins = bg_bins)

    if verbose & 1:
        timeS = time.time()
        print('shape:', img_shape) 
        print('background:', bg) 
    gi = GaussImaging(target = image, bg = bg, _prev = _prev, 
                      window_size = window_size, random_type = random_type, min_size = min_size,
                      ssim_alpha = ssim_alpha)
    if not(weight is None):
        gi.setWeight(weight)

    opt = gi.hillClimb
    _arg_min = gi.tensor(gi.argMin).unsqueeze(dim = -1)
    _arg_max = gi.tensor(gi.argMax).unsqueeze(dim = -1)

    if not (arg is None):
        n = arg.shape[0]
        random_n = n
        gi.setInitialArg(arg)
        i_best  = 0
    elif not best_random:
        random_n = n
        gi.generateInitialArg(random_n, factor_u = factor_u, factor_s = factor_s, min_size = min_size, shrink_rate = 0.9, shrink_min = 0.05, shrink_n = shrink_n)  
        i_best  = 0

    iR = 0

    factor_smin_ratio = 0.8
    if size_limit:
        factor_smin = factor_s * factor_smin_ratio   
        _arg_max[2] = width  * factor_s
        _arg_max[3] = height * factor_s
        _arg_min[2] = width  * factor_smin
        _arg_min[3] = height * factor_smin        
    else:
        factor_smin = 0.0
    while iR < n:
        #timeS = time.time() 
        if arg is None and best_random:

            gi.generateInitialArg(random_n, factor_u = factor_u, factor_s = factor_s, min_size = min_size)  
            i_best, _I_best, _err = gi.bestSearch(gi._Aini, gi._I, primitive = primitive, loss = loss)
        else:
            i_best = iR
            _err = None
            _I_best = None

        #if (_min_err is None) or (_err < _min_err and i_best < random_n):  
        if i_best < random_n:  
            #gi.debug = True
            _arg_opt, gi._I, gi._err = opt(gi._Aini[i_best], gi._I, _arg_min = _arg_min, _arg_max = _arg_max, _current_err = _err, 
                                         acceleration = acc, primitive = primitive, loss = loss, 
                                         gray = gray,
                                         min_decline = min_decline, rounds = rounds)

            #print('decomposite::_arg_opt', _arg_opt)
            _min_err = gi._err
            gi.appendArg(_arg_opt)

            if gi.debug:
                #gi.debug = False
                Err = toNumpy(gi._debugData)
                Rounds = range(Err.size)
                plt.plot(Rounds, Err, 'go--', linewidth=1, markersize=3)
                plt.show()
            iR += 1

            #if iR >= 100:
            #    size_limit = False
            #    factor_smin = 0.0
            #    _arg_min = gi.tensor(gi.argMin).unsqueeze(dim = -1)
            #    _arg_max = gi.tensor(gi.argMax).unsqueeze(dim = -1)

            if random_type & 2 and (iR >= 500):
                gi.random_type = 0


            if iR % shrink_n == 0:
                factor_s = max(factor_s * 0.9, 0.0)
                #factor_s = max(factor_s * 0.95, 0.0)   
                if size_limit:                    
                    width_max = width  * factor_s
                    height_max = height * factor_s
                    _arg_max[2] = width_max
                    _arg_max[3] = height_max
                    if _arg_max[2] <= _arg_min[2]:
                        _arg_max[2] = _arg_min[2]
                    if _arg_max[3] <= _arg_min[3]:
                        _arg_max[3] = _arg_min[3]    

                    #factor_smin = factor_s * factor_smin_ratio
                    #width_min = width  * factor_smin
                    #height_min = height  * factor_smin                                            
                    #_arg_min[2] = width_min
                    #_arg_min[3] = height_min
                    if verbose & 1:
                        print('size_limit ratio (max,min):', factor_s, factor_smin, 
                              ', min (w,h):', width_min, height_min, 
                              ', max (w,h):',  width_max, height_max)

            if iR % verbose_rounds[0] == 0:
                if verbose & 1:
                    t = time.time() - timeS
                    _ssim = gi.lossSSIM(gi._I)
                    _mse = gi.lossMSE(gi._I)
                    _l1 = gi.lossL1(gi._I)
                    err_R = toNumpy(_min_err)
                    ssim_R = toNumpy(_ssim)[0]
                    mse_R = toNumpy(_mse)[0]
                    l1_R = toNumpy(_l1)[0]
                    print(loss, 'n:', iR)
                    print('err_R', err_R)
                    print('ssim_R', ssim_R)
                    print('mse_R', mse_R)
                    print('L1_R', l1_R)
                    print('time:', t)
                    print()

                    if not(log is None):
                        log_t += [[err_R, ssim_R, mse_R, l1_R, t]]

                if verbose & 2:
                    I = gi.image()
                    vgi.showImage(I)

            if iR == ms_n:
                if loss == 'msssim':
                    loss = 'ssim'
                elif loss == 'msssimL1':
                    loss = 'ssimL1'

            if (reopt_type & 1) and (iR % reopt_n == 0):
                for i in range(reopt):
                    gi = reoptimize(gi, n = reopt_n, loss = loss, primitive = primitive, optimizer = optimizer,
                                    rounds = rounds, min_decline = min_decline, batch_size = 10, 
                                    gray = gray,
                                    verbose = verbose, verbose_rounds = verbose_rounds, log = log)                

    # round loop
    if not(log is None):
        log += [log_t]

    if reopt_type & 2:
        for i in range(reopt):
            if verbose & 1:
                print('reoptimize', i)
            gi = reoptimize(gi, loss = loss, primitive = primitive, optimizer = optimizer,
                            rounds = rounds, min_decline = min_decline, batch_size = 10, 
                            gray = gray,
                            verbose = verbose, verbose_rounds = verbose_rounds, log = log)
    return  gi
# decomposition

# re-optimize the decomposed primitives
# Forward alpha compositing with hill climb
def reoptimize(gi, loss = 'ssim', primitive = 'Gaussian', optimizer = 'hill', 
                rounds = 100, min_decline = 0.000001, batch_size = 10, n = 0, clamp = True,
                gray = False,
                verbose = 1, verbose_rounds = (10, 50), log =  None):

        
    shape_I = gi._I.shape
    images, channels, height, width = shape_I
    primitives, dimensions, _ = gi._A.shape
    pixels = height * width 
    if n <= 0:
        n = primitives
    log_t = None
    if not(log is None):
        log_t = [] 

    if verbose & 1:
        timeS = time.time()
        #print('reoptimize #primitives:', primitives)     

    draw = gi.getDrawMethod(primitive)
    lf = gi.getLoss(loss) 
    opt = gi.hillClimb
    acc = 1.2

    _arg_min = gi.tensor(gi.argMin).unsqueeze(dim = -1)
    _arg_max = gi.tensor(gi.argMax).unsqueeze(dim = -1)   

    _A = clone(gi._A)
    _Ibg_px = clone(gi._I.flatten(start_dim = 2))
    _Ifg_px = gi.zeros((_Ibg_px.shape))
    _f_kp = gi.ones((1, pixels))

    k = primitives - 1
    n_prim = 0
    ib_s = primitives - batch_size
    ib_e = primitives   
    while k >= 0:
        ib_s = max(ib_s, 0) 
        n_b = ib_e - ib_s
        _Ab = _A[ib_s:ib_e]
        _I_Ga, _f_each, _G = draw(_Ab)
            #  _I_Ga: (primitives, channels, pixels), the images of all Gaussians, alpha * G
            #  _f_each: (primitives, pixels), 1 - alpha * Q    
            #  _G: (primitives, pixels)
        i = n_b - 1
        while i >= 0:
            _I_Ga_k = _I_Ga[i].unsqueeze(0)
            _f_k = _f_each[i].unsqueeze(0)

            # Backward alpha compositing: I_k  = \beta_k \alpha_k G_k + (1 - \alpha_k G_k)I_{k-1}
            #   Inverse:  I_{k-1} = (I_k - \beta_k \alpha_k G_k) / (1 - \alpha_k G_k)
            #print('reoptimize::_Ibg_px before', torch.min(_Ibg_px), torch.max(_Ibg_px))
            #print('reoptimize::_I_Ga_k', torch.min(_I_Ga_k), torch.max(_I_Ga_k))
            #print('reoptimize::_f_k', torch.min(_f_k), torch.max(_f_k))
            _Ibg_px = (_Ibg_px - _I_Ga_k) / _f_k
            if clamp:
                torch.clamp(_Ibg_px, min=0.0, max=1.0, out=_Ibg_px)

            #print('reoptimize::_Ibg_px after', torch.min(_Ibg_px), torch.max(_Ibg_px))
            _Ibg = _Ibg_px.reshape(shape_I)
            _foreground = (_Ifg_px, _f_kp)

            # Updating
            _arg_k, gi._I, gi._err = opt(_Ab[i], _Ibg, _arg_min = _arg_min, _arg_max = _arg_max, _current_err = gi._err, 
                                         acceleration = acc, primitive = primitive, loss = loss, 
                                         min_decline = min_decline, rounds = rounds,
                                         gray = gray,
                                         _foreground = _foreground)            
            _A[k] = _arg_k
            _I_Ga_k, _f_k, _G_k = draw(_arg_k.unsqueeze(0))
            
            # Forward alpha compositing: J_k = J_{k - 1} + \beta_k \alpha_k G_k * f_{k - 1}
            _Ifg_px += _I_Ga_k * _f_kp
            
            _f_kp *= _f_k

            n_prim += 1
            if n_prim % verbose_rounds[0] == 0:
                if verbose & 1:
                    t = time.time() - timeS
                    _ssim = gi.lossSSIM(gi._I)
                    _mse = gi.lossMSE(gi._I)
                    _l1 = gi.lossL1(gi._I)
                    err_R = toNumpy(gi._err)
                    ssim_R = toNumpy(_ssim)[0]
                    mse_R = toNumpy(_mse)[0]
                    l1_R = toNumpy(_l1)[0]
                    print(loss, 'n:', n_prim)
                    print('err_R', err_R)
                    print('ssim_R', ssim_R)
                    print('mse_R', mse_R)
                    print('L1_R', l1_R)
                    print('time:', t)
                    print()

                    if not(log is None):
                        log_t += [[err_R, ssim_R, mse_R, l1_R, t]]

                if verbose & 2:
                    I = gi.image()
                    #print(vgi.metric(I))
                    vgi.showImage(I)        
            i -= 1
            k -= 1   
            if n_prim >= n:
                break; 
        # batch loop
        ib_e -= n_b 
        ib_s -= batch_size 
        if n_prim >= n:
            break;         
    # primitive loop

    if not(log is None):
        log += [log_t]

    gi._A = clone(_A)
    return gi


# ---------------------------------------------------------


# Unfinished work: bad accuracy, bad performance, and difficult to find the step size
# Image decomposition by primitives (batch decomposition)
# verbose = 1~255 for users
# verbose > 255 for developers
def decompositeBatch(image, n, loss = 'ssim', primitive = 'Gaussian', 
                     bg = None, _prev = None, rounds = 100, min_norm = 0.0001, 
                     step_size = None,
                     arg_init = None, batch_size = 10, alpha_max = 0.9,
                     factor_u = 0.45, factor_s = 0.5, # for argument limitaion
                     random_mode = 2, # 1: uniform; 2: noraml; 3: normal + uniform
                     random_seed = 249,  window_size = 7,
                     verbose = 1, verbose_rounds = (1, 10), log =  None):
    time_s = time.time()

    if not(random_seed is None):
        np.random.seed(random_seed)
    acc = 1.2
    _min_err = None

    shape_I = image.shape
    height, width, channels = shape_I
    pixels = height * width
    if bg is None:
        bins = 256
        bg = vgi.mostRGB(image, bins = bins)
    if verbose & 1:
        timeS = time.time()
        print('shape:', shape_I) 
        print('background:', bg) 
    gi = GaussImaging(target = image, bg = bg, _prev = _prev, window_size = window_size)

    # initial arguments setting
    if not (arg_init is None):
        n = arg_init.shape[0]
        gi.setInitialArg(arg_init)
        random_mode = 0

    n_q = 0
    if verbose & 4:
        gi.debug = True

    ##
    # The limitaion of arguments, do not change them!
    arg_min = np.array([-width * factor_u, -height * factor_u, 1, 1, -np.pi / 2.0, alpha_max / 2, 0.0, 0.0, 0.0])
    arg_max = np.array([ width * factor_u,  height * factor_u, width * factor_s, height * factor_s, np.pi / 2.0, alpha_max, 1.0, 1.0, 1.0])
    arg_range = np.array([width / 2, height / 2, width / 2, height / 2, np.pi / 2.0, 1.0, 1.0, 1.0, 1.0])

    _arg_min = gi.tensor(arg_min).unsqueeze(dim = -1)
    _arg_max = gi.tensor(arg_max).unsqueeze(dim = -1)
    _arg_range = gi.tensor(arg_range).unsqueeze(dim = -1)

    # For uniform random generating 
    arg_umin = np.array(arg_min)
    arg_umax = np.array(arg_max)
    # For normal distribution random generating 
    arg_mu = (arg_umax + arg_umin) / 2.0
    arg_stdv = arg_max - arg_mu

    # Optimization parameters
    #_step_size = gi._step_size
    if step_size is None:
        step_size = np.array([pixels, pixels, pixels, pixels, pixels, 1.0, 1.0, 1.0, 1.0])
    
    _step_size = gi.tensor(step_size).unsqueeze(dim = -1)

    _min_norm = gi.tensor([min_norm]) # L-inf norm, the maximum updating value

    #print('decompositeBatch::arg_min, arg_max', arg_min, arg_max)
    #print('decompositeBatch::arg_mu, arg_stdv', arg_mu, arg_stdv)

    n_proc_batches = 0
    _A = None
    while n_q < n:
        # batch initialization
        n_qn = n_q + batch_size
        if n_qn > n:
            n_qb = n - n_q
            n_q = n
        else:
            n_qb = batch_size
            n_q = n_qn

        if random_mode > 0:
            if not(_A is None):
                arg_mu = toNumpy(_A.mean(dim = 0).squeeze())
                arg_stdv = toNumpy(_A.std(dim = 0).squeeze())
                arg_stdv2 = arg_stdv + arg_stdv
                arg_umin = arg_mu - arg_stdv2
                arg_umax = arg_mu + arg_stdv2
            # Initializing the arguments of a batch by random number generating
            if random_mode == 1:
                arg_b = np.random.uniform(arg_umin, arg_umax, [n_qb, ARGN]) 
            elif random_mode == 2:
                arg_b = vgi.truncatedNormal(arg_min, arg_max, arg_mu, arg_stdv, n_qb)
            elif random_mode == 3:
                n_qbn = n_qb // 2
                n_qbu = n_qb - n_qbn
                arg_b = np.concatenate((vgi.truncatedNormal(arg_min, arg_max, arg_mu, arg_stdv, n_qbn), 
                                        np.random.uniform(arg_umin, arg_umax, [n_qbu, ARGN]) ))             
            #print(arg_b)
            #gi.setInitialArg(arg_b)
            _arg_b = gi.tensor(arg_b).unsqueeze(-1)
            if verbose & 256:
                print('decompositeBatch::_arg_b', _arg_b)

        #  Optimization
        

        _arg_bu, _I = gi.updateGaussianBatch(_arg_b, _arg_min, _arg_max, _arg_range, 
                               rounds = rounds, _step_size = _step_size, _min_norm = _min_norm, loss =  loss,
                               verbose = verbose, verbose_rounds = verbose_rounds[1], log = log)
        if _A is None:
            _A = _arg_bu
        else:
            _A = torch.cat((_A, _arg_bu))

        # Storing
        gi._A = _A
        gi._I = _I
        # vebosing 
        n_proc_batches += 1
        if verbose > 0 and (n_proc_batches % verbose_rounds[0] == 0):    
            print('decompositeBatch::n_q', gi._A.shape[0])
            t = time.time() - time_s
            print('time:', t)
    # batch loop 
    return  gi
# decompositeBatch

def drawPrimitives(arg, shape, bg = [0.0, 0.0, 0.0], _prev = None, max_n = -1, 
                   batch = 50, save_batch = True, scale_x = None, scale_y = None, 
                   primitive = 'Gaussian', weight_only = False, gamma = None, 
                   tensor = False, gi = None):
    img_shape = shape
    if not(scale_x is None):
        img_shape[3] = int(img_shape[3] * scale_x)        
        arg[:, 0] *= scale_x
        arg[:, 2] *= scale_x
    if not(scale_y is None):                
        img_shape[2] = int(img_shape[2] * scale_y)
        arg[:, 1] *= scale_y
        arg[:, 3] *= scale_y        
    iB = 0
    iQ = 0
    if gi is None:
        gi = GaussImaging(shape = img_shape, bg = bg, _prev = _prev)
    else:
        gi.setBackground(bg)

    n = arg.shape[0]
    if save_batch:
        imgSet = []

    if max_n >= 0:
        n = min(arg.shape[0], max_n)
    while iQ < n:
        iQe = min(iQ + batch, n)
        argB = arg[iQ:iQe, :]
        gi.setArg(argB)
        if weight_only:
            gi.compositeWeightOnly(primitive = primitive)
        else:
            gi.composite(primitive = primitive)
        if not(gamma is None):
            gi.gamma = gamma
        if save_batch:
            I =  gi._I if tensor else gi.image()
            imgSet += [I]
        iB += 1
        iQ = iQe
    gi.setArg(arg[:n])
    I = None
    if save_batch:
        I = imgSet
    else:
        I = gi._I if tensor else gi.image()
 
    return I

# drawPrimitives  

def drawJson(file, max_n = -1, batch = 50, _prev = None, 
              save_batch = True, scale_x = None, scale_y = None, 
              primitive = 'Gaussian', weight_only = False, gamma = None, 
              tensor = False, gi = None):
    with open(file) as jFile:
        jData = jFile.read()
        data = json.loads(jData)
        img_shape = data['shape']
        bg = data['bg']
        arg = np.array(data['arg'])
        return drawPrimitives(arg, img_shape, bg = bg, _prev = _prev, max_n = max_n, batch = batch, save_batch = save_batch, 
                              scale_x = scale_x, scale_y = scale_y, primitive = primitive, weight_only = weight_only, gamma = gamma, 
                              tensor = tensor, gi = gi)

# drawGaussianJson    