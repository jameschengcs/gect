# Gaussian ellipse reconstruction for Fan-beam tomography with flat detector array 
# (c) 2022, Chang-Chieh Cheng, jameschengcs@nycu.edu.tw

import numpy as np
import copy
import torch
from torch.nn import functional as F
import json
import time
import sys
sys.path.append('../')
import vgi
from vgi.ct import FanRec
from vgi.ssim import SSIM

__all__ = ('FanTomoG',)

# g is a vector to represent the parameters of a Gaussoan ellips
# g = [[ux], [uy], [sx], [sy], [theta], [alpha], [beta]]

def randg(g_min, g_max, n = 1):
    return np.random.uniform(g_min, g_max, [n, g_max.size]) 

# the shape of the input image must be (height, width)
class FanTomoG(FanRec):		
    def tensor(self, data):
        return torch.tensor(data, dtype = torch.float, device = self.device)
    def zeros(self, shape):
        return torch.zeros(shape, dtype = torch.float, device = self.device)
    def ones(self, shape):
        return torch.ones(shape, dtype = torch.float, device = self.device)   
    def full(self, shape, value):
        return torch.full(shape, value, dtype = torch.float, device = self.device) 
    def toTorchImage(self, image):
        return vgi.toTorchImage(image = image, dtype = torch.float, device = self.device)
    
    def __init__(self, rec_shape, sino_shape, 
                 ssim_window = 7, g_min = None, g_max = None, 
                 scan_range = (0, np.pi), angles = None, sino = None, 
                 proj_mode = 'fanflat', 
                 det_width = 1.0, source_origin = 256., origin_det = 256.,
                 algo = 'FBP', iterations = 10, gpu = True        
        ):
        super().__init__(rec_shape = rec_shape, sino_shape = sino_shape, scan_range = scan_range, 
                         angles = angles, sino = sino, proj_mode = proj_mode, 
                         det_width = det_width, source_origin = source_origin, origin_det = origin_det,
                         algo = algo, iterations = iterations, gpu = gpu)
        if self.gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # Constants            
        self._one = self.tensor(1.0)    
        self._zero = self.tensor(0.0)             
        self.pi2 = np.pi + np.pi
        self.pih = np.pi / 2
        self.piq = np.pi / 4         
        self.seed = 8051
        self._expThres = self.tensor(-15.0)
        self._idx_off = None # for hill climb's index offset, see self.hillClimb
        self.ssim_window = ssim_window

        # Geometry
        self.rec_t_shape = (1, 1, self.rec_shape[0], self.rec_shape[1])
        self.img_min_p = -(self.rec_shape[0] - 1) // 2
        self.img_max_p = (self.rec_shape[0] + 1) // 2
        self.rec_boundary = vgi.imageBoundary((self.rec_shape[0], self.rec_shape[1]))

        # g = [[ux], [uy], [sx], [sy], [theta], [alpha], [beta]]
        if g_min is None:
            self.g_min = np.array([self.img_min_p, self.img_min_p, 0.5, 0.5, -self.piq, 0.01, 0.0])
        else:
            self.g_min = np.array(g_min)
        if g_max is None:
            self.g_max = np.array([self.img_max_p, self.img_max_p, self.img_max_p, self.img_max_p, self.piq, 0.9, 1.0])
        else:
            self.g_max = np.array(g_max)

        self._g_min = self.tensor(self.g_min).unsqueeze(-1)
        self._g_max = self.tensor(self.g_max).unsqueeze(-1)   
        self._step_size = torch.tensor( [0.5, 0.5, 0.5, 0.5, np.pi / 180., 1/256, 1/256], device = self.device).unsqueeze(dim = -1)

        # self._X: [H, W], X coordinates
        # self._Y: [H, W], Y coordinates 
        self._Y, self._X = torch.meshgrid(torch.arange(self.rec_boundary[0], self.rec_boundary[1], device = self.device), 
                                          torch.arange(self.rec_boundary[2], self.rec_boundary[3], device = self.device),
                                          indexing = 'ij')
        self._Y = self._Y.flatten() # (pixels)
        self._X = self._X.flatten() # (pixels)        
        self._g = None
        self.bg = 0.0
        if self.sino is None:            
            self._target = None
        else:
            #bg_bins = 256           
            #self.bg = vgi.mostIntensity(image, bins = bg_bins)            
            self._target = self.tensor(self.sino).unsqueeze(0).unsqueeze(0)   # From (n_phi, n_tau) to (1, 1, n_phi, n_tau)
      
        self.ssimL1_alpha = 0.8
        self._mse = torch.nn.MSELoss()
        self._L1 = torch.nn.L1Loss()
        if self._target is None:
            self._SSIM = SSIM(val_range = 1.0, window_size = self.ssim_window, mean_axis = (1, 2, 3))
        else:
            self._SSIM = SSIM(img2 = self._target, window_size = self.ssim_window, mean_axis = (1, 2, 3), val_range = 1.0)
                
    # FanTomoG::__init__

    def setMinSize(self, sx, sy):
        self.g_min[2] = sx
        self.g_min[3] = sy
        self._g_min = self.tensor(self.g_min).unsqueeze(-1)
        self._g_max = self.tensor(self.g_max).unsqueeze(-1)         
    
    def lossL1(self, _I, mean_axis = (1, 2, 3)):
        _loss = (_I -  self._target).abs()
        if mean_axis == 0:
            _loss = _loss.mean()    
        elif not(mean_axis is None):
            _loss = _loss.mean(dim = mean_axis) 
        return _loss                 
    
    def lossMSE(self, _I, mean_axis = (1, 2, 3) ):
        _loss = ((_I -  self._target) ** 2.0)
        #if not(self._weight is None):
        #    _loss *= self._weight
        if mean_axis == 0:
            _loss = _loss.mean()    
        elif not(mean_axis is None):
            _loss = _loss.mean(dim = mean_axis) 
        _loss = _loss ** 0.5
        return _loss  
    
    def lossSSIM(self, _I, mean_axis = (1, 2, 3) ):
        self._SSIM.mean_axis = mean_axis
        _loss = self._one - self._SSIM.forward(_I)           
        return _loss
    
    def lossSSIML1(self, _I, mean_axis = (1, 2, 3)):
        self._SSIM.L1 = True
        self._SSIM.alpha = self.ssimL1_alpha
        self._SSIM.mean_axis = mean_axis
        _loss = self._SSIM.alpha - self._SSIM.forward(_I)                 
        self._SSIM.L1 = False
        return _loss  
    
    def getLoss(self, loss = 'ssim'):
        lf = None
        if loss == 'ssim':
            lf = self.lossSSIM
        elif loss == 'mse':   
            lf = self.lossMSE
        elif loss == 'L1':   
            lf = self.lossL1
        elif loss == 'ssimL1':
            lf = self.lossSSIML1 
        return lf     
    # FanTomoG::getLoss
    
    # the shape of g should be [n_g, n_arg, 1]
    def validateg(self, _g):
        n_dim = len(_g.shape)
        if n_dim == 2:
            _g = _g.unsqueeze(-1)
        elif n_dim == 1:
            _g = _g.unsqueeze(-1).unsqueeze(0) 
        _g = torch.clamp(_g, self._g_min, self._g_max)
        return _g
    # FanTomoG::validateg

    # _g: a set of argument vectors of primitives, (n, dimensions, 1)    
    def primitiveMeta(self, _g):
        # (primitives, 1)
        _ux = _g[:, 0, :] 
        _uy = _g[:, 1, :] 
        _sx = _g[:, 2, :]
        _sy = _g[:, 3, :]
        _sxsx = torch.square(_sx) 
        _sysy = torch.square(_sy) 
        _theta = _g[:, 4, :]
        _cosT = torch.cos(_theta) 
        _sinT = torch.sin(_theta) 
        _alpha = _g[:, 5, :]
        _beta = _g[:, 6, :]
        
        _Xu = self._X - _ux # (primitives, pixels)
        _Yu = self._Y - _uy

        #_Xt = _Xu * _cosT + _Yu * _sinT
        #_Yt = -_Xu * _sinT + _Yu * _cosT
        _Xt = _Xu * _cosT - _Yu * _sinT
        _Yt = _Xu * _sinT + _Yu * _cosT

        return _Xt, _Yt, _sx, _sy, _sxsx, _sysy, _sinT, _cosT, _alpha, _beta    
    # FanTomoG::primitiveMeta

    # Drawing a batch of n Gaussian ellipses to an image
    # _g: the parameters of n Gaussian ellipses
    # _I: the initial image
    # sum_all
    #   True: Drawing all ellipses to an image, the output is the result image
    #   False: Drawing each ellipse to an image, the number of output images is n.
    def drawBatch(self, _g = None, _I = None, sum_all = True):
        if _g is None:
            _g = self._g
        _Xt, _Yt, _sx, _sy, _sxsx, _sysy, _sinT, _cosT, _alpha, _beta = self.primitiveMeta(_g = _g)
        n = _sx.shape[0]
        _2sxsx = _sxsx + _sxsx
        _2sysy = _sysy + _sysy
        _XtXt = _Xt * _Xt
        _YtYt = _Yt * _Yt        
        _Z = -(_XtXt / _2sxsx + _YtYt / _2sysy)
        _G = torch.where(_Z > self._expThres, torch.exp(_Z), self._zero)  # (n, pixels)
        _aG = _G * _alpha
        if _I is None:
            _I = self.zeros([_G.shape[1]])
        else:
            _I = _I.flatten()
        if sum_all:
            for i in range(n):
                _I = _aG[i] * _beta[i] + (1.0 - _aG[i]) * _I
            _I = _I.reshape(self.rec_t_shape)
            return _I
        else:
            n = _g.shape[0]
            _I = _aG * _beta + (1.0 - _aG) * _I
            _I = _I.reshape((n,) + self.rec_t_shape[1:])
            return _I
    # FanTomoG::drawBatch       

    # returns: 
    #  _I_Ga: (n, channels, width, height), the images of all Gaussians, alpha * G
    #  _f: (n, 1, width, height), 1 - alpha * Q    
    #  _G: (n, 1, width, height)
    def drawGaussian(self, _g = None, weight_only = False, f_only = False):    
        n = _g.shape[0]
        _Xt, _Yt, _sx, _sy, _sxsx, _sysy, _sinT, _cosT, _alpha, _beta = self.primitiveMeta(_g = _g)            
        _2sxsx = _sxsx + _sxsx
        _2sysy = _sysy + _sysy
        _XtXt = _Xt * _Xt
        _YtYt = _Yt * _Yt        
        _Z = -(_XtXt / _2sxsx + _YtYt / _2sysy)
        _G = torch.where(_Z > self._expThres, torch.exp(_Z), self._zero) # (n, pixels)
        ret = None
        out_shape = (n, 1, self.rec_shape[0], self.rec_shape[1])
        if weight_only:
            ret = _G.reshape(out_shape)    
        else:
            _Ga = _G * _alpha
            _f = self._one - _Ga # (n, pixels)
            if f_only:                
                ret = _f.reshape(out_shape)                     
            else:          
                _I_Ga = (_Ga * _beta) # (n, pixels)
                ret = _I_Ga.reshape(out_shape), _f.reshape(out_shape), _G.reshape(out_shape)
        return ret 
    # FanTomoG::drawGaussian    
    
    def draw(self, _g = None, _I = None, batch_size = 0, sum_all = True):
        _g = self.validateg(_g)
        n = _g.shape[0]
        if batch_size <= 0:
            batch_size = n        
        remain = n
        i_b = 0       
        while remain > 0:            
            n_b = batch_size if remain > batch_size else remain
            i_bn = i_b + n_b
            _gb = _g[i_b: i_bn]
            _I = self.drawBatch(_g = _gb, _I = _I, sum_all = sum_all)
            i_b = i_bn
            remain -= n_b
        return _I  
    # FanTomoG::draw


    # Projecting a PyTorch image.
    def projectTorch(self, _I):
        n = _I.shape[0]
        _proj = None
        I = vgi.toNumpy(_I)
        for i in range(n):      
            img = I[i, 0] 
            proj = self.project(img)
            _p = self.tensor(proj).unsqueeze(0).unsqueeze(0)
            if _proj is None:
                _proj = _p
            else:
                _proj = torch.cat([_proj, _p])
        return _proj
    # FanTomoG::projectTorch    

    # draw all primitives of _g to _I, and apply the fan-beam projection with a flat detector array.
    def recProject(self, _g, batch_size = 0, _I = None, sum_all = True, _foreground = None):
        _proj = None
        if not(_g is None):
            _I = self.draw(_g = _g, batch_size = batch_size, _I = _I, sum_all = sum_all)

        if not(_foreground is None):
            _Ifg = _foreground[0]            
            if _I is None:
            	_I = _Ifg 
            else:
            	_f_k = _foreground[1]
            	_I = _Ifg + _f_k * _I

        _proj = self.projectTorch(_I)
        return _I, _proj 
    # FanTomoG::recProject   

    def bestSearch(self, _g, _I, loss = 'ssim'):
        images, channels, height, width = _I.shape
        n = _g.shape[0] 
        lf = self.getLoss(loss)
        _I_g, _proj_g = self.recProject(_g = _g, _I = _I, sum_all = False)        
         
        i_best = n
        _proj_best = None
        _min_err = None
     
        _err = lf(_proj_g)
        i_best = torch.argmin(_err)
        _min_err = _err[i_best]
        _proj_best = _proj_g[i_best].unsqueeze(0)
        _I_best = _I_g[i_best].unsqueeze(0)
        return i_best, _proj_best, _min_err, _I_best
    # FanTomoG::bestSearch    

    # ***************
    # _g [n, dimensions, 1],  _current_err [1] must be torch.tensor
    # updating by stepSize * (+/-)acceleration and stepSize * (+/-)(1 / acceleration)
    # shrinking when updating failed
    def hillClimb(self, _g, _I, _g_min, _g_max, _current_err = None, 
                  step_size = None, acceleration = 1.2, min_decline = 0.00001, _foreground = None,
                  lf = None, rounds = 100):
        n, dimensions, _ = _g.shape     
        if step_size is None:
            _step_size = vgi.clone(self._step_size)
        else:
            _step_size = self.tensor(step_size) # [dimensions, 1]
        _step_size = _step_size.unsqueeze(0).repeat((n, 1, 1)) # [n, dimensions, 1]

        _min_decline = self.tensor(min_decline)
        _g_cur = vgi.clone(_g)
        _shrink_rate = self.tensor(acceleration)
        _candidate = self.tensor([[[-acceleration], [-1.0/acceleration], [1.0/acceleration], [acceleration]]]) # [1, 4, 1]
        _min_err = _current_err
        i_r = 0

        if self._idx_off is None or self._idx_off.size != n:
            self._idx_off = torch.arange(start = 0, end = n * 4, step = 4, dtype = torch.int, device = self.device)
        
        if lf is None:
            lf = self.getLoss('ssim')
        while i_r < rounds:
            if _min_err is None:
                #_before_err = None
                _I_upd, _proj_upd = self.recProject(_g = _g_cur, _I = _I, sum_all = True, _foreground = _foreground)
                _err = lf(_proj_upd)  
                _min_err = _err
                _before_err = _err              
            else:
                _before_err = vgi.clone(_min_err)            

            for i in range(dimensions):          
                _step = (_step_size[:, i].unsqueeze(-1) * _candidate) # [n, 1, 1] * [1, 4, 1] = [n, 4, 1]
                _g_candidate = _g_cur.unsqueeze(1).repeat((1, 4, 1, 1)) # [n, 4, dimensions, 1]
                _g_candidate[:, :, i, :] += _step                
                _g_candidate = torch.clip(_g_candidate, _g_min, _g_max)

                _E = torch.cat( [lf(self.recProject (_g = _g_candidate[:, 0], _I = _I, sum_all = False, _foreground = _foreground)[1]).unsqueeze(0),
                                 lf(self.recProject (_g = _g_candidate[:, 1], _I = _I, sum_all = False, _foreground = _foreground)[1]).unsqueeze(0),
                                 lf(self.recProject (_g = _g_candidate[:, 2], _I = _I, sum_all = False, _foreground = _foreground)[1]).unsqueeze(0),
                                 lf(self.recProject (_g = _g_candidate[:, 3], _I = _I, sum_all = False, _foreground = _foreground)[1]).unsqueeze(0)] ) # [4, n]
                _idx_E = torch.argmin(_E, dim = 0) # [n]
                _idx_E += self._idx_off

                _g_candidate = _g_candidate.flatten(start_dim=0, end_dim=1) # (n * 4, 6, 1)
                _g_cur = torch.index_select(_g_candidate, 0, _idx_E) # (n, 6, 1)
                _I_upd, _proj_upd = self.recProject (_g = _g_cur, _I = _I, sum_all = True, _foreground = _foreground)
                _err = lf(_proj_upd)
                if ((_min_err is None) or (_err < _min_err)) :
                    _min_err = _err
                    _step_flat = _step.flatten(start_dim=0, end_dim=1) # (n * 4, 1)
                    _step_upd = torch.index_select(_step_flat, 0, _idx_E)
                    _step_size[:, i] = _step_upd #bestStep                 
                else:  # updating failed then shrinking            
                    _step_size[:, i] /= _shrink_rate 
            # dimension loop                  
            if not (_before_err is None) and ((_before_err - _min_err) < _min_decline):
                i_r = rounds
            else:
                i_r += 1
        # round loop  
        return _g_cur, _proj_upd, _min_err, _I_upd
    # FanTomoG::hillClimb      
            
   # 
    def reconstruct(self, n = 200, random_n = 50, 
                    shrink_rate = 0.95, shrink_batches = 1, 
                    loss = 'ssim', _I = None,
                    g_rand_min = None, g_rand_max = None, _foreground = None,
                    min_decline = 0.0000001,
                    cycle_n = 0, blur_n = 0,
                    verbose = 1, log_n = 1): 
        _g = None
        lf = self.getLoss(loss)
        opt = self.hillClimb
        if _I is None:
            _I = self.full(self.rec_t_shape, self.bg)        
        _, _proj = self.recProject(_g = None, _I = _I, sum_all = True)
        n_g = 0
        n_cycle_g = 0
        n_blur_g = 0
        if g_rand_min is None:
            g_rand_min = np.array(self.g_min)   
        g_rand_min0 = np.array(g_rand_min)         
        if g_rand_max is None:
            g_rand_max = np.array(self.g_max)
        g_rand_max0 = np.array(g_rand_max)
        _err = lf(_proj)
        print('reconstructRadon::init_err', _err)
        time_s = time.time()

        while n_g < n:
            n_g += 1  
            n_cycle_g += 1 
            n_blur_g += 1

            candidates = randg(g_rand_min, g_rand_max, random_n)
            _candidates = self.validateg(self.tensor(candidates))
            i_best, _proj_gb, _min_err, _I_gb = self.bestSearch(_g = _candidates, _I = _I, loss = loss)
            _gb = _candidates[i_best].unsqueeze(0)
                          
            if verbose & 256 and n_g % log_n == 0:
                proj_gb = vgi.toNumpyImage(_proj_gb)
                print('proj_gb', vgi.metric(proj_gb))
                proj_gb = vgi.normalize(proj_gb)
                vgi.showImage(proj_gb[0])
                
            # updating
            _gb_upd, _proj_upd, _upd_err, _I_upd = opt(_g = _gb, _I = _I, 
                                                       _g_min = self._g_min, _g_max = self._g_max, 
                                                       _current_err = None, lf = lf, 
                                                       _foreground = _foreground,
                                                       min_decline = min_decline)
        

            #g_upd = vgi.toNumpy(_gb_upd)
            if _g is None:
                _g = _gb_upd
            else:
                _g = torch.cat((_g, _gb_upd))
            _proj = _proj_upd
            _I = _I_upd
            _err = vgi.clone(_upd_err)
            if verbose & 1 and n_g % log_n == 0: 
                print('n: ', n_g)
                print('error: ', _err)
                t = time.time() - time_s
                print('time:', t)                
                
            if verbose & 256 and n_g % log_n == 0:
                proj = vgi.toNumpyImage(_proj)
                print('after proj_gb', vgi.metric(proj))
                proj = vgi.normalize(proj)
                vgi.showImage(proj[0])                
                
            if n_g > n:
                n_g = n
                n_cycle_g = 0
                n_blur_g = 0
            else: 
                if n_g % shrink_batches == 0:
                    g_rand_max[2] *= shrink_rate
                    g_rand_max[3] *= shrink_rate                
                    if g_rand_max[2] <= g_rand_min[2]:
                        g_rand_max[2] = g_rand_min[2]
                    if g_rand_max[3] <= g_rand_min[3]:
                        g_rand_max[3] = g_rand_min[3]                

                if cycle_n > 0 and n_cycle_g >= cycle_n:
                    g_rand_min = np.array(g_rand_min0)         
                    g_rand_max = np.array(g_rand_max0)  
                    n_cycle_g = 0  
                if blur_n > 0 and n_blur_g >= blur_n:
                    _I = self._SSIM.blur(_I)
                    n_blur_g = 0
        # while n_g < n
        torch.cuda.empty_cache()
        self._g = _g
        return _g, _proj, _I
    # FanTomoG::reconstruct    


    # re-optimize the decomposed primitives
    # Forward alpha compositing with hill climb
    def reoptimize(self, _g, _proj, _I, loss = 'ssim', rounds = 100, min_decline = 0.000001, batch_size = 10, n = 0, clamp = True,
                    verbose = 1, verbose_rounds = (10, 50), log =  None):

        shape_I = self.rec_t_shape
        images, channels, height, width = shape_I
        n = _g.shape[0]
        pixels = height * width 
        log_t = None
        if not(log is None):
            log_t = [] 

        if verbose & 1:
            time_s = time.time()
            print('reoptimize n:', n)     

        lf = self.getLoss(loss) 
        opt = self.hillClimb
        acc = 1.2

        _g_min = vgi.clone(self._g_min)
        #_g_min[2] = 0.5
        #_g_min[3] = 0.5
        _g_max = vgi.clone(self._g_max)

        _g_out = None
        _Ibg = vgi.clone(_I)
        _Ifg = self.zeros(_Ibg.shape)
        _f_kp = self.ones(_Ibg.shape)

        k = n - 1
        n_prim = 0
        ib_s = n - batch_size
        ib_e = n   
        _err = lf(_proj)
        while k >= 0:
            ib_s = max(ib_s, 0) 
            n_b = ib_e - ib_s
            _gb = _g[ib_s:ib_e]
            _I_Ga, _f_each, _G = self.drawGaussian(_gb)

            i = n_b - 1
            while i >= 0:
                _I_Ga_k = _I_Ga[i].unsqueeze(0)
                _f_k = _f_each[i].unsqueeze(0)
                if verbose & 512:
                    print('_Ibg before', torch.min(_Ibg), torch.max(_Ibg))
                    print('_I_Ga_k', torch.min(_I_Ga_k), torch.max(_I_Ga_k)) 
                    vgi.showTorchImage(_I_Ga_k) 
                    print('_f_k', torch.min(_f_k), torch.max(_f_k)) 
                    vgi.showTorchImage(_f_k)               
                _Ibg = (_Ibg - _I_Ga_k) / _f_k
                if clamp:
                    torch.clamp(_Ibg, min=0.0, max=1.0, out=_Ibg)

                _foreground = (_Ifg, _f_kp)

                # Updating  
                _g_k = _gb[i].unsqueeze(0)
                _g_k, _proj, _err, _I = opt(_g = _g_k, _I = _Ibg, 
                                                    _g_min = _g_min, _g_max = _g_max, 
                                                    _current_err = _err, lf = lf, 
                                                    _foreground = _foreground)            
                if _g_out is None:
                    _g_out = _g_k
                else:
                    _g_out = torch.cat([_g_k, _g_out])
                _I_Ga_k, _f_k, _G_k = self.drawGaussian(_g_k)
                
                # Forward alpha compositing: J_k = J_{k - 1} + \beta_k \alpha_k G_k * f_{k - 1}
                _Ifg += _I_Ga_k * _f_kp
                _f_kp *= _f_k

                n_prim += 1
                if n_prim % verbose_rounds[0] == 0:
  
                    if verbose & 1:
                        t = time.time() - time_s
                        
                        _ssim = self.lossSSIM(_proj)
                        _mse = self.lossMSE(_proj)
                        _l1 = self.lossL1(_proj)
                        err_R = vgi.toNumpy(_err)[0]
                        ssim_R = vgi.toNumpy(_ssim)[0]
                        mse_R = vgi.toNumpy(_mse)[0]
                        l1_R = vgi.toNumpy(_l1)[0]
                        print(loss, 'n:', n_prim)
                        print('err_R', err_R)
                        print('ssim_R', ssim_R)
                        print('mse_R', mse_R)
                        print('L1_R', l1_R)
                        print('time:', t)
                        print()

                        if not(log is None):
                            log_t += [[err_R, ssim_R, mse_R, l1_R, t]]

                    if verbose & 256:
                        print('_Ibg', torch.min(_Ibg), torch.max(_Ibg))
                        print('_Ibg', torch.min(_Ibg), torch.max(_Ibg))
                        vgi.showTorchImage(_Ibg)
                        print('_Ifg', torch.min(_Ifg), torch.max(_Ifg))
                        vgi.showTorchImage(_Ifg)
                        vgi.showTorchImage(_proj) 
                        vgi.showTorchImage(_I)  
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

        return _g_out, _proj, _I
    # FanTomoG::reoptimize              
# @FanTomoG  