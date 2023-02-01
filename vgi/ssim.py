# forked from https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py

import torch
import torch.nn.functional as F
from math import exp
import numpy as np
import vgi

#img is a torch.tensor with the shape of (n_images, height, width, channels) 
def packBatch(img):
    n_dim = len(img.shape)
    if n_dim == 2:
        return img.unsqueeze(0).unsqueeze(0)
    elif n_dim == 3:
        return img.permute(2, 0, 1).unsqueeze(0)
    elif n_dim == 4:
        return img.permute(0, 3, 1, 2)

def unpackBatch(img):
    return img.permute(0, 2, 3, 1)        

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channels=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channels, 1, window_size, window_size).contiguous()
    return window # channels, 1, kernel_size, kernel_size

def padding(img, padd, mode='replicate'):
    return F.pad(img, pad = (padd, padd, padd, padd), mode='replicate')    

def unfoldGaussian(kernel_size, channels = 1, device = None):
    window = create_window(kernel_size, channels=channels).to(device)
    return window.flatten(start_dim = 2)  # channels, 1, kernel_size * kernel_size

# img must be padded by window_size
#  (batch_size, channels, height + window_size - 1, width + window_size - 1)
def ssim_terms(img, window, channels):
    # img.shape is (batch_size, channels, height, width)
    padd = 0 # padd must be zero for previous padding (James Cheng)
    mu = F.conv2d(img, window, padding=padd, groups=channels) #(batch_size, channels, height, width)
    mu_sq = mu * mu
    sigma_sq = F.conv2d(img * img, window, padding=padd, groups=channels) - mu_sq
    return mu, mu_sq, sigma_sq # thier shapes are (batch_size, channels, height, width)

# img must be padded by window_size
#  (batch_size, channels, height + window_size - 1, width + window_size - 1)
# mu cannot be padded, (batch_size, channels, height, width)
def dIqMu(img, mu, window_size):
    _I_uf = vgi.unfoldImage(img, kernel_size = window_size) # (batch_size, channels, patches, patch_size)
    _mu_uf = vgi.unfoldCenterImage(mu, pad_size = 0) # (batch_size, channels, patches, 1)
    return _I_uf - _mu_uf  # (batch_size, channels, patches, patch_size)

# img1 and img2 must be padded by window_size
#  (batch_size, channels, height + window_size - 1, width + window_size - 1)
def ssim_lcs(img1, img2,
             mu1, mu1_sq, sigma1_sq,  
             mu2, mu2_sq, sigma2_sq, window, val_range = None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    (_, channels, _, _) = img1.shape
    mu1_mu2 = mu1 * mu2
    sigma12 = F.conv2d(img1 * img2, window, groups=channels) - mu1_mu2

    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range    

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    cs_num = sigma12 + sigma12 + C2 # Numerators
    cs_den = sigma1_sq + sigma2_sq + C2 # denominator 
    cs = cs_num / cs_den  # contrast sensitivity

    l_num = mu1_mu2 + mu1_mu2 + C1 # Numerators
    l_den = mu1_sq + mu2_sq + C1 # denominator 

    l = l_num  / l_den
    return l, cs, l_den, cs_den #(batch_size, channels, height, width)

# img1 and img2 must be padded by window_size
#  (batch_size, channels, height + window_size - 1, width + window_size - 1)
def ssim(img1, img2, window_size=11, window=None, mean_axis = 0, full=False, val_range=None, 
        L1 = False, alpha = 0.84,
        mu2 = None, mu2_sq = None, sigma2_sq = None):  

    (_, channels, _, _) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channels = channels).to(img1.device)
    if mu2 is None:
        mu2, mu2_sq, sigma2_sq = ssim_terms(img2, window, channels)
    mu1, mu1_sq, sigma1_sq = ssim_terms(img1, window, channels)

    l, cs, _, _ = ssim_lcs(img1 = img1, img2 = img2, 
                     mu1 = mu1, mu1_sq = mu1_sq, sigma1_sq = sigma1_sq, 
                     mu2 = mu2, mu2_sq = mu2_sq, sigma2_sq = sigma2_sq, 
                     window = window, val_range = val_range)  
    _ssim  = l * cs

    if L1:
        _diff = F.conv2d(torch.abs(img1 - img2), window, groups=channels)
        _ssim = _ssim * alpha - _diff * (1.0 - alpha)

    # cs: (batch_size, channels, height, width)
    #print('ssim::window', window.shape)
    #print('ssim::window_uf', window.flatten(start_dim = 2).shape)
    #print('ssim::img1', img1.shape)
    #print('ssim::_ssim', _ssim.shape)
    #print('ssim::cs', cs.shape)
    #print('ssim::cs.mean(dim = (1, 2, 3))', cs.mean(dim = (1, 2, 3)).shape)
    if mean_axis == 0:
        if full:
            cs = cs.mean()
        ret = _ssim.mean()
    elif not(mean_axis is None):
        # ret = _ssim
        if full:
            cs = cs.mean(dim = mean_axis)
        ret = _ssim.mean(dim = mean_axis)
    else:
        ret = _ssim


    if full:
        return ret, cs
    return ret


#  returns (batch_size, channels, patches, patch_size)
#   \pds{SSIM(p)}{x(q)} = \pds{l(p)}{x(q)} cs(p) + l(p) \dps{cs(p)}{x(q)} 
#       \pds{l(p)}{x(q)} = 2G(p-q) \left( \frac{\mu_y - \mu_x l(p)}{\mu_x^2 + \mu_y^2 + C_1} \right)
#       \pds{cs(p)}{x(q)} = \frac{2G(p-q)}{\sigma_x^2 + \sigma_y^2 +  C_2} 
#                           \left( (y(q) - mu_y) - cs(p)(x(q) - mu_x) \right)
def dssim(img1, img2, window_size=11, window=None, full=False, val_range=None, 
        mu2 = None, mu2_sq = None, sigma2_sq = None, d_Iq2_Mu2 = None):  
    (_, channels, _, _) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channels = channels).to(img1.device)
    if mu2 is None:
        mu2, mu2_sq, sigma2_sq = ssim_terms(img2, window, channels)
    mu1, mu1_sq, sigma1_sq = ssim_terms(img1, window, channels) # (batch_size, channels, height, width)

    l, cs, l_den, cs_den = ssim_lcs(img1 = img1, img2 = img2, 
                                mu1 = mu1, mu1_sq = mu1_sq, sigma1_sq = sigma1_sq, 
                                mu2 = mu2, mu2_sq = mu2_sq, sigma2_sq = sigma2_sq, 
                                window = window, val_range = val_range)  
        # (batch_size, channels, height, width)
    G2 = window.flatten(start_dim = 2) * 2.0 # channels, 1, kernel_size * kernel_size 

    # Term1
    term1 = G2 * vgi.unfoldCenterImage((mu2 - mu1 * l) / l_den * cs) # (batch_size, channels, patches, patche_size)

    # Term2
    d_Iq1_Mu1 = dIqMu(img1, mu1, window_size) # (batch_size, channels, patches, patch_size)
    if d_Iq2_Mu2 is None:
        d_Iq2_Mu2 = dIqMu(img2, mu2, window_size)
    term2 = ((d_Iq2_Mu2 - vgi.unfoldCenterImage(cs) * d_Iq1_Mu1) / vgi.unfoldCenterImage(cs_den)) * G2
    ret = term1 + term2

    #print('dssim:ret', ret.shape)
    return ret    

# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, mean_axis=0, val_range=None, img2 = None, padding = True, loss = False):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.patch_size = self.window_size * self.window_size
        self.mean_axis = mean_axis
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channels = 1
        self.L1 = False
        self.alpha = 0.84
        self.img2 = None
        self.mu2 = None
        self.mu2_sq = None
        self.sigma2_sq = None
        self.d_Iq2_Mu2 = None
        self.padd = 0
        self.padding = padding
        self.loss = loss

        self.window = None
        if not(img2 is None):
            (_, self.channels, _, _) = img2.size()
            self.window = create_window(self.window_size, self.channels).to(img2.device).type(img2.dtype)
            if self.padding:
                self.padd = window_size // 2
            self.setImg2(img2)


    # The second image is the target
    def setImg2(self, img):
        self.img2 = img
        (_, self.channels, _, _) = self.img2.size()     
        if self.padding:
            self.img2 = padding(self.img2, self.padd)
            #self.img2 = F.pad(self.img2, pad = (self.padd, self.padd, self.padd, self.padd), mode='replicate')
        self.mu2, self.mu2_sq, self.sigma2_sq = ssim_terms(self.img2, self.window, self.channels)
        self.d_Iq2_Mu2 = dIqMu(self.img2, self.mu2, self.window_size)

    def blur(self, img):
        (_, channels, _, _) = img.size()
        if channels == self.channels and self.window.dtype == img.dtype:
            if self.padding:
                img = padding(img, self.padd)            
            img = F.conv2d(img, self.window, padding=self.padd, groups=channels) #(batch_size, channels, height, width)
            img = img[:, :, self.padd:-self.padd, self.padd:-self.padd]
            return img

    def forward(self, img1, img2 = None):
        (_, channels, _, _) = img1.size()
        if self.padding:
            img1 = padding(img1, self.padd)
        if not(self.window is None) and channels == self.channels and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channels).to(img1.device).type(img1.dtype)
            self.window = window
            self.channels = channels
        if img2 is None:
            img2 = self.img2

        v = ssim(img1, img2, window=window, window_size=self.window_size, mean_axis=self.mean_axis, 
                    val_range = self.val_range, L1 = self.L1, alpha = self.alpha,
                    mu2 = self.mu2, mu2_sq = self.mu2_sq, sigma2_sq = self.sigma2_sq)   
        if self.loss:
            v  = 1.0 - v
        return v

def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=None, 
    L1 = False, alpha = 0.84, weights = None):
    device = img1.device
    if weights is None:
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        #weights = torch.FloatTensor([0., 0.25, 0.25, 0.25]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for iLv in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range, L1 = L1, alpha = alpha)
        #print('msssim', iLv, sim )
        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims) # [levels, 1]
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    # TODO: remove support for normalize == True (kept for backward support)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1]) * pow2[-1]
    return output

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channels=3, val_range=None):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channels = channels
        self.val_range = val_range
        self.L1 = False
        self.alpha = 0.84      
        self.weights = None  

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, 
            val_range = self.val_range, L1 = self.L1, alpha = self.alpha, weights = self.weights)
