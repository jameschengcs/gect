# Fan-beam tomography 
# (c) 2022, Chang-Chieh Cheng, jameschengcs@nycu.edu.tw

import numpy as np
import copy
import sys
sys.path.append('../')
import vgi
import astra

__all__ = ('FanRec',)
 

# ----------------------------------------------------
class FanRec:
    def __init__(self, rec_shape, sino_shape, scan_range = (0, np.pi), angles = None, sino = None, 
                 proj_mode = 'fanflat', 
                 det_width = 1.0, source_origin = 512., origin_det = 512.,
                 algo = 'FBP', iterations = 10, gpu = True):
        self.gpu = gpu
        self.rec_shape = rec_shape
        self.sino_shape = sino_shape
        self.scan_range = scan_range
        self.proj_mode = proj_mode
        self.vol_geom = astra.create_vol_geom(self.rec_shape[0], self.rec_shape[1])
        # create_proj_geom('parallel', detector_spacing, det_count, angles)
        if angles is None:
            self.angles = np.linspace(self.scan_range[0], self.scan_range[1], self.sino_shape[0], False)
        else:
            self.angles = angles
        self.det_width = det_width
        self.source_origin = source_origin
        self.origin_det = origin_det

        self.proj_geom = astra.create_proj_geom(self.proj_mode, 
                                                self.det_width,
                                                self.sino_shape[1], 
                                                self.angles, 
                                                (self.source_origin + self.origin_det) / self.det_width,
                                                0.)

        # For CPU-based algorithms, a "projector" object specifies the projection
        # model used. In this case, we use the "strip" model.
        # Available algorithms:
        # ART, SART, SIRT, CGLS, FBP
        if self.gpu:
            self.algo = algo + '_CUDA'
            self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)
            self.cfg = astra.astra_dict(self.algo)
        else:
            self.algo = algo
            self.proj_id = astra.create_projector('strip', self.proj_geom, self.vol_geom)
            self.cfg = astra.astra_dict(self.algo)
        self.rec_id = astra.data2d.create('-vol', self.vol_geom)    
        self.sino = sino
        self.sino_id  = astra.data2d.create('-sino', self.proj_geom, data = sino)
    
        self.cfg['ProjectorId'] = self.proj_id
        self.cfg['ReconstructionDataId'] = self.rec_id    
        self.cfg['ProjectionDataId'] = self.sino_id
        self.alg_id = astra.algorithm.create(self.cfg)  
        self.iterations = iterations

    @classmethod
    def create512f(cls, n_angles = 720, algo = 'FBP', iterations = 10, gpu = True):
        ang_range = np.pi * 2
        return cls(rec_shape = (512, 512), sino_shape = (n_angles, 768), 
                 scan_range = (0, ang_range), 
                 angles = np.linspace(0,  ang_range, num = n_angles, endpoint = False), 
                 sino = None, 
                 proj_mode = 'fanflat', 
                 det_width = 1.0, source_origin = 512., origin_det = 512.,
                 algo = algo, iterations = iterations, gpu = gpu)
    @classmethod
    def create256f(cls, n_angles = 360, algo = 'FBP', iterations = 10, gpu = True):
        ang_range = np.pi * 2
        return cls(rec_shape = (256, 256), sino_shape = (n_angles, 384), 
                 scan_range = (0, ang_range), 
                 angles = np.linspace(0,  ang_range, num = n_angles, endpoint = False), 
                 sino = None, 
                 proj_mode = 'fanflat', 
                 det_width = 1.0, source_origin = 256., origin_det = 256.,
                 algo = algo, iterations = iterations, gpu = gpu)                      
    
    def project(self, img, keep_id = False):
        sinogram_id, sinogram = astra.create_sino(img, self.proj_id)
        if keep_id:
            return sinogram_id, sinogram
        else:
            sinogram = np.array(sinogram)
            astra.data2d.delete(sinogram_id)
            return sinogram

    def reconstruct(self, sino = None):
        if not(sino is None):
            astra.data2d.store(self.sino_id, sino)
            astra.algorithm.run(self.alg_id, self.iterations)
            rec = astra.data2d.get(self.rec_id)   
            rec = rec.astype(np.float32)
            return rec
    # Calling destructor
    def release(self):
        astra.data2d.delete(self.sino_id)
        astra.algorithm.delete(self.alg_id)
        astra.data2d.delete(self.rec_id)
        astra.projector.delete(self.proj_id) 
 