## Creating the testdata
The test sinograms were created by [ASTRA Toolbox](https://www.astra-toolbox.com/) with *create_proj_geom* and the following configuration:
```
astra.create_proj_geom(
    'fanflat', 
    det_width = 1.0, 
    det_count = 384, 
    angles = np.linspace(0.0, np.pi * 2, v, False), 
    source_origin = 256. * 2, 
    origin_det = 0.0)
```
where *v* indicates the number of projection views.

## Ground truth images:
012_100_gt.npy

025_128_gt.npy

## Sinograms of 60 projection views
012_100_60_sino.npy 

025_128_60_sino.npy

## Sinograms of 30 projection views
012_100_30_sino.npy 

025_128_30_sino.npy


