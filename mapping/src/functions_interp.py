#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bellemva
"""


from pyinterp import fill, Axis, TemporalAxis, Grid3D, Grid2D
n_workers = 10

def fill_nan(da):
    # INTERPOLATION OF NaNs # 
    x_axis = Axis(da.longitude.values)
    y_axis = Axis(da.latitude.values)
    t_axis = TemporalAxis(da.time.values)

    grid = Grid3D(y_axis, x_axis, t_axis, da.values.transpose(1,2,0))
    has_converged, filled = fill.gauss_seidel(grid,num_threads=n_workers)

    return da.copy(data = filled.transpose(2,0,1))

    # return  xr.DataArray(filled.transpose(2,0,1),dims=["time_counter", "x", "y"])