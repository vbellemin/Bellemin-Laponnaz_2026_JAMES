#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 22:36:20 2021

@author: leguillou
"""

import jax
jax.config.update("jax_enable_x64", True)

from importlib.machinery import SourceFileLoader 
import sys
import xarray as xr
import numpy as np
import os
from math import pi
from datetime import timedelta
import matplotlib.pyplot as plt 

import jax.numpy as jnp 
from jax import jit
from jax import jvp,vjp
from jax.lax import scan

from copy import deepcopy

from functools import partial

from . import  grid
from . import switchvar

from .exp import Config as Config

import warnings 

import time

import functools

from scipy.signal import convolve2d
from scipy.special import factorial

def Model(config, State, verbose=True):
    """
    NAME
        Model

    DESCRIPTION
        Main function calling subclass for specific models
    """
    if config.MOD is None:
        return
    
    elif config.MOD.super is None:
        return Model_multi(config,State)

    elif config.MOD.super is not None:
        if verbose:
            print(config.MOD)

        if config.MOD.super=='MOD_DIFF':
            return Model_diffusion(config,State)
        
        elif config.MOD.super=='MOD_QG1L_JAX':
            return Model_qg1l_jax(config,State)
        
        elif config.MOD.super=='MOD_SW1L_JAX':
            return Model_sw1l_jax(config,State)
        
        else:
            sys.exit(config.MOD.super + ' not implemented yet')
    else:
        sys.exit('super class if not defined')
    
class M:

    def __init__(self,config,State):
        
        # Time parameters
        self.dt = config.MOD.dtmodel
        if self.dt>0:
            self.nt = 1 + int((config.EXP.final_date - config.EXP.init_date).total_seconds()//self.dt)
        else:
            self.nt = 1
        self.T = np.arange(self.nt) * self.dt
        self.ny = State.ny
        self.nx = State.nx
        
        # Construct timestamps
        if self.dt>0:
            self.timestamps = [] 
            t = config.EXP.init_date
            while t<=config.EXP.final_date:
                self.timestamps.append(t)
                t += timedelta(seconds=self.dt)
            self.timestamps = np.asarray(self.timestamps)
        else:
            self.timestamps = np.array([config.EXP.init_date])

        # Model variables
        self.name_var = config.MOD.name_var
        self.var_to_save = []
        if config.MOD.var_to_save is not None: # Only specified variables are saved 
            for name in config.MOD.var_to_save:
                self.var_to_save.append(self.name_var[name])
        else: # All variables are saved 
            for name in self.name_var:
                self.var_to_save.append(self.name_var[name])


    def init(self, State, t0=0):
        return
    
    def set_bc(self,time_bc,var_bc):

        return

    def ano_bc(self,t,State,sign):

        return
        
    
    def step(self,State,nstep=1,t=None):

        return 
    
    def step_tgl(self,dState,State,nstep=1,t=None):

        return

    def step_adj(self,adState,State,nstep=1,t=None):

        return
    
    def save_output(self,State,present_date,name_var=None,t=None):
        State.save_output(present_date,name_var)
    

###############################################################################
#                            Diffusion Models                                 #
###############################################################################
        
class Model_diffusion(M):
    
    def __init__(self,config,State):

        super().__init__(config,State)
        
        self.Kdiffus = config.MOD.Kdiffus
        self.SIC_mod = config.MOD.SIC_mod
        self.dx = State.DX
        self.dy = State.DY

        # Initialization 
        if (config.GRID.super == 'GRID_FROM_FILE') and (config.MOD.name_init_var is not None):
            dsin = xr.open_dataset(config.GRID.path_init_grid)
            for name in self.name_var:
                if name in config.MOD.name_init_var:
                    var_init = dsin[config.MOD.name_init_var[name]]
                    if len(var_init.shape)==3:
                        var_init = var_init[0,:,:]
                    if config.GRID.subsampling is not None:
                        var_init = var_init[::config.GRID.subsampling,::config.GRID.subsampling]
                    dsin.close()
                    del dsin
                    State.var[self.name_var[name]] = var_init.values
                else:
                    State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))
        else:
            for name in self.name_var:  
                State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))
        
        # Model Parameters (Flux)
        for name in self.name_var:
            State.params[self.name_var[name]] = np.zeros((State.ny,State.nx))

        # Initialize boundary condition dictionnary for each model variable
        self.bc = {}
        for _name_var_mod in self.name_var:
            self.bc[_name_var_mod] = {}
        self.init_from_bc = config.MOD.init_from_bc
        
        # Weight map to apply BC in a smoothed way
        if config.MOD.dist_sponge_bc is not None:
            Wbc = grid.compute_weight_map(State.lon, State.lat, +State.mask, config.MOD.dist_sponge_bc)
        else:
            Wbc = np.zeros((State.ny,State.nx)) 
            if State.mask is not None:
                for i,j in np.argwhere(State.mask):
                    for p1 in [-1,0,1]:
                        for p2 in [-1,0,1]:
                            itest=i+p1
                            jtest=j+p2
                            if ((itest>=0) & (itest<=State.ny-1) & (jtest>=0) & (jtest<=State.nx-1)):
                                if Wbc[itest,jtest]==0:
                                    Wbc[itest,jtest] = 1
        self.Wbc = Wbc
        
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            print('Tangent test:')
            tangent_test(self,State)
            print('Adjoint test:')
            adjoint_test(self,State)

    

    def init(self, State, t0=0):

        if type(self.init_from_bc)==dict:
            for name in self.init_from_bc:
                if self.init_from_bc[name] and name in self.bc and t0 in self.bc[name]:
                    State.setvar(self.bc[name][t0], self.name_var[name])
        elif self.init_from_bc:
            for name in self.name_var: 
                if t0 in self.bc[name]:
                     State.setvar(self.bc[name][t0], self.name_var[name])

    def set_bc(self,time_bc,var_bc):
        
        for _name_var_bc in var_bc:
            for _name_var_mod in self.name_var:
                if _name_var_bc==_name_var_mod:
                    for i,t in enumerate(time_bc):
                        self.bc[_name_var_mod][t] = var_bc[_name_var_bc][i]


    def step(self,State,nstep=1,t=None):

        # Loop on model variables
        for name in self.name_var:

            # Get state variable
            var0 = State.getvar(self.name_var[name])
            
            # Init
            var1 = +var0

            # Time propagation
            if self.Kdiffus>0:
                for _ in range(nstep):
                    var1[1:-1,1:-1] += self.dt*self.Kdiffus*(\
                        (var1[1:-1,2:]+var1[1:-1,:-2]-2*var1[1:-1,1:-1])/(self.dx[1:-1,1:-1]**2) +\
                        (var1[2:,1:-1]+var1[:-2,1:-1]-2*var1[1:-1,1:-1])/(self.dy[1:-1,1:-1]**2))
            
            # Update state
            if self.name_var[name] in State.params:
                # params = State.params[self.name_var[name]]
                params = np.asarray(State.params[self.name_var[name]], dtype=np.float64)
                var1 += (1-self.Wbc)*nstep*self.dt/(3600*24) * params

            State.setvar(var1, self.name_var[name])
        


    def step_tgl(self,dState,State,nstep=1,t=None):

        # Loop on model variables
        for name in self.name_var:

            # Get state variable
            var0 = dState.getvar(self.name_var[name])
            
            # Init
            var1 = +var0
            
            # Time propagation
            if self.Kdiffus>0:
                for _ in range(nstep):
                    var1[1:-1,1:-1] += self.dt*self.Kdiffus*(\
                        (var1[1:-1,2:]+var1[1:-1,:-2]-2*var1[1:-1,1:-1])/(self.dx[1:-1,1:-1]**2) +\
                        (var1[2:,1:-1]+var1[:-2,1:-1]-2*var1[1:-1,1:-1])/(self.dy[1:-1,1:-1]**2))
            

            # Update state
            if self.name_var[name] in dState.params:
                params = dState.params[self.name_var[name]]
                var1 += (1-self.Wbc)*nstep*self.dt/(3600*24) * params

            dState.setvar(var1,self.name_var[name])
        
    def step_adj(self,adState,State,nstep=1,t=None):

        # Loop on model variables
        for name in self.name_var:

            # Get state variable
            advar0 = adState.getvar(self.name_var[name])

            # Init
            advar1 = +advar0
            
            # Time propagation
            if self.Kdiffus>0:
                for _ in range(nstep):
                    
                    advar1[1:-1,2:] += self.dt*self.Kdiffus/(self.dx[1:-1,1:-1]**2) * advar0[1:-1,1:-1]
                    advar1[1:-1,:-2] += self.dt*self.Kdiffus/(self.dx[1:-1,1:-1]**2) * advar0[1:-1,1:-1]
                    advar1[1:-1,1:-1] += -2*self.dt*self.Kdiffus/(self.dx[1:-1,1:-1]**2) * advar0[1:-1,1:-1]
                    
                    advar1[2:,1:-1] += self.dt*self.Kdiffus/(self.dy[1:-1,1:-1]**2) * advar0[1:-1,1:-1]
                    advar1[:-2,1:-1] += self.dt*self.Kdiffus/(self.dy[1:-1,1:-1]**2) * advar0[1:-1,1:-1]
                    advar1[1:-1,1:-1] += -2*self.dt*self.Kdiffus/(self.dy[1:-1,1:-1]**2) * advar0[1:-1,1:-1]
                    
                    advar0 = +advar1
                

            # Update state and parameters
            if self.name_var[name] in State.params:
                adState.params[self.name_var[name]] += (1-self.Wbc)*nstep*self.dt/(3600*24) * advar0 
            
            advar1[np.isnan(advar1)] = 0
            adState.setvar(advar1,self.name_var[name])

class Model_diffusion_jax(Model_diffusion):
    def __init__(self,config,State):
        super().__init__(config,State)

    def step(self, t, State_var, State_params, nstep=1):

        # Loop on model variables
        for name in self.name_var:

            # Get state variable
            var0 = State_var[self.name_var[name]]
            
            # Init
            var1 = +var0

            # Time propagation
            if self.Kdiffus>0:
                for _ in range(nstep):
                    var1[1:-1,1:-1] += self.dt*self.Kdiffus*(\
                        (var1[1:-1,2:]+var1[1:-1,:-2]-2*var1[1:-1,1:-1])/(self.dx[1:-1,1:-1]**2) +\
                        (var1[2:,1:-1]+var1[:-2,1:-1]-2*var1[1:-1,1:-1])/(self.dy[1:-1,1:-1]**2))
            
            # Update state
            if self.name_var[name] in State_params:
                params = State_params[self.name_var[name]]
                var1 += (1-self.Wbc)*nstep*self.dt/(3600*24) * params

            State_var1 = State_var.copy()
            State_var1[self.name_var[name]] = var1
            
            return State_var1

##############################################################################
#                       Quasi-Geostrophic Model                              #
##############################################################################

class Model_qg1l_jax(M):

    def __init__(self,config,State):

        super().__init__(config,State)

        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

        # Model specific libraries
        if config.MOD.dir_model is None:
            dir_model = os.path.realpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','models','model_qg1l'))
        else:
            dir_model = config.MOD.dir_model  
        qgm = SourceFileLoader("qgm",f'{dir_model}/jqgm.py').load_module() 
        model = getattr(qgm, config.MOD.name_class)

        # Coriolis
        if config.MOD.f0 is not None and config.MOD.constant_f:
            self.f = config.MOD.f0
        else:
            self.f = State.f
            f0 = np.nanmean(self.f)
            self.f[np.isnan(self.f)] = f0
            
        # Open Rossby Radius if provided
        if config.MOD.filec_aux is not None and os.path.exists(config.MOD.filec_aux):

            ds = xr.open_dataset(config.MOD.filec_aux)
            name_lon = config.MOD.name_var_c['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
            ds = ds.sortby(name_lon)    

            self.c = grid.interp2d(ds,
                                   config.MOD.name_var_c,
                                   State.lon,
                                   State.lat)
            
            if config.MOD.cmin is not None:
                self.c[self.c<config.MOD.cmin] = config.MOD.cmin
                self.c[np.isnan(self.c)] = config.MOD.cmin
            
            if config.MOD.cmax is not None:
                self.c[self.c>config.MOD.cmax] = config.MOD.cmax
            
            if config.EXP.flag_plot>0:
                plt.figure()
                plt.pcolormesh(self.c)
                plt.colorbar()
                plt.title('Rossby phase velocity')
                plt.show()
                
        else:
            self.c = config.MOD.c0 * np.ones((State.ny,State.nx))
        
        # Open MDT map if provided
        if config.MOD.path_mdt is not None and os.path.exists(config.MOD.path_mdt):
                      
            ds = xr.open_dataset(config.MOD.path_mdt)
            name_lon = config.MOD.name_var_mdt['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
            ds = ds.sortby(name_lon)    

            self.mdt = grid.interp2d(ds,
                                   config.MOD.name_var_mdt,
                                   State.lon,
                                   State.lat)

            if config.EXP.flag_plot>0:
                plt.figure()
                plt.pcolormesh(self.mdt)
                plt.colorbar()
                plt.title('MDT')
                plt.show()

        else:
            self.mdt = None
            
        # Initialize model state
        if (config.GRID.super == 'GRID_FROM_FILE') and (config.MOD.name_init_var is not None):
            dsin = xr.open_dataset(config.GRID.path_init_grid)
            for name in self.name_var:
                if name in config.MOD.name_init_var:
                    var_init = dsin[config.MOD.name_init_var[name]]
                    if len(var_init.shape)==3:
                        var_init = var_init[0,:,:]
                    if config.GRID.subsampling is not None:
                        var_init = var_init[::config.GRID.subsampling,::config.GRID.subsampling]
                    dsin.close()
                    del dsin
                    State.var[self.name_var[name]] = var_init.values
                else:
                    State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))
        else:
            for name in self.name_var:  
                State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))
                if State.mask is not None:
                    State.var[self.name_var[name]][State.mask] = np.nan

        # Initialize model Parameters (Flux on SSH and tracers)
        for name in self.name_var:
            State.params[self.name_var[name]] = np.zeros((State.ny,State.nx))

        # Initialize boundary condition dictionnary for each model variable
        self.bc = {}
        self.forcing = {}
        for _name_var_mod in self.name_var:
            self.bc[_name_var_mod] = {}
            self.forcing[_name_var_mod] = {}
        self.init_from_bc = config.MOD.init_from_bc
        self.Wbc = np.zeros((State.ny,State.nx))
        if config.MOD.dist_sponge_bc is not None and State.mask is not None:
            if config.MOD.advect_tracer and config.MOD.bc_trac=='OBC':
                bc = False # No sponge band for open boundaries
            else:
                bc = True
            self.Wbc = grid.compute_weight_map(State.lon, State.lat, deepcopy(State.mask), config.MOD.dist_sponge_bc, bc=bc)
            if config.EXP.flag_plot>1:
                plt.figure()
                plt.pcolormesh(self.Wbc)
                plt.colorbar()
                plt.title('Wbc')
                plt.show()

        # Use boundary conditions as mean field (for 4Dvar only)
        if config.INV is not None and config.INV.super=='INV_4DVAR':
            self.anomaly_from_bc = config.INV.anomaly_from_bc
        else:
            self.anomaly_from_bc = False

        # Tracer advection flag
        self.advect_pv = config.MOD.advect_pv
        self.advect_tracer = config.MOD.advect_tracer
        self.forcing_tracer_from_bc = config.MOD.forcing_tracer_from_bc

        # Ageostrophic velocity flag
        if 'U' in self.name_var and 'V' in self.name_var:
            self.ageo_velocities = True
        else:
            self.ageo_velocities = False

       # Masked array for model initialization
        SSH0 = State.getvar(name_var=self.name_var['SSH'])
            
        # Model initialization
        self.qgm = model(dx=State.DX,
                         dy=State.DY,
                         dt=self.dt,
                         SSH=SSH0,
                         c=self.c,
                         upwind=config.MOD.upwind,
                         time_scheme=config.MOD.time_scheme,
                         g=State.g,
                         f=self.f,
                         Wbc=self.Wbc,
                         Kdiffus=config.MOD.Kdiffus,
                         Kdiffus_trac=config.MOD.Kdiffus_trac,
                         bc_trac=config.MOD.bc_trac,
                         advect_pv=self.advect_pv,
                         ageo_velocities=self.ageo_velocities,
                         constant_c=config.MOD.constant_c,
                         constant_f=config.MOD.constant_f,
                         solver=config.MOD.solver,
                         tile_size=config.MOD.tile_size,
                         tile_overlap=config.MOD.tile_overlap,
                         mdt=self.mdt)

        # Model functions initialization
        self.qgm_step = self.qgm.step_jit
        self.qgm_step_tgl = self.qgm.step_tgl_jit
        self.qgm_step_adj = self.qgm.step_adj_jit

        self.step_jax_jit = jit(self.step_jax, static_argnums=[2,3])

        
        # Tests tgl & adj
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            print('Tangent test:')
            #tangent_test(self,State,nstep=100)
            print('Adjoint test:')
            adjoint_test(self,State,nstep=100)
    
    def init(self, State, t0=0):

        if self.anomaly_from_bc:
            return
        elif type(self.init_from_bc)==dict:
            for name in self.init_from_bc:
                if self.init_from_bc[name] and t0 in self.bc[name]:
                    State.setvar(self.bc[name][t0], self.name_var[name])
        elif self.init_from_bc:
            for name in self.name_var: 
                if t0 in self.bc[name]:
                     State.setvar(self.bc[name][t0], self.name_var[name])

    
    def save_output(self,State,present_date,name_var=None,t=None):
        # Add geostrophic current to ageostrophic velocities
        if self.ageo_velocities:
            State0 = State.copy()
            # Get ageostrophic velocites
            ua = State0.getvar(name_var=self.name_var['U'])
            va = State0.getvar(name_var=self.name_var['V'])
            # Compute geostrophic current from ssh
            ssh = State0.getvar(name_var=self.name_var['SSH'])
            ug,vg = switchvar.ssh2uv(ssh,State0)
            # Set total current
            State0.setvar(ug+ua, name_var=self.name_var['U'])
            State0.setvar(vg+va, name_var=self.name_var['V'])
            State0.save_output(present_date,name_var)
            State0.plot()
        else:
            State.save_output(present_date,name_var)

    def set_bc(self,time_bc,var_bc):

        for _name_var_bc in var_bc:
            for _name_var_mod in self.name_var:
                if _name_var_bc==_name_var_mod:
                    for i,t in enumerate(time_bc):
                        var_bc_t = +var_bc[_name_var_bc][i]
                        # Remove nan
                        var_bc_t[np.isnan(var_bc_t)] = 0.
                        # Fill bc dictionnary
                        self.bc[_name_var_mod][t] = var_bc_t
                elif _name_var_bc==f'{_name_var_mod}_params':
                    for i,t in enumerate(time_bc):
                        var_bc[_name_var_bc][i][np.isnan(var_bc[_name_var_bc][i])] = 0.
                        self.forcing[_name_var_mod][t] = var_bc[_name_var_bc][i]

    def ano_bc(self,t,State,sign):

        if not self.anomaly_from_bc:
            return
        else:
            for name in self.name_var:
                if t in self.bc[name]:
                    State.var[self.name_var[name]] += sign * self.bc[name][t]
            
    def _apply_bc(self,t0,t1):
        
        Xb = jnp.zeros((self.ny,self.nx,))

        if 'SSH' not in self.bc:
            return Xb
        elif len(self.bc['SSH'].keys())==0:
             return Xb
        elif t0 not in self.bc['SSH']:
            # Find closest time
            t_list = np.array(list(self.bc['SSH'].keys()))
            idx_closest = jnp.argmin(jnp.abs(t_list-t0))
            t0 = t_list[idx_closest]

        Xb = self.bc['SSH'][t0]

        if self.advect_tracer:
            Xb = Xb[np.newaxis,:,:]
            for name in self.name_var:
                if name!='SSH' and name in self.bc and len(self.bc[name].keys())>0:
                    if t1 in self.bc[name]: 
                        Cb = self.bc[name][t1]
                    else:
                        # Find closest time
                        t_list = np.array(list(self.bc['SSH'].keys()))
                        idx_closest = np.argmin(np.abs(t_list-t1))
                        new_t1 = t_list[idx_closest]
                        Cb = self.bc[name][new_t1]
                    Xb = np.append(Xb, Cb[np.newaxis,:,:], axis=0)     
        
        return Xb
    
    def step(self,State,nstep=1,t=0):
 
        # Get full field from anomaly 
        self.ano_bc(t,State,+1)

        # Boundary field
        Xb = self._apply_bc(t,int(t+nstep*self.dt))

        # Get state variable(s)
        X0 = State.getvar(name_var=self.name_var['SSH'])
        if self.advect_tracer:
            X0 = X0[np.newaxis,:,:]
            # Ageostrophic velocities
            if self.ageo_velocities:
                U = State.getvar(name_var=self.name_var['U'])[np.newaxis,:,:]
                V = State.getvar(name_var=self.name_var['V'])[np.newaxis,:,:]
                X0 = np.append(X0, U, axis=0)
                X0 = np.append(X0, V, axis=0)
            # Tracers
            for name in self.name_var:
                if name not in ['SSH', 'U', 'V']:
                    C0 = State.getvar(name_var=self.name_var[name])[np.newaxis,:,:]
                    X0 = np.append(X0, C0, axis=0)
        
        # init
        X1 = +X0.astype('float64')

        # Time propagation
        X1 = self.qgm_step(X1,Xb,nstep=nstep)
        t1 = t + nstep*self.dt

        # Convert to numpy array
        X1 = np.array(X1).astype('float64')
        
        # Update state
        if self.name_var['SSH'] in State.params:
            # Fssh = State.params[self.name_var['SSH']].astype('float64') # Forcing term for SSH
            Fssh = np.asarray(State.params[self.name_var['SSH']], dtype=np.float64)
            if self.advect_tracer:
                X1[0] += nstep*self.dt/(3600*24) * Fssh 
                State.setvar(X1[0], name_var=self.name_var['SSH'])
                for i,name in enumerate(self.name_var):
                    if name!='SSH':
                        Fc = +State.params[self.name_var[name]] # Forcing term for tracer or ageostrophic velocities
                        # Add Nudging to BC 
                        if self.forcing_tracer_from_bc:
                            X1[i] += nstep*self.dt/(3600*24) * (1-self.Wbc)  * Fc * (Xb[i] - X0[i]) 
                        # Only forcing flux
                        else:
                            X1[i] += nstep*self.dt/(3600*24) * (1-self.Wbc)  * Fc 
                        State.setvar(X1[i], name_var=self.name_var[name])
            else:
                X1 += nstep*self.dt/(3600*24) * Fssh
                State.setvar(X1, name_var=self.name_var['SSH'])

        # Get anomaly from full field
        self.ano_bc(t1,State,-1)
    
    def step_jax(self,State_vars,State_params,nstep=1,t=0):

        # Boundary field
        Xb = self._apply_bc(t,t+nstep*self.dt)

        # Get state variable(s)
        X0 = State_vars[self.name_var['SSH']]
        if self.advect_tracer:
            X0 = X0[jnp.newaxis,:,:]
            # Tracers
            for name in self.name_var:
                if name not in ['SSH', 'U', 'V']:
                    C0 = State_vars[self.name_var[name]][jnp.newaxis,:,:]
                    X0 = jnp.append(X0, C0, axis=0)
        
        # init
        X1 = +X0

        # Time propagation
        X1 = self.qgm_step(X1,Xb,nstep=nstep)

        # Update state
        if self.name_var['SSH'] in State_params:
            Fssh = State_params[self.name_var['SSH']] # Forcing term for SSH
            if self.advect_tracer:
                X1[0] += nstep*self.dt/(3600*24) * Fssh 
                State_vars[self.name_var['SSH']] = X1[0]
                for i,name in enumerate(self.name_var):
                    if name!='SSH':
                        Fc = State_params[self.name_var[name]] # Forcing term for tracer or ageostrophic velocities
                        # Only forcing flux
                        X1[i] += nstep*self.dt/(3600*24) * (1-self.Wbc)  * Fc 
                        State_vars[self.name_var[name]] = X1[i]
            else:
                X1 += nstep*self.dt/(3600*24) * Fssh
                State_vars[self.name_var['SSH']] = X1
            
        return State_vars

    def step_tgl(self,dState,State,nstep=1,t=0):

        # Get full field from anomaly 
        self.ano_bc(t,State,+1)

        # Boundary field
        Xb = self._apply_bc(t,int(t+nstep*self.dt))
        
        # Get state variable
        dX0 = dState.getvar(name_var=self.name_var['SSH']).astype('float64')
        X0 = State.getvar(name_var=self.name_var['SSH']).astype('float64')
        if self.advect_tracer:
            dX0 = dX0[np.newaxis,:,:]
            X0 = X0[np.newaxis,:,:]
            # Ageostrophic velocities
            if self.ageo_velocities:
                U = State.getvar(name_var=self.name_var['U'])[np.newaxis,:,:]
                V = State.getvar(name_var=self.name_var['V'])[np.newaxis,:,:]
                dU = dState.getvar(name_var=self.name_var['U'])[np.newaxis,:,:]
                dV = dState.getvar(name_var=self.name_var['V'])[np.newaxis,:,:]
                X0 = np.append(X0, U, axis=0)
                X0 = np.append(X0, V, axis=0)
                dX0 = np.append(dX0, dU, axis=0)
                dX0 = np.append(dX0, dV, axis=0)
            # Tracers
            for name in self.name_var:
                if name not in ['SSH', 'U', 'V']:
                    dC0 = dState.getvar(name_var=self.name_var[name])[np.newaxis,:,:]
                    dX0 = np.append(dX0, dC0, axis=0)
                    C0 = State.getvar(name_var=self.name_var[name])[np.newaxis,:,:]
                    X0 = np.append(X0, C0, axis=0)
        
        # init
        dX1 = +dX0.astype('float64')
        X1 = +X0.astype('float64')

        # Time propagation
        dX1 = self.qgm_step_tgl(dX1,X1,hb=Xb,nstep=nstep)

        # Convert to numpy and reshape
        dX1 = np.array(dX1).astype('float64')

        # Update state
        if self.name_var['SSH'] in dState.params:
            dFssh = dState.params[self.name_var['SSH']].astype('float64') # Forcing term for SSH
            if self.advect_tracer:
                dX1[0] +=  nstep*self.dt/(3600*24) * dFssh  
                dState.setvar(dX1[0], name_var=self.name_var['SSH'])
                for i,name in enumerate(self.name_var):
                    if name!='SSH':
                        dFc = dState.params[self.name_var[name]] # Forcing term for tracer or ageostrophic velocities
                        # Add Nudging to BC 
                        if self.forcing_tracer_from_bc:
                            Fc = State.params[self.name_var[name]] 
                            dX1[i] +=  nstep*self.dt/(3600*24) * (1-self.Wbc) *\
                                  (dFc * (Xb[i] - X0[i]) - Fc * dX0[i])
                        # Only forcing flux
                        else:
                            dX1[i] +=  nstep*self.dt/(3600*24) * dFc  * (1-self.Wbc)
                        dState.setvar(dX1[i], name_var=self.name_var[name])
            else:
                dX1 += nstep*self.dt/(3600*24) * dFssh  
                dState.setvar(dX1, name_var=self.name_var['SSH'])

        # Get anomaly from full field
        self.ano_bc(t,State,-1)

    def step_adj(self,adState,State,nstep=1,t=0):
        
        # Get full field from anomaly 
        self.ano_bc(t,State,+1)

        # Boundary field
        Xb = self._apply_bc(t,int(t+nstep*self.dt))

        # Get state variable
        adSSH0 = adState.getvar(name_var=self.name_var['SSH']).astype('float64')
        SSH0 = State.getvar(name_var=self.name_var['SSH']).astype('float64')
        if self.advect_tracer:
            adX0 = adSSH0[np.newaxis,:,:].astype('float64')
            X0 = SSH0[np.newaxis,:,:].astype('float64')
            # Ageostrophic velocities
            if self.ageo_velocities:
                U = State.getvar(name_var=self.name_var['U'])[np.newaxis,:,:]
                V = State.getvar(name_var=self.name_var['V'])[np.newaxis,:,:]
                adU = adState.getvar(name_var=self.name_var['U'])[np.newaxis,:,:]
                adV = adState.getvar(name_var=self.name_var['V'])[np.newaxis,:,:]
                X0 = np.append(X0, U, axis=0)
                X0 = np.append(X0, V, axis=0)
                adX0 = np.append(adX0, adU, axis=0)
                adX0 = np.append(adX0, adV, axis=0)
            # Tracers
            for name in self.name_var:
                if name not in ['SSH', 'U', 'V']:
                    adC0 = adState.getvar(name_var=self.name_var[name])[np.newaxis,:,:]
                    adX0 = np.append(adX0, adC0, axis=0)
                    C0 = State.getvar(name_var=self.name_var[name])[np.newaxis,:,:]
                    X0 = np.append(X0, C0, axis=0)
        else:
            adX0 = adSSH0
            X0 = SSH0

        # Init
        adX1 = +adX0
        X1 = +X0

        # Time propagation
        adX1 = self.qgm_step_adj(adX1,X1,Xb,nstep=nstep)

        # Convert to numpy and reshape
        adX1 = np.array(adX1).squeeze().astype('float64')

        # Update state and parameters
        if self.name_var['SSH'] in adState.params:
            for i,name in enumerate(self.name_var):
                adparams = nstep*self.dt/(3600*24) *\
                    adState.getvar(name_var=self.name_var[name]).astype('float64') 
                if name!='SSH':
                    adparams *= (1-self.Wbc)
                    if self.forcing_tracer_from_bc:
                        Fc = State.params[self.name_var[name]] 
                        adparams *=  (Xb[i] - X0[i])
                        adX1[i] += -nstep*self.dt/(3600*24) * (1-self.Wbc) * Fc * adX0[i]
                adState.params[self.name_var[name]] += adparams  
                
        if self.advect_tracer:
            adState.setvar(adX1[0],self.name_var['SSH'])
            for i,name in enumerate(self.name_var):
                if name!='SSH':
                    adState.setvar(adX1[i],self.name_var[name])
        else:
            adState.setvar(adX1,self.name_var['SSH'])

##############################################################################
#                         Shallow Water Model                                #
##############################################################################
  
class Model_sw1l_jax(M):

    def __init__(self,config,State):

        super().__init__(config,State)

        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

        self.config = config

        ############################
        ### MODEL SPECIFICATIONS ###
        ############################

        # Time integration scheme 
        self.time_scheme = config.MOD.time_scheme

        # Boundary condition types
        self.bc_kind = config.MOD.bc_kind # For the domain boundaries
        self.bc_island = config.MOD.bc_island # For the islands and continents 

        # Grid specifications
        self.ny = State.ny
        self.nx = State.nx

        # Coriolis
        self.f = State.f
        f0 = np.nanmean(self.f)
        self.f[np.isnan(self.f)] = f0

        # Gravity
        self.g = State.g

        # Tidal frequency components 
        self.omegas = np.asarray(config.MOD.w_waves)
        self.omega_names = config.MOD.w_names

        # Tidal velocities 
        self.init_tidal_velocity(config,State)

        # Bathymetry field 
        self.init_bathy(config,State)

        ####################################
        ### INITIALIZING MODEL VARIABLES ###
        ####################################

        # List of variable names
        self.name_var = config.MOD.name_var

        # Setting Model mask # 
        self.mask = {}
        self.set_mask(State.mask,config)

        self.init_variables(config,State)

        #############################################
        ### INITIALIZING MODEL CONTROL PARAMETERS ###
        #############################################

        # List of parameter names
        self.name_params = config.MOD.name_params
                
        # Initializing model params
        self.init_params(config,State)

        #################################
        ### LOADING MODEL PYTHON FILE ### 
        #################################

        if config.MOD.dir_model is None:
            dir_model = os.path.realpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','models','model_sw1l'))
        else:
            dir_model = config.MOD.dir_model
        
        swm = SourceFileLoader("swm", 
                                dir_model + "/jswm.py").load_module()

        # Model initialization
        self.swm = swm.Swm(Model = self,
                           State = State) 

        # Model functions initialization
        if config.INV is not None and config.INV.super in ['INV_4DVAR','INV_4DVAR_PARALLEL']:
            self.swm_step = self.swm.step_jit
            self.swm_step_tgl = self.swm.step_tgl_jit
            self.swm_step_adj = self.swm.step_adj_jit
        else:
            self.swm_step = self.swm.step_jit

        # Tests tgl & adj
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            print('Tangent test:')
            print('Tangest test is commented.')
            # tangent_test(self,State,nstep=100)
            print('Adjoint test:')
            print('Tangest test is commented.')
            # adjoint_test(self,State,nstep=100)

    def init (self,State,t0=0):

        return

    

    def init_tidal_velocity(self,config,State):
        """
        NAME
            init_tidal_velocity

        DESCRIPTION
            Reads tidal velocity file, interpolate it to the grid
        """

        # Read tidal velocities
        if config.MOD.path_tidal_velocity is not None and os.path.exists(config.MOD.path_tidal_velocity): 
            
            # Variables
            
            self.tidal_U = np.zeros((len(self.omega_names), # Number of tidal components
                                     State.lat[:,0].size, # Number of latitude grid points 
                                     State.lon[0,:].size, # Number of longitude grid points 
                                     ))
            
            self.tidal_V = np.zeros((len(self.omega_names), # Number of tidal components
                                     State.lat[:,0].size, # Number of latitude grid points
                                     State.lon[0,:].size, # Number of longitude grid points  
                                     ))
            
            for (i,name) in enumerate(self.omega_names):
                self.tidal_U[i,:,:] = self.open_interpolate(config,name,"U",State)
                self.tidal_V[i,:,:] = self.open_interpolate(config,name,"V",State)
        
        else: # No tidal velocity file prescripted
            warnings.warn("No tidal velocity field prescribed. This is not suitable if Internal Tide generation ('itg') is being controlled.")
            return None 
        

    def init_bathy(self,config,State):

        """
        NAME
            init_bathy

        DESCRIPTION
            Read bathymetry file, interpolate it to the grid
        """

        # Read bathymetry 
        if config.MOD.path_bathymetry is not None and os.path.exists(config.MOD.path_bathymetry):
            ds = xr.open_dataset(config.MOD.path_bathymetry).squeeze()
            name_lon = config.MOD.name_var_bathy['lon']
            name_lat = config.MOD.name_var_bathy['lat']
            name_elevation = config.MOD.name_var_bathy['var']

        else: # No bathymetry file prescripted
            warnings.warn("No bathymetry field prescribed.")
            return None 

        # Convert longitudes
        if np.sign(ds[name_lon].data.min())==-1 and State.lon_unit=='0_360':
            ds = ds.assign_coords({name_lon:((name_lon, ds[name_lon].data % 360))})
        elif np.sign(ds[name_lon].data.min())==1 and State.lon_unit=='-180_180':
            ds = ds.assign_coords({name_lon:((name_lon, (ds[name_lon].data + 180) % 360 - 180))})
        ds = ds.sortby(ds[name_lon])   


        dlon =  np.nanmax(State.lon[:,1:] - State.lon[:,:-1])
        dlat =  np.nanmax(State.lat[1:,:] - State.lat[:-1,:])
        dlon +=  np.nanmax(ds[name_lon].data[1:] - ds[name_lon].data[:-1])
        dlat +=  np.nanmax(ds[name_lat].data[1:] - ds[name_lat].data[:-1])

        ds = ds.sel(
            {name_lon:slice(State.lon_min-dlon,State.lon_max+dlon),
                name_lat:slice(State.lat_min-dlat,State.lat_max+dlat)})

        ds = ds.interp(coords={name_lon:State.lon[0,:],name_lat:State.lat[:,0]},method='cubic')

        ds = ds.where(ds.elevation<0,0) # replacing the continents (where ds.elevation>0) with 0 

        self.bathymetry = ds[name_elevation].values

        # Calculating bathymetry gradient 
        # X component of gradient
        grad_x = np.zeros(State.X.shape)
        grad_x[:,1:-1] = (self.bathymetry[:,2:]-self.bathymetry[:,0:-2])/(State.X[:,2:]-State.X[:,0:-2]) # inner part of gradient 
        grad_x[:,0] = (self.bathymetry[:,1]-self.bathymetry[:,0])/(State.X[:,1]-State.X[:,0])
        grad_x[:,-1] = (self.bathymetry[:,-1]-self.bathymetry[:,-2])/(State.X[:,-1]-State.X[:,-2])
        
        # Y component of gradient
        grad_y = np.zeros(State.Y.shape)
        grad_y[1:-1,:] = (self.bathymetry[2:,:]-self.bathymetry[0:-2,:])/(State.Y[2:,:]-State.Y[0:-2,:])
        grad_y[0,:] = (self.bathymetry[1,:]-self.bathymetry[0,:])/(State.Y[1,:]-State.Y[0,:])
        grad_y[-1,:] = (self.bathymetry[-1,:]-self.bathymetry[-2,:])/(State.Y[-1,:]-State.Y[-2,:])

        # Applying bathymetry smoothing if prescribed 
        if config.MOD.smooth_wavelength != None and np.round(config.MOD.smooth_wavelength/State.dx).astype(np.int32) > 0 : 
            N_pixel = np.round(config.MOD.smooth_wavelength/State.dx).astype(np.int32)
            array_pascal = factorial(N_pixel-1)/(factorial(np.ones((1,N_pixel))*(N_pixel-1)-np.arange(0,N_pixel).reshape((1,N_pixel)))*factorial(np.arange(0,N_pixel).reshape((1,N_pixel))))
            gaussian_kernel = (1/array_pascal.sum()**2)*array_pascal.T*array_pascal
            grad_x = convolve2d(grad_x,gaussian_kernel,mode='same', boundary='fill', fillvalue=0)
            grad_y = convolve2d(grad_y,gaussian_kernel,mode='same', boundary='fill', fillvalue=0)
        
        self.grad_bathymetry_x = grad_x
        self.grad_bathymetry_y = grad_y

    def open_interpolate(self,config,name,direction,State):
        """
        NAME
            open_interpolate

        DESCRIPTION
            Opens and interpolates the tidal velocity files 

        ARGUMENT 
            - config : config python file 
            - name (str) : name of the tidal component 
            - direction (str) :  velocity direction, either "U" or "V"
        """

        if direction == "U":
            ds = xr.open_dataset(os.path.join(config.MOD.path_tidal_velocity,"eastward_velocity",name+".nc")).squeeze()
        elif direction == "V":
            ds = xr.open_dataset(os.path.join(config.MOD.path_tidal_velocity,"northward_velocity",name+".nc")).squeeze()
        
        # Convert longitudes
        if np.sign(ds["lon"].data.min())==-1 and State.lon_unit=='0_360':
            ds = ds.assign_coords({"lon":(("lon", ds["lon"].data % 360))})
        elif np.sign(ds["lon"].data.min())==1 and State.lon_unit=='-180_180':
            ds = ds.assign_coords({"lon":(("lon", (ds["lon"].data + 180) % 360 - 180))})
        ds = ds.sortby(ds["lon"])   


        dlon =  np.nanmax(State.lon[:,1:] - State.lon[:,:-1])
        dlat =  np.nanmax(State.lat[1:,:] - State.lat[:-1,:])
        dlon +=  np.nanmax(ds["lon"].data[1:] - ds["lon"].data[:-1])
        dlat +=  np.nanmax(ds["lat"].data[1:] - ds["lat"].data[:-1])

        ds = ds.sel(
            {"lon":slice(State.lon_min-dlon,State.lon_max+dlon),
                "lat":slice(State.lat_min-dlat,State.lat_max+dlat)})

        ds =ds.fillna(0)

        ds = ds.interp(coords={"lon":State.lon[0,:],"lat":State.lat[:,0]},method='cubic')
        
        if direction == "U":
            return ds["Ua"].values*1E-2 # Converting into m/s
        elif direction == "V":
            return ds["Va"].values*1E-2 # Converting into m/s

    def init_variables(self,config,State) : 

        """
        Initialize model state and auxiliary variables based on the configuration file.
        This method initializes the model state variables in the State object, and the auxiliary variables in the Model object.

        Args:
        -----
            config (Config): The configuration file.
            State (State): A State object which will be modified to include the model state variables.

        Notes:
            - Coastal pixel indexes and auxiliary variables are set up only if `State.mask` is not None.
        """

        ##################################################
        ### - INITIALIZING THE MODEL STATE VARIABLES - ###
        ##################################################

        # Iinitializing from a specified grid file 
        if (config.GRID.super == 'GRID_FROM_FILE') and (config.MOD.name_init_var is not None):
            dsin = xr.open_dataset(config.GRID.path_init_grid)
            for name in self.name_var:
                if name in config.MOD.name_init_var:
                    var_init = dsin[config.MOD.name_init_var[name]]
                    if len(var_init.shape)==3:
                        var_init = var_init[0,:,:]
                    if config.GRID.subsampling is not None:
                        var_init = var_init[::config.GRID.subsampling,::config.GRID.subsampling]
                    dsin.close()
                    del dsin
                    State.var[self.name_var[name]] = var_init.values
        # Iinitializing with zeros 
        else:
            for name in self.name_var:
                State.var[self.name_var[name]] = np.zeros((State.ny,State.nx),dtype='float64')
                State.var[self.name_var[name]][State.mask] = np.nan

        ######################################################
        ### - INITIALIZING THE MODEL AUXILIARY VARIABLES - ###
        ######################################################

        # Coastal pixel indexes 
        self.idxcoast = {}
        # Coastal pixel values 
        self.auxvar = {}

        # Setting coastal pixel indexes 
        if State.mask is not None : 

            idxcoastN = np.where( np.invert(State.mask[:-1,:]) * State.mask[1:,:]  )
            idxcoastS = np.where( np.invert(State.mask[1:,:])  * State.mask[:-1,:] )
            idxcoastW = np.where( np.invert(State.mask[:,1:])  * State.mask[:,:-1] )
            idxcoastE = np.where( np.invert(State.mask[:,:-1]) * State.mask[:,1:]  )

            # NORTH coastal indexes # 
            self.idxcoast["vN"] = idxcoastN
            self.idxcoast["hN"] = (idxcoastN[0]+1,idxcoastN[1])
            # SOUTH coast variables # 
            self.idxcoast["vS"] = idxcoastS
            self.idxcoast["hS"] = (idxcoastS[0],idxcoastS[1])
            # WEST coast variables #  
            self.idxcoast["uW"] = idxcoastW
            self.idxcoast["hW"] = (idxcoastW[0],idxcoastW[1])
            # EAST coast variables # 
            self.idxcoast["uE"] = idxcoastE
            self.idxcoast["hE"] = (idxcoastE[0],idxcoastE[1]+1)

            if self.bc_island == "radiative" : 
            # Auxiliary variables to store ghost pixels (ssh at the coast and orthogonal values) 
                # NORTH coastal indexes # 
                self.auxvar["vN"] = np.zeros(idxcoastN[0].shape,dtype='float64')
                self.auxvar["hN"] = np.zeros(idxcoastN[0].shape,dtype='float64')
                # SOUTH coastal indexes # 
                self.auxvar["vS"] = np.zeros(idxcoastS[0].shape,dtype='float64')
                self.auxvar["hS"] = np.zeros(idxcoastS[0].shape,dtype='float64')
                # WEST coastal indexes # 
                self.auxvar["uW"] = np.zeros(idxcoastW[0].shape,dtype='float64')
                self.auxvar["hW"] = np.zeros(idxcoastW[0].shape,dtype='float64')
                # EAST coastal indexes # 
                self.auxvar["uE"] = np.zeros(idxcoastE[0].shape,dtype='float64')
                self.auxvar["hE"] = np.zeros(idxcoastE[0].shape,dtype='float64')

    def init_params(self,config,State) :

        """
        Initializes the model controlled parameters information based on the configuration file. 
        This method also initializes the parameters in the State object. 

        Args:
        -----
            config (Config): The configuration file.
            State (State): A State object which will be modified to include the initialized parameters.

        Notes:
        ------
            Parameters in `self.name_params` should be one of :
                - 'He' : Equivalent Height
                - 'hbcx' : Height Boundary Conditions along x axis
                - 'hbcy' : Height Boundary Conditions along y axis
                - 'itg' : Internal Tide Generation

        """

        # Dictionary containing the shape of the parameters
        self.shape_params = {} 
        # Dictionary containing the slices of the parameters
        self.slice_params = {}

        #######################################
        ### - INITIALIZING SPECIFICATIONS - ###
        #######################################

        # - Equivalent Height He background 
        if config.MOD.He_data is not None and os.path.exists(config.MOD.He_data['path']):
            ds = xr.open_dataset(config.MOD.He_data['path'])
            self.Heb = ds[config.MOD.He_data['var']].values
        else:
            self.Heb = config.MOD.He_init
        
        # Height boundary condition hbc structure  
        if 'HBCX' in self.name_params and 'HBCY' in self.name_params :
            if config.MOD.Ntheta>0:
                theta_p = np.arange(0,pi/2+pi/2/config.MOD.Ntheta,pi/2/config.MOD.Ntheta)
                self.bc_theta = np.append(theta_p-pi/2,theta_p[1:]) 
            else:
                self.bc_theta = np.array([0])
        elif 'HBCX' in self.name_params or 'HBCY' in self.name_params :
            warnings.warn("Only partly controlling boundary conditions (either just x or y)", Warning)
            if config.MOD.Ntheta>0:
                theta_p = np.arange(0,pi/2+pi/2/config.MOD.Ntheta,pi/2/config.MOD.Ntheta)
                self.bc_theta = np.append(theta_p-pi/2,theta_p[1:]) 
            else:
                self.bc_theta = np.array([0])

        ###############################
        ### - INITIALIZING SHAPES - ###
        ###############################               

        for param in self.name_params : 

            # If the parameter is not implemented 
            if param not in ['HE','HE_OFFSET','HBCX','HBCY','ITG'] : 
                sys.exit(param+" not implemented. Please choose parameters among ['HE','HE_OFFSET','HBCX','HBCY','ITG'].")

            # - Equivalent Height : He 
            elif param =='HE' : 
                self.shape_params['HE'] = [State.ny,    # - Number of grid points along y axis.
                                           State.nx]    # - Number of grid points along x axis.

            # - Equivalent Height Offset : He_offset
            elif param =='HE_OFFSET' : 
                self.shape_params['HE_OFFSET'] = [State.ny,    # - Number of grid points along y axis.
                                                  State.nx]    # - Number of grid points along x axis.
                
            # - Height Boundary Conditions along x : hbcx 
            elif param =='HBCX' : 
                self.shape_params['HBCX'] = [len(self.omegas),      # - Number of tidal frequency components 
                                            2,                      # - Number of boundaries (North & South)
                                            2,                      # - Number of controlled components (cos & sin)
                                            len(self.bc_theta),     # - Number of angles
                                            State.nx]               # - Number of gridpoints along x axis
                
            # - Height Boundary Conditions along y : hbcy
            elif param =='HBCY' :
                self.shape_params['HBCY'] = [len(self.omegas),      # - Number of tidal frequency components 
                                            2,                      # - Number of boundaries (East & West)
                                            2,                      # - Number of controlled components (cos & sin)
                                            len(self.bc_theta),     # - Number of angles
                                            State.ny]               # - Number of gridpoints along y axis
            
            # - Internal Tide Generation itg 
            elif param =='ITG' :
                self.shape_params['ITG'] = [len(self.omegas),       # - Number of tidal frequency components 
                                            4,                      # - Number of estimated parameter (cos and sin for x and y axis)
                                            State.ny,               # - Number of grid points along y axis.
                                            State.nx]               # - Number of grid points along x axis.      

        #####################################################
        ### - INITIALIZING SLICE AND NUMBER INFORMATION - ###
        #####################################################

        # Number of parameters 
        self.nparams = sum(list(map(np.prod,list(self.shape_params.values()))))

        # Slices of parameters
        idx = 0 
        for param in self.name_params : 
            self.slice_params[param] = slice(idx, idx + np.prod(self.shape_params[param]))
            idx += np.prod(self.shape_params[param])

        #######################################################
        ### - INITIALIZING PARAMETERS IN THE STATE OBJECT - ###
        #######################################################      

        for param in self.name_params :     
            State.params[self.name_params[param]] = np.zeros((self.shape_params[param]),dtype='float64')

    def _detect_coast(self,mask,axis):
        """
        NAME
            _detect_coast

        ARGUMENT 
            mask : mask of continents (N,n) shaped array
            axis : either "x" or "y"
    
        DESCRIPTION
            Detects coast between pixels. 

        RETURNS 
            (N-1,n) or (N,n-1) array with True if it is a coast (transisition continent - ocean) False otherwise. 
        """
        if axis == "x": 
            a1 = mask[:,1:]
            a2 = mask[:,:-1]
        elif axis == "y": 
            a1 = mask[1:,:]
            a2 = mask[:-1,:]
        p1 = np.logical_and(a1,np.invert(a2))
        p2 = np.logical_and(a2,np.invert(a1))
        return np.logical_or(p1,p2)
    
    def set_mask(self,mask,config) : 
        """
        NAME
            set_mask

        ARGUMENT 
            mask : mask to set 
    
        DESCRIPTION
            Sets the mask attribute of the Model object. The mask is a dictionnary containing the masks of variables, represented by an int (0,1 or NaN). 
            For "SSH" the mask is : 
                - 1 if ocean 
                - 999 if continent (NaN value)
            For "U" and "V" the mask is : 
                - 1 if ocean 
                - 999 if continent (NaN value)
                - 0 if normal to the coast    
        """

        for varname in config.MOD.name_var:

            if varname == "SSH" : 
                mask_ssh = np.ones(mask.shape,dtype='float')
                mask_ssh[mask==True]=np.nan
                self.mask[config.MOD.name_var[varname]] = mask_ssh

            elif varname == "U" :
                mask_u = np.ones(mask[:,1:].shape,dtype='float')
                mask_u[np.logical_and(mask[:,1:],mask[:,:-1])]=np.nan
                mask_u[self._detect_coast(mask,"x")]=0
                self.mask[config.MOD.name_var[varname]] = mask_u

            elif varname == "V" : 
                mask_v = np.ones(mask[1:,:].shape,dtype='float')
                mask_v[np.logical_and(mask[1:,:],mask[:-1,:])]=np.nan
                mask_v[self._detect_coast(mask,"y")]=0
                self.mask[config.MOD.name_var[varname]] = mask_v

    def step(self,State,nstep=1,t=0):

        ############################
        ###   INITIALIZATION    ####
        ############################

        X0 = self.init_array(State,t)

        #############################
        ###   TIME PROPAGATION   ####
        #############################

        # X1 = self.swm_step(X0,nstep=nstep)
        # Init
        X1 = +X0
        for _ in range(nstep):
            X1 = self.swm.one_step_jit(X1)
        
        # Remove time in output array
        X1 = X1[1:]

        ##################
        ###   SAVING   ###
        ##################
        
        self.save_variables(State, X1)
    

    def step_tgl(self,dState,State,nstep=1,t=0):
        
        ############################
        ###   INITIALIZATION    ####
        ############################

        X0 = self.init_array(State,t)
        dX0 = self.init_array(dState,t)

        #############################
        ###   TIME PROPAGATION   ####
        #############################

        # dX1 = self.swm_step_tgl(dX0,X0,nstep=nstep)

        dX1 = +dX0
        X1 = +X0
        for i in range(nstep):
            # One timestep
            dX1 = self.swm.step_tgl_jit(dX1,X1)
            if i<nstep-1:
                X1 = self.swm.one_step_jit(X1)

        # Convert to numpy and reshape
        dX1 = np.array(dX1).astype('float64')

        # Remove time in control vector
        dX1 = dX1[1:]

        ##################
        ###   SAVING   ###
        ##################

        self.save_variables(dState,dX1)

        # return 

    def step_adj(self,adState,State,nstep=1,t=0): 

        ############################
        ###   INITIALIZATION    ####
        ############################

        X0 = self.init_array(State,t)
        adX0 = self.init_array(adState,t)

        #############################
        ###   TIME PROPAGATION   ####
        #############################

        # adX1 = self.swm_step_adj(adX0,X0,nstep=nstep)

        # Init
        adX1 = +adX0
        X1 = +X0

        # adX1 = adX1[1:]
        
        traj = [X1]
        if nstep>1:
            for i in range(nstep):
                # One timestep
                X1 = self.swm.one_step_jit(X1)
                if i<nstep-1:
                    traj.append(+X1)
            
        # Reversed time propagation
        # Add time in control vector (for JAX)
        # adX1 = np.append(traj[-1][0],adX1)
        adX1[0] = traj[-1][0]
        for i in reversed(range(nstep)):
            X1 = traj[i]
            # One timestep
            adX1 = self.swm.step_adj_jit(adX1,X1)

        # Convert to numpy and reshape
        adX1 = np.array(adX1).astype('float64')

        # Remove time in control vector
        adX1 = adX1[1:]

        ##################
        ###   SAVING   ###
        ##################

        self.save_params(adState,adX1)
        self.save_variables(adState,adX1)

    def init_array(self,State,t=0):
        """
        NAME
            init_array

        ARGUMENT 
            State : State object 
            t : time step 
    
        DESCRIPTION
            Initializes the array X0, which is composed by the State variable and State parameters at time step t. 

        RETURNS
            X0 : array 
        """

        # - Get state variable
        u0 = State.getvar(self.name_var['U'])[:,:-1].flatten()
        v0 = State.getvar(self.name_var['V'])[:-1,:].flatten()
        h0 = State.getvar(self.name_var['SSH'],vect = True)

        # - test - # 
        u0[np.isnan(u0)]=0
        v0[np.isnan(v0)]=0
        h0[np.isnan(h0)]=0

        # - Get auxiliary variables - coastal values
        # if dirichlet condition : all auxiliary variables set to zero
        len_varcoast = self.idxcoast["vN"][0].size + self.idxcoast["hN"][0].size +\
                       self.idxcoast["vS"][0].size + self.idxcoast["hS"][0].size +\
                       self.idxcoast["uW"][0].size + self.idxcoast["hW"][0].size +\
                       self.idxcoast["uE"][0].size + self.idxcoast["hE"][0].size 
        varcoast = np.zeros((len_varcoast,),dtype='float64')
        # if radiative conditions : auxiliary variables are stored in self.auxvar
        if self.bc_island == "radiative": 
            varcoast = np.concatenate(list(self.auxvar.values()))
        
        # - Create state vector X0 
        X0 = np.concatenate((u0,v0,h0,varcoast))

        # - Get parameters variable 
        if State.params is not None:
            for param in self.name_params : 
                params = +State.getparams(self.name_params[param],vect=True)
                X0 = np.concatenate((X0,params))

        # - Add time in input array
        X0 = np.append(t,X0)

        return X0

    def save_variables(self,State,X1):

        """
        NAME
            save_variables

        ARGUMENT 
            State : State object 
            X1 : array
    
        DESCRIPTION
            Saves the state variables of the array X1 onto the State object.  
        """
        
        # - u, v, and h   
        u1 = np.array(X1[self.swm.sliceu]).reshape(self.swm.shapeu)
        v1 = np.array(X1[self.swm.slicev]).reshape(self.swm.shapev)
        h1 = np.array(X1[self.swm.sliceh]).reshape(self.swm.shapeh)


        if np.any(State.mask): # if mask is containing pixels to mask 
            # Masking the variables 
            u1[State.mask[:,0:-1]] = np.nan
            v1[State.mask[0:-1,:]] = np.nan
            h1[State.mask] = np.nan

            # setting coastal variables 
            self.auxvar["vN"] = np.array(X1[self.swm.slicevN])
            self.auxvar["hN"] = np.array(X1[self.swm.slicehN])

            self.auxvar["vS"] = np.array(X1[self.swm.slicevS])
            self.auxvar["hS"] = np.array(X1[self.swm.slicehS])

            self.auxvar["uW"] = np.array(X1[self.swm.sliceuW])
            self.auxvar["hW"] = np.array(X1[self.swm.slicehW])

            self.auxvar["uE"] = np.array(X1[self.swm.sliceuE])
            self.auxvar["hE"] = np.array(X1[self.swm.slicehE])

        # setting u, v, and h in State 
        State.var[self.name_var['U']][:,0:-1] = u1
        State.var[self.name_var['V']][0:-1,:] = v1
        State.var[self.name_var['SSH']] = h1

    def save_params(self,State,X1):

        """
        NAME
            save_params

        ARGUMENT 
            State : State object 
            X1 : array
    
        DESCRIPTION
            Saves the control parameters of the array X1 onto the State object.  
        """

        #setting params 
        params = X1[self.swm.nstates:]
        for param in self.name_params :    
            State.params[self.name_params[param]] = params[self.slice_params[param]].reshape(self.shape_params[param])

###############################################################################
#                          Multi-models class                                 #
###############################################################################      

class Model_multi:

    def __init__(self,config,State):

        self.Models = []
        _config = config.copy()

        for _MOD in config.MOD:
            _config.MOD = config.MOD[_MOD]
            self.Models.append(Model(_config,State))
            print()

        # Time parameters
        self.dt = int(np.max([M.dt for M in self.Models])) # We take the longer timestep 
        self.nt = 1 + int((config.EXP.final_date - config.EXP.init_date).total_seconds()//self.dt)
        self.T = np.arange(self.nt) * self.dt
        self.timestamps = [] 
        t = config.EXP.init_date
        while t<=config.EXP.final_date:
            self.timestamps.append(t)
            t += timedelta(seconds=self.dt)

        # Model variables: for each variable ('SSH', 'SST', 'Chl' etc...), 
        # we initialize a new variable for the sum of the different contributions
        self.name_var = {}
        self.name_var_tot = {}
        _name_var_tmp = []
        self.var_to_save = []
        for M in self.Models:
            self.var_to_save = np.concatenate((self.var_to_save, M.var_to_save))
            for name in M.name_var:
                if name not in _name_var_tmp:
                    _name_var_tmp.append(name)
                else:
                    # At least two component for the same variable, so we initialize a global variable
                    new_name = f'{name}_tot'
                    self.name_var[name] = new_name
                    self.name_var_tot[name] = new_name
                    # Initialize new State variable
                    if new_name in State.var:
                        State.var[new_name] += State.var[M.name_var[name]]
                    else:
                        State.var[new_name] = State.var[M.name_var[name]].copy()
                    if M.name_var[name] in M.var_to_save and new_name not in self.var_to_save:
                        self.var_to_save = np.append(self.var_to_save,new_name)
        self.var_to_save = list(self.var_to_save)

        for M in self.Models:
            for name in M.name_var:
                if name not in self.name_var_tot:
                    self.name_var[name] = M.name_var[name]

        # Tests tgl & adj
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            #for M in self.Models:
                #print('Tangent test:')
                #tangent_test(M,State,nstep=10)
                #print('Adjoint test:')
                #adjoint_test(M,State,nstep=10)
            print('MultiModel Tangent test:')
            tangent_test(self,State,nstep=10)
            print('QG1L_JAX Adjoint test:')
            adjoint_test(self,State,nstep=10)

    def init(self,State,t0=0):

        # Intialization
        var_tot_tmp = {}
        for name in self.name_var:
            var_tot_tmp[name] = np.zeros_like(State.var[self.name_var[name]]) 

        for M in self.Models:
            M.init(State, t0=t0)
            for name in self.name_var:
                if name in M.name_var:
                    var_tot_tmp[name] += State.var[M.name_var[name]]

        # Update state
        for name in self.name_var:
            State.var[self.name_var[name]] = var_tot_tmp[name]

    def set_bc(self,time_bc,var_bc):

        for M in self.Models:
            M.set_bc(time_bc,var_bc)

    def save_output(self,State,present_date,name_var=None,t=None):

        for M in self.Models:
            M.save_output(State,present_date,name_var,t)

    def step(self,State,nstep=1,t=None):

        # Intialization
        var_tot_tmp = {}
        for name in self.name_var_tot:
            var_tot_tmp[name] = jnp.zeros_like(State.var[self.name_var[name]]) 
        
        # Loop over models
        for M in self.Models:
            _nstep = nstep*self.dt//M.dt
            # Forward propagation
            M.step(State,nstep=_nstep,t=t)
            # Add to total variables
            for name in self.name_var:
                if name in M.name_var and (name in self.name_var_tot):
                    var_tot_tmp[name] += +State.var[M.name_var[name]]
                    
        # Update state
        for name in self.name_var_tot:
            State.var[self.name_var_tot[name]] = var_tot_tmp[name]

    def step_tgl(self,dState,State,nstep=1,t=None):

        # Intialization
        var_tot_tmp = {}
        for name in self.name_var_tot:
            var_tot_tmp[name] = np.zeros_like(State.var[self.name_var[name]]) 

        # Loop over models
        for M in self.Models:
            _nstep = nstep*self.dt//M.dt
            # Tangent propagation
            M.step_tgl(dState,State,nstep=_nstep,t=t)
            # Add to total variables
            for name in self.name_var:
                if name in M.name_var and name in var_tot_tmp:
                    var_tot_tmp[name] += dState.var[M.name_var[name]]
        
        # Update state
        for name in self.name_var_tot:
            dState.var[self.name_var_tot[name]] = var_tot_tmp[name]

    def step_adj(self,adState,State,nstep=1,t=None):

        # Intialization
        var_tot_tmp = {}
        for name in self.name_var_tot:
            var_tot_tmp[name] = adState.var[self.name_var_tot[name]]
        
        # Loop over models
        for M in self.Models:
            _nstep = nstep*self.dt//M.dt
            # Add to local variable
            for name in self.name_var:
                if name in M.name_var and name in self.name_var_tot:
                    adState.var[M.name_var[name]] += var_tot_tmp[name]  
            # Adjoint propagation
            M.step_adj(adState,State,nstep=_nstep,t=t)
        
        for name in self.name_var_tot:
            adState.var[self.name_var_tot[name]] *= 0 
             
###############################################################################
#                       Tangent and Adjoint tests                             #
###############################################################################     
    
def tangent_test(M,State,t0=0,nstep=1):

    # Boundary conditions
    var_bc = {}
    for name in M.name_var:
        var_bc[name] = {0:np.random.random((M.ny,M.nx)).astype('float64'),
                        1:np.random.random((M.ny,M.nx)).astype('float64')}
    M.set_bc([t0,t0+nstep*M.dt],var_bc)

    State0 = State.random()
    dState = State.random()
    State0_tmp = State0.copy()
    
    M.step(t=t0,State=State0_tmp,nstep=nstep)
    X2 = State0_tmp.getvar(vect=True) 
    
    for p in range(10):
        
        lambd = 10**(-p)
        
        State1 = dState.copy()
        State1.scalar(lambd)
        State1.Sum(State0)

        M.step(t=t0,State=State1,nstep=nstep)
        X1 = State1.getvar(vect=True)
        
        dState1 = dState.copy()
        dState1.scalar(lambd)
        M.step_tgl(t=t0,dState=dState1,State=State0,nstep=nstep)

        dX = dState1.getvar(vect=True)
        
        mask = np.isnan(X1+X2+dX)
        
        ps = np.linalg.norm(X1[~mask]-X2[~mask]-dX[~mask])/np.linalg.norm(dX[~mask])
    
        print('%.E' % lambd,'%.E' % ps)
        
def adjoint_test(M,State,t0=0,nstep=1):

    # Boundary conditions
    var_bc = {}
    for name in M.name_var:
        var_bc[name] = {0:np.random.random((M.ny,M.nx)).astype('float64'),
                        1:np.random.random((M.ny,M.nx)).astype('float64')}
    M.set_bc([t0,t0+nstep*M.dt],var_bc)
    
    # Current trajectory
    State0 = State.random()
    
    # Perturbation
    dState = State.random()
    dX0 = np.concatenate((dState.getvar(vect=True),dState.getparams(vect=True)))
    
    # Adjoint
    adState = State.random()
    adX0 = np.concatenate((adState.getvar(vect=True),adState.getparams(vect=True)))
    
    # Run TLM
    M.step_tgl(t=t0,dState=dState,State=State0,nstep=nstep)
    dX1 = np.concatenate((dState.getvar(vect=True),dState.getparams(vect=True)))
    
    # Run ADJ
    M.step_adj(t=t0,adState=adState,State=State0,nstep=nstep)
    adX1 = np.concatenate((adState.getvar(vect=True),adState.getparams(vect=True)))
    
    mask = np.isnan(adX0+dX0)
    
    ps1 = np.inner(dX1[~mask],adX0[~mask])
    ps2 = np.inner(dX0[~mask],adX1[~mask]) 
    
    print(ps1/ps2)
