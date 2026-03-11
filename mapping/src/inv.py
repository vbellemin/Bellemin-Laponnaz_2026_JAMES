#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:59:11 2021

@author: leguillou
"""

import sys
import numpy as np 
import os
import matplotlib.pylab as plt
from datetime import datetime,timedelta
import scipy.optimize as opt
import gc
import xarray as xr
import glob
from importlib.machinery import SourceFileLoader 

from . import grid

import time
import subprocess

import jax.numpy as jnp


def Inv(config, State=None, Model=None, dict_obs=None, Obsop=None, Basis=None,X=None, Bc=None, ssh_truth=None, *args, **kwargs):

    """
    NAME
        Inv

    DESCRIPTION
        Main function calling subfunctions for specific Inversion algorithms
    """
    
    if config.INV.super=='INV_FORWARD':
        return Inv_forward(config, State=State, Model=Model,X=X,Basis=Basis, Bc=Bc, Obsop = Obsop,ssh_truth=ssh_truth)

    print(config.INV)
    
    if config.INV.super=='INV_OI':
        return Inv_oi(config, State=State, dict_obs=dict_obs)
    
    elif config.INV.super=='INV_4DVAR':
        return Inv_4Dvar(config, State=State, Model=Model, dict_obs=dict_obs, Obsop=Obsop, Basis=Basis, Bc=Bc)
    
    else:
        sys.exit(config.INV.super + ' not implemented yet')
     
def Inv_forward(config,State,Model,Basis,X,Bc,Obsop,ssh_truth=None):
    
    """
    NAME
        Inv_forward

    DESCRIPTION
        Run a model forward integration  
    
    """    
    
    # Compute checkpoints when the cost function will be evaluated 
    nstep_check = int(config.INV.timestep_checkpoint.total_seconds()//Model.dt)
    checkpoints = [0]
    time_checkpoints = [np.datetime64(Model.timestamps[0])]
    t_checkpoints = [Model.T[0]]
    check = 0
    for i,t in enumerate(Model.timestamps[:-1]):
        if i>0 and (Obsop.is_obs(t) or check==nstep_check):
            checkpoints.append(i)
            time_checkpoints.append(np.datetime64(t))
            t_checkpoints.append(Model.T[i])
            if check==nstep_check:
                check = 0
        check += 1 
    checkpoints.append(len(Model.timestamps)-1) # last timestep
    time_checkpoints.append(np.datetime64(Model.timestamps[-1]))
    t_checkpoints.append(Model.T[-1])
    checkpoints = np.asarray(checkpoints)
    time_checkpoints = np.asarray(time_checkpoints)
    print(f'--> {checkpoints.size} checkpoints to evaluate the cost function')

    # Boundary conditions
    if Bc is not None:
        var_bc = Bc.interp(time_checkpoints)
        Model.set_bc(t_checkpoints,var_bc)

    # Observations operator 
    if config.INV.anomaly_from_bc: # Remove boundary fields if anomaly mode is chosen
        time_obs = [np.datetime64(date) for date in Obsop.date_obs]
        var_bc = Bc.interp(time_obs)
    else:
        var_bc = None
    
    # Initial model state
    Model.init(State)
    State.plot(title='Init State')

    # Set Reduced Basis
    if Basis is not None:
        time_basis = np.arange(0,Model.T[-1]+nstep_check*Model.dt,nstep_check*Model.dt)/24/3600 # Time (in days) for which the basis components will be compute (at each timestep_checkpoint)
        Xb, Q = Basis.set_basis(time_basis,return_q=True) # Q is the standard deviation. To get the variance, use Q^2
    else:
        sys.exit('4Dvar only works with reduced basis!!')
    
    # Covariance matrix
    from .tools_4Dvar import Cov
    if config.INV.sigma_B is not None:     
        print('Warning: sigma_B is prescribed --> ignore Q of the reduced basis')
        # Least squares
        B = Cov(config.INV.sigma_B)
        R = Cov(config.INV.sigma_R)
    else:
        B = Cov(Q)
        R = Cov(config.INV.sigma_R)
    
    ####################################################
    ########     - INITIAL CONTROL VECTOR -     ########
    #################################################### 

    # - Prescribed vector - # 
    if X is not None : 
        print("Prescribed control parameter vector with X will be used to start the integration.")
        Xopt = X
    else : 
        print(f"Null control parameter vector will be used to start the integration.")
        Xopt = np.zeros((Xb.size,))
            
    Xres = +Xopt
    
    if config.INV.prec:
        Xa = Xb + B.sqr(Xres)
    else:
        Xa = Xb + Xres

    # Init
    State0 = State.copy()
    present_date = config.EXP.init_date

    nstep = int(config.EXP.saveoutput_time_step.total_seconds()//Model.dt)

    square_diff = 0 
    i = 1

    while present_date < config.EXP.final_date :
        
        # State0.plot(present_date)

        # current time in secondes
        t = (present_date - config.EXP.init_date).total_seconds()
        
        if t%int(config.INV.timestep_checkpoint.total_seconds())==0:

            # Reduced basis
            Basis.operg(t/3600/24,Xa,State=State0)

        # Save
        if config.EXP.saveoutputs:
            # print(State0.var)
            # print(State0.params)
            Model.save_output(State0,present_date,name_var=Model.var_to_save,t=t)

        # Propagation
        Model.step(State0,nstep,t=t)

        if t%(15*24*3600)==0: # plotting everyn 15 days 
            State0.plot(present_date)

        #####################################################
        # Calculating rmse with truth ssh - TEST for BM dev #
        if ssh_truth is not None : 

            square_diff += np.nansum((State0.var['ssh_bm']-ssh_truth[i,:,:].values)**2)

            # if i == 300 :
            #     plt.figure()
            #     plt.pcolormesh(State0.var['ssh_bm'],vmin=0.65,vmax=1.05)
            #     plt.title("State SSH BM")
            #     plt.colorbar()
            #     plt.show()

            #     plt.figure()
            #     plt.pcolormesh(ssh_truth[i,:,:].values,vmin=0.65,vmax=1.05)
            #     plt.title("SSH truth BM")
            #     plt.colorbar()
            #     plt.show()

            i += 1 
        #####################################################

        # Time increment
        present_date += timedelta(seconds=nstep*Model.dt)
        t += nstep*Model.dt

    #   Last timestep 
    if config.EXP.saveoutputs:
        Model.save_output(State0,present_date,name_var=Model.var_to_save,t=t)
      
    if ssh_truth is not None:
        # Returning the  rmse with truth ssh - TEST for BM dev
        return np.sqrt(square_diff/(i*State0.nx*State0.ny))
    else : 
        return  

    # del State, State0, Xa, B, R
    # gc.collect()

    # return res
         
def Inv_oi(config,State,dict_obs):
    
    """
    NAME
        Inv_oi

    DESCRIPTION
        Run an optimal interpolation experiment 
    
    """
    
    from . import obs
    
    # Initialize variables (normally this step is done in mod.py, but here no Model object is provided)
    for name in config.INV.name_var:
        State.var[config.INV.name_var[name]] = np.zeros((State.ny,State.nx))

    # Boundary box
    box = [State.lon.min(),State.lon.max(),State.lat.min(),State.lat.max(),
           None, None]
    
    # Time parameters
    ndays = (config.EXP.final_date-config.EXP.init_date).total_seconds()/3600/24
    dt = config.EXP.saveoutput_time_step.total_seconds()/3600/24
    times = np.arange(0, ndays + dt, dt)

    # Coordinates
    lon1d = State.lon.flatten()
    lat1d = State.lat.flatten()
    
    # Time loop
    for t in times:

        for name in config.INV.name_var:
            
            # Time boundary
            box[4] = config.EXP.init_date + timedelta(days=t-config.INV.Lt)
            box[5] = config.EXP.init_date + timedelta(days=t+config.INV.Lt)
            
            # Get obs in (time x lon x lat) cube
            obs_val, obs_coords, _ = obs.get_obs(dict_obs, box, config.EXP.init_date, name)
            obs_lon = obs_coords[0]
            obs_lat = obs_coords[1]
            obs_time = obs_coords[2]
            
            # Perform the optimal interpolation 
            BHt = np.exp(-((t - obs_time[np.newaxis,:])/config.INV.Lt)**2 - 
                        ((lon1d[:,np.newaxis] - obs_lon[np.newaxis,:])/config.INV.Lx)**2 - 
                        ((lat1d[:,np.newaxis] - obs_lat[np.newaxis,:])/config.INV.Ly)**2)
            HBHt = np.exp(-((obs_time[np.newaxis,:] - obs_time[:,np.newaxis])/config.INV.Lt)**2 -
                        ((obs_lon[np.newaxis,:] - obs_lon[:,np.newaxis])/config.INV.Lx)**2 -
                        ((obs_lat[np.newaxis,:] - obs_lat[:,np.newaxis])/config.INV.Ly)**2) 
            R = np.diag(np.full((len(obs_val)), config.INV.sigma_R**2))
            Coo = HBHt + R
            Mi = np.linalg.inv(Coo)
            sol = np.dot(np.dot(BHt, Mi), obs_val).reshape((State.ny,State.nx))
            
            # Set estimated variable
            State.setvar(sol,name_var=config.INV.name_var[name])

        # Save estimated fields for date t
        date = config.EXP.init_date + timedelta(days=t)
        State.save_output(date)

    return
    
def Inv_4Dvar(config=None,State=None,Model=None,dict_obs=None,Obsop=None,Basis=None,Bc=None,verbose=True,gpu_device=None) : 
    '''
    Run a 4Dvar analysis
    '''

    #if 'JAX' in config.MOD.super:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    if gpu_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device
        
    
    # Module initializations
    if Model is None:
        # initialize Model operator 
        from . import mod
        Model = mod.Model(config, State, verbose=verbose)
    if Bc is None:
        # initialize Bc 
        from . import bc
        Bc = bc.Bc(config, verbose=verbose)
    if dict_obs is None:
        # initialize Obs
        from . import obs
        dict_obs = obs.Obs(config, State)
    if Obsop is None:
        # initialize Obsop
        from . import obsop
        Obsop = obsop.Obsop(config, State, dict_obs, Model, verbose=verbose)
    if Basis is None:
        # initialize Basis
        from . import basis
        Basis = basis.Basis(config, State, verbose=verbose)
    
    
    # Compute checkpoints when the cost function will be evaluated 
    nstep_check = int(config.INV.timestep_checkpoint.total_seconds()//Model.dt)
    checkpoints = [0]
    time_checkpoints = [np.datetime64(Model.timestamps[0])]
    t_checkpoints = [Model.T[0]]
    check = 0
    for i,t in enumerate(Model.timestamps[:-1]):
        if i>0 and (Obsop.is_obs(t) or check==nstep_check):
            checkpoints.append(i)
            time_checkpoints.append(np.datetime64(t))
            t_checkpoints.append(Model.T[i])
            if check==nstep_check:
                check = 0
        check += 1 
    checkpoints.append(len(Model.timestamps)-1) # last timestep
    time_checkpoints.append(np.datetime64(Model.timestamps[-1]))
    t_checkpoints.append(Model.T[-1])
    checkpoints = np.asarray(checkpoints)
    time_checkpoints = np.asarray(time_checkpoints)
    print(f'--> {checkpoints.size} checkpoints to evaluate the cost function')

    # Boundary conditions
    if Bc is not None:
        var_bc = Bc.interp(time_checkpoints)
        Model.set_bc(t_checkpoints,var_bc)
    
    # Observations operator 
    if config.INV.anomaly_from_bc: # Remove boundary fields if anomaly mode is chosen
        time_obs = [np.datetime64(date) for date in Obsop.date_obs]
        var_bc = Bc.interp(time_obs)
    else:
        var_bc = None
    print('process observation operators')
    Obsop.process_obs(var_bc)

    # ### TEST : CORRECTING THE OBSERVATIONS, FOR IMPLEMENTING SEQUENTIAL MAPPING ### --> very preliminary, only functions for multiple OBSOP 
    # if config.OBSOP.super is None: 
    #     for k,_OBSOP in enumerate(config.OBSOP):
    #         if config.OBSOP[_OBSOP].super == "OBSOP_INTERP_L4" and config.OBSOP[_OBSOP].file_corr is not None:
    #             print("Correcting obs ...")
    #             array_t_obs = np.zeros(Obsop.Obsop[k].t_obs_jax.shape,dtype="datetime64[s]")
    #             for i,dt in enumerate(np.array(Obsop.Obsop[k].t_obs_jax)):
    #                 # print(np.datetime64(config.EXP.init_date)+np.timedelta64(dt,'s'))
    #                 array_t_obs[i]=np.datetime64(config.EXP.init_date)+np.timedelta64(dt,'s')
                
    #             ds_corr = xr.open_mfdataset(config.OBSOP[_OBSOP].file_corr)
    #             # print("File correction field opened")
    #             for _var in config.OBSOP[_OBSOP].name_var_corr:
    #                 field_corr = ds_corr[config.OBSOP[_OBSOP].name_var_corr[_var]].load()
    #                 field_corr = field_corr.interp({"time":array_t_obs})
    #                 Obsop.Obsop[k].varobs -= field_corr.values.reshape(field_corr.shape[0],field_corr.shape[1]*field_corr.shape[2])
                # print("Obs corrected")

                # plt.figure()
                # plt.pcolormesh(Obsop.Obsop[k].varobs[12,:].reshape(161,161))
                # plt.colorbar()
                # plt.show()

    # Initial model state
    Model.init(State)
    State.plot(title='Init State')

    # Set Reduced Basis
    if Basis is not None:
        time_basis = np.arange(0,Model.T[-1]+nstep_check*Model.dt,nstep_check*Model.dt)/24/3600 # Time (in days) for which the basis components will be compute (at each timestep_checkpoint)
        Xb, Q = Basis.set_basis(time_basis, return_q=True) # Q is the standard deviation. To get the variance, use Q^2
    else:
        sys.exit('4Dvar only work with reduced basis!!')
    
    # Covariance matrix
    from .tools_4Dvar import Cov
    if config.INV.sigma_B is not None:     
        print('Warning: sigma_B is prescribed --> ignore Q of the reduced basis')
        # Least squares
        B = Cov(config.INV.sigma_B)
        R = Cov(config.INV.sigma_R)
    else:
        B = Cov(Q)
        R = Cov(config.INV.sigma_R)
        
    # Variational object initialization
    if config.INV.flag_full_jax:
        from .tools_4Dvar import Variational_jax as Variational
    else:
        from .tools_4Dvar import Variational as Variational

    var = Variational(
        config=config, M=Model, H=Obsop, State=State, B=B, R=R, Basis=Basis, Xb=Xb, checkpoints=checkpoints, nstep=nstep_check, freq_it_plot=config.INV.freq_it_plot)
    
    # Initial Control vector 
    if config.INV.path_init_4Dvar is None:
        if config.INV.flag_full_jax:
            Xopt = jnp.zeros((Xb.size,))
        else:
            Xopt = np.zeros((Xb.size,))
    else:
        # Read previous minimum 
        print('Read previous minimum:',config.INV.path_init_4Dvar)
        ds = xr.open_dataset(config.INV.path_init_4Dvar)
        Xopt = var.Xb*0
        Xopt[:ds.res.size] = ds.res.values
        ds.close()
        if config.INV.prec:
            Xopt = B.invsqr(Xopt - var.Xb)
    
    # Path where to save the control vector at each 4Dvar iteration 
    # (carefull, depending on the number of control variables, these files may use large disk space)
    if config.INV.path_save_control_vectors is not None:
        path_save_control_vectors = config.INV.path_save_control_vectors
    else:
        path_save_control_vectors = config.EXP.tmp_DA_path
    if not os.path.exists(path_save_control_vectors):
        os.makedirs(path_save_control_vectors)

    # Restart mode
    maxiter = config.INV.maxiter
    if config.INV.restart_4Dvar:
        tmp_files = sorted(glob.glob(os.path.join(path_save_control_vectors,'X_it*.nc')))
        if len(tmp_files)>0:
            print('Restart at:',tmp_files[-1])
            try:
                ds = xr.open_dataset(tmp_files[-1])
            except:
                if len(tmp_files)>1:
                    ds = xr.open_dataset(tmp_files[-2])
            try:
                Xopt = ds.res.values
                maxiter = max(config.INV.maxiter - len(tmp_files), 0)
                ds.close()
            except:
                Xopt = +Xopt
            
    if not (config.INV.restart_4Dvar and maxiter==0):
        print('\n*** Minimization ***\n')
        ###################
        # Minimization    #
        ###################

        # Main function
        if config.INV.flag_full_jax:
            from jax import jit, value_and_grad
            fun = jit(value_and_grad(var.cost))
        else:
            fun = var.cost_and_grad

        # Callback function called at every minimization iterations
        # def callback(XX):
        #     if config.INV.save_minimization:

        #         now = datetime.now()
        #         current_time = now.strftime("%Y-%m-%d_%H%M%S")
        #         ds = xr.Dataset({'res':(('x',),XX)})
        #         ds.to_netcdf(os.path.join(config.EXP.tmp_DA_path,'X_it-'+current_time+'.nc'))
        #         ds.close()

        #         # ds = xr.Dataset({'res':(('x',),XX)})
        #         # ds.to_netcdf(os.path.join(path_save_control_vectors,'X_it.nc'))
        #         # ds.close()

        # Calback version from Val version # 
        list_J=[]
        iteration = 0
        n_consecutive = config.INV.n_consecutive
        ftol = config.INV.ftol
        
        def callback(intermediate_result):
            nonlocal list_J
            nonlocal iteration
            nonlocal ftol
            nonlocal n_consecutive
            # Iteration +=1
            iteration += 1

            # print(f"\n \n At iterate   {iteration} \n")
            # print(f"Jtot=  {intermediate_result.fun:.5e}")

            # Printing status # 
            # print("success : ",intermediate_result.success)
            # print("status : ",intermediate_result.status)
            # print("message : ",intermediate_result.message)


            # Saving the control parameters

            if config.INV.save_minimization:

                now = datetime.now()
                current_time = now.strftime("%Y-%m-%d_%H%M%S")
                ds = xr.Dataset({'res':(('x',),intermediate_result.x)})
                ds.to_netcdf(os.path.join(config.EXP.tmp_DA_path,'X_it-'+current_time+'.nc'))
                ds.close()
            
            # Saving the 
            list_J.append(intermediate_result.fun)
            if (n_consecutive is not None) and (ftol is not None) and len(list_J)>n_consecutive+1:
                arrayJ_k =  np.array(list_J[-n_consecutive-1:-2])
                arrayJ_k1 =  np.array(list_J[-n_consecutive:-1])
                d_J = np.abs(arrayJ_k1-arrayJ_k)/np.maximum(arrayJ_k1,arrayJ_k)
                if np.all(d_J<ftol):
                    print(f"Criterion on ftol has been satisfied for a consecutive number of {n_consecutive} iterations (n_consecutive specified).")
                    raise StopIteration

        # Minimization options
        options = {}
        if verbose:
            options['disp'] = True
        else:
            options['disp'] = False
        options['maxiter'] = maxiter

        # If ftol stopping criterion should be raised 
        # at the first iteration it happens (n_consecutive is None) 
        if config.INV.ftol is not None and config.INV.n_consecutive is None : 
            options['ftol'] = config.INV.ftol

        # if config.INV.ftol is not None:
        #     options['ftol'] = config.INV.ftol

        if config.INV.gtol is not None:
            _, g0 = fun(Xopt*0.)
            projg0 = np.max(np.abs(g0))
            options['gtol'] = config.INV.gtol*projg0
        
        # Run minimization 
        from decimal import Decimal
        import time
        class Wrapper:
            def __init__(self):
                J0, G0 = fun(Xopt)
                self.cache = {
                    'cost':J0,
                    'grad':G0
                }
                self.J_list = []
                self.G_list = []
                self.time = time.time()
                self.it = 1
                if 'gtol' in options:
                    self.gtol = options['gtol']
                else:
                    self.gtol = None
                self.filename_out = os.path.join(path_save_control_vectors, 'iterations.txt')
                with open(self.filename_out, "w") as f:
                    f.write("Minimization\n")  # Header
                

            def __call__(self, x, *args):
                cost, grad = fun(x)
                ftol = (cost - self.cache['cost']) / max(cost, self.cache['cost'], 1)
                mean_grad = np.mean(np.abs(grad))
                mean_grad_previous = np.mean(np.abs(self.cache['grad']))
                gtol = (mean_grad - mean_grad_previous) / max(mean_grad, mean_grad_previous, 1)
                self.cache['cost'] = cost
                self.cache['grad'] = grad
                time0 = time.time()
                text = "computed in %.2E second:" % (time0 - self.time) + ', x=%.2E' % Decimal(float(x.mean()))  + ', J=%.2E' % Decimal(float(cost)) + ', G=%.2E' % Decimal(float(mean_grad))  + ', ftol=%.2E' % Decimal(abs(float(ftol)))  + ', gtol=%.2E' % Decimal(abs(float(gtol))) 
                print(f"* iteration {self.it}", text)
                with open(self.filename_out, "a") as f:
                    f.write(f"iteration {self.it}, {text}\n")
                self.time = time0
                self.it += 1

                self.J_list.append(float(cost))
                self.G_list.append(float(mean_grad))

                return cost

            def jac(self, x, *args):
                return self.cache['grad']
        
        wrapper = Wrapper()
        print('Xopt:',Xopt.max(),Xopt.min(),Xopt.mean())
        res = opt.minimize(wrapper, Xopt,
                        method=config.INV.opt_method,
                        jac=wrapper.jac,
                        options=options,
                        callback=callback)
        

        print ('\nIs the minimization successful? {}'.format(res.success))
        print ('\nFinal cost function value: {}'.format(res.fun))
        print ('\nNumber of iterations: {}'.format(res.nit))
        print ('\nMessage: {}'.format(res.message))
        print ('\nStatus: {}'.format(res.status))
        
        # Save minimization trajectory
        if config.INV.save_minimization:
            ds = xr.Dataset({'cost':(('i'),np.array(wrapper.J_list)),
                             'grad':(('i'),np.array(wrapper.G_list))
                             })
            ds.to_netcdf(os.path.join(path_save_control_vectors,'minimization_trajectory.nc'))
            ds.close()

        Xres = res.x
    else:
        print('You ask for restart_4Dvar and maxiter==0, so we save directly the trajectory')
        Xres = +Xopt
        
    ########################
    #    Saving trajectory #
    ########################
    print('\n*** Saving trajectory ***\n')
    
    if config.INV.prec:
        Xa = var.Xb + B.sqr(Xres)
    else:
        Xa = var.Xb + Xres
        
    # Save minimum for next experiments
    ds = xr.Dataset({'res':(('x',), Xa)})
    ds.to_netcdf(os.path.join(path_save_control_vectors, 'Xres.nc'))
    ds.close()

    # Init
    State0 = State.copy()
    Model.init(State0)
    date = config.EXP.init_date
    Model.save_output(State0, date, name_var=Model.var_to_save, t=0) 
    
    nstep = min(nstep_check, int(config.EXP.saveoutput_time_step.total_seconds()//Model.dt))
    # Forward propagation
    while date<config.EXP.final_date:
        
        # current time in secondes
        t = (date - config.EXP.init_date).total_seconds()
        
        # Reduced basis
        if t%int(config.INV.timestep_checkpoint.total_seconds())==0:
            Basis.operg(t/3600/24, Xa, State=State0)

        # Forward propagation
        Model.step(t=t, State=State0, nstep=nstep)
        date += timedelta(seconds=nstep*Model.dt)

        # Save output
        if (((date - config.EXP.init_date).total_seconds()
            /config.EXP.saveoutput_time_step.total_seconds())%1 == 0)\
            & (date>=config.EXP.init_date) & (date<=config.EXP.final_date) :
            Model.save_output(State0, date, name_var=Model.var_to_save, t=t) 
        
    del State, State0, Xa, dict_obs, B, R, Model, Basis, var, Xopt, Xres, checkpoints, time_checkpoints, t_checkpoints
    gc.collect()
    print()
