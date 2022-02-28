import jax.numpy as jnp
from jax import jit
from jax.config import config
config.update("jax_enable_x64", True)
import optim_3D

def update_inner_params(num, num_obs, n_agents, a_obs, b_obs, lw, bw, hw, n_d, rho_obs, x, y, z, x_obs, y_obs, z_obs, A_obs, lamda_x, lamda_y, lamda_z, ext_rad, ext_rad_sq):
    #################### Obstacle
        wc_alpha = (jnp.expand_dims(x,1)-x_obs)
        ws_alpha = (jnp.expand_dims(y,1)-y_obs)
        alpha_obs = jnp.arctan2( ws_alpha, wc_alpha)

        wc_beta = (jnp.expand_dims(z,1)-z_obs)
        ws_beta = wc_alpha/jnp.cos(alpha_obs)
        beta_obs = jnp.arctan2( ws_beta, wc_beta)
       
        c1_d_dyn = 1.0*rho_obs*ext_rad_sq*(jnp.cos(alpha_obs)**2*jnp.sin(beta_obs)**2 + jnp.sin(alpha_obs)**2*jnp.sin(beta_obs)**2 + jnp.cos(beta_obs)**2)
        # c1_d_static = 1.0*rho_obs*lw**2
        
        c2_d_dyn = 1.0*ext_rad*rho_obs*(wc_alpha*jnp.cos(alpha_obs)*jnp.sin(beta_obs) + ws_alpha*jnp.sin(alpha_obs)*jnp.sin(beta_obs) + wc_beta*jnp.cos(beta_obs))
        # d_temp_1 = c2_d_dyn[:int(n_agents),:(n_agents-1),:num]/c1_d_dyn[:int(n_agents),:(n_agents-1),:num]
        # d_temp_2 = c2_d_dyn[:int(n_agents),(n_agents-1):(n_agents-1+n_d),:num]/c1_d_dyn[:int(n_agents),(n_agents-1):(n_agents-1+n_d),:num]
        
        # d_temp = jnp.hstack((d_temp_1,d_temp_2))
        d_temp = c2_d_dyn/c1_d_dyn
        d_obs_temp = jnp.maximum(jnp.ones((n_agents, num_obs+n_d,  num)), d_temp)
        d_obs = d_obs_temp

        # print (d_obs.shape)

        res_x_obs_vec = wc_alpha-ext_rad*d_obs*jnp.cos(alpha_obs)*jnp.sin(beta_obs)
        res_y_obs_vec = ws_alpha-ext_rad*d_obs*jnp.sin(alpha_obs)*jnp.sin(beta_obs)
        res_z_obs_vec = wc_beta-ext_rad*d_obs*jnp.cos(beta_obs)

        # print (res_x_obs_vec.shape)

        lamda_x = lamda_x-rho_obs*jnp.dot(A_obs.T, res_x_obs_vec.reshape((n_agents,num*(num_obs+n_d))).T).T
        lamda_y = lamda_y-rho_obs*jnp.dot(A_obs.T, res_y_obs_vec.reshape((n_agents,num*(num_obs+n_d))).T).T
        lamda_z = lamda_z-rho_obs*jnp.dot(A_obs.T, res_z_obs_vec.reshape((n_agents,num*(num_obs+n_d))).T).T
    
        
        # print (res_obs.shape)

        return alpha_obs, beta_obs, d_obs,lamda_x,lamda_y, lamda_z, res_x_obs_vec, res_y_obs_vec, res_z_obs_vec, rho_obs

def wrapper_circle(optim_jit, update_inner_jit, num_obs, n_d, maxiter_circle, num, nvar, n_agents, cost_smoothness, x_obs, y_obs, z_obs, a_obs, alpha_obs, beta_obs, b_obs, rho_obs, d_obs, lamda_x, lamda_y, lamda_z, x_iter, y_iter, z_iter, A_eq, b_x_eq, b_y_eq, b_z_eq, P, Pdot, Pddot, A_obs, lw, bw, hw, ext_rad, ext_rad_sq):
    
    
    # d_min = ones(maxiter_circle)
    # res_obs = ones(maxiter_circle)
        
    for i in range(0, maxiter_circle):
        x,y,z = optim_jit(num_obs, num, nvar, n_agents, n_d, cost_smoothness, x_obs, y_obs, z_obs, a_obs, alpha_obs, beta_obs, b_obs, rho_obs, d_obs, lamda_x, lamda_y, lamda_z, A_eq, b_x_eq, b_y_eq, b_z_eq, P, Pdot, Pddot,A_obs,lw,bw,hw,ext_rad)
        alpha_obs,beta_obs,d_obs,lamda_x,lamda_y, lamda_z, res_obs_x, res_obs_y, res_obs_z,rho_obs = update_inner_jit(num, num_obs, n_agents, a_obs, b_obs, lw, bw, hw, n_d, rho_obs, x, y, z, x_obs, y_obs, z_obs, A_obs, lamda_x, lamda_y, lamda_z, ext_rad, ext_rad_sq)
        # rho_obs = 100 if rho_obs*1.05 > 100 else rho_obs*1.05

        
    # print (res_obs[i])        
    x_iter = x
    y_iter = y
    z_iter = z
    max_x  = jnp.sum(jnp.linalg.norm(res_obs_x,axis=2),axis=1)
    max_y  = jnp.sum(jnp.linalg.norm(res_obs_y,axis=2),axis=1)
    max_z  = jnp.sum(jnp.linalg.norm(res_obs_z,axis=2),axis=1)
    # print (max_x.shape)

    return max_x, max_y, max_z, x_iter, y_iter, z_iter, lamda_x, lamda_y, lamda_z

