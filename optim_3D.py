import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

def compute_circle_traj(num_obs, num, nvar, n_agents, n_d, cost_inv, x_obs, y_obs, z_obs, a_obs, alpha_obs, beta_obs, b_obs, rho_obs, d_obs, lamda_x, lamda_y, lamda_z, A_eq, b_x_eq, b_y_eq, b_z_eq, P, Pdot, Pddot,A_obs,lw,bw,hw,ext_rad):

    temp_x_obs = d_obs*jnp.cos(alpha_obs)*jnp.sin(beta_obs)*ext_rad
    b_obs_x = x_obs.reshape((n_agents,num*(num_obs+n_d)))+temp_x_obs.reshape((n_agents,num*(num_obs+n_d)))

    # print (dot(A_obs.T, b_obs_x.T).T.shape)
    # print (lamda_x.shape)
    # print (b_x_eq.shape)

    temp_y_obs = d_obs*jnp.sin(alpha_obs)*jnp.sin(beta_obs)*ext_rad
    b_obs_y = y_obs.reshape((n_agents,num*(num_obs+n_d)))+temp_y_obs.reshape((n_agents,num*(num_obs+n_d)))

    temp_z_obs = d_obs*jnp.cos(beta_obs)*ext_rad
    b_obs_z = z_obs.reshape((n_agents,num*(num_obs+n_d)))+temp_z_obs.reshape((n_agents,num*(num_obs+n_d)))

    # lincost_x = -lamda_x-rho_obs*dot(A_obs.T, b_obs_x) + weight_r*dot(x_all[agent_index].T,P)
    # lincost_y = -lamda_y-rho_obs*dot(A_obs.T, b_obs_y) + weight_r*dot(y_all[agent_index].T,P)
    lincost_x = -lamda_x-rho_obs*jnp.dot(A_obs.T, b_obs_x.T).T
    lincost_y = -lamda_y-rho_obs*jnp.dot(A_obs.T, b_obs_y.T).T 
    lincost_z = -lamda_z-rho_obs*jnp.dot(A_obs.T, b_obs_z.T).T 
    
    sol_x = jnp.dot(cost_inv, jnp.hstack(( -lincost_x, b_x_eq )).T) #check

    x = jnp.dot(P, sol_x[0:nvar]).T
    
    sol_y = jnp.dot(cost_inv, jnp.hstack(( -lincost_y, b_y_eq )).T) #check

    y = jnp.dot(P, sol_y[0:nvar]).T

    sol_z = jnp.dot(cost_inv, jnp.hstack(( -lincost_z, b_z_eq )).T) #check

    z = jnp.dot(P, sol_z[0:nvar]).T

    # print (x.shape)

    # print ("Found 1 successfully")


    return x,y,z

    #[
    #block1: [2:,3:,4:,,,,32:.1s,2s]
    #block2: [1:,3:,4:,,,,,32:,.1s,2s]
    #]
