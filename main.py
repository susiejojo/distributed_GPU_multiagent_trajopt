from importlib.machinery import OPTIMIZED_BYTECODE_SUFFIXES
from numpy.lib.npyio import save
import jax.numpy as jnp
import jax
from jax import jit, vmap
from jax.config import config
config.update("jax_enable_x64", True)

import bernstein_coeff_order10_arbitinterval
import wrapper_circle_initialization_3D
from scipy.io import savemat
import json, time
from tqdm import tqdm
import optim_3D
import matplotlib.pyplot as plt 

cmaps = {
            4: ["red","blue","green","purple"],
            8: ["red","pink","orange","green","grey","navy","yellow","purple"],
            14:["red","pink","orange","green","grey","navy","yellow","purple","darkgreen","teal","salmon","black","cyan","brown"],
            15:["red","pink","orange","green","grey","navy","yellow","purple","darkgreen","teal","salmon","black","cyan","brown","coral"],
            16:["red","pink","orange","green","grey","navy","yellow","purple","darkgreen","teal","salmon","black","cyan","brown","coral","lime"],
            18:["red","pink","orange","green","grey","navy","yellow","purple","darkgreen","teal","salmon","black","cyan","brown","coral","lime","khaki","deeppink"],
            30:["red","pink","orange","green","grey","navy","yellow","purple","darkgreen","teal","salmon","black","cyan","brown","coral","lime",
                "khaki","deeppink","darkorange","crimson","darkgrey","tomato","plum","hotpink","limegreen","peru","olive","wheat","blue","orchid"],
            32:["red","pink","orange","green","grey","navy","yellow","purple","darkgreen","teal","salmon","black","cyan","brown","coral","lime",
                "khaki","deeppink","darkorange","crimson","darkgrey","tomato","plum","hotpink","limegreen","peru","olive","wheat","blue","orchid","gold","limegreen"],
            64:["red","pink","orange","green","grey","navy","yellow","purple","darkgreen","teal","salmon","black","cyan","brown","coral","lime",
                "khaki","deeppink","darkorange","crimson","darkgrey","tomato","plum","hotpink","limegreen","peru","olive","wheat","blue","orchid","gold","limegreen",
                "darkslategrey","yellowgreen","tomato","peachpuff","lightgreen","darkviolet","powderblue","hotpink","magenta","thistle","crimson","lightpink","plum",
                "darkblue","slateblue","royalblue","slategrey","forestgreen","turquoise","dodgerblue","sandybrown","sienna","coral","darkred","firebrick","rosybrown","darkorange",
                "tan","darkkhaki","olivedrab","palegreen","lightgreen"]
        }

def plot_cuboid(center, size, ax,color):
    u = jnp.linspace(0, 2 * jnp.pi, 100)
    v = jnp.linspace(0, jnp.pi, 100)
    r = size[0]

    x = center[0]+r * jnp.outer(jnp.cos(u), jnp.sin(v))
    y = center[1]+r * jnp.outer(jnp.sin(u), jnp.sin(v))
    z = center[2]+r * jnp.outer(jnp.ones(jnp.size(u)), jnp.cos(v))
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=color, linewidth=0, alpha=1)

def apply_mask(a,remove_idx):
    return jnp.take(a,remove_idx,axis=0)

def delete_per_row_3D(n_agents, a_iter,a_obs):
    a_new = jnp.array(a_iter)
    a_it = jnp.vstack((a_new,a_obs))
    a = jnp.expand_dims(a_it[1:,:],0)
    for i in range(1,n_agents):
        obs_mat = jnp.vstack((a_new[:i,:],a_new[i+1:,:]))
        obs_mat = jnp.vstack((obs_mat,a_obs))
        obs_mat = jnp.expand_dims(obs_mat,0)
        a = jnp.vstack((a,obs_mat))
    return a

def update_params(rho_obs, n_agents, num_obs, n_d, ext_rad, ext_rad_sq, x_guess_expand, y_guess_expand, z_guess_expand, x_obs, y_obs, z_obs):
    

    wc_alpha = (x_guess_expand-x_obs)
    ws_alpha = (y_guess_expand-y_obs)
    alpha_obs = jnp.arctan2( ws_alpha, wc_alpha)

    wc_beta = (z_guess_expand-z_obs)
    ws_beta = wc_alpha/jnp.cos(alpha_obs)
    beta_obs = jnp.arctan2( ws_beta, wc_beta)
    
    c1_d_dyn = 1.0*rho_obs*ext_rad_sq*(jnp.cos(alpha_obs)**2*jnp.sin(beta_obs)**2 + jnp.sin(alpha_obs)**2*jnp.sin(beta_obs)**2 + jnp.cos(beta_obs)**2)
    
    c2_d_dyn = 1.0*ext_rad*rho_obs*(wc_alpha*jnp.cos(alpha_obs)*jnp.sin(beta_obs) + ws_alpha*jnp.sin(alpha_obs)*jnp.sin(beta_obs) + wc_beta*jnp.cos(beta_obs))
    d_temp = c2_d_dyn/c1_d_dyn
    d_obs_temp = jnp.maximum(jnp.ones((n_agents, num_obs+n_d,  num)), d_temp)
    d_obs = d_obs_temp

    return wc_alpha,ws_alpha,alpha_obs,wc_beta,ws_beta,beta_obs,c1_d_dyn,c2_d_dyn,d_obs

def generate_trajectories(config_name,robot_config,obst_config,obst_rad, weight_smoothness, rho_obs, sim=False, save_path = False, save_residuals = False, show_plots = False):
    robot_config_file = open(robot_config)
    data = json.loads(robot_config_file.read())
    agents = data["agents"]
    n_agents = len(agents)
    agent_rad = data["agents"][0]["radius"]
    x_all = []
    y_all = []
    z_all = []

    rob_coords = [[agents[i]["start"][0],agents[i]["start"][1],agents[i]["start"][2]] for i in range(n_agents)]
    goals = [[agents[i]["goal"][0],agents[i]["goal"][1],agents[i]["goal"][2]] for i in range(n_agents)]
    for i in range(n_agents):
        x_agent = jnp.linspace(rob_coords[i][0],goals[i][0],num)
        y_agent = jnp.linspace(rob_coords[i][1],goals[i][1],num)
        z_agent = jnp.linspace(rob_coords[i][2],goals[i][2],num)
        x_all.append(jnp.array(x_agent))
        y_all.append(jnp.array(y_agent))
        z_all.append(jnp.array(z_agent))
    x_init = [agents[i]["start"][0] for i in range(n_agents)]
    y_init = [agents[i]["start"][1] for i in range(n_agents)]
    z_init = [agents[i]["start"][2] for i in range(n_agents)]

    ##############final positions#######
    x_fin = [agents[i]["goal"][0] for i in range(n_agents)]
    y_fin = [agents[i]["goal"][1] for i in range(n_agents)]
    z_fin = [agents[i]["goal"][2] for i in range(n_agents)]

    vx_init = [agents[i]["v_init"][0] for i in range(n_agents)]
    vy_init = [agents[i]["v_init"][1] for i in range(n_agents)]
    vz_init = [agents[i]["v_init"][2] for i in range(n_agents)]

    ax_init = [agents[i]["a_init"][0] for i in range(n_agents)]
    ay_init = [agents[i]["a_init"][1] for i in range(n_agents)]
    az_init = [agents[i]["a_init"][2] for i in range(n_agents)]

    vx_fin = [agents[i]["v_fin"][0] for i in range(n_agents)]
    vy_fin = [agents[i]["v_fin"][1] for i in range(n_agents)]
    vz_fin = [agents[i]["v_fin"][2] for i in range(n_agents)]

    ax_fin = [agents[i]["a_fin"][0] for i in range(n_agents)]
    ay_fin = [agents[i]["a_fin"][1] for i in range(n_agents)]
    az_fin = [agents[i]["a_fin"][2] for i in range(n_agents)]

    ###########################static obstacles########################
    x_init_obs = obst_config["x"]
    y_init_obs = obst_config["y"]
    z_init_obs = obst_config["z"]
    

    n_d = x_init_obs.shape[0]

    x_obs_static = x_init_obs[0]*jnp.ones(num)
    y_obs_static = y_init_obs[0]*jnp.ones(num)
    z_obs_static = z_init_obs[0]*jnp.ones(num)
    for i in range(1,n_d):
        x_obs_static = jnp.vstack((x_obs_static,x_init_obs[i]*jnp.ones(num)))
        y_obs_static = jnp.vstack((y_obs_static,y_init_obs[i]*jnp.ones(num)))
        z_obs_static = jnp.vstack((z_obs_static,z_init_obs[i]*jnp.ones(num)))


    lw = 2*obst_rad #diameter of obstacle
    bw = 2*obst_rad
    hw = 2*obst_rad
    a_static = lw

    start = 0
    cost_smoothness = weight_smoothness*jnp.dot(Pddot.T, Pddot)

    A_eq = jnp.vstack(( P[0], Pdot[0], Pddot[0], P[-1], Pdot[-1], Pddot[-1])) 

    b_x_eq = jnp.vstack(( x_init, vx_init, ax_init, x_fin, vx_fin, ax_fin)).T
    b_y_eq = jnp.vstack(( y_init, vy_init, ay_init, y_fin, vy_fin, ay_fin)).T
    b_z_eq = jnp.vstack(( z_init, vz_init, az_init, z_fin, vz_fin, az_fin)).T

    lamda_x = jnp.ones((n_agents,nvar))
    lamda_y = jnp.ones((n_agents,nvar))
    lamda_z = jnp.ones((n_agents,nvar))

    num_obs = (n_agents-1)

    a_obs = agent_rad*2
    a_obs += 0.04 #padding

    A_obs = jnp.tile(P, (num_obs+n_d, 1))
    cost = cost_smoothness+rho_obs*jnp.dot(A_obs.T, A_obs) 
    cost_mat = jnp.vstack((jnp.hstack(( cost, A_eq.T )), jnp.hstack(( A_eq, jnp.zeros(( jnp.shape(A_eq)[0], jnp.shape(A_eq)[0] )) )) ))
    cost_mat_inv = jnp.linalg.inv(cost_mat)

    parta = a_obs*jnp.ones(((n_agents-1),num))
    partb = lw*jnp.ones((n_d,num))
    each_agent = jnp.vstack((parta,partb))
    ext_rad = jnp.repeat(each_agent[jnp.newaxis, :, :], n_agents, axis=0)
    parta_sq = a_obs**2*jnp.ones(((n_agents-1),num))
    partb_sq = lw**2*jnp.ones((n_d,num))
    each_agent_sq = jnp.vstack((parta_sq,partb_sq))
    ext_rad_sq = jnp.repeat(each_agent_sq[jnp.newaxis, :, :], n_agents, axis=0)

    remove_idx = jnp.arange(1,n_agents)

    for i in range(1,n_agents):
        remove_row = jnp.arange(n_agents)
        remove_row = remove_row[remove_row!=i]
        remove_idx = jnp.vstack((remove_idx,remove_row))

    optim_jit = jit(optim_3D.compute_circle_traj, static_argnums=([0,1,2,3,4]))
    filter_jit = jit(apply_mask)
    delete_jit = jit(delete_per_row_3D,static_argnums=([0]))
    update_jit = jit(update_params, static_argnums=([0,1,2,3]))
    update_inner_jit = jit(wrapper_circle_initialization_3D.update_inner_params, static_argnums=([0,1,2,3,4,5,6,7,8,9]))

    x_others = delete_jit(n_agents, x_all,x_obs_static)
    y_others = delete_jit(n_agents, y_all,y_obs_static)
    z_others = delete_jit(n_agents, z_all,z_obs_static)
    alpha_start,beta_start,d_start,_,_, _, res_x_start, res_y_start, res_z_start,_ = update_inner_jit(num, num_obs, n_agents, a_obs, a_obs, lw, bw, hw, n_d, rho_obs, jnp.array(x_all), jnp.array(y_all), jnp.array(z_all), x_others, y_others, z_others, A_obs, lamda_x, lamda_y, lamda_z, ext_rad, ext_rad_sq)

    res_x_start  = jnp.sum(jnp.linalg.norm(res_x_start,axis=2),axis=1)
    res_y_start  = jnp.sum(jnp.linalg.norm(res_y_start,axis=2),axis=1)
    res_z_start  = jnp.sum(jnp.linalg.norm(res_z_start,axis=2),axis=1)

    residuals_x = [[] for i in range(max_outer_iter+1)]
    residuals_y = [[] for i in range(max_outer_iter+1)]
    residuals_z = [[] for i in range(max_outer_iter+1)]
    residuals_x[0].append(jnp.mean(res_x_start))
    residuals_y[0].append(jnp.mean(res_y_start))
    residuals_z[0].append(jnp.mean(res_z_start))


    residuals_std_x = [[] for i in range(max_outer_iter+1)]
    residuals_std_y = [[] for i in range(max_outer_iter+1)]
    residuals_std_z = [[] for i in range(max_outer_iter+1)]

    residuals_std_x[0].append(jnp.std(res_x_start))
    residuals_std_y[0].append(jnp.std(res_y_start))
    residuals_std_z[0].append(jnp.std(res_z_start))

    for iter in tqdm(range(1,max_outer_iter+1)):

        if (iter==4): start = time.time()

        x_iter = x_all
        y_iter = y_all
        z_iter = z_all

        ###################################### Guess Trajectory

        x_guess = jnp.array(x_iter)
        y_guess = jnp.array(y_iter)
        z_guess = jnp.array(z_iter)

        ###################################### Obstacle Information

        #Get the guess/updated trajectories for obstacles other than current agent

        x_obs = delete_jit(n_agents, x_iter,x_obs_static)
        y_obs = delete_jit(n_agents, y_iter,y_obs_static)
        z_obs = delete_jit(n_agents, z_iter,z_obs_static)

        x_guess_expand = jnp.expand_dims(x_guess,1)
        y_guess_expand = jnp.expand_dims(y_guess,1)
        z_guess_expand = jnp.expand_dims(z_guess,1)

        # print (ext_rad)

        wc_alpha,ws_alpha,alpha_obs,wc_beta,ws_beta,beta_obs,c1_d,c2_d,d_obs = update_jit(rho_obs, n_agents, num_obs, n_d, ext_rad, ext_rad_sq, x_guess_expand,y_guess_expand,z_guess_expand, x_obs,y_obs,z_obs)
        
        ###################################### Trajectory with circle approximation

        res_x, res_y, res_z, x_iter, y_iter, z_iter, lamda_x,lamda_y,lamda_z = wrapper_circle_initialization_3D.wrapper_circle(optim_jit, update_inner_jit, num_obs, n_d, maxiter_circle, num, nvar, n_agents, cost_mat_inv,x_obs, y_obs, z_obs, a_obs, alpha_obs, beta_obs, a_obs, rho_obs, d_obs, lamda_x, lamda_y, lamda_z, x_iter, y_iter, z_iter, A_eq, b_x_eq, b_y_eq, b_z_eq, P, Pdot, Pddot, A_obs,lw,bw,hw, ext_rad, ext_rad_sq)
        residuals_x[iter].append(jnp.mean(res_x))
        residuals_y[iter].append(jnp.mean(res_y))
        residuals_z[iter].append(jnp.mean(res_z))

        residuals_std_x[iter].append(jnp.std(res_x))
        residuals_std_y[iter].append(jnp.std(res_y))
        residuals_std_z[iter].append(jnp.std(res_z))

        x_all = x_iter
        y_all = y_iter
        z_all = z_iter

    print ("Time elapsed for planning: ",time.time()-start)
    x_all = jnp.array(x_all)
    y_all = jnp.array(y_all)
    z_all = jnp.array(z_all)

    dist_plot = []
    dist_obst = []
    coll_violate = []
    for i in range(n_agents):
        x_1 = x_all[i,:]
        y_1 = y_all[i,:]
        z_1 = z_all[i,:]
        for j in range(n_agents):
            if (i!=j):
                x_2 = x_all[j,:]
                y_2 = y_all[j,:]
                z_2 = z_all[j,:]
                dist = jnp.square((x_2-x_1))+jnp.square((y_2-y_1)+jnp.square((z_2-z_1)))
                dist_plot.append(jnp.sqrt(dist))
                coll_violate.append(sum(jnp.sqrt(dist)<agent_rad*2))
        
    print (sum(coll_violate)//2 ,"violations among agents out of",(n_agents*(n_agents-1))//2*num)
    dist_plot = jnp.array(dist_plot)
    dist_plot = jnp.min(dist_plot,axis=0)
    coll_violate = []

    for i in range(n_agents):
        x_1 = x_all[i,:]
        y_1 = y_all[i,:]
        z_1 = z_all[i,:]
        for j in range(n_d):
            x_2 = x_obs_static[j,:]
            y_2 = y_obs_static[j,:]
            z_2 = z_obs_static[j,:]
            dist = jnp.square((x_2-x_1))+jnp.square((y_2-y_1)+jnp.square((z_2-z_1)))
            dist_obst.append(jnp.sqrt(dist))
            coll_violate.append(sum(jnp.sqrt(dist)<(agent_rad+lw/2)))

    print (sum(coll_violate) ,"violations with obstacles")
    dist_obst = jnp.array(dist_obst)
    dist_obst = jnp.min(dist_obst,axis=0)

    if (sim):
        for t in range(num):
            figure = plt.figure(figsize=plt.figaspect(1))
            ax = figure.add_subplot(111, projection='3d')
            ax.set_aspect('auto')

            for i in range(n_agents):
                center = jnp.hstack((x_all[i,t],y_all[i,t],z_all[i,t]))
                plot_cuboid(center,(agent_rad, agent_rad, agent_rad), ax, cmaps[n_agents][i])
                ax.plot(x_all[i,:t],y_all[i,:t],z_all[i,:t],cmaps[n_agents][i],linewidth = 0.2)


            for i in range(n_d):
                center = jnp.hstack((x_obs_static[i,0],y_obs_static[i,0],z_obs_static[i,0]))
                plot_cuboid(center,(lw/2, bw/2, hw/2), ax, 'b')

            ax.set_zlim3d(-16,16)
            ax.set_xlabel('x[m]', fontweight ='bold')  
            ax.set_ylabel('y[m]', fontweight ='bold')  
            ax.set_zlabel('z[m]', fontweight ='bold') 
            filename = "16_agents_"+str(t)+".jpg"
            plt.savefig("data/"+filename,dpi=200)
            plt.cla()
            plt.close(figure)

    if (save_path):
        savemat(paths+config_name+'_x_dist.mat', {'x': x_all})
        savemat(paths+config_name+'_y_dist.mat', {'y': y_all})
        savemat(paths+config_name+'_z_dist.mat', {'z': z_all})  

    epoch = list(range(num+1))

    if (show_plots):
        plt.plot(epoch[1:],jnp.array(dist_plot),"red",linewidth=4)
        plt.plot(epoch[1:],jnp.array(dist_obst),"green",linewidth=4)
        plt.plot(epoch[1:],agent_rad*2*jnp.ones(num),"blue",linewidth=4)
        plt.plot(epoch[1:],(agent_rad+lw/2)*jnp.ones(num),"orange",linewidth=4)
        plt.xlabel("Time steps",fontsize=20,fontweight ='bold')
        plt.ylabel("Distances",fontsize=20,fontweight ='bold')
        plt.legend(["Min distance between robot centres","Min distance between robot and obstacle centres","Sum of robot radii", "Sum of radii of robot and obstacle"],fontsize=14,loc="upper right")
        plt.xticks(fontsize=16,fontweight ='bold')
        plt.yticks(fontsize=18,fontweight ='bold')
        # plt.savefig("distances_"+str(config)+".jpg")
        plt.show()

        fig = plt.figure(figsize=(20,10))
        plt.plot(jnp.array(residuals_x),"red")
        plt.plot(jnp.array(residuals_y),"blue")
        plt.plot(jnp.array(residuals_z),"green")
        plt.legend(["Residual $f_c^x$","Residual $f_c^y$","Residual $f_c^z$"],fontsize=18)
        plt.xlabel("Iterations",fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=18)
        # plt.savefig("residuals_"+str(config)+".jpg")
        plt.show()

    if (save_residuals):
        savemat("residuals/"+config_name+'_x_dist.mat', {'x': jnp.array(residuals_x)})
        savemat("residuals/"+config_name+'_y_dist.mat', {'y': jnp.array(residuals_y)})
        savemat("residuals/"+config_name+'_z_dist.mat', {'z': jnp.array(residuals_z)})  

if __name__=="__main__":
    max_outer_iter = 100
    maxiter_circle = 1

    obst_config_file = "obst_configs.json"
    with open(obst_config_file, 'r') as f:
        obst_config = json.load(f)
    for obstacles in obst_config["configs"]:
        t_fin = 10
        num = 100
        t = t_fin/num	    # dt
        tot_time = jnp.linspace(0.0, t_fin, num)
        tot_time_copy = tot_time.reshape(num,1)
        P, Pdot, Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)
        nvar = jnp.shape(P)[1]
        paths = "our_paths/" #path to save the generated trajectories in

        robot_config = obstacles["robot_config"]
        weight_smoothness = obstacles["weight_smoothness"]
        rho_obs = obstacles["rho_obs"]
        config_name = obstacles["name"]
        print ("Configuration: ",config_name)
        x_init_obs = jnp.hstack(obstacles["x_init_obs"])
        y_init_obs = jnp.hstack(obstacles["y_init_obs"])
        z_init_obs = jnp.hstack(obstacles["z_init_obs"])
        obst_config = {"x":x_init_obs,"y":y_init_obs,'z':z_init_obs}
        robot_rad = obstacles["robot_rad"]
        obst_rad = obstacles["obst_rad"]
        generate_trajectories(config_name,robot_config,obst_config,obst_rad, weight_smoothness, rho_obs, show_plots=True)

