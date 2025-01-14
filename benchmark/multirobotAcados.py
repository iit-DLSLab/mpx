import numpy as np
import casadi as cs
from acados_template import *
from scipy.spatial.transform import Rotation as R
class ocp_formulation:

    def __init__(self,args):

        # OCP parameters
        self.N_ = args['N'] #length of the horizon
        self.dt_ = args['dt'] # delta time between the integration node

        self.nx_ = 0
        self.nu_ = 0

        self.lh = []
        self.uh = []

        # model = pinocchio.buildModelFromUrdf(self.model_path_wb_ )

        # self.cmodel = cpin.Model(model)
        # self.cdata =  self.cmodel.createData()

    def getOptimalProblem(self, dynamic_model="DISCRETE", model_name = "robot", make_model=True,n_robot=1,n_batch=1):

        mass = 24
        print('mass:\n',mass)
        inertia = np.array([[ 2.5719824e-01,  1.3145953e-03, -1.6161108e-02],[ 1.3145991e-03,  1.0406910e+00,  1.1957530e-04],[-1.6161105e-02,  1.1957530e-04,  1.0870107e+00]])
        print('inertia',inertia)

        inertia_inv = np.linalg.inv(inertia)
        print("Creating optimal control problem for the robot model: ", model_name)
        # if(dynamic_model == "CONTINUOUS"):
        #     print("changing to discrete dynamics model with semi-implicit euler, \n TO DO continuos")
            # dynamic_model = "DISCRETE"

        ocp = AcadosOcp()
        ocp.model.name = model_name

        n_contact = 4

        # Define the optimization variables depending on the dynamic model

                                            ##========  STATE ======== ##
        p =  cs.SX.sym("position",3*n_robot,1) #trunk positon
        dp = cs.SX.sym("linear_speed",3*n_robot,1) #trunk speed

        rpy = cs.SX.sym("orientation",3*n_robot,1) # psi in R^3 agular variation with the nominal
        omega = cs.SX.sym("angular_speed",3*n_robot,1) #angular speed


        grf = cs.SX.sym("ground_reaction_forces",3*n_contact*n_robot,1) #Ground Reaction Forces

        foot_ref = cs.SX.sym("foot_position_reference",3*(n_contact)*n_robot,1) #position of the leg
        contact_state = cs.SX.sym("contact_state",n_contact*n_robot,1) #contact state 1 in contact 0 if not

        dt = cs.SX.sym("dt",1,1) #Ground discretization time

        state = cs.vertcat(p,rpy,dp,omega)
        nx = 12*n_robot
        self.nx_ = nx

        control = cs.vertcat(grf)
        nu =  3*n_contact*n_robot
        # control = cs.vertcat(tau,grf)
        # nu =  n_joints + 3*n_contact
        self.nu_ = nu

        parameter = cs.vertcat(contact_state,foot_ref,dt)

        ocp.model.x = state
        ocp.model.u = control
        ocp.model.p = parameter
        ocp.parameter_values = np.zeros((n_contact*n_robot+3*n_contact*n_robot+1,1))

                                            ##========  DYNAMICS ======== ##

        ##========SYSTEM WHOLE BODY DYNAMICS DISCRETE========
        dp_next = cs.SX.zeros(3*n_robot,1)
        omega_next = cs.SX.zeros(3*n_robot,1)
        p_new = cs.SX.zeros(3*n_robot,1)
        rpy_new = cs.SX.zeros(3*n_robot,1)
        for i in range(n_robot):
            dp_next[i*3:3+i*3] = dp[i*3:3+i*3] + (np.array([0, 0, -9.81]) + (1 / mass) * (grf[12*i:3+12*i]*contact_state[4*i] + grf[3+12*i:6+12*i]*contact_state[4*i+1] + grf[6+12*i:9+12*i]*contact_state[2+3*i] + grf[9+12*i:12+12*i]*contact_state[3+3*i])) * dt
            p0 = foot_ref[12*i:3+12*i]
            p1 = foot_ref[3+12*i:6+12*i]
            p2 = foot_ref[6+12*i:9+12*i]
            p3 = foot_ref[9+12*i:12+12*i]
            omega_next[i*3:3+i*3] = omega[i*3:3+i*3] + inertia_inv@((cs.cross(p0 - p[i*3:3+i*3], grf[12*i:3+12*i])*contact_state[4*i] + cs.cross(p1 - p[i*3:3+i*3], grf[3+12*i:6+12*i])*contact_state[1+4*i] + cs.cross(p2 - p[i*3:3+i*3], grf[6+12*i:9+12*i])*contact_state[2+4*i] + cs.cross(p3 - p[i*3:3+i*3], grf[9+12*i:12+12*i])*contact_state[3+4*i]))*dt

            ##========INTERGRATOR========
            #semi-implicit euler
            p_new[i*3:3+i*3] = p[i*3:3+i*3] + dp_next[i*3:3+i*3] * dt
            conj_euler_rates = cs.SX.zeros(3,3)
            conj_euler_rates[0,0] = 1
            conj_euler_rates[0,2] = -cs.sin(rpy[1+i*3])
            conj_euler_rates[1,1] = cs.cos(rpy[i*3])
            conj_euler_rates[1,2] = cs.cos(rpy[1+i*3]) * cs.sin(rpy[i*3])
            conj_euler_rates[2,1] = -cs.sin(rpy[i*3])
            conj_euler_rates[2,2] = cs.cos(rpy[1+i*3]) * cs.cos(rpy[i*3])
            inv_conj_euler_rates = cs.inv(conj_euler_rates)
            rpy_new[i*3:3+i*3] = rpy[i*3:3+i*3] + inv_conj_euler_rates@omega[i*3:3+i*3] * dt
            # p_legs_new = p_legs# + dp_leg * dt

        x_next = cs.vertcat(p_new, rpy_new, dp_next, omega_next)

        ocp.model.disc_dyn_expr = x_next

                                            ##========CONSTRAINTS========#
        # ## == FRICTION CONE ===
        mu = 0.5
        Fmin = 0
        Fmax = 500
        expr_fc = cs.SX.ones(5*n_contact*n_robot,1)
        uh_fc = np.zeros(5*n_contact*n_robot)
        lh_fc = np.zeros(5*n_contact*n_robot)
        kk = 0
        for i in range(n_robot):
            for idx in range(n_contact):
                    expr_fc[5*kk : 5 + 5*kk] = cs.vertcat(mu*grf[3*idx+2+12*i]+grf[3*idx+12*i],
                                                         mu*grf[3*idx+2+12*i]-grf[3*idx],
                                                         mu*grf[3*idx+2+12*i]+grf[3*idx+1+12*i],
                                                         mu*grf[3*idx+2+12*i]-grf[3*idx+1+12*i],
                                                         grf[3*idx+2+12*i])
                    kk = kk + 1
        uh_fc = np.ones(5*n_contact*n_robot)*Fmax
        lh_fc = np.array([0,0,0,0,Fmin]*n_contact*n_robot)

        ng_fc = 5*n_contact*n_robot
        # ====== TOTAL CONSTRAINTS =====

        ocp.model.con_h_expr = expr_fc
        ocp.constraints.uh = uh_fc
        ocp.constraints.lh = lh_fc
        ng_c = ng_fc

        # # ## === add slack for non liear constraints === ##
        ocp.constraints.lsh = np.zeros(ng_c)             # Lower bounds on slacks corresponding to soft lower bounds for nonlinear constraints
        ocp.constraints.ush = np.zeros(ng_c)             # Lower bounds on slacks corresponding to soft upper bounds for nonlinear constraints
        ocp.constraints.idxsh = np.array(range(ng_c))    # Jsh

        ocp.cost.zl = 1000 * np.ones((ng_c,)) # gradient wrt lower slack at intermediate shooting nodes (1 to N-1)
        ocp.cost.Zl = 1 * np.ones((ng_c,))    # diagonal of Hessian wrt lower slack at intermediate shooting nodes (1 to N-1)
        ocp.cost.zu = 1000 * np.ones((ng_c,))
        ocp.cost.Zu = 1 * np.ones((ng_c,))

        #                                     ##========OBJECTIVE FUNCTION========##

        cost_type = 'NONLINEAR_LS'
        # cost_type = 'LINEAR_LS'

        ocp.cost.cost_type = cost_type
        ocp.cost.cost_type_e = cost_type


        ## running cost (for state and control)

        ocp.model.cost_y_expr = cs.vertcat(p,rpy,dp,omega,grf)

        ## final cost (only state)
        ocp.model.cost_y_expr_e = cs.vertcat(p,rpy,dp,omega)

        #COST
        Qp =  np.tile(np.array([0, 0, 10000]),n_robot)
        Qdp = np.tile(np.array([1000, 1000, 1000]),n_robot)
        Qomega = np.tile(np.array([100, 100, 10]),n_robot)
        Rgrf = np.tile(np.ones(3 * n_contact) * 1e-3,n_robot)
        Qrpy = np.tile(np.array([500,500,0]),n_robot)


        Q = np.concatenate((Qp,Qrpy,Qdp,Qomega))
        R = Rgrf
        W = np.diag(np.concatenate((Q,R)))

        ocp.cost.W = W
        ocp.cost.W_e = np.diag(Q)

        ocp.cost.yref = np.zeros(cs.SX.size(ocp.model.cost_y_expr,1))
        ocp.cost.yref_e = np.zeros(cs.SX.size(ocp.model.cost_y_expr_e,1))
                                                ## == INITIAL CONDITION == ##
        ocp.constraints.x0 = np.zeros((nx,1))
                                                  ## == SOLVER OPTIONS == ##
        tol = 1e-3                
        shooting_nodes = np.linspace(0,self.N_*self.dt_,self.N_+1)
        print(shooting_nodes)
        tol = 1e-3
        ocp.dims.N = self.N_
        ocp.solver_options.shooting_nodes = shooting_nodes
        # ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES , PARTIAL_CONDENSING_HPIPM
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # GAUSS_NEWTON, EXACT
        ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI or SQP
        ocp.solver_options.nlp_solver_max_iter = 1
        ocp.solver_options.ext_fun_compile_flags = '-O3'
        ocp.solver_options.integrator_type = "DISCRETE"
        # ocp.solver_options.tol = tol
        if n_batch > 1:
            ocp.solver_options.num_threads_in_batch_solve = n_batch
        ocp.solver_options.print_level = 0

        ocp.solver_options.tf = shooting_nodes[self.N_]
        # ocp.translate_to_feasibility_problem(keep_x0=True, keep_cost=True)
        if make_model :
            if n_batch > 1:
                ocp_solver = AcadosOcpBatchSolver(ocp, n_batch, json_file = model_name+'acados_ocp.json')
            else :
                ocp_solver =  AcadosOcpSolver(ocp, json_file = model_name+'acados_ocp.json')

        else :
            if n_batch > 1:
                ocp_solver = AcadosOcpBatchSolver(ocp, n_batch, json_file = model_name+'acados_ocp.json', build = False, generate = True)
            else :
                ocp_solver =  AcadosOcpSolver(ocp, json_file = model_name+'acados_ocp.json', build = False, generate = True)


        return ocp_solver
    
# args = {}
# N = 50
# dt = 0.01
# problems = []

# args['N'] = N # Horizon lenght
# args['dt'] = dt # delta time between the integration node

# srbd_acados = ocp_formulation(args)
# srbd_acados_solver = srbd_acados.getOptimalProblem(model_name = "srbd")