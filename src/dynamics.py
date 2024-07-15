import numpy as np

def integrate_dynamic_step(theta_vel_0, theta_0, theta_acc,
                           com_vel_0, com_0, com_acc,
                           dt):
        
        theta_vel = theta_vel_0 + theta_acc * dt
        theta = theta_0 + theta_vel * dt

        com_vel = com_vel_0 + com_acc * dt
        com = com_0 + com_vel * dt        

        return theta_vel, theta, com_vel, com

def dynamic_step(F1_mod: float, F2_mod:float, F3_mod: float, 
                 alpha1: float, alpha2: float, alpha3: float,
                 mass: float, width: float, height: float,
                 r1: tuple, r2: tuple, r3: tuple,
                 theta_vel_0: float, theta_0: float,
                 com_vel_0: np.array, com_0: np.array,
                 dt: float):
        
        F1 = F1_mod * np.array([np.cos(alpha1 * np.pi/180),
                                np.sin(alpha1 * np.pi/180)])
        
        F2 = F2_mod * np.array([np.cos(alpha2 * np.pi/180),
                                np.sin(alpha2 * np.pi/180)])
        
        F3 = F3_mod * np.array([np.cos(alpha3 * np.pi/180),
                                np.sin(alpha3 * np.pi/180)])
        
        #F_grav = np.array([0, -9.81]) * mass/ distanza_terra^2 + np.array([0, -9.81]) * mass/ distanza_luna^2

        F_net = F1 + F2 + F3

        tau_net = np.sum([r1[0]*F1[1] - r1[1]*F1[0],
                          r2[0]*F2[1] - r2[1]*F2[0],
                          r3[0]*F3[1] - r3[1]*F3[0]])

        I = mass * (width**2 + height**2) / 12
        
        theta_acc = tau_net / I
        com_acc = F_net / mass

        com_acc_projected =  com_acc * np.array([np.cos(theta_0 * np.pi/180),
                                                 np.sin(theta_0 * np.pi/180)])

        theta_vel, theta, com_vel, com = integrate_dynamic_step(theta_vel_0, theta_0, theta_acc,
                                                                com_vel_0, com_0, com_acc_projected,
                                                                dt)

        return theta_vel, theta, com_vel, com