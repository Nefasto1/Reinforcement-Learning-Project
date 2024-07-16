import numpy as np

def integrate_dynamic_step(theta_vel_0, theta_0, theta_acc,
                           com_vel_0, com_0, com_acc,
                           dt):
        
        theta_vel = theta_vel_0 + theta_acc * dt
        theta = theta_0 + theta_vel * dt

        com_vel = com_vel_0 + com_acc * dt
        com = com_0 + com_vel * dt        

        return theta_vel, theta, com_vel, com

def integrate_dynamic_step2(vel_0, x_0, acc,
                           dt):
        
        vel = vel_0 + acc * dt
        x = x_0 + vel * dt

        return vel, x

def dynamic_step(F1_mod: float, F2_mod:float, F3_mod: float, 
                 alpha1: float, alpha2: float, alpha3: float,
                 mass_shuttle: float, width: float, height: float,
                 r1: tuple, r2: tuple, r3: tuple,
                 theta_vel_0: float, theta_0: float,
                 com_vel_0: np.array, com_0: np.array,
                 moon_coords: np.array, earth_coords: np.array,
                 dt: float):
        
        F1 = F1_mod * np.array([np.cos(alpha1 * np.pi/180),
                                np.sin(alpha1 * np.pi/180)])
        
        F2 = F2_mod * np.array([np.cos(alpha2 * np.pi/180),
                                np.sin(alpha2 * np.pi/180)])
        
        F3 = F3_mod * np.array([np.cos(alpha3 * np.pi/180),
                                np.sin(alpha3 * np.pi/180)])
        
        #distance_shuttle_moon = np.sqrt(np.sum(com_0 - moon_coords)**2)
        #distance_shuttle_earth = np.sqrt(np.sum(com_0 - earth_coords)**2)

        #G = 6.674*1e-11 # Gravitational constant
        #unit_vector_shuttle_moon = (com_0 - moon_coords)/distance_shuttle_moon
        #unit_vector_shuttle_earth = (com_0 - earth_coords)/distance_shuttle_moon

        #mass_moon = 200
        #mass_earth = 1000

        #F_grav = G * mass_shuttle * (mass_moon * unit_vector_shuttle_moon / distance_shuttle_moon**2 + mass_earth * unit_vector_shuttle_earth / distance_shuttle_earth**2)

        #print(F_grav)
    

        F_net = F1*0.5 + F2 + F3*0.5 #+ F_grav

        tau_net = np.sum([r1[0]*F1[1] - r1[1]*F1[0],
                          r2[0]*F2[1] - r2[1]*F2[0],
                          r3[0]*F3[1] - r3[1]*F3[0]])

        I = mass_shuttle * (width**2 + height**2) / 12
        
        theta_acc = tau_net /20#/ I
        com_acc = F_net /20#/ mass_shuttle

        theta_vel, theta = integrate_dynamic_step2(theta_vel_0, theta_0, theta_acc,
                                                                dt)
    
        com_acc_projected =  com_acc * np.array([np.cos(theta * np.pi/180),
                                                 np.sin(theta * np.pi/180)])
        com_acc_projected +=  com_acc[::-1] * np.array([np.cos(theta * np.pi/180),
                                                 np.sin(theta * np.pi/180)])
    
        com_vel, com = integrate_dynamic_step2(com_vel_0, com_0, com_acc_projected,
                                                                dt)

        return theta_vel, theta, com_vel, com
