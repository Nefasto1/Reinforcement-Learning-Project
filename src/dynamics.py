import numpy as np

def integrate_dynamic_step(vel_0, x_0, acc, dt):
        
        vel = vel_0 + acc * dt
        x = x_0 + vel * dt

        return vel, x

def distance(x1, x2):
    
    d = np.sqrt(np.sum((x1-x2)**2))
    
    return d
    

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

        #Constants
        G                   = 6.674 * 1e-11 # N m^2 kg^-2
        mass_moon           = 7.342 * 1e22  # kg
        mass_earth          = 5.972 * 1e24  # kg
        pixel_side          = 3.624 * 1e5   # m
        
        distance_shuttle_moon  = distance(com_0, moon_coords) * pixel_side 
        distance_shuttle_earth = distance(com_0, earth_coords) * pixel_side

        # print()
        # print(com_0)
        # print(distance_shuttle_moon)
        # print(distance_shuttle_earth)

        # print(np.arccos((moon_coords - com_0)[1]/distance(com_0, moon_coords)))
    
        F_grav_moon = G * mass_shuttle * mass_moon / distance_shuttle_moon**2.0#1.8
        beta  = np.arccos((moon_coords - com_0)[0]*pixel_side/distance_shuttle_moon) 
        if (moon_coords - com_0)[1] < 0:
            beta *= -1 
        # print(f"beta : {beta  * 180 / np.pi}")
        
        # print()
        F_grav_moon_x = F_grav_moon * np.cos(beta)
        F_grav_moon_y = F_grav_moon * np.sin(beta)
        # print(F_grav_moon_x, F_grav_moon_y)

        
        F_grav_earth = G * mass_shuttle * mass_earth / distance_shuttle_earth**2.1
        gamma = np.arccos((earth_coords-com_0)[0]*pixel_side/distance_shuttle_earth)
        if (earth_coords - com_0)[1] < 0:
            gamma *= -1 
        F_grav_earth_x = F_grav_earth * np.cos(gamma)
        F_grav_earth_y = F_grav_earth * np.sin(gamma)
        # print(F_grav_earth_x, F_grav_earth_y)

        F_grav = np.array((F_grav_moon_x + F_grav_earth_x,
                           F_grav_moon_y + F_grav_earth_y))
    
        # print(F_grav)
        # print((earth_coords[0]*F_grav_earth_y - earth_coords[1]*F_grav_earth_x))
        # print(moon_coords[0]*F_grav_moon_y - moon_coords[1]*F_grav_moon_x)
        # print(r1[0]*F1[1] - r1[1]*F1[0])
        # print(F1)
                                         
        F_net = 300*F1 + 500*F2 + 300*F3 #+ 400*F_grav
        # print(F_grav/F_net *  100)


        tau_net = np.sum([r1[0]*F1[1] - r1[1]*F1[0],
                          r2[0]*F2[1] - r2[1]*F2[0],
                          r3[0]*F3[1] - r3[1]*F3[0],
                          # (earth_coords[0]*F_grav_earth_y - earth_coords[1]*F_grav_earth_x) / 25,
                          # (moon_coords[0]*F_grav_moon_y - moon_coords[1]*F_grav_moon_x) * -1,
                         ])

        I = (width**3 + height) / 12

        theta_acc = tau_net / I
        com_acc = F_net / mass_shuttle

        theta_vel, theta = integrate_dynamic_step(theta_vel_0, theta_0, theta_acc, dt)
    
        com_acc_projected =  com_acc * np.array([np.cos(theta * np.pi/180),
                                                 np.sin(theta * np.pi/180)]) 
        
        com_acc_projected += com_acc[::-1] * np.array([np.cos(theta * np.pi/180),
                                                       np.sin(theta * np.pi/180)])

        com_acc_projected += F_grav / mass_shuttle
    
        com_vel, com = integrate_dynamic_step(com_vel_0, com_0, com_acc_projected, dt)
        # print(com_vel)

        return theta_vel, theta, com_vel, com
