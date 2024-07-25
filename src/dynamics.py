import numpy as np

# Constants
mass_moon:  float = 7.342 * 1e22  # kg
mass_earth: float = 5.972 * 1e24  # kg
G:          float = 6.674 * 1e-11 # N m^2 kg^-2
pixel_side: float = 3.624 * 1e5   # m

def integrate_dynamic_step(vel_0: tuple, 
                           x_0:   tuple, 
                           acc:   tuple, 
                           dt:    tuple
                          ):
    
    vel = vel_0 + acc * dt
    x = x_0 + vel * dt
    
    return vel, x

def distance(x1: tuple, 
             x2: tuple):
    return np.sqrt(np.sum((x1-x2)**2))

def get_power_components(module: float, 
                         alpha:  float
                        ):
    return module * np.array([np.cos(alpha),
                              np.sin(alpha)])

def get_power_angle(mass1:    float, mass2:    float,
                    coords1:  tuple, coords2:  tuple,
                    distance: float,
                    exponent: float
                   ):

    # Determine the gravity for the moon
    F_grav: float = G * mass1 * mass2 / distance ** exponent
    
    dx:     float = (coords1 - coords2)[0]          
    
    angle:  float = np.arccos( dx*pixel_side / distance )
    
    # Change the sign if the positive y_axis
    if (coords1 - coords2)[1] < 0:
        angle *= -1 

    return F_grav, angle

def get_gravity_force(com_0:        tuple,
                      moon_coords:  tuple,
                      earth_coords: tuple,
                      mass_shuttle: float,
                     ):

    # Evaluate the distances
    distance_shuttle_moon:  float = distance(com_0, moon_coords)  * pixel_side 
    distance_shuttle_earth: float = distance(com_0, earth_coords) * pixel_side

    F_grav_moon, beta = get_power_angle(mass_moon, mass_shuttle, 
                                        moon_coords, com_0, 
                                        distance_shuttle_moon, 
                                        2.0)
    # Get the components
    F_grav_moon = get_power_components(F_grav_moon, beta)

    # Determine the gravity for the earth
    F_grav_earth, gamma = get_power_angle(mass_earth, mass_shuttle, 
                                          earth_coords, com_0, 
                                          distance_shuttle_earth, 
                                          2.1)
    # Get the components
    F_grav_earth = get_power_components(F_grav_earth, gamma)

    return F_grav_moon, F_grav_earth


def dynamic_step(F1_mod:             float, F2_mod:          float, F3_mod:         float, 
                 alpha_to_left:      float, alpha_center:    float, alpha_to_right: float,
                 mass_shuttle:       float, width:           float, height:         float,
                 main_engine:        tuple,
                 t_r:                tuple, b_l:             tuple, 
                 t_l:                tuple, b_r:             tuple,
                 angular_speed_0:    float, theta_0:         float,
                 com_vel_0:       np.array, com_0:        np.array,
                 moon_coords:     np.array, earth_coords: np.array,
                 dt:                 float
                ):

        # Power to turn left, the top right engine and the bottom left engine
        F1_top:    np.array = get_power_components(F1_mod, alpha_to_left  * np.pi/180)
        F1_bottom: np.array = get_power_components(F1_mod, alpha_to_right * np.pi/180)

        # Power for the main engine
        F2: np.array = get_power_components(F2_mod, alpha_center * np.pi/180)     
    
        # Power to turn right, the bottom right engine and the top left engine
        F3_bottom: np.array = get_power_components(F3_mod, alpha_to_left  * np.pi/180)  
        F3_top:    np.array = get_power_components(F3_mod, alpha_to_right * np.pi/180)  

        # Get the gravity force
        F_grav_moon, F_grav_earth = get_gravity_force(com_0, moon_coords, earth_coords, mass_shuttle)
        F_grav = F_grav_moon + F_grav_earth

        # Obtain the final Power
        F_net = 100*F2
    
        tau_net: float = np.sum([main_engine[0]*F2[1]        - main_engine[1]*F2[0],
                                  t_r[0]       *F1_top[1]    - t_r[1]        *F1_top[0],
                                  b_l[0]       *F1_bottom[1] - b_l[1]        *F1_bottom[0],
                                  t_l[0]       *F3_top[1]    - t_l[1]        *F3_top[0],
                                  b_r[0]       *F3_bottom[1] - b_r[1]        *F3_bottom[0],
                                 ])

        # Evaluate the torque inertia
        I: float = (width**3 + height) / 12

    
        # Evaluate the accellerations
        theta_acc: float = tau_net / I
        com_acc:   float = F_net   / mass_shuttle

        # Evaluate the new angle
        angular_speed, theta = integrate_dynamic_step(angular_speed_0, theta_0, theta_acc, dt)
    
        # Project the power in the system coordinates
        com_acc_projected  = get_power_components(com_acc,       theta * np.pi/180)
        com_acc_projected += get_power_components(com_acc[::-1], theta * np.pi/180)
        com_acc_projected += F_grav / mass_shuttle

        # Evaluate the new coordinates
        com_vel, com = integrate_dynamic_step(com_vel_0, com_0, com_acc_projected, dt)
    

        return angular_speed, theta, com_vel, com
