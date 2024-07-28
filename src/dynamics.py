import numpy as np

# Constants
mass_moon:  float = 7.342 * 1e22  # kg
mass_earth: float = 5.972 * 1e24  # kg
G:          float = 6.674 * 1e-11 # N m^2 kg^-2
pixel_side: float = 3.624 * 1e5   # m

def distance(coords_1: np.array, coords_2: np.array):
    return np.sqrt(np.sum((coords_1-coords_2)**2))
def get_power_components(module: float, alpha: float):
    return module * np.array([np.cos(alpha), np.sin(alpha)])

def integrate_dynamic_step(vel_0: np.array, 
                           x_0:   np.array, 
                           acc:   np.array, 
                           dt:    float
                          ):
    """
    Evaluate the new speed and coordinates given the previous ones

    Parameters
    ----------
    vel_0: np.array
        Components of the initial speeds
    x_0: np.array
        Components of the initial coordinates
    acc: np.array
        Components of the accelleration
    dt: float
        The time distance from the start of the current accelleration to the end

    Returns
    -------
    vel: np.array
        Updated components for the speed
    x: np.array
        Updated components for the coordinates
    """
    
    vel = vel_0 + acc * dt
    x = x_0 + vel * dt
    
    return vel, x

def get_gravity_power_angle(mass_1:   float, mass_2:   float,
                            coords_1: tuple, coords_2: tuple,
                            distance: float,
                            exponent: float
                           ):
    """
    Get the gravitational power and angle between two bodies

    Parameters
    ----------
    mass_1: float
        The mass for the first body
    mass_2: float
        The mass for the second body
    coords_1: tuple
        The coordinates for the first body
    coords_2: tuple
        The coordinates for the second body
    distance: float
        The distance between the two bodies
    exponent: float
        The exponent to use to evaluate the Gravitational force 
        (Usually equal to two but our case is not in scale)

    Returns
    -------
    F_grav: float
        The gravitational force between the two bodies
    angle: float
        The application angle of the force
    """

    # Determine the gravity for the moon
    F_grav: float = G * mass_1 * mass_2 / distance ** exponent
    
    # Determine the application angle
    dx:     float = (coords_1 - coords_2)[0]          
    angle:  float = np.arccos( dx*pixel_side / distance )
    
    # Change the sign if the positive y_axis
    if (coords_1 - coords_2)[1] < 0:
        angle *= -1 

    return F_grav, angle

def get_gravity_force(com_0:        tuple,
                      moon_coords:  tuple,
                      earth_coords: tuple,
                      mass_shuttle: float,
                     ):
    """
    Get the gravitational power applied from the moon and the earth to the shuttle

    Parameters
    ----------
    com_0: tuple
        The initial center of mass of the shuttle
    moon_coords: tuple
        The moon's coordinates
    earth_coords: tuple
        The earth's coordinates
    mass_shuttle: float
        The shuttle's mass

    Returns
    -------
    F_grav_moon: np.array
        The gravitational force applied from the moon to the shuttle
    F_grav_earth: np.array
        The gravitational force applied from the earth to the shuttle
    """
    
    # Evaluate the distances
    distance_shuttle_moon:  float = distance(com_0, moon_coords)  * pixel_side 
    distance_shuttle_earth: float = distance(com_0, earth_coords) * pixel_side

    F_grav_moon, beta = get_gravity_power_angle(mass_moon, mass_shuttle, 
                                                moon_coords, com_0, 
                                                distance_shuttle_moon, 
                                                2.0
                                               )
    # Get the components
    F_grav_moon = get_power_components(F_grav_moon, beta)

    # Determine the gravity for the earth
    F_grav_earth, gamma = get_gravity_power_angle(mass_earth, mass_shuttle, 
                                                  earth_coords, com_0, 
                                                  distance_shuttle_earth, 
                                                  2.1
                                                 )
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
        """
        Function to evaluate the movement of the shuttle in the space determined by the engines and gravitational forces

        Parameters
        ----------
        F1_mod: float
            Module of the force to turn left
        F2_mod: float
            Module of the force to go straight
        F3_mod: float
            Module of the force to turn right
        alpha_left: float
            Application angle for the forces of the engines to the left of the shuttle
        alpha_center: float
            Application angle for the forces of the engines below the shuttle
        alpha_right: float
            Application angle for the forces of the engines to the right of the shuttle
        mass_shuttle: float
            The shuttle's mass
        width: float
            The shuttle's width
        height: float
            The shuttle's height
        main_engine: tuple
            The main engine coordinates
        t_r: tuple
            The top-right engine coordinates
        b_l: tuple
            The bottom-left engine coordinates
        t_l: tuple
            The top-left engine coordinates
        b_r: tuple
            The bottom-right engine coordinates
        angular_speed_0: float
            The initial angular speed
        theta_0: float
            The initial shuttle's angle
        com_vel_0: np.array
            The initial shuttle's center of mass speed components
        com_0: np.array
            The initial shuttle's center of mass coordinates 
        moon_coords: np.array
            The moon's coordinates 
        earth_coords: np.array
            The earth's coordinates
        dt: float
            The time distance from the start of the current accelleration to the end

        Returns
        -------
        angular_speed: float
            The final angular speed
        theta: float
            The final shuttle's angle
        com_vel: np.array
            The final center of mass speed
        com: np.array
            The final center of mass coordinates
        """

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
        F_grav                    = F_grav_moon + F_grav_earth

        # Obtain the final Power 
        # (The gravitational power is added when the force is projected to the referement system of the environment)
        # (Actual referement system: the center of mass of the shuttle)
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
