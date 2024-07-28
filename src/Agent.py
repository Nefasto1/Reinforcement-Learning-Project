###########################################################
##                   AGENT DEFINITION                    ##
###########################################################
from src.dynamics import dynamic_step    # To simulate the dynamics
from src.dynamics import distance        # To evaluate the distance
import numpy as np                       # To work with multidimensional arrays

class Agent():
    def __init__(self, 
                 angle:                       int, 
                 earth_size:                  int, 
                 earth_coords:              tuple,
                 inertia:                   float,
                 init_coords:        tuple | None = None,
                 init_angular_speed: int | None = None
                ):

        # Initialize the Agent information with the input data
        # Physical informations of the shuttle
        self.w: float = 10
        self.h: float = 100
        self.m: float = 1000
        
        # Additional informations
        self.angle:         float = angle
        self.speed:         tuple = (0, 0)
        self.angular_speed: float = 0 if init_angular_speed is None else init_angular_speed
        self.dt:            float = 1   
        self.fuel:            int = 3000
                    
        # Earth, Moon and shuttle coordinates
        shuttle_x = earth_coords[0] - 512*earth_size*np.sin((self.angle - 90) * np.pi/180) if init_coords is None else init_coords[0]
        shuttle_y = earth_coords[0] + 512*earth_size*np.cos((self.angle - 90) * np.pi/180) if init_coords is None else init_coords[1]
        
        self.earth_coords = np.array((150, 150))
        self.moon_coords  = np.array((900, 900))
        self.coords       = np.array((shuttle_x, shuttle_y))

        # Engines Coordinates
        self.top_right_engine_coords:    tuple = ( self.w/2,  self.h/2)
        self.bottom_left_engine_coords:  tuple = (-self.w/2, -self.h/2)
        self.top_left_engine_coords:     tuple = (-self.w/2,  self.h/2)
        self.bottom_right_engine_coords: tuple = ( self.w/2, -self.h/2)
        self.main_engine:                tuple = (        0, -self.h/2)

        # Engine Angles
        self.alpha_to_left:  float = 180
        self.alpha_center:   float = 90
        self.alpha_to_right: float = 0 

    def __update_fuel(self, 
                      left_speed:   float, 
                      middle_speed: float, 
                      right_speed:  float
                     ):  
        """
        Update the fuel counter and check if the engine can be activated
        (To go left you activate the right engine and viceversa)

        Parameters
        ----------
        left_speed: float
            Amount of energy in the right engine
        middle_speed: float
            Amount of energy in the middle engine
        right_speed: float
            Amount of energy in the left engine

        Returns
        -------
        left_speed: float
            Amount of energy in the right engine
            0 if not enough fuel, input otherwise
        middle_speed: float
            Amount of energy in the middle engine
            0 if not enough fuel, input otherwise
        right_speed: float
            Amount of energy in the left engine
            0 if not enough fuel, input otherwise
        """
        # Determine if the shuttle has enough fuel to activate the left engine
        left_speed  = 0 if self.fuel - 2 * left_speed < 0 else left_speed
        self.fuel  -= 2 * left_speed

        # Determine if the shuttle has enough fuel to activate the right engine
        right_speed  = 0 if self.fuel - 2 * right_speed < 0 else right_speed
        self.fuel   -= 2 * right_speed

        # Determine if the shuttle has enough fuel to activate the middle engine
        middle_speed  = 0 if self.fuel - 5 * middle_speed < 0 else middle_speed
        self.fuel    -= 5 * middle_speed

        return left_speed, middle_speed, right_speed

    def step(self, 
             action: int
            ):
        """
        Function to move the agent based on the selected action

        Parameters
        ----------
        action: int
            An integer to select the action
                - Do Nothing:  0
                - Turn Left:   1
                - Go Straight: 2
                - Turn Right:  3
        Returns
        -------
        F1_mod: float
            Module for the engines to turn left (just for rendering purposes)
        F2_mod: float
            Module for the engines to go straight (just for rendering purposes)
        F3_mod: float
            Module for the engines to turn right (just for rendering purposes)
        """

        if action == 1:
            action = [1, 0, 0]
        elif action == 2:
            action = [0, 1, 0]
        elif action == 3:
            action = [0, 0, 1]
        else:
            action = [0, 0, 0]
        
        # Determine if the engines are over 0.5 and get their shifted value
        F1_mod = (action[0] > 0.5) * (action[0] - 0.5)
        F2_mod = (action[1] > 0.5) * (action[1] - 0.5)
        F3_mod = (action[2] > 0.5) * (action[2] - 0.5)

        # Update the fuel counter and check if the engines can be activated
        F1_mod, F2_mod, F3_mod = self.__update_fuel(F1_mod, F2_mod, F3_mod)

        # Dynamic step
        self.angular_speed, self.angle, self.speed, self.coords = dynamic_step(F1_mod, F2_mod, F3_mod,
                                                                               self.alpha_to_left,             self.alpha_center, self.alpha_to_right,
                                                                               self.m,                         self.w,            self.h,
                                                                               self.main_engine,
                                                                               self.top_right_engine_coords,   self.bottom_left_engine_coords, 
                                                                               self.top_left_engine_coords,    self.bottom_right_engine_coords,
                                                                               self.angular_speed,             self.angle,
                                                                               self.speed,                     self.coords,
                                                                               self.moon_coords,               self.earth_coords,
                                                                               self.dt
                                                                              )
        
        return F1_mod, F2_mod, F3_mod

    def get_angle(self):
        return self.angle-90
    def get_coords(self):
        return self.coords
    def get_fuel(self):
        return self.fuel
    def get_speed(self):
        return self.speed
    def get_angular_speed(self):
        return self.angular_speed
    def distance_from(self, obj_coords):
        return distance(self.coords, obj_coords)
    def get_speed_module(self):
        shuttle_speed_x, shuttle_speed_y = self.speed
        return np.sqrt( shuttle_speed_x**2 + shuttle_speed_y**2 )
        
        
    def get_state(self):
        shuttle_x, shuttle_y = self.coords
        shuttle_speed_x, shuttle_speed_y = self.speed
        return (int(shuttle_x), int(shuttle_y), int(self.angle), int(shuttle_speed_x), int(shuttle_speed_y), int(self.angular_speed), self.fuel)
