###########################################################
##                                                       ##
##      File containing the environment for the          ##
##      lunar landing problem and the shuttle agent      ##
##                                                       ##
###########################################################

import numpy as np                                           # To work with multidimensional arrays

import matplotlib.pyplot as plt                              # To visualize the images
import matplotlib.patches as patches                         # To add the images
from matplotlib.image import imread                          # To open the images
from matplotlib.offsetbox import OffsetImage, AnnotationBbox # To Resize and add the images

from PIL import Image                                        # To Rotate the images

from src.dynamics import dynamic_step    # To simulate the dynamics

###########################################################
##                   AGENT DEFINITION                    ##
###########################################################

class Agent():
    def __init__(self, 
                 angle:                     int, 
                 earth_size:                int, 
                 earth_coords:            tuple,
                 inertia:                 float,
                 init_coords:      tuple | None = None,
                 init_angular_speed: int | None = None
                ):

        # Initialize the Agent information with the input data
        self.angle: int = angle

        self.w: float = 10
        self.h: float = 100
        self.m: float = 1000
        
        self.top_right_engine_coords:    tuple = ( self.w/2,  self.h/2)
        self.bottom_left_engine_coords:  tuple = (-self.w/2, -self.h/2)
        self.top_left_engine_coords:     tuple = (-self.w/2,  self.h/2)
        self.bottom_right_engine_coords: tuple = ( self.w/2, -self.h/2)
        self.main_engine:                tuple = (        0, -self.h/2)
        
        self.alpha_to_left:  float = 180  #(alpha)
        self.alpha_center:   float = 90
        self.alpha_to_right: float = 0 #(180-alpha)
        
        self.earth_coords: np.array = np.array((150, 150))
        self.moon_coords:  np.array = np.array((900, 900))
        
        shuttle_x = earth_coords[0] - 512*earth_size*np.sin((self.angle - 90) * np.pi/180) if init_coords is None else init_coords[0]
        shuttle_y = earth_coords[0] + 512*earth_size*np.cos((self.angle - 90) * np.pi/180) if init_coords is None else init_coords[1]
        
        self.coords:      np.array = np.array((shuttle_x, shuttle_y))
        self.prev_coords: np.array = np.array((shuttle_x, shuttle_y))

        self.speed:         tuple = (0, 0)
        self.angular_speed: float = 0 if init_angular_speed is None else init_angular_speed
        self.fuel:            int = 3000
        self.dt:            float = 1   

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
        left_speed = 0 if self.fuel - 2 * left_speed < 0 else left_speed
        self.fuel -= 2 * left_speed

        # Determine if the shuttle has enough fuel to activate the right engine
        right_speed = 0 if self.fuel - 2 * right_speed < 0 else right_speed
        self.fuel -= 2 * right_speed

        # Determine if the shuttle has enough fuel to activate the middle engine
        middle_speed = 0 if self.fuel - 5 * middle_speed < 0 else middle_speed
        self.fuel -= 5 * middle_speed

        return left_speed, middle_speed, right_speed

    def step(self, 
             action: int
            ):

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

        self.prev_coords = self.coords
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
    def get_prev_coords(self):
        return self.prev_coords
    def get_fuel(self):
        return self.fuel
    def get_speed(self):
        return self.speed
    def get_angular_speed(self):
        return self.angular_speed
    def get_state(self):
        # Get the shuttle informations as the current state
        shuttle_x, shuttle_y = self.coords
        shuttle_speed_x, shuttle_speed_y = self.speed

        return (int(shuttle_x), int(shuttle_y), int(self.angle), int(shuttle_speed_x), int(shuttle_speed_y), int(self.angular_speed), self.fuel)
        
###########################################################
##                ENVIRONMENT DEFINITION                 ##
###########################################################


class Environment():
    def __init__(self):
        # Load the images
        self.__load_assets()

        # Initialize the images size
        self.image_size:   int   = 1300
        self.earth_size:   float = 1/4
        self.moon_size:    float = 1/8
        self.shuttle_size: float = 1/16
        self.fire_size:    float = 1/51
        self.flag_size:    float = 1/51

        # Force to use Render method to can obtain the initial state
        self.__initialized: bool = False

    
    #######################################################################################################################################
    ##-----------------------------------------------------------------------------------------------------------------------------------##
    ##-----------------------------------------------------------| PRIVATE METHODS |-----------------------------------------------------##
    ##-----------------------------------------------------------------------------------------------------------------------------------##
    #######################################################################################################################################

    def __load_assets(self):
        # Load the images
        assets:       list = ["earth", "fire", "moon", "spaceship", "flag", "explode"]
        asset_images: dict = {asset:imread(f"./assets/{asset}.png") for asset in assets}

        # Convert the images to numpy array
        earth:     np.array = (asset_images["earth"]     * 255).astype(np.uint8)
        moon:      np.array = (asset_images["moon"]      * 255).astype(np.uint8)
        flag:      np.array = (asset_images["flag"]      * 255).astype(np.uint8)
        shuttle:   np.array = (asset_images["spaceship"] * 255).astype(np.uint8)
        fire:      np.array = (asset_images["fire"]      * 255).astype(np.uint8)
        explosion: np.array = (asset_images["explode"]   * 255).astype(np.uint8)

        self.earth      = Image.fromarray(earth)
        self.moon       = Image.fromarray(moon)
        self.flag       = Image.fromarray(flag)
        self.shuttle    = Image.fromarray(shuttle)
        self.fire_main  = Image.fromarray(fire)
        self.fire_left  = Image.fromarray(fire)
        self.fire_right = Image.fromarray(fire)
        self.explosion  = explosion

    def __state(self):
        return self.shuttle_agent.get_state() + self.flag_coords

    def __rotate(self):
        # Get the shuttle angle
        shuttle_angle: int = self.shuttle_agent.get_angle()

        # Rotate the images
        earth_rotated      = self.earth     .rotate(self.earth_angle,    expand=True)
        moon_rotated       = self.moon      .rotate(self.moon_angle,     expand=True)
        flag_rotated       = self.flag      .rotate(self.flag_angle,     expand=True)
        shuttle_rotated    = self.shuttle   .rotate(shuttle_angle,       expand=True)
        fire_main_rotated  = self.fire_main .rotate(shuttle_angle + 180, expand=True)
        fire_left_rotated  = self.fire_left .rotate(shuttle_angle + 180, expand=True)
        fire_right_rotated = self.fire_right.rotate(shuttle_angle + 180, expand=True)

        # Convert the images to numpy array for easier management
        earth_rotated      = np.array(earth_rotated)
        moon_rotated       = np.array(moon_rotated)
        flag_rotated       = np.array(flag_rotated)
        shuttle_rotated    = np.array(shuttle_rotated)
        fire_main_rotated  = np.array(fire_main_rotated)
        fire_left_rotated  = np.array(fire_left_rotated)
        fire_right_rotated = np.array(fire_right_rotated)

        # Resize the images
        earth      = OffsetImage(earth_rotated,      zoom=self.earth_size)
        moon       = OffsetImage(moon_rotated,       zoom=self.moon_size)
        flag       = OffsetImage(flag_rotated,       zoom=self.flag_size)
        shuttle    = OffsetImage(shuttle_rotated,    zoom=self.shuttle_size)
        fire_main  = OffsetImage(fire_main_rotated,  zoom=self.fire_size)
        fire_left  = OffsetImage(fire_left_rotated,  zoom=self.fire_size * 0.5)
        fire_right = OffsetImage(fire_right_rotated, zoom=self.fire_size * 0.5)
        explosion  = OffsetImage(self.explosion,     zoom=self.shuttle_size)

        return earth, moon, flag, shuttle, fire_main, fire_left, fire_right, explosion

    def __reward(self):
        """
        The reward penalize the fuel consumption by 10 (15 for main engine)
        The reward penalize the crash of the shuttle (or if it goes outside the bounds) by 100
        The reward is increased by 100 if the shuttle lands on the moon
        The reward is increased by 100 if the shuttle is near the flag

        Returns
        -------
        reward: int
            The state reward
        """        
        # Get the system coordinates
        shuttle_x, shuttle_y             = self.shuttle_agent.get_coords()
        prev_shuttle_x, prev_shuttle_y   = self.shuttle_agent.get_prev_coords()
        flag_x, flag_y                   = self.flag_coords
        earth_x, earth_y                 = self.earth_coords
        moon_x, moon_y                   = self.moon_coords

        # Get the shuttle speed
        shuttle_x_speed, shuttle_y_speed       = self.shuttle_agent.get_speed()
        speed_module:                    float = np.sqrt(shuttle_x_speed ** 2 + shuttle_y_speed ** 2)

        # Get the shuttle angular speed
        shuttle_angular_speed: float = self.shuttle_agent.get_angular_speed()
        shuttle_angle:         float = self.shuttle_agent.get_angle()

        # Evaluate the distances
        flag_distance:      float = np.sqrt( (shuttle_x     -flag_x) **2 + (shuttle_y     -flag_y) **2 )
        prev_flag_distance: float = np.sqrt( (prev_shuttle_x-flag_x) **2 + (prev_shuttle_y-flag_y) **2 )
        
        earth_distance:      float = np.sqrt( (shuttle_x     -earth_x) **2 + (shuttle_y     -earth_y) **2 )
        prev_earth_distance: float = np.sqrt( (prev_shuttle_x-earth_x) **2 + (prev_shuttle_y-earth_y) **2 )

        moon_distance:      float = np.sqrt( (shuttle_x     -moon_x) **2 + (shuttle_y     -moon_y) **2 )
        prev_moon_distance: float = np.sqrt( (prev_shuttle_x-moon_x) **2 + (prev_shuttle_y-moon_y) **2 )

        # Initialize the rewards
        reward: int = 0

        # Give the Agent a reward based on its distance from the flag
        reward += 5  if flag_distance < 650  else 0
        reward += 5  if flag_distance < 550  else 0
        reward += 5  if flag_distance < 450  else 0
        reward += 5  if flag_distance < 350  else 0
        reward += 5  if flag_distance < 250  else 0
        reward += 5  if flag_distance < 150  else 0
        reward += 5  if flag_distance < 100  else 0
        reward += 5  if flag_distance < 50   else 0
        reward += 10 if flag_distance < 20   else 0

        # Give rewards if the speed is under a certain value
        reward += 15 if speed_module < 0.5 else 0
        reward += 15 if speed_module < 0.5 and flag_distance < 200 else 0

        # Give a reward if the distance from the flag is decreased
        # reward += 20 if prev_flag_distance - flag_distance > 0 else 0

        # Give a reward if the angle of the shuttle is correct for the landing
        reward += 15 if flag_distance < 150 and abs(shuttle_angle % 360 - self.flag_angle % 360) < 25 else 0

        # Give a reward if the shuttle angular speed is small

        # Give a big reward if the shuttle has landed
        reward += 10000 if self.__is_landed() else 0

        # Give a big penalty if the shuttle has crashed near the earth
        reward -= 10000 if self.__is_crashed() and flag_distance > 500 else 0
        
        # Give a relative big penalty if the shuttle has crashed near the moon but away from the flag
        reward -= 1000  if self.__is_crashed() and flag_distance < 500 and flag_distance > 200 else 0
        reward = reward if abs(shuttle_angular_speed) < 1.5 else 0

        return reward

    def __is_done(self):
        """
        Returns
        -------
        done: bool
            True if the game is done (The shuttle has landed or crashed)
        """
        return self.__is_crashed() or self.__is_landed()

    def __is_crashed(self):
        """
        Returns
        -------
        crashed: bool
            True if the shuttle has crashed or if it goes outside the bounds or is out of fuel
        """

        # Get the shuttle, moon and earth coordinates
        shuttle_x, shuttle_y = self.shuttle_agent.get_coords()
        moon_x, moon_y       = self.moon_coords
        earth_x, earth_y     = self.earth_coords

        # Evaluate the distances
        moon_distance:  float = np.sqrt( (shuttle_x - moon_x)  **2 + (shuttle_y - moon_y)  **2 )
        earth_distance: float = np.sqrt( (shuttle_x - earth_x) **2 + (shuttle_y - earth_y) **2 )

        return self.shuttle_agent.get_fuel()            < 2 \
            or shuttle_x < 0 or shuttle_x > self.image_size \
            or shuttle_y < 0 or shuttle_y > self.image_size \
            or moon_distance                         <=  50 \
            or earth_distance                        <= 110 \
            or self.stall                             > 100

    def __is_landed(self):
        """
        Returns
        -------
        landed: bool
            True if the shuttle has landed on the moon
        """

        # Get the shuttle informations
        shuttle_x, shuttle_y                   = self.shuttle_agent.get_coords()
        shuttle_angle:                     int = self.shuttle_agent.get_angle()
        shuttle_speed_x, shuttle_speed_y       = self.shuttle_agent.get_speed()
        shuttle_angular_speed:           float = self.shuttle_agent.get_angular_speed()
        speed_module:                    float = np.sqrt(shuttle_speed_x ** 2 + shuttle_speed_y ** 2)

        # Get the moon informations
        moon_x, moon_y = self.moon_coords
        
        # Get the flag informations
        flag_x, flag_y       = self.flag_coords
        flag_angle:    float = self.flag_angle
        
        # Evaluate the distances
        target_distance: float = np.sqrt( (shuttle_x - flag_x) **2 + (shuttle_y - flag_y) **2 )
        moon_distance:   float = np.sqrt( (shuttle_x - moon_x) **2 + (shuttle_y - moon_y) **2 )

        return self.__is_near()                                 \
           and moon_distance                            >   50  \
           and abs(flag_angle%360 - shuttle_angle%360)  <   25  \
           and speed_module                             <=  0.5 \
           and shuttle_angular_speed                    <=  0.5 

    def __is_near(self):
        """
        Returns
        -------
        near: bool
            True if the shuttle is near the flag
        """
        # Get the shuttle and flag coordinates
        shuttle_x, shuttle_y      = self.shuttle_agent.get_coords()
        flag_x, flag_y            = self.flag_coords
        flag_distance:      float = np.sqrt( (shuttle_x - flag_x) **2 + (shuttle_y - flag_y) **2 )

        return flag_distance <= 50

    #######################################################################################################################################
    ##-----------------------------------------------------------------------------------------------------------------------------------##
    ##------------------------------------------------------------| PUBLIC METHODS |-----------------------------------------------------##
    ##-----------------------------------------------------------------------------------------------------------------------------------##
    #######################################################################################################################################


    def reset(self, 
              angle:                 int   = 90, 
              flag_angle:            int   = 90,
              inertia:               float = 0.,
              init_coords:    tuple | None = None,
              init_angular_speed: tuple | None = None
             ):
        """
        Reset the episode state

        Parameters
        ----------
        angle: int
            The angle of the shuttle
        flag_angle: int
            The angle of the flag
        inertia: float
            The inertia of the system
        init_coords: tuple | None
            The initial shuttle coordinates
        init_angular_speed: tuple | None
            The initial angular speed

        Returns
        -------
        state: tuple
            The initial state
        """
        self.__initialized: bool = True
        # Initialize the images angle
        self.earth_angle:    int = 0
        self.moon_angle:     int = 0
        self.flag_angle:     int = flag_angle - 90

        # Initialize the images coordinates
        self.earth_coords: tuple = (150, 150)
        self.moon_coords:  tuple = (900, 900)
        self.flag_coords:  tuple = (int(self.moon_coords[0] - 512*self.moon_size*np.sin((self.flag_angle) * np.pi/180)) ,
                                    int(self.moon_coords[1] + 512*self.moon_size*np.cos((self.flag_angle) * np.pi/180)) )

        # Initializate the fire to deactivate
        # (left, main, right)
        self.actives_fire: list = [False, False, False]

        # Initialize the shuttle agent
        self.shuttle_agent = Agent(angle, self.earth_size, self.earth_coords, inertia, init_coords, init_angular_speed)

        self.done:   bool = False
        self.landed: bool = False
        self.stall:   int = 0
        
        return self.__state()

    def step(self, 
             action: list
            ):
        """
        Method to take an action

        Parameters
        ----------
        action: list
            List containg the power of the engines to use
            Values between 0 and 1
            The engine is active when the data are greater than 0.5
        inertia: float
            The inertia factor

        Returns
        -------
        observation: tuple
            Shuttle state containing 9 values: 
            (x_coord, y_coord, angle, speed_x, speed_y, angular_speed, fuel, flag_x_coord, flag_y_coord)
        reward: int
            The reward of taking the action in the current state
        done: bool
            Boolean which indicates that the episode is ended
        landed: bool
            Boolean which indicates that the shuttle is landed safely
        """
        
        # If the episode is still going on
        if self.__initialized and not self.done:
            # Take the action and obtain the speeds
            initial_coords: tuple = self.shuttle_agent.get_coords()
            
            left_speed, middle_speed, right_speed = self.shuttle_agent.step(action)
            
            # Determine the active engines
            self.actives_fire: list = [left_speed > 0, middle_speed > 0, right_speed > 0]

            # Update the internal informations
            self.landed: bool = self.__is_landed()
            self.done:   bool = self.__is_done()
            reward:      int  = self.__reward()

            final_coords = self.shuttle_agent.get_coords()

            if initial_coords[0] == final_coords[0] and initial_coords[1] == final_coords[1]:
                self.stall += 1
            else:
                self.stall = 0

            return self.__state(), reward, self.done, self.landed

        if self.done:
            print("Episode Ended!!!")
            return self.__state(), None, self.done, self.landed

        print("Call Reset function first!!!")
        return None, None, None, None

    def render(self, 
               array: bool = False, 
               debug: bool = False):
        """
        Method to render the current state

        Parameters
        ----------
        array: bool
            If True return the plot as a numpy array
            otherwise return the plot as a matplotlib figure
        debug: bool
            If True print the collision boxes and the shuttle coordinates

        Returns
        -------
        image: numpy array
            The current state as a numpy array
        """
        if self.__initialized:
            # Rotate the images
            earth, moon, flag, shuttle, fire_main, fire_left, fire_right, explosion = self.__rotate()

            # Get the shuttle coordinates
            shuttle_coords: tuple = self.shuttle_agent.get_coords()

            # Recover the shuttle informations
            angle:        float = self.shuttle_agent.get_angle()*np.pi/180
            sin_angle:    float = np.sin(angle)
            cos_angle:    float = np.cos(angle)
            shuttle_size: float = 512*self.shuttle_size

            # Determine the fire offsets
            main_fire_angle_offset      = np.array([ (shuttle_size + 5)  *sin_angle, -(shuttle_size + 5)  *cos_angle ])
            secondary_fire_angle_offset = np.array([ (shuttle_size)      *sin_angle, -(shuttle_size)      *cos_angle ])
            secondary_fire_offset       = np.array([ (shuttle_size * 0.4)*cos_angle,  (shuttle_size * 0.4)*sin_angle ])

            # Determine the fire coordinates
            fire_main_coords  = shuttle_coords + main_fire_angle_offset
            fire_left_coords  = shuttle_coords + secondary_fire_angle_offset + secondary_fire_offset
            fire_right_coords = shuttle_coords + secondary_fire_angle_offset - secondary_fire_offset

            # Create figure
            fig = plt.figure(figsize=(self.image_size / 100, self.image_size / 100), dpi=100)
            plt.style.use('dark_background')

            ax = plt.gca()
            ax.set_xlim(0, self.image_size)
            ax.set_ylim(0, self.image_size)
            ax.axis('off')

            # Add the images into the correct coordinates
            earth_img      = AnnotationBbox(earth,      self.earth_coords,   frameon=False)
            moon_img       = AnnotationBbox(moon,       self.moon_coords,    frameon=False)
            flag_img       = AnnotationBbox(flag,       self.flag_coords,    frameon=False)
            shuttle_img    = AnnotationBbox(shuttle,    shuttle_coords,      frameon=False)
            fire_main_img  = AnnotationBbox(fire_main,  fire_main_coords,    frameon=False)
            fire_left_img  = AnnotationBbox(fire_left,  fire_left_coords,    frameon=False)
            fire_right_img = AnnotationBbox(fire_right, fire_right_coords,   frameon=False)
            explosion_img  = AnnotationBbox(explosion,  shuttle_coords,      frameon=False)

            # Add the images to the plot
            ax.add_artist(earth_img)
            ax.add_artist(moon_img)
            ax.add_artist(flag_img)

            # If episode not ended and not in debug draw the flames and the shuttle
            if not debug and (not self.done or self.__is_landed):
                ax.add_artist(shuttle_img)
                if self.actives_fire[0]:
                    ax.add_artist(fire_left_img)
                if self.actives_fire[1]:
                    ax.add_artist(fire_main_img)
                if self.actives_fire[2]:
                    ax.add_artist(fire_right_img)

            vel_x, vel_y = self.shuttle_agent.get_speed()
            speed_module = np.sqrt(vel_x ** 2 + vel_y ** 2)
            vel_w = self.shuttle_agent.get_angular_speed()
            plt.text(shuttle_coords[0], shuttle_coords[1], f"s={speed_module}, w={vel_w}")
            
            # If debug draw the explosion
            if not debug and self.done and not self.__is_landed():
                ax.add_artist(explosion_img)

            # If debug draw the collision boxes
            if debug:
                # Obtain the shuttle informations
                shuttle_x, shuttle_y       = self.shuttle_agent.get_coords()

                # Draw the collision boxes for the penalities
                ax.add_patch( patches.Circle(self.earth_coords, 110, edgecolor='red', facecolor="None", lw=3, zorder=5))
                ax.add_patch( patches.Circle(self.moon_coords,   50,  edgecolor='red', facecolor="None", lw=3, zorder=5))

                # Draw the rewards areas
                ax.add_patch( patches.Circle(self.flag_coords,  650,  edgecolor='blue', facecolor="None", lw=3, zorder=5))
                ax.add_patch( patches.Circle(self.flag_coords,  550,  edgecolor='blue', facecolor="None", lw=3, zorder=5))
                ax.add_patch( patches.Circle(self.flag_coords,  450,  edgecolor='blue', facecolor="None", lw=3, zorder=5))
                ax.add_patch( patches.Circle(self.flag_coords,  350,  edgecolor='blue', facecolor="None", lw=3, zorder=5))
                ax.add_patch( patches.Circle(self.flag_coords,  250,  edgecolor='blue', facecolor="None", lw=3, zorder=5))
                ax.add_patch( patches.Circle(self.flag_coords,  150,  edgecolor='blue', facecolor="None", lw=3, zorder=5))
                ax.add_patch( patches.Circle(self.flag_coords,  100,  edgecolor='blue', facecolor="None", lw=3, zorder=5))
                ax.add_patch( patches.Circle(self.flag_coords,   50,  edgecolor='blue', facecolor="None", lw=3, zorder=5))
                ax.add_patch( patches.Circle(self.flag_coords,   20,  edgecolor='blue', facecolor="None", lw=3, zorder=5))

                ax.axvline(0,               color="red", lw=3, zorder=5)
                ax.axvline(self.image_size, color="red", lw=3, zorder=5)
                ax.axhline(0,               color="red", lw=3, zorder=5)
                ax.axhline(self.image_size, color="red", lw=3, zorder=5)

                # Draw the shuttle center position
                ax.scatter(shuttle_x, shuttle_y, color="red", lw=1, zorder=5)

            # If specified, return the plot array
            # Otherwise plot the element
            if array:
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]

                plt.close()
                return data
            else:
                plt.show()
                plt.close()
                return None

        # If the episode does not exits
        print("Call Reset function first!!!")
        return None
