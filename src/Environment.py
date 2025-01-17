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

from PIL import Image                                        # To Rotate the images

from src.utils import get_boxes
from src.utils import get_reward
from src.Agent import Agent
        
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
        self.fire       = Image.fromarray(fire)
        self.explosion  = explosion

    def __state(self):
        return self.shuttle_agent.get_state() + self.flag_coords

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
        # Get the shuttle speed
        speed_module:          float = self.shuttle_agent.get_speed_module()
        
        # Get the shuttle angular speed
        shuttle_angular_speed: float = self.shuttle_agent.get_angular_speed()
        shuttle_angle:         float = self.shuttle_agent.get_angle()

        # Evaluate the distances
        flag_distance:         float = self.shuttle_agent.distance_from(self.flag_coords)

        
        return get_reward(flag_distance, speed_module, shuttle_angle, self.flag_angle, shuttle_angular_speed, self.__is_landed, self.__is_crashed)

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
        shuttle_x, shuttle_y = self.shuttle_agent.get_coords()
        
        # Evaluate the distances
        moon_distance:  float = self.shuttle_agent.distance_from(self.moon_coords)
        earth_distance: float = self.shuttle_agent.distance_from(self.earth_coords)

        return shuttle_x < 0 or shuttle_x > self.image_size \
            or shuttle_y < 0 or shuttle_y > self.image_size \
            or self.shuttle_agent.get_fuel()          <   2 \
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
        shuttle_angle:         float = self.shuttle_agent.get_angle()
        shuttle_angular_speed: float = self.shuttle_agent.get_angular_speed()
        speed_module:          float = self.shuttle_agent.get_speed_module()

        # Get the flag informations
        flag_angle:    float = self.flag_angle
        
        # Evaluate the distances
        moon_distance: float = self.shuttle_agent.distance_from(self.moon_coords)

        return self.__is_near()                                 \
           and moon_distance                            >   50  \
           and abs(flag_angle%360 - shuttle_angle%360)  <   25  \
           and speed_module                             <=  0.5 \
           and abs(shuttle_angular_speed)               <=  0.5 

    def __is_near(self):
        """
        Returns
        -------
        near: bool
            True if the shuttle is near the flag
        """
        # Get the shuttle and flag coordinates
        flag_distance: float = self.shuttle_agent.distance_from(self.flag_coords)

        return flag_distance <= 50

    #######################################################################################################################################
    ##-----------------------------------------------------------------------------------------------------------------------------------##
    ##------------------------------------------------------------| PUBLIC METHODS |-----------------------------------------------------##
    ##-----------------------------------------------------------------------------------------------------------------------------------##
    #######################################################################################################################################


    def reset(self, 
              angle:                     int   = 90, 
              flag_angle:                int   = 90,
              inertia:                   float = 0.,
              init_coords:        tuple | None = None,
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
        self.actives_fire:  list = [False, False, False]

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
            initial_coords:   tuple = self.shuttle_agent.get_coords()
            
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
            # Get the shuttle coordinates
            shuttle_coords: tuple = self.shuttle_agent.get_coords()
            shuttle_angle:  float = self.shuttle_agent.get_angle()

            vel_w:          float = self.shuttle_agent.get_angular_speed()
            vel_x, vel_y          = self.shuttle_agent.get_speed()            
            speed_module:   float = np.sqrt(vel_x ** 2 + vel_y ** 2)
            
            # Get the images boxes
            earth, moon, flag, shuttle,   \
            fire_main, fire_left, fire_right, \
            fire_br, fire_bl, fire_tl, fire_tr, explosion = get_boxes(shuttle_angle, self.earth_angle, self.moon_angle, self.flag_angle,
                                                                      self.shuttle_size, self.earth_size, self.moon_size, self.flag_size, self.fire_size,
                                                                      self.shuttle, self.earth, self.moon, self.flag, self.fire, self.explosion,
                                                                      shuttle_coords, self.earth_coords, self.moon_coords, self.flag_coords,
                                                                     )

            # Create figure
            fig = plt.figure(figsize=(self.image_size / 100, self.image_size / 100), dpi=100)
            plt.style.use('dark_background')

            ax = plt.gca()
            ax.set_xlim(0, self.image_size)
            ax.set_ylim(0, self.image_size)
            ax.axis('off')

            # Add the images to the plot
            ax.add_artist( earth)
            ax.add_artist( moon )
            ax.add_artist( flag )            
            plt.text(shuttle_coords[0]+20, shuttle_coords[1], f"s={speed_module: .2f}, w={vel_w: .2f}")

            # If episode not ended and not in debug draw the flames and the shuttle
            if not debug and (not self.done or self.__is_landed):
                ax.add_artist(shuttle)
                if self.actives_fire[0]:
                    ax.add_artist(fire_bl)
                    ax.add_artist(fire_tr)
                if self.actives_fire[1]:
                    ax.add_artist(fire_left)
                    ax.add_artist(fire_main)
                    ax.add_artist(fire_right)
                if self.actives_fire[2]:
                    ax.add_artist(fire_br)
                    ax.add_artist(fire_tl)
            
            # If debug draw the explosion
            if not debug and self.done and not self.__is_landed():
                ax.add_artist(explosion)

            # If debug draw the collision boxes
            if debug:
                # Obtain the shuttle informations
                shuttle_x, shuttle_y = self.shuttle_agent.get_coords()

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
