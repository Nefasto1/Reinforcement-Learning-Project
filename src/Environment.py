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

###########################################################
##                   AGENT DEFINITION                    ##
###########################################################

class Agent():
    def __init__(self, angle, earth_size, earth_coords):
        # Initialize the Agent information with the input data
        self.angle = angle
        self.coords = (earth_coords[0] - 512*earth_size*np.sin((self.angle - 90) * np.pi/180) ,
                       earth_coords[1] + 512*earth_size*np.cos((self.angle - 90) * np.pi/180) )

        self.speed = (0, 0)
        self.angular_speed = 0
        self.fuel = 200

    def step(self, action, alpha):
        # Determine if the engines are over 0.5 and get their shifted value
        left_speed   = (action[0] > 0.5) * (action[0] - 0.5)
        middle_speed = (action[1] > 0.5) * (action[1] - 0.5)
        right_speed  = (action[2] > 0.5) * (action[2] - 0.5)

        # Determine if the shuttle has enough fuel to activate the left engine
        left_speed = 0 if self.fuel - 2 * left_speed < 0 else left_speed
        self.fuel -= 2 * left_speed

        # Determine if the shuttle has enough fuel to activate the right engine
        right_speed = 0 if self.fuel - 2 * right_speed < 0 else right_speed
        self.fuel -= 2 * right_speed

        # Determine if the shuttle has enough fuel to activate the middle engine
        middle_speed = 0 if self.fuel - 5 * middle_speed < 0 else middle_speed
        self.fuel -= 5 * middle_speed

        # Determine the angular speed and update the shuttle angle
        self.angular_speed = self.angular_speed*alpha + left_speed * 5 - right_speed * 5
        self.angle += self.angular_speed

        # Determine the new shuttle speed
        new_x_speed = (self.speed[0]*alpha + left_speed *0.6 + middle_speed * np.cos(self.angle*np.pi/180) *1.4)
        new_y_speed = (self.speed[1]*alpha + right_speed*0.6 + middle_speed * np.sin(self.angle*np.pi/180) *1.4)
        self.speed = (new_x_speed, new_y_speed)

        # Update the shuttle coordinates
        self.coords = (self.coords[0] + self.speed[0]*20, self.coords[1] + self.speed[1]*20)

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

###########################################################
##                ENVIRONMENT DEFINITION                 ##
###########################################################


class Environment():
    def __init__(self):
        # Load the images
        self.__load_assets()

        # Initialize the images size
        self.image_size   = 1000
        self.earth_size   = 1/4
        self.moon_size    = 1/8
        self.shuttle_size = 1/16
        self.fire_size    = 1/51
        self.flag_size    = 1/51

        # Force to use Render method to can obtain the initial state
        self.__initialized = False

    
    #######################################################################################################################################
    ##-----------------------------------------------------------------------------------------------------------------------------------##
    ##-----------------------------------------------------------| PRIVATE METHODS |-----------------------------------------------------##
    ##-----------------------------------------------------------------------------------------------------------------------------------##
    #######################################################################################################################################

    def __load_assets(self):
        # Load the images
        assets = ["earth", "fire", "moon", "spaceship", "flag", "explode"]
        asset_images = {asset:imread(f"./assets/{asset}.png") for asset in assets}

        # Convert the images to numpy array
        earth     = (asset_images["earth"]     * 255).astype(np.uint8)
        moon      = (asset_images["moon"]      * 255).astype(np.uint8)
        flag      = (asset_images["flag"]      * 255).astype(np.uint8)
        shuttle   = (asset_images["spaceship"] * 255).astype(np.uint8)
        fire      = (asset_images["fire"]      * 255).astype(np.uint8)
        explosion = (asset_images["explode"]   * 255).astype(np.uint8)

        self.earth      = Image.fromarray(earth)
        self.moon       = Image.fromarray(moon)
        self.flag       = Image.fromarray(flag)
        self.shuttle    = Image.fromarray(shuttle)
        self.fire_main  = Image.fromarray(fire)
        self.fire_left  = Image.fromarray(fire)
        self.fire_right = Image.fromarray(fire)
        self.explosion  = explosion

    def __state(self):
        # Get the shuttle informations as the current state
        shuttle_x, shuttle_y = self.shuttle_agent.get_coords()
        shuttle_angle        = self.shuttle_agent.get_angle()
        shuttle_speed_x, shuttle_speed_y = self.shuttle_agent.get_speed()
        shuttle_angular_speed = self.shuttle_agent.get_angular_speed()
        shuttle_fuel = self.shuttle_agent.get_fuel()

        return (shuttle_x, shuttle_y, shuttle_angle, shuttle_speed_x, shuttle_speed_y, shuttle_angular_speed, shuttle_fuel)

    def __rotate(self):
        # Get the shuttle angle
        shuttle_angle = self.shuttle_agent.get_angle()

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

        reward = 0
        if self.actives_fire[0]:
            reward -= 10
        if self.actives_fire[1]:
            reward -= 15
        if self.actives_fire[2]:
            reward -= 10
        if self.__is_crashed():
            reward -= 100
        if self.__is_near():
            reward += 100
        if self.__is_landed():
            reward += 100

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

        return self.shuttle_agent.get_fuel() <= 0 \
            or shuttle_x < 0 or shuttle_x > self.image_size \
            or shuttle_y < 0 or shuttle_y > self.image_size \
            or abs( np.sqrt( shuttle_x**2 + shuttle_y**2 ) - np.sqrt( moon_x**2  + moon_y**2  ) ) <=  self.moon_size*430 \
            or abs( np.sqrt( shuttle_x**2 + shuttle_y**2 ) - np.sqrt( earth_x**2 + earth_y**2 ) ) <=  self.earth_size*430

    def __is_landed(self):
        """
        Returns
        -------
        landed: bool
            True if the shuttle has landed on the moon
        """

        # Get the shuttle, moon informations
        shuttle_x, shuttle_y = self.shuttle_agent.get_coords()
        shuttle_angle        = self.shuttle_agent.get_angle()
        shuttle_speed_x, shuttle_speed_y = self.shuttle_agent.get_speed()
        shuttle_angular_speed = self.shuttle_agent.get_angular_speed()

        moon_x, moon_y       = self.moon_coords

        # Determine the target position of the shuttle based on its angle to have a safe landing
        target_x = self.moon_coords[0] - 512*self.moon_size*np.sin((shuttle_angle) * np.pi/180)
        target_y = self.moon_coords[1] + 512*self.moon_size*np.cos((shuttle_angle) * np.pi/180)

        return abs( np.sqrt( shuttle_x**2 + shuttle_y**2 ) - np.sqrt( moon_x**2 + moon_y**2 ) )     <=  self.moon_size*600 \
           and abs( np.sqrt( shuttle_x**2 + shuttle_y**2 ) - np.sqrt( moon_x**2 + moon_y**2 ) )     >   self.moon_size*430 \
           and abs( np.sqrt( shuttle_x**2 + shuttle_y**2 ) - np.sqrt( target_x**2 + target_y**2 ) ) <=  self.shuttle_size*430 \
           and shuttle_speed_x <= 0.5 and shuttle_speed_y <= 0.5 and shuttle_angular_speed <= 0.5

    def __is_near(self):
        """
        Returns
        -------
        near: bool
            True if the shuttle is near the flag
        """

        # Get the shuttle and flag coordinates
        shuttle_x, shuttle_y = self.shuttle_agent.get_coords()
        flag_x, flag_y       = self.flag_coords

        return abs( np.sqrt( shuttle_x**2 + shuttle_y**2 ) - np.sqrt( flag_x**2 + flag_y**2 ) ) <=  self.flag_size*2000

    #######################################################################################################################################
    ##-----------------------------------------------------------------------------------------------------------------------------------##
    ##------------------------------------------------------------| PUBLIC METHODS |-----------------------------------------------------##
    ##-----------------------------------------------------------------------------------------------------------------------------------##
    #######################################################################################################################################


    def reset(self, angle = 90, flag_angle = 90):
        """
        Reset the episode state

        Parameters
        ----------
        angle: int
            The angle of the shuttle
        flag_angle: int
            The angle of the flag

        Returns
        -------
        state: tuple
            The initial state
        """
        self.__initialized = True
        # Initialize the images angle
        self.earth_angle   = 0
        self.moon_angle    = 0
        self.flag_angle    = flag_angle - 90

        # Initialize the images coordinates
        self.earth_coords   = (150, 150)
        self.moon_coords    = (900, 900)
        self.flag_coords    = (self.moon_coords[0] - 512*self.moon_size*np.sin((self.flag_angle) * np.pi/180) ,
                               self.moon_coords[1] + 512*self.moon_size*np.cos((self.flag_angle) * np.pi/180) )

        # Initializate the fire to deactivate
        # (left, main, right)
        self.actives_fire = [False, False, False]

        # Initialize the shuttle agent
        self.shuttle_agent = Agent(angle, self.earth_size, self.earth_coords)

        self.done   = False
        self.landed = False

        return self.__state()

    def step(self, action, alpha):
        # If the episode is still going on
        if self.__initialized and not self.done:
            # Take the action
            self.shuttle_agent.step(action, alpha)
            self.actives_fire = [action[0] > 0.5, action[1] > 0.5, action[2] > 0.5]

            # Update the internal informations
            self.done   = self.__is_done()
            self.landed = self.__is_landed()
            reward      = self.__reward()

            return self.__state(), reward, self.done, self.landed

        if self.done:
            print("Episode Ended!!!")
            return self.__state(), None, self.done, self.landed

        print("Call Reset function first!!!")
        return None, None, None, None

    def render(self, array=False, debug=False):
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
            shuttle_coords = self.shuttle_agent.get_coords()

            # Recover the shuttle informations
            angle        = self.shuttle_agent.get_angle()*np.pi/180
            sin_angle    = np.sin(angle)
            cos_angle    = np.cos(angle)
            shuttle_size = 512*self.shuttle_size

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
            if not debug and not self.done:
                ax.add_artist(shuttle_img)
                if self.actives_fire[0]:
                    ax.add_artist(fire_left_img)
                if self.actives_fire[1]:
                    ax.add_artist(fire_main_img)
                if self.actives_fire[2]:
                    ax.add_artist(fire_right_img)

            # If debug draw the explosion
            if not debug and self.done:
                ax.add_artist(explosion_img)

            # If debug draw the collision boxes
            if debug:
                # Obtain the shuttle informations
                shuttle_x, shuttle_y = self.shuttle_agent.get_coords()
                shuttle_angle        = self.shuttle_agent.get_angle()

                # Determine the target position for the safe landing
                target_x = self.moon_coords[0] - 512*self.moon_size*np.sin((shuttle_angle) * np.pi/180)
                target_y = self.moon_coords[1] + 512*self.moon_size*np.cos((shuttle_angle) * np.pi/180)

                # Draw the collision boxes for the penalities
                ax.add_patch( patches.Circle(self.earth_coords, self.earth_size*430, edgecolor='red', facecolor="None", lw=3, zorder=5))
                ax.add_patch( patches.Circle(self.moon_coords,  self.moon_size*430,  edgecolor='red', facecolor="None", lw=3, zorder=5))

                ax.axvline(0,               color="red", lw=3, zorder=5)
                ax.axvline(self.image_size, color="red", lw=3, zorder=5)
                ax.axhline(0,               color="red", lw=3, zorder=5)
                ax.axhline(self.image_size, color="red", lw=3, zorder=5)

                # Draw the collision boxes for the positive rewards
                ax.add_patch( patches.Circle(self.flag_coords, self.flag_size*2000, edgecolor='green', facecolor="None", lw=3, zorder=5))
                ax.add_patch( patches.Circle((target_x, target_y), self.shuttle_size*430, edgecolor='green', facecolor="None", lw=3, zorder=5))

                # Draw the shuttle center position
                ax.scatter(shuttle_x, shuttle_y, color="red", lw=1, zorder=5)

            # If specified, return the plot array
            # Otherwise plot the element
            if array:
                fig.canvas.draw()
                # print(np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).shape)
                # print(np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).shape)
                # data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
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