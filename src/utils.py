import numpy as np
from PIL import Image                                        # To Rotate the images
from matplotlib.offsetbox import OffsetImage, AnnotationBbox # To Resize and add the images

def evaluate_fire_coords(shuttle_angle:     float, 
                         shuttle_size:      float,
                         shuttle_coords: np.array
                        ):
    """
    Evaluate and returns the coordinates of the engines' fires, relative to the current shuttle position (just for rendering)
    """
    # Recover the shuttle informations
    sin_angle:    float = np.sin(shuttle_angle)
    cos_angle:    float = np.cos(shuttle_angle)

    # Determine the fire offsets
    main_fire_angle_offset      = np.array([ (shuttle_size + 5)  *sin_angle, -(shuttle_size + 5)  *cos_angle ])
    secondary_fire_angle_offset = np.array([ (shuttle_size)      *sin_angle, -(shuttle_size)      *cos_angle ])
    secondary_fire_offset       = np.array([ (shuttle_size * 0.4)*cos_angle,  (shuttle_size * 0.4)*sin_angle ])

    # Lateral fires
    tr_fire_angle_offset     = np.array([ -(shuttle_size * 0.3)*sin_angle,  (shuttle_size * 0.3)*cos_angle ])
    tr_fire_offset           = np.array([  (shuttle_size * 0.7)*cos_angle,  (shuttle_size * 0.7)*sin_angle ])
    
    tl_fire_angle_offset     = np.array([ -(shuttle_size * 0.3)*sin_angle,  (shuttle_size * 0.3)*cos_angle ])
    tl_fire_offset           = np.array([ -(shuttle_size * 0.7)*cos_angle, -(shuttle_size * 0.7)*sin_angle ])
    
    br_fire_angle_offset     = np.array([  (shuttle_size * 0.3)*sin_angle, -(shuttle_size * 0.3)*cos_angle ])
    br_fire_offset           = np.array([  (shuttle_size * 0.7)*cos_angle,  (shuttle_size * 0.7)*sin_angle ])
    
    bl_fire_angle_offset     = np.array([  (shuttle_size * 0.3)*sin_angle, -(shuttle_size * 0.3)*cos_angle ])
    bl_fire_offset           = np.array([ -(shuttle_size * 0.7)*cos_angle, -(shuttle_size * 0.7)*sin_angle ])

    # Determine the fire coordinates
    fire_main_coords         = shuttle_coords + main_fire_angle_offset
    fire_left_coords         = shuttle_coords + secondary_fire_angle_offset + secondary_fire_offset
    fire_right_coords        = shuttle_coords + secondary_fire_angle_offset - secondary_fire_offset

    fire_bottom_right_coords = shuttle_coords + br_fire_angle_offset + br_fire_offset
    fire_bottom_left_coords  = shuttle_coords + bl_fire_angle_offset + bl_fire_offset
    fire_top_left_coords     = shuttle_coords + tl_fire_angle_offset + tl_fire_offset
    fire_top_right_coords    = shuttle_coords + tr_fire_angle_offset + tr_fire_offset

    return fire_main_coords, fire_left_coords, fire_right_coords, \
           fire_bottom_right_coords, fire_bottom_left_coords, fire_top_left_coords, fire_top_right_coords


def rotate(shuttle_angle: float, earth_angle: float, moon_angle: float, flag_angle: float,
           shuttle_size: float, earth_size: float, moon_size: float, flag_size: float, fire_size: float,
           shuttle: Image, earth: Image, moon: Image, flag: Image, fire: Image, explosion: Image
          ):
    """
    Function to rotate the images of a specified angle (just for rendering)
    """
    # Rotate the images
    earth_rotated      = earth  .rotate(earth_angle,         expand=True)
    moon_rotated       = moon   .rotate(moon_angle,          expand=True)
    flag_rotated       = flag   .rotate(flag_angle,          expand=True)
    shuttle_rotated    = shuttle.rotate(shuttle_angle,       expand=True)
    fire_main_rotated  = fire   .rotate(shuttle_angle + 180, expand=True)
    fire_left_rotated  = fire   .rotate(shuttle_angle + 90,  expand=True)
    fire_right_rotated = fire   .rotate(shuttle_angle - 90,  expand=True)

    # Convert the images to numpy array for easier management
    earth_rotated      = np.array(earth_rotated)
    moon_rotated       = np.array(moon_rotated)
    flag_rotated       = np.array(flag_rotated)
    shuttle_rotated    = np.array(shuttle_rotated)
    fire_main_rotated  = np.array(fire_main_rotated)
    fire_left_rotated  = np.array(fire_left_rotated)
    fire_right_rotated = np.array(fire_right_rotated)

    # Resize the images
    earth           = OffsetImage(earth_rotated,      zoom=earth_size)
    moon            = OffsetImage(moon_rotated,       zoom=moon_size)
    flag            = OffsetImage(flag_rotated,       zoom=flag_size)
    shuttle         = OffsetImage(shuttle_rotated,    zoom=shuttle_size)
    fire_main       = OffsetImage(fire_main_rotated,  zoom=fire_size)
    fire_secondary  = OffsetImage(fire_main_rotated,  zoom=fire_size * 0.6)
    explosion       = OffsetImage(explosion,          zoom=shuttle_size)
    fire_left       = OffsetImage(fire_left_rotated,  zoom=fire_size * 0.7)
    fire_right      = OffsetImage(fire_right_rotated, zoom=fire_size * 0.7)

    return earth, moon, flag, shuttle, fire_main, fire_secondary, fire_left, fire_right, explosion

def get_boxes(shuttle_angle: float, earth_angle: float, moon_angle: float, flag_angle: float,
              shuttle_size: float, earth_size: float, moon_size: float, flag_size: float, fire_size: float,
              shuttle: Image, earth: Image, moon: Image, flag: Image, fire: Image, explosion: Image,
              shuttle_coords: tuple, earth_coords: tuple, moon_coords: tuple, flag_coords: tuple,
             ):
    """
    Function to generate the images to add to the render
    """
    # Rotate the images
    earth, moon, flag, shuttle, fire_main, fire_secondary, fire_left, fire_right, explosion = rotate(shuttle_angle, earth_angle, moon_angle, flag_angle,
                                                                                                     shuttle_size, earth_size, moon_size, flag_size, fire_size,
                                                                                                     shuttle, earth, moon, flag, fire, explosion
                                                                                                    )
    # Evaluate the fire coordinates
    fire_main_coords, fire_left_coords, fire_right_coords, \
    fire_br_coords, fire_bl_coords, fire_tl_coords, fire_tr_coords = evaluate_fire_coords(shuttle_angle*np.pi/180, shuttle_size*512, shuttle_coords)
    
    earth_img             = AnnotationBbox(earth,          earth_coords,      frameon=False)
    moon_img              = AnnotationBbox(moon,           moon_coords,       frameon=False)
    flag_img              = AnnotationBbox(flag,           flag_coords,       frameon=False)
    shuttle_img           = AnnotationBbox(shuttle,        shuttle_coords,    frameon=False)
    fire_main_img         = AnnotationBbox(fire_main,      fire_main_coords,  frameon=False)
    fire_left_img         = AnnotationBbox(fire_secondary, fire_left_coords,  frameon=False)
    fire_right_img        = AnnotationBbox(fire_secondary, fire_right_coords, frameon=False)
    fire_bottom_right_img = AnnotationBbox(fire_right,     fire_br_coords,    frameon=False)
    fire_bottom_left_img  = AnnotationBbox(fire_left,      fire_bl_coords,    frameon=False)
    fire_top_left_img     = AnnotationBbox(fire_left,      fire_tl_coords,    frameon=False)
    fire_top_right_img    = AnnotationBbox(fire_right,     fire_tr_coords,    frameon=False)
    explosion_img         = AnnotationBbox(explosion,      shuttle_coords,    frameon=False)

    return earth_img, moon_img, flag_img, shuttle_img,   \
           fire_main_img, fire_left_img, fire_right_img, \
           fire_bottom_right_img, fire_bottom_left_img, fire_top_left_img, fire_top_right_img, explosion_img

def get_reward(flag_distance:         float, 
               speed_module:          float, 
               shuttle_angle:         float, 
               flag_angle:            float, 
               shuttle_angular_speed: float,
               is_landed:              bool, 
               is_crashed:             bool
              ):
    """
    Function to evaluate the rewards obtained from the agent after an action
    """
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
    
    # Give a reward if the angle of the shuttle is correct for the landing
    reward += 15 if flag_distance < 150 and abs(shuttle_angle % 360 - flag_angle % 360) < 25 else 0
    
    # Give a reward if the shuttle angular speed is small
    
    # Give a big reward if the shuttle has landed
    reward += 1e6 if is_landed() else 0
    
    # Give a big penalty if the shuttle has crashed near the earth
    reward -= 1e6 if is_crashed() and flag_distance > 500 else 0
    
    # Give a relative big penalty if the shuttle has crashed near the moon but away from the flag
    reward -= 1e3 if is_crashed() and flag_distance < 500 and flag_distance > 200 else 0
    reward = reward if abs(shuttle_angular_speed) < 1.5 else 0

    return reward
