## define required functions for solid body transformation 
## (rotation followed by translation)

##------------------------------------------------------------------------------

def measure_z_axis_rotation_angle(a,b):
    
    # remove z-axis component of bead point arrays
    a_xy = a[:,1:]
    b_xy = b[:,1:]

    # measure angles between corresponding point pairs of a & b (measurements in radians)
    angles = np.zeros((a_xy.shape[0],1))
    for i in range(a.shape[0]):
    
        a_pt = Point(a_xy[i,...])
        b_pt = Point(b_xy[i,...])

        a_line = Line(Point(0,0), a_pt)
        b_line = Line(Point(0,0), b_pt)

        angles[i,0] = float((N(a_line.angle_between(b_line))))

    return angles

##------------------------------------------------------------------------------

def rotate_around_z_axis(image_coordinates, angle):

    # this is usually the x_rot_mat, but axes swap in python requires reordering
    z_rot_mat = np.array([[1,             0,              0],
                          [0, np.cos(angle), -np.sin(angle)],
                          [0, np.sin(angle),  np.cos(angle)]])

    # use @ for matrix multiplication
    rotated_coordinates =  image_coordinates @ z_rot_mat 
    
    return(rotated_coordinates)

##------------------------------------------------------------------------------

def measure_y_axis_rotation_angle(a,b):

    # remove y-axis component of bead point arrays
    a_zx = np.delete(a, 1, 1)
    b_zx = np.delete(b, 1, 1)
    
    # measure angles between corresponding point pairs of a & b
    angles = np.zeros((a_zx.shape[0],1))
    for i in range(4):
    
        a_pt = Point(a_zx[i,...])
        b_pt = Point(b_zx[i,...])

        a_line = Line(Point(0,0), a_pt)
        b_line = Line(Point(0,0), b_pt)

        angles[i,0] = float((N(a_line.angle_between(b_line))))
    
    return angles

##------------------------------------------------------------------------------

def rotate_around_y_axis(image_coordinates, angle):

    y_rot_mat = np.array([[np.cos(angle), 0, -np.sin(angle)],
                          [            0, 1,              0],
                          [np.sin(angle), 0,  np.cos(angle)]])

    rotated_coordinates =  image_coordinates @ y_rot_mat

    #print(a,b)

    return(rotated_coordinates)

##------------------------------------------------------------------------------

def measure_x_axis_rotation_angle(a,b):

    # remove z-axis component of bead point arrays
    a_zy = a[:,:-1]
    b_zy = b[:,:-1]

    # measure angles between corresponding point pairs of a & b
    angles = np.zeros((a_zy.shape[0],1))
    for i in range(4):
    
        a_pt = Point(a_zy[i,...])
        b_pt = Point(b_zy[i,...])

        a_line = Line(Point(0,0), a_pt)
        b_line = Line(Point(0,0), b_pt)

        angles[i,0] = float((N(a_line.angle_between(b_line))))
    
    return angles

##------------------------------------------------------------------------------

def rotate_around_x_axis(image_coordinates, angle):

    # this is usually the z_rot_mat, but axes swap in python requires reordering
    x_rot_mat = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle),  np.cos(angle), 0],
                          [            0,              0, 1]])

    rotated_coordinates = image_coordinates @ x_rot_mat

    return rotated_coordinates