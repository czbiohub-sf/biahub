import numpy as np

def compute_rigid_body_rotation_matrix_3d(z_angle, y_angle, x_angle):

    '''
    computes the rotation matrix for rotating about the z-, y- and then x-axis.
    
    input(s):
    ---------
    z_angle: float value of angle to rotate about the z-axis. must be in radians.
    y_angle: float value of angle to rotate about the y-axis. must be in radians
    x_angle: must be in radians
    
    return(s):
    ----------
    rot_mat: rotation matrix with shape (3,3). datatype = np.float64
    
    '''

    ## terms are long so assigned individually so that it is more legible
    ## (could be skipped in future if wanted)
    mat_elem_a0 = np.cos(x_angle)*np.cos(y_angle)
    mat_elem_a1 = (-1*np.sin(x_angle)*np.cos(y_angle)) - (np.cos(x_angle)*np.sin(y_angle)*np.sin(z_angle))
    mat_elem_a2 = (  np.sin(x_angle)*np.sin(z_angle)) - (np.cos(x_angle)*np.sin(y_angle)*np.cos(z_angle))
    mat_elem_b0 = np.sin(x_angle)*np.cos(y_angle)
    mat_elem_b1 = (   np.cos(x_angle)*np.cos(z_angle)) + (np.sin(x_angle)*np.sin(y_angle)*np.sin(z_angle))
    mat_elem_b2 = (-1*np.cos(x_angle)*np.sin(z_angle)) + (np.sin(x_angle)*np.sin(y_angle)*np.cos(z_angle))
    mat_elem_c0 = -1*np.sin(y_angle)
    mat_elem_c1 = np.cos(y_angle)*np.sin(z_angle)
    mat_elem_c2 = np.cos(y_angle)*np.cos(z_angle)

    ## fill matrix with terms
    rot_mat = np.array(
        [
            [mat_elem_a0, mat_elem_a1, mat_elem_a2],
            [mat_elem_b0, mat_elem_b1, mat_elem_b2],
            [mat_elem_c0, mat_elem_c1, mat_elem_c2]
        ]
    )
    
    return rot_mat