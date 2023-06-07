import numpy as np
from typing import List, Tuple
import cv2

from cv2 import cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

'''
Please do Not change or add any imports. 
'''

#task1

def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles along x, y and z axis respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from xyz to XYZ.

    '''
    rot_xyz2XYZ = np.eye(3).astype(float)
    # Your implementation

    #convert degree to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    #convert rotation matrix from xyz to XYZ
    rotation1 = rotZ_Axis(alpha)
    rotation2 = rotX_Axis(beta)
    rotation3 = rotZ_Axis(gamma)

    rot_xyz2XYZ = np.dot(rotation3,np.dot(rotation2,rotation1))

    return rot_xyz2XYZ


def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles of the 3 step respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from XYZ to xyz.

    '''
    rot_XYZ2xyz = np.eye(3).astype(float)

    # Your implementation

    #convert degree to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    #convert rotation matrix from xyz to XYZ
    rotation1 = rotZ_Axis(-alpha)
    rotation2 = rotX_Axis(-beta)
    rotation3 = rotZ_Axis(-gamma)

    rot_XYZ2xyz = np.dot(rotation1,np.dot(rotation2,rotation3))
    
    return rot_XYZ2xyz


"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above "findRot_xyz2XYZ()" and "findRot_XYZ2xyz()" functions are the only 2 function that will be called in task1.py.
"""

# Your functions for task1

# Rotation around Z-axis
def rotZ_Axis(theta: float):
    return np.array([[np.cos(theta), -np.sin(theta), 0 ], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

# Rotation around X-axis
def rotX_Axis(theta: float):
	return np.array([[1,0,0], [0, np.cos(theta), -np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])




#--------------------------------------------------------------------------------------------------------------
# task2:

def find_corner_img_coord(image: np.ndarray) -> np.ndarray:
    '''
    Args: 
        image: Input image of size MxNx3. M is the height of the image. N is the width of the image. 3 is the channel of the image.
    Return:
        A numpy array of size 32x2 that represents the 32 checkerboard corners' pixel coordinates. 
        The pixel coordinate is defined such that the of top-left corner is (0, 0) and the bottom-right corner of the image is (N, M). 
    '''
    img_coord = np.zeros([32, 2], dtype=float)

    # Your implementation
    
    chessboard_size = (4,9)

    image_gray = cvtColor(image, COLOR_BGR2GRAY)

    image_shape = image.shape
    height = image_shape[0]
    width = image_shape[1]

    found, corners = findChessboardCorners(image_gray,chessboard_size,None)

    criteria = (TERM_CRITERIA_EPS+TERM_CRITERIA_MAX_ITER,100,0.01)
    corners = cornerSubPix(image_gray,corners, (11,11), (height,width), criteria)

    corners = np.delete(corners,[16,17,18,19],axis = 0)

    img_coord = corners

    return img_coord


def find_corner_world_coord(img_coord: np.ndarray) -> np.ndarray:
    '''
    You can output the world coord manually or through some algorithms you design. Your output should be the same order with img_coord.
    Args: 
        img_coord: The image coordinate of the corners. Note that you do not required to use this as input, 
        as long as your output is in the same order with img_coord.
    Return:
        A numpy array of size 32x3 that represents the 32 checkerboard corners' pixel coordinates. 
        The world coordinate or each point should be in form of (x, y, z). 
        The axis of the world coordinate system are given in the image. The output results should be in milimeters.
    '''
    world_coord = np.zeros([32, 3], dtype=float)

    # Your implementation
    world_coord=np.array([[40,0,40],[40,0,30],[40,0,20],[40,0,10],
    [30,0,40], [30,0,30],[30,0,20],[30,0,10],
    [20,0,40], [20,0,30],[20,0,20],[20,0,10],
    [10,0,40], [10,0,30],[10,0,20],[10,0,10],
    [0,10,40], [0,10,30],[0,10,20],[0,10,10],
    [0,20,40], [0,20,30],[0,20,20],[0,20,10],
    [0,30,40], [0,30,30],[0,30,20],[0,30,10],
    [0,40,40],[0,40,30],[0,40,20],[0,40,10]])
    

    return world_coord


def find_intrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[float, float, float, float]:
    '''
    Use the image coordinates and world coordinates of the 32 point to calculate the intrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        fx, fy: Focal length. 
        (cx, cy): Principal point of the camera (in pixel coordinate).
    '''

    fx: float = 0
    fy: float = 0
    cx: float = 0
    cy: float = 0

    # Your implementation

    img_coord = np.reshape(img_coord,(32,2))

    length = len(img_coord)
    A = np.zeros((2*length, 12))
    
    # print(img_coord)
    for i in range(length):
        # image coordinates
        x, y = img_coord[i]  
        # world coordinates
        X, Y, Z = world_coord[i] 

        # 1 pair of rows in M for one pair of img and world coordinates
        var1 = np.array([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x])
        var2 = np.array([0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y])
        A[2*i] = var1
        A[(2*i)+1] = var2
    
    u,s,v = np.linalg.svd(A)
    v_shape = v.shape
    last_row = v[v_shape[1] - 1]

    last_row_3ele = last_row[8:11]

    lambdaI = 1/np.sqrt(np.sum(last_row_3ele**2))

    global M 
    
    M = lambdaI*last_row

    m1 = [M[0], M[1], M[2]]
    m2 = [M[4], M[5], M[6]]
    m3 = [M[8], M[9], M[10]]

    cx = np.sum([np.dot(m1[0],m3[0]),np.dot(m1[1],m3[1]),np.dot(m1[2],m3[2])])
    cy = np.sum([np.dot(m2[0],m3[0]),np.dot(m2[1],m3[1]),np.dot(m2[2],m3[2])])

    print(cx,cy)

    fx = np.sqrt(np.sum([np.dot(m1[0],m1[0]),np.dot(m1[1],m1[1]),np.dot(m1[2],m1[2])]) - np.square(cx))
    fy = np.sqrt(np.sum([np.dot(m2[0],m2[0]), np.dot(m2[1],m2[1]), np.dot(m2[2],m2[2])]) - np.square(cy))

    print(fx,fy)

    return fx, fy, cx, cy

def find_extrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Use the image coordinates, world coordinates of the 32 point and the intrinsic parameters to calculate the extrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        R: The rotation matrix of the extrinsic parameters. It is a 3x3 numpy array.
        T: The translation matrix of the extrinsic parameters. It is a 1-dimensional numpy array with length of 3.
    '''

    R = np.eye(3).astype(float)
    T = np.zeros(3, dtype=float)


    # Your implementation

    fx, fy, cx, cy = find_intrinsic(img_coord,world_coord)

    I_matrix = np.zeros((3, 3))
    I_matrix[0][0] = fx
    I_matrix[0][2] = cx
    I_matrix[1][1] = fy
    I_matrix[1][2] = cy
    I_matrix[2][2] = 1

    print(I_matrix)
    
    E_matrix = np.dot(np.linalg.inv(I_matrix),M.reshape(3,4))

    # print(E_matrix)

    R = E_matrix[0:3,0:3]
    T = E_matrix[0:3,3:]

    print(M)
    print(R)
    print(T)
    return R, T


"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above 4 functions are the only ones that will be called in task2.py.
"""

# Your functions for task2







#---------------------------------------------------------------------------------------------------------------------