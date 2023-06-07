**Camera Calibration using Chessboard Corners**

This project provides Python code for camera calibration using chessboard corners. The calibration process involves estimating the intrinsic and extrinsic parameters of a camera, which are necessary for correcting lens distortions and obtaining accurate measurements in computer vision applications.

**Prerequisites**

Before running the code, ensure that you have the following:

- Python installed on your system
- OpenCV library installed (can be installed via pip install opencv-python)

**Usage**

- Capture Calibration Images: 
  Take multiple images of a chessboard pattern using the camera you want to calibrate. Make sure the chessboard is fully visible in each image and covers a significant portion of the frame.
- Implement the Required Functions: 
  Open the Python file containing the functions for camera calibration. You will find two tasks in the code: task 1 and task 2.
- Task 1: 
  In this task, you need to implement two functions: findRot_xyz2XYZ and findRot_XYZ2xyz. These functions calculate the rotation matrices between xyz and XYZ coordinate systems. Implement the rotation calculations as per the provided instructions.
- Task 2: 
  In this task, you need to implement four functions: find_corner_img_coord, find_corner_world_coord, find_intrinsic, and find_extrinsic.
- find_corner_img_coord: 
  This function takes an image as input and returns the pixel coordinates of the 32 chessboard corners in the image.
- find_corner_world_coord: 
  This function returns the world coordinates (x, y, z) of the 32 chessboard corners. You can manually define the coordinates or design your own algorithm to calculate them.
- find_intrinsic: 
  This function uses the image and world coordinates to calculate the intrinsic parameters of the camera, including the focal length and principal point.
- find_extrinsic: 
  This function utilizes the image and world coordinates along with the intrinsic parameters to compute the extrinsic parameters, which include the rotation matrix and translation vector.
- Run the Code: 
  Once you have implemented the required functions, save the Python file and execute it. The code will run the camera calibration process using the provided chessboard images and compute the intrinsic and extrinsic parameters.
- Calibration Results: 
  The code will display the calculated intrinsic parameters (focal length, principal point) and the extrinsic parameters (rotation matrix, translation vector). These results represent the properties of your camera and can be used for subsequent image undistortion or other computer vision tasks.
