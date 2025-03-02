import cv2
import numpy as np

def kalman_filter():
    """
    The function defines a Kalman filter. In each sample we will get three values: the center of the ball(x,y) and the radius of the ball.
    The filter will track eight variables:
    """
    kalman = cv2.KalmanFilter(8, 3)     # define Kalman filter, in each sample we recieved :x,y,r. follow 8 parameter:x,y,r,Vx,Vy,Vr,Ax,Ay

    # measurment matrik (x,y,r)
    kalman.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],  # x
        [0, 1, 0, 0, 0, 0, 0, 0],  # y
        [0, 0, 1, 0, 0, 0, 0, 0]   # r
    ], dtype=np.float32)

    
    dt = 1/30  # sample rate

    kalman.transitionMatrix = np.array([
        [1, 0, 0, dt, 0, 0, 0.5 * dt**2, 0],  # x' = x + Vx*dt + 0.5Ax*dt^2
        [0, 1, 0, 0, dt, 0, 0, 0.5 * dt**2],  # y' = y + Vy*dt + 0.5Ay*dt^2
        [0, 0, 1, 0, 0, dt, 0, 0],            # r' = r + Vr*dt
        [0, 0, 0, 1, 0, 0, dt, 0],            # Vx' = Vx + Ax*dt
        [0, 0, 0, 0, 1, 0, 0, dt],            # Vy' = Vy + Ay*dt
        [0, 0, 0, 0, 0, 1, 0, 0],             # Vr' = Vr
        [0, 0, 0, 0, 0, 0, 1, 0],             # Ax
        [0, 0, 0, 0, 0, 0, 0, 1]              # Ay 
    ], dtype=np.float32)

    # The process noise matrix represents the uncertainty in the process, that is, unexpected changes such as non-linear fluctuations or external disturbances
    # High value cause more weight on the measurment
    kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-1  #  8×8

    # The measurement noise matrix is ​​a matrix that represents the level of noise in the measurements.
    # The smaller the noise, the more the filter will rely on the measurement
    kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-4  #  3×3

    # The error matrix of the state after the update.
    # It represents how confident the filter is in the current values ​​of the state (statePost).
    kalman.errorCovPost = np.eye(8, dtype=np.float32)  # 8×8

    # It represents the position, speed or any other variable we are tracking, after we have weighted the new measurement
    kalman.statePost = np.zeros((8, 1), dtype=np.float32)

    return kalman

def Kalman_update(KalmanFilter,center,radius):
    """
        Input: measurmant(center and radius of the ball)
    """
    KalmanFilter.statePost[:3] = np.array([[center[0]], [center[1]], [radius]], dtype=np.float32)
    yellow_measured = np.array([[np.float32(center[0])],
                                [np.float32(center[1])],
                                [np.float32(radius)]])
    
 
    KalmanFilter.correct(yellow_measured)
    predicted = KalmanFilter.predict()      # Get the predicted place in the next frame
    return predicted

def IsBallIsBack(predicted,last_vx,last_vy):
    current_vx = predicted[3][0]  
    current_vy = predicted[4][0]  
    IsChange= (current_vy*last_vy<-400)
    last_vx=current_vx
    last_vy=current_vy
    return IsChange,last_vx,last_vy