import numpy as np

#Ix = 0.2*M*(b**2+c**2)
#Iy = 0.2*M*(a**2+c**2)
#Iz = 0.2*M*(b**2+a**2)
    #when main is executed the mass is the mass as def in environment
#mass = 1000
    #w_i should be sampled from w0 to wmax; are functions of time in original article

#I = np.array([Ix,Iy,Iz])

#vx0 = 0.3
#vy0 = 0.3
#vz0 = 0.3

#m0 = 450 #initial mass of spacecraft

#E = 1/2*(Ix*wx**2+Iy*wy**2+Iz*wz**2)

def EoM(position,velocity,gravity_Accel,ellipsoid_Angular_Velocity,action_Taken):
    x = position[0]
    y = position[1]
    z = position[2]
    vx = velocity[0]
    vy = velocity[1]
    vz = velocity [2]
    gx = gravity_Accel[0]
    gy = gravity_Accel[1]
    gz = gravity_Accel[2]
    wx = ellipsoid_Angular_Velocity[0]
    wy = ellipsoid_Angular_Velocity[1]
    wz = ellipsoid_Angular_Velocity[2]
    Tx = action_Taken[0]
    Ty = action_Taken[1]
    Tz = action_Taken[2]
    m0 = 450.0

    accelaration_x = -2*(wy*vz - vy*wz) - y*wx*wy + x*wx**2 + x*wz**2 - z*wx*wz + gx + Tx/m0 
    accelaration_y = -2*(wz*vx - wx*vz) - z*wz*wy + y*wz**2 + y*wx**2 - x*wy*wx + gy + Ty/m0
    accelaration_z = -2*(wx*vy - vx*wy) - x*wz*wx + z*wx**2 + z*wy**2 - y*wz*wy + gz + Tz/m0 
    accelaration = np.array([accelaration_x,accelaration_y,accelaration_z])
    #print(gx,gy,gz)
    return accelaration
