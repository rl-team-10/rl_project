from environ import Environ3D #import environment
import random
import matplotlib.pyplot as plt
import numpy as np
def main():
    seed = 3
    np.random.seed(seed)
    NUMBER_OF_STEPS = 100000
    x = []
    y = []
    z = []
    environment=Environ3D(seed)
    environment.reset()

    for step in range(NUMBER_OF_STEPS):
        done = False

       #implement action-taking policy - for now it's random:
        action = random.randint(0,5)

       #advance the environment
        current_State, reward, done = environment.step(action)
       #print(environment.ellipsoid_Angular_Velocity)
       #print(ellipsoid_Angular_Velocity[1]*velocity[2] - velocity[1]*ellipsoid_Angular_Velocity[2],position[1]*ellipsoid_Angular_Velocity[0]*ellipsoid_Angular_Velocity[1],position[0]*ellipsoid_Angular_Velocity[0]**2,position[0]*ellipsoid_Angular_Velocity[2]**2,position[2]*ellipsoid_Angular_Velocity[0]*ellipsoid_Angular_Velocity[2],gravity_Accel[0],action_Taken[0]/m0)
       #save x,y,z coordinates for every step for visualisation
        x.append(current_State[0][0])
        y.append(current_State[0][1])
        z.append(current_State[0][2])
        if done:
            print("Steps taken", step)
            break
    ###########################################################################
    #mathplotlib shit (plots 3D now!)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.grid(True)
    ax.plot(x,y,z, marker='.', markevery=5)
    ax.plot(environment.target_Point[0],environment.target_Point[1], '-xr')
   #trying to plot elipsoid
    #coefs = (environment.ellipsoid_Semiaxis_a, environment.ellipsoid_Semiaxis_b,
    #        environment.ellipsoid_Semiaxis_c)
    coefs = (1/environment.ellipsoid_Semiaxis_a**2,
            1/environment.ellipsoid_Semiaxis_b**2,
            1/environment.ellipsoid_Semiaxis_c**2)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1
   #Radii corresponding to the coefficients:
    rx, ry, rz = 1/np.sqrt(coefs)
   #Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
   # Cartesian coordinates that correspond to the spherical angles:
   # (this is the equation of an ellipsoid):
    xe = rx * np.outer(np.cos(u), np.sin(v))
    ye = ry * np.outer(np.sin(u), np.sin(v))
    ze = rz * np.outer(np.ones_like(u), np.cos(v))
    # Plot:
    ax.plot_surface(xe, ye, ze,  rstride=2, cstride=4, color='grey')
    plt.show()
    ###########################################################################

    #constructor for main
if __name__=="__main__":
    main()
