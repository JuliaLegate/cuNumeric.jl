# python equivalent of gray-scott.jl to test the GC problem

import cupynumeric as np


# import matplotlib.animation as animation
# from IPython.display import HTML
# import matplotlib.pyplot as plt

def greyScottSys(u, v, dx, dt, c_u, c_v, f, k):
    # u,v are arrays
    # dx,dt are space and time steps
    # c_u, c_v, f, k are constant paramaters

     #create new u array
    u_new = np.zeros_like(u)
    v_new = np.zeros_like(v)

    #calculate F_u and F_v functions
    F_u = (-u[1:-1,1:-1]*(v[1:-1,1:-1]**2)) + f*(1-u[1:-1,1:-1])
    F_v = (u[1:-1,1:-1]*(v[1:-1,1:-1]**2)) - (f+k)*v[1:-1,1:-1]

    # 2-D Laplacian of f using array slicing, excluding boundaries
    # For an N x N array f, f_lap is the N-1 x N-1 array in the "middle"
    u_lap = (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1]) / dx**2\
            + (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2]) / dx**2
    v_lap = (v[2:,1:-1] - 2*v[1:-1,1:-1] + v[:-2,1:-1]) / dx**2\
            + (v[1:-1,2:] - 2*v[1:-1,1:-1] + v[1:-1,:-2]) / dx**2

    # Forward-Euler time step for all points except the boundaries
    u_new[1:-1,1:-1] = ((c_u * u_lap) + F_u)*dt + u[1:-1,1:-1]
    v_new[1:-1,1:-1] = ((c_v * v_lap) + F_v)*dt + v[1:-1,1:-1]

    # Apply periodic boundary conditions
    u_new[:,0] = u[:,-2]
    u_new[:,-1] = u[:,1]
    u_new[0,:] = u[-2,:]
    u_new[-1,:] = u[1,:]
    v_new[:,0] = v[:,-2]
    v_new[:,-1] = v[:,1]
    v_new[0,:] = v[-2,:]
    v_new[-1,:] = v[1,:]

    return u_new, v_new



# initial conditions and discretizaiton
dx = 1
dt = dx/5
u = np.ones((4000,4000))
v = np.zeros((4000,4000))
u[:150,:150] = np.random.rand(150,150)
v[:150,:150] = np.random.rand(150,150)


# fig = plt.figure()

c_u = 1
c_v = 0.3
f = 0.03
k = 0.06

# t_final = 1000

# ims = []
n_steps = 100 # number of steps to take
frame_interval = 200 # steps to take between making plots

# build a list of images
for n in range(n_steps) :

    ## This may need to be changed.
    u,v = greyScottSys(u, v, dx, dt, c_u, c_v, f, k)

    # ## Store frames when n is a multiple of frame_interval
    # if n%frame_interval == 0:
    #     im = plt.imshow(u, vmin=0, vmax=1) # Show a plot of u.
    #     ims.append([im]) # append single image to the list of images

# anim = animation.ArtistAnimation(fig, ims, interval=100, repeat=False)
# HTML(anim.to_jshtml())
