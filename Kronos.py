#Import our libraries
import numpy as np
import matplotlib.pyplot as plt

"""# Defining Functions

We define our own functions that we have written and will be utilizing throughout the code.
"""

#Runge-Kutta Method
def rk4(u_k, delta_t):
    k1 = delta_t*f_true(u_k)
    k2 = delta_t*f_true(u_k+k1/2)
    k3 = delta_t*f_true(u_k+k2/2)
    k4 = delta_t*f_true(u_k+k3)

    ukplus1 = u_k + (k1 + 2*k2 + 2*k3 + k4)/6
    return ukplus1

#Find the evolution of our system over time using the RK4 Method
def ivp_rk4(u_0, T, delta_t):
    times = []
    u = []
    num_elements = int(T / delta_t) + 1
    sol_r = []
    sol_time = 0
    threshold = np.radians(10)
    first = False
    for t_k in np.linspace(0,T,num_elements):
        r = u_0[0]
        dist = []
        N = r.shape[0]
        if first != True:
            for i in range(1,N):
                dist.append(r[i]-r[0])
            for i in range(1, N-1):
                cross = np.cross(dist[i],dist[i-1])/(np.linalg.norm(dist[i])*np.linalg.norm(dist[i-1]))
                theta = np.arcsin(np.linalg.norm(cross))
                if theta > threshold or theta < -threshold:
                    break
                elif i == N-2:
                    sol_r = r[2]
                    sol_time = t_k
                    first = True
        times.append(t_k)
        u.append(u_0)
        u_k = rk4(u_0, delta_t)
        u_0 = u_k
    return np.array(u), times, sol_r, sol_time

#Our true function of our system
def f_true(u):
    r = u[0]
    v = u[1]
    N = r.shape[0]
    rddot = np.zeros((N,3))
    for i in range(0,N):
        for j in range(0,N):
            if i == j:
                continue
            rddot[i] += G*m[j]*(r[j]-r[i])/(np.linalg.norm(r[j]-r[i])**3)
    udot = np.zeros(u.shape)
    udot[0] = v
    udot[1] = rddot
    return udot

#Adams-Bashforth Four Step Method
def ab4(u_0,T,delta_t):
    k = int(T/delta_t) + 1
    times = []
    u = []
    u.append(u_0)
    u.append(rk4(u[0],delta_t))
    u.append(rk4(u[1],delta_t))
    u.append(rk4(u[2],delta_t))
    for i in range (4,k):
        u.append(u[i-1] + (delta_t/24) * (55*f_true(u[i-1]) - 59*f_true(u[i-2]) + 37*f_true(u[i-3]) - 9*f_true(u[i-4])))
    num_elements = int(T / delta_t) + 1
    sol_r = []
    sol_time = 0
    threshold = np.radians(10)
    first = False
    for t_k in np.linspace(0,T,num_elements):
        r = u_0[0]
        dist = []
        N = r.shape[0]
        if first != True:
            for i in range(1,N):
                dist.append(r[i]-r[0])
            for i in range(1, N-1):
                cross = np.cross(dist[i],dist[i-1])/(np.linalg.norm(dist[i])*np.linalg.norm(dist[i-1]))
                theta = np.arcsin(np.linalg.norm(cross))
                if theta > threshold or theta < -threshold:
                    break
                elif i == N-2:
                    sol_r = r[2]
                    sol_time = t_k
                    first = True
        times.append(t_k)
        u.append(u_0)
        u_k = rk4(u_0, delta_t)
        u_0 = u_k
    return np.array(u),times, sol_r, sol_time

"""# Variables

After we have defined our functions, we need to set establish our variables and their set values, along with some intiial conditions for our system.
"""

#Gravitational Constant
G = 6.674e-11


#Final Time and Time Steps
T, dt = 24*60*60*365*10, 60*30


#Masses and Orbital Radii of our Bodies
mSaturn = 568.32e24 # kg
mTitan = 1.345e23 # kg
rTitan = 1.22e9 # m
mDeathStar = 2.24e23 # m
mMimas = 3.75e19 # kg
rMimas = 189e6 # m
mHyperion = 5.55e18 # kg
rHyperion = 1.5e9 # m
mIapetus = 1.806e21 # kg
rIapetus = 3.56e9 #m
mFenrir = 1e13 # kg
rFenrir = 22.5e9 # m
m = np.array([mSaturn, mTitan, mDeathStar, mHyperion, mIapetus, mFenrir, mMimas]) # 

#Initializing our Radii of our Bodies
r0 = np.zeros((np.size(m),3))
r0[0] = [0.0, 0.0, 0.0]
r0[1] = [rTitan, 0.0,0.0]
r0[3] = [rHyperion, 0.0, 0.0]
r0[4] = [rIapetus, 0.0, 0.0]
r0[5] = [rFenrir, 0.0, 0.0]
r0[6] = [rMimas, 0.0, 0.0]
rDeathStar = np.max(r0)*2
r0[2] = [rDeathStar*np.sin(np.pi/4), rDeathStar*np.cos(np.pi/4), 0.0]

#Determining the Velocities of our Bodies
v0 = np.zeros((np.size(m),3))
v0[0] = [0.0, 0.0, 0.0]
v0[1] = [0.0, -np.sqrt(G*m[0]/np.linalg.norm(r0[1])), 0.0]
vDeathStar = np.sqrt(G*m[0]/np.linalg.norm(r0[2]))
v0[2] = [vDeathStar*np.cos(np.pi/4), -vDeathStar*np.sin(np.pi/4), 0.0]
v0[3] = [0.0, -np.sqrt(G*m[0]/np.linalg.norm(r0[3])), 0.0]
v0[4] = [0.0, -np.sqrt(G*m[0]/np.linalg.norm(r0[4])), 0.0]
v0[5] = [0.0, -np.sqrt(G*m[0]/np.linalg.norm(r0[5])), 0.0]
v0[6] = [0.0, -np.sqrt(G*m[0]/np.linalg.norm(r0[6])), 0.0]

"""# Finding our Positions

We utilize our numerical methods to iterate through our system and find the positions of our bodies, along with the point in which all of the bodies align collinearly.
"""

#Initial Positions and Velocities
u0 = np.array([r0, v0])


#Producing the Positions, Velocities and Times of the System using RK4
  #---RK4---#
u_rk, times, sol_r, sol_time = ivp_rk4(u0, T, dt)

  #---AB4---#
u_ab, times_ab, sol_r_ab, sol_time_ab = ab4(u0, T, dt)

#Splitting into Positions and Velocities
  #----RK4----#
ur = u_rk[:,0]
uv = u_rk[:,1]

  #----AB4----#
ur_ab = u_ab[:,0]
uv_ab = u_ab[:,1]

#Assign the Positions to their Respective Bodies
  #----RK4----#
r1 = []
r2 = []
r3 = []
r4 = []
r5 = []
r6 = []
r7 = []

  #----AB4----#
r1ab = []
r2ab = []
r3ab = []
r4ab = []
r5ab = []
r6ab = []
r7ab = []

#Finding and Appending the Positional Array that Correlates to Each Body
for i in range(ur.shape[0]):
    #----RK4----#
    r1.append(ur[i][0])
    r2.append(ur[i][1])
    r3.append(ur[i][2])
    r4.append(ur[i][3])
    r5.append(ur[i][4])
    r6.append(ur[i][5])
    r7.append(ur[i][6])

    #----AB4----#
    r1ab.append(ur_ab[i][0])
    r2ab.append(ur_ab[i][1])
    r3ab.append(ur_ab[i][2])
    r4ab.append(ur_ab[i][3])
    r5ab.append(ur_ab[i][4])
    r6ab.append(ur_ab[i][5])
    r7ab.append(ur_ab[i][6])

  #----RK4----#
r1 = np.array(r1)
r2 = np.array(r2)
r3 = np.array(r3)
r4 = np.array(r4)
r5 = np.array(r5)
r6 = np.array(r6)
r7 = np.array(r7)

  #----AB4----#
r1ab = np.array(r1ab)
r2ab = np.array(r2ab)
r3ab = np.array(r3ab)
r4ab = np.array(r4ab)
r5ab = np.array(r5ab)
r6ab = np.array(r6ab)
r7ab = np.array(r7ab)


#Print Solutions of our System
print(sol_r)
print(sol_time)
sol_r = np.array(sol_r)
sol = np.vstack((sol_r, np.zeros(sol_r.shape)))

sol_r_ab = np.array(sol_r_ab)
sol_ab = np.vstack((sol_r_ab, np.zeros(sol_r_ab.shape)))
print(sol_r_ab)
print(sol_time_ab)

"""# Simulation

We simulate our data through plotting the trajectory of our bodies and the our true solution.
"""

#Initializing our Graph
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax_ab = fig.add_subplot(1, 2, 2, projection='3d')


#Plotting the Positions of our Bodies
  #----RK4----#
ax.scatter(r1[:,0], r1[:,1], r1[:,2], color="maroon", s=2, label="Saturn")
ax.plot(r2[:,0], r2[:,1], r2[:,2], color="orangered", label="Titan")
ax.plot(r3[:,0], r3[:,1], r3[:,2], color="black", label="Death Star")
ax.plot(r4[:,0], r4[:,1], r4[:,2], color="goldenrod", label="Hyperion")
ax.plot(r5[:,0], r5[:,1], r5[:,2], color="darkolivegreen", label="Iapetus")
ax.plot(r6[:,0], r6[:,1], r6[:,2], color="darkblue", label="Fenrir")
ax.plot(r7[:,0], r7[:,1], r7[:,2], color="magenta", label="Mimas")

  #----AB4----#
ax_ab.scatter(r1ab[:,0], r1ab[:,1], r1ab[:,2], color="maroon", s=2, label="Saturn")
ax_ab.plot(r2ab[:,0], r2ab[:,1], r2ab[:,2], color="orangered", label="Titan")
ax_ab.plot(r3ab[:,0], r3ab[:,1], r3ab[:,2], color="black", label="Death Star")
ax_ab.plot(r4ab[:,0], r4ab[:,1], r4ab[:,2], color="goldenrod", label="Hyperion")
ax_ab.plot(r5ab[:,0], r5ab[:,1], r5ab[:,2], color="darkolivegreen", label="Iapetus")
ax_ab.plot(r6ab[:,0], r6ab[:,1], r6ab[:,2], color="darkblue", label="Fenrir")
ax_ab.plot(r7ab[:,0], r7ab[:,1], r7ab[:,2], color="magenta", label="Mimas")

#Plotting a Solution
fig.suptitle("Project Kronos")
  #----RK4----#
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('RK4')

  #----AB4----#
ax_ab.set_xlabel('X')
ax_ab.set_ylabel('Y')
ax_ab.set_zlabel('Z')
ax_ab.set_title('AB4')

plt.legend()
#Solution for both RK4 and AB4 Method
ax.plot(sol[:,0], sol[:,1], sol[:,2], color="lime", label="Super Laser")
ax_ab.plot(sol_ab[:,0], sol_ab[:,1], sol_ab[:,2], color="lime", label="Super Laser")

plt.legend()
# plt.show()

"""# Error Plots
Through iterating through different timestep values, we can generate a plot that showcases the magnitude of error for our different methods.
"""

# #Timestep List
dt_list = np.array([60*30,60*40,60*50,60*60,60*70,60*80])
delta_t = 60
#Set up our Error arrays and our Error Baselines
errk = np.zeros(6)
base = ivp_rk4(u0, T, delta_t)[0]
errab4 = np.zeros(6)
baseab4 = ab4(u0, T, delta_t)[0]
Kb = T/delta_t

# #Find the Error of our Methods
for i in range (0,6):
    K = T/dt_list[i]
    temp = ivp_rk4(u0, T, dt_list[i])[0]
    errk[i] = np.linalg.norm(temp[-1,:]-base[-1,:])/np.linalg.norm(base[-1,:])
    tempab = ab4(u0, T, dt_list[i])[0]
    errab4[i] = np.linalg.norm(tempab[-1,:]-baseab4[-1,:])/np.linalg.norm(baseab4[-1,:])

# #Plot the Errors
fig = plt.figure(figsize=(6, 4))
ax1=fig.add_subplot(111)
ax1.plot(dt_list,errk,'b-^', label = "rk4")
ax1.plot(dt_list,errab4,'r-^', label = "ab4")
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlabel('$\Delta$' + '$t$')
ax1.set_ylabel('Error')
ax1.set_title('Error for RK4 and AB4 model')
plt.legend()

"""# Stability Plots
We are able to model the regions of stability of our graphs, which is the area where the plots are unlikely to grow chaotically.
"""

#Define our function R which will be used to determine our Stability Area
def R(z):
    return 1 + z + (z**2)/2 + (z**3)/6 + (z**4)/24

x = np.linspace(-4, 4, 400)
y = np.linspace(-4, 4, 400)
X, Y = np.meshgrid(x, y)

axisbox = [-1.5, 1.5, -1, 1]
xa, xb, ya, yb = axisbox
npts = 50
theta = np.linspace(0, 2 * np.pi, 2 * npts + 1)
z = np.exp(1j * theta)
Z = X + 1j*Y
W = R(Z)

nu2 = (z**2 - z) / ((3 * z - 1) / 2)

nu3 = (z**3 - z**2) / ((5 - 16 * z + 23 * z**2) / 12)

nu4 = (z**4 - z**3) / ((55 * z**3 - 59 * z**2 + 37 * z - 9) / 24)

# nu5 = 1 + z + (z**2)/2 + (z**3)/6 + (z**4)/24

nu4_list = list(nu4)

#Finding the Area in which the Methods are Stable
for k in range(len(nu4_list) - 1):
    z_ = np.array([np.real(nu4_list), np.imag(nu4_list)])
    iloop = []
    for j in range(k + 2, len(nu4_list) - 1):
        lam = np.linalg.inv(np.column_stack((z_[:, k] - z_[:, k + 1], z_[:, j + 1] - z_[:, j]))) @ (z_[:, j + 1] - z_[:, k + 1])
        if np.all(lam >= 0) and np.all(lam <= 1):
            iloop = list(range(k + 1, j + 1))
            zint = lam[0] * z_[:, k] + (1 - lam[0]) * z_[:, k + 1]
            break
    if iloop:
        zcp = complex(zint[0], zint[1])
        nu4_list[iloop[0]] = zcp
        for index in reversed(iloop[1:]):
            del nu4_list[index]

nu4 = np.array(nu4_list)


#Plot our Stability Graphs
plt.figure(figsize=(8, 6))
plt.plot(np.real(nu2), np.imag(nu2), 'g-', linewidth=2)
plt.fill_between(np.real(nu2), np.imag(nu2), color='green', label = 'AB2')
plt.plot(np.real(nu3), np.imag(nu3), 'b-', linewidth=2)
plt.fill_between(np.real(nu3), np.imag(nu3), color='blue', label = 'AB3')
plt.plot(np.real(nu4), np.imag(nu4), 'r-', linewidth=2)
plt.fill_between(np.real(nu4), np.imag(nu4), color='red', label = 'AB4')
C = plt.contourf(X, Y, np.abs(W), levels=[0, 1], colors=['yellow', 'yellow'],alpha = 0.3)
plt.contour(X, Y, np.abs(W), levels=[1], colors='yellow')

handles_fill, labels_fill = plt.gca().get_legend_handles_labels()

handle_RK4 = [plt.Line2D([0], [0], color='yellow', lw=4)]
label_RK4 = ['RK4']
handles_all = handles_fill + handle_RK4
labels_all = labels_fill + label_RK4

plt.legend(handles_all, labels_all, loc='upper left')

plt.plot([xa, xb], [0, 0], 'k-', linewidth=2)
plt.plot([0, 0], [ya, yb], 'k-', linewidth=2)
plt.title('Region of absolute stability')
plt.xlabel('Re(hλ)')
plt.ylabel('Im(hλ)')
plt.xlim(-4,0.5)
plt.ylim(-3,3)
plt.grid(True)
plt.show()




