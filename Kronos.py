import numpy as np
import matplotlib.pyplot as plt

def rk4(u_k, delta_t):
    k1 = delta_t*f_true(u_k)
    k2 = delta_t*f_true(u_k+k1/2)
    k3 = delta_t*f_true(u_k+k2/2)
    k4 = delta_t*f_true(u_k+k3)

    ukplus1 = u_k + (k1 + 2*k2 + 2*k3 + k4)/6
    return ukplus1

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

G = 6.674e-11
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

def ab4(u_0,T,delta_t):
    k = int(T/delta_t) + 1
    times = np.linspace(0, T, k)
    u = []
    u.append(u_0)
    u.append(rk4(u[0],delta_t))
    u.append(rk4(u[1],delta_t))
    u.append(rk4(u[2],delta_t))
    for i in range (4,k):
        u.append(u[i-1] + (delta_t/24) * (55*f_true(u[i-1]) - 59*f_true(u[i-2]) + 37*f_true(u[i-3]) - 9*f_true(u[i-4])))
    return np.array(u),times

# T, dt = 300 * 365 * 24 * 3600, 60*60*24*100
T, dt = 24*60*60*365*10, 60*60
mSaturn = 568.32e24 # kg
mTitan = 1.345e23 # kg
rTitan = 1.22e9 # m
mDeathStar = 2.24e23 # m
# mearth = 5.97e24 # kg
mMimas = 3.75e19 # kg
rMimas = 189e6 # m
mHyperion = 5.58*1080 # kg
rHyperion = 1.5e9 # m
mIapetus = 1.806*1021 # kg
rIapetus = 3.56e9 #m
mFenrir = 1e13 # kg
rFenrir = 22.5e9 # m
m = np.array([mSaturn,mTitan,mDeathStar, mHyperion, mIapetus]) # , mFenrir
r0 = np.zeros((np.size(m),3))
r0[0] = [0.0, 0.0, 0.0]
r0[1] = [rTitan, 0.0,0.0]
r0[3] = [rHyperion, 0.0, 0.0]
r0[4] = [rIapetus, 0.0, 0.0]
# r0[5] = [rFenrir, 0.0, 0.0]

rDeathStar = np.max(r0)*1.2
r0[2] = [rDeathStar*np.sin(np.pi/4), rDeathStar*np.cos(np.pi/4), 0.0]

# print(r0)
v0 = np.zeros((np.size(m),3))
v0[0] = [0.0, 0.0, 0.0]
v0[1] = [0.0, -np.sqrt(G*m[0]/np.linalg.norm(r0[1])), 0.0]
vDeathStar = np.sqrt(G*m[0]/np.linalg.norm(r0[2]))
v0[2] = [vDeathStar*np.cos(np.pi/4), -vDeathStar*np.sin(np.pi/4), 0.0]
v0[3] = [0.0, -np.sqrt(G*m[0]/np.linalg.norm(r0[3])), 0.0]
v0[4] = [0.0, -np.sqrt(G*m[0]/np.linalg.norm(r0[4])), 0.0]
# v0[5] = [0.0, -np.sqrt(G*m[0]/np.linalg.norm(r0[5])), 0.0]

u0 = np.array([r0, v0])
u_rk, times, sol_r, sol_time = ivp_rk4(u0, T, dt)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ur = u_rk[:,0]
uv = u_rk[:,1]

r1 = []
r2 = []
r3 = []
r4 = []
r5 = []
# r6 = []

for i in range(ur.shape[0]):
    r1.append(ur[i][0])
    r2.append(ur[i][1])
    r3.append(ur[i][2])
    r4.append(ur[i][3])
    r5.append(ur[i][4])
    # r6.append(ur[i][5])
r1 = np.array(r1)
r2 = np.array(r2)
r3 = np.array(r3)
r4 = np.array(r4)
r5 = np.array(r5)
# r6 = np.array(r6)

print(sol_r)
print(sol_time)
sol_r = np.array(sol_r)
sol = np.vstack((sol_r, np.zeros(sol_r.shape)))
print(sol)
ax.scatter(r1[:,0], r1[:,1], r1[:,2], color="goldenrod", s=20, label="Saturn")
ax.scatter(r2[:,0], r2[:,1], r2[:,2], color="blue", s=1, label="Titan")
ax.scatter(r3[:,0], r3[:,1], r3[:,2], color="dimgray", s=20, label="Death Star")
ax.scatter(r4[:,0], r4[:,1], r4[:,2], color="cyan", s=1, label="Hyperion")
ax.scatter(r5[:,0], r5[:,1], r5[:,2], color="pink", s=1, label="Iapetus")
# ax.scatter(r6[:,0], r6[:,1], r6[:,2], color="orange", s=1, label="Fenrir")
ax.plot(sol[:,0], sol[:,1], sol[:,2], color="lime", label="Super Laser")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.suptitle("Project Kronos")
# plt.xlim(-1.22e12,1.22e9)
# plt.ylim(0, 1.22e9)
plt.legend()
# plt.show()

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

