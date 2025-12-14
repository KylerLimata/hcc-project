import control as ctrl
import numpy as np
import matplotlib.pyplot as plt

# The goal here is to design a PID controller for use with
# the HybridSteeringAgent. We want to have a short rise time

# Define design parameters
Tr = 0.2 # sec, rise time
Ovsh = 5 # percent overshoot
sse = 0.0 # steady-state error

# Compute zeta and psi
log_os = np.log(Ovsh/100)
zeta = -log_os/np.sqrt(np.pi**2+log_os**2)

print(f'zeta = {zeta:.3f}')

psi_mag = 2.05/Tr # also omega_n
psi_angle_min = np.pi - np.acos(zeta)
psi_re = psi_mag*np.cos(psi_angle_min)
psi_im = psi_mag*np.sin(psi_angle_min)
psi = psi_re + 1j*psi_im

latex_string = r"$\theta_{\psi, min}$"
print(f'psi = {psi_re:.3f} + {psi_im:.3f}j')
print(f'||psi|| = {psi_mag:.3f}, {latex_string} = {psi_angle_min:.3f} rad')

# The output of the neural network will be the target steering power
# and the output of the plant will be the current steering power.
# Therefore, G is a linear system f(x) = x, effectively the ramp function.
G = ctrl.tf([1], [1, 0, 0]) # Laplace transform of f(x) = x for s > 0
plt.figure()
roots, gains = ctrl.root_locus(G, plot=True)

print(f'G = {G}')
ax = plt.gca()  # get current axes

ax.set_title(r'Root Locus of $G(s) = 1/(s^2)$')
ax.set_xlabel('Real Axis')
ax.set_ylabel('Imaginary Axis')
ax.plot(psi_re, psi_im, 'ro')
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.grid(True)
plt.show()

# Getting as close as possible to psi gives us a gain of 55.02
# Proportional compensation
K1 = 55.02
G_P = ctrl.feedback(K1*G, 1)

# Derivative compensation
K2 = 1 # Set to 1 for now
theta_c = np.pi - np.angle(ctrl.evalfr(G, psi))
print(f"theta_c = {theta_c:.3f} rad, {np.degrees(theta_c):.3f} deg")

zD = psi_re - abs(psi_im)/np.tan(theta_c)
print(f"zD = {zD:.3f}")

## Root locus with K2 = 1
C_D_sans = ctrl.zpk([zD], [], 1)
plt.figure()
roots, gains = ctrl.root_locus(K1*C_D_sans*G, plot=True)

ax = plt.gca()  # get current axes

ax.set_title(r'Root Locus of plant with Derivative Control')
ax.set_xlabel('Real Axis')
ax.set_ylabel('Imaginary Axis')
ax.plot(psi_re, psi_im, 'ro')
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.grid(True)
plt.show()

## Compute K2
K2 = 1/np.abs(ctrl.evalfr(K1*C_D_sans*G, psi))
C_D = K1*K2*C_D_sans
G_PD = ctrl.feedback(C_D*G, 1)

## Step response for P and PD control
t_P, y_P = ctrl.step_response(G_P)
t_PD, y_PD = ctrl.step_response(G_PD)
stepinfo_PD = ctrl.step_info(G_PD)
print("Stepinfo for PD")
print(stepinfo_PD)

plt.figure()
plt.plot(t_P, y_P, label='P Control')
plt.plot(t_PD, y_PD, label='PD Control')
plt.title('Step Response for PD control')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()

# We can see that the steady state error is 0
# but the transient response is not entirely
# sufficient, so let's add integral control

# Integral compensation
zI = -5 # Starting point
C_I_sans = ctrl.zpk([zI], [0], 1)

G_PID1 = ctrl.feedback(C_I_sans*C_D*G, 1)

## Simulate initial design
t_PID1, y_PID1 = ctrl.step_response(G_PID1)
stepinfo_PID1 = ctrl.step_info(G_PID1)
print("Stepinfo for PID1")
print(stepinfo_PID1)

plt.figure()
plt.plot(t_PID1, y_PID1, label='PID1 Control')
plt.title('Step Response for PID1')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()

## This ends up making the steady-state performance
## worse, so let's move the zero
zI2 = -6 # Starting point
C_I_sans2 = ctrl.zpk([zI2], [0], 1)

G_PID2 = ctrl.feedback(C_I_sans2*C_D*G, 1)

## Simulate initial design
t_PID2, y_PID2 = ctrl.step_response(G_PID2)
stepinfo_PID2 = ctrl.step_info(G_PID2)
print("Stepinfo for PID2")
print(stepinfo_PID2)

plt.figure()
plt.plot(t_PID2, y_PID2, label='PID2 Control')
plt.title('Step Response for PID2')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()

## This doesn't help much, so lets' tweak the zeros and gain directly
K3 = 0.75
zD3 = -0.8
zI3 = -0.8
C_D_sans3 = ctrl.zpk([zD3], [], 1)
C_I_sans3 = ctrl.zpk([zI3], [0], 1)
G_PID3 = ctrl.feedback(K1*K2*K3*C_I_sans3*C_D_sans3*G,1)

## And plot again
t_PID3, y_PID3 = ctrl.step_response(G_PID3)
stepinfo_PID3 = ctrl.step_info(G_PID3)
print("Stepinfo for PID3")
print(stepinfo_PID3)

plt.figure()
plt.plot(t_PID3, y_PID3, label='PID3 Control')
plt.title('Step Response for PID3')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()

## While this isn't entirely what we wanted, it will suffice
## for this project, so lets extract Kp, Ki, and Kd
C_PID3 = K1*K2*K3*C_I_sans3*C_D_sans3
num, den = ctrl.tfdata(C_PID3)
num = np.squeeze(num)
den = np.squeeze(den)

print("PID Numerator coefficients:", num)
print("PID Denominator coefficients:", den)

Kd = num[0]
Kp = num[1]
Ki = num[2]

print(f"Kp = {Kp:.3f}, Ki = {Ki:.3f}, Kd = {Kd:.3f}")
# Kp = 16.977, Ki = 6.791, Kd = 10.61