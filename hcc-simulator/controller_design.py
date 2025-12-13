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
