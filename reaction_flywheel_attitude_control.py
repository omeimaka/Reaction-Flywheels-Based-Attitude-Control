"""
Experiment 3: Reaction Flywheel Attitude Control
Implements:
- Decoupled small‑angle dynamics
- PID controller (continuous torque from flywheels)
- Flywheel angular velocity  (3000 rpm limit)
- Pointing accuracy calculation (per axis & overall)
- Bandwidth estimation from step response
- Solar radiation disturbance
- Stabilisation and maneuvre cases
- Downloadable plots and results
"""

import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import os

# ================== PARAMETERS ==================
# Spacecraft & flywheel
I_body = np.array([300.0, 300.0, 50.0])   # kg·m²
I_wheel = 1.0                              # kg·m² (per axis)
max_rpm = 3000
max_omega_wheel = max_rpm * (2 * np.pi / 60)   # rad/s ≈ 314.16

# PID gains (tuned per axis for reasonable response)
Kp = np.array([0.5, 0.5, 1.0])      # N·m/rad
Kd = np.array([10.0, 10.0, 20.0])   # N·m/(rad/s)
Ki = np.array([0.01, 0.01, 0.02])   # N·m/(rad·s)

# Solar radiation disturbance (actual data from guidebook: Chateristic area of spacecraft =50 m²)
solar_pressure = 4.65e-6            # N/m² (I/c)
area = 50.0                         # m²
arm = 0.5                           # m (assumed moment arm)
rho = 0.0                           # reflectance factor
T_solar_amp = solar_pressure * area * (1 + rho) * arm   # ≈ 2.325e-4 N·m

# Orbit
orbit_period = 24 * 3600.0
omega_orbit = 2 * np.pi / orbit_period

# Simulation times
t_end_stab = 2000.0        # seconds (stabilisation)
t_end_man  = 4000.0        # seconds (manoeuvre)
dt = 0.1

time_stab = np.arange(0, t_end_stab, dt)
time_man  = np.arange(0, t_end_man, dt)

# Initial and target attitudes (degrees, then converted to radians)
init_stab_deg = np.array([0.5, 1.0, 2.0])
init_stab_rad = init_stab_deg * np.pi / 180.0
target_stab_rad = np.zeros(3)

init_man_rad = init_stab_rad.copy()
target_man_deg = np.array([5.0, 10.0, 20.0])
target_man_rad = target_man_deg * np.pi / 180.0

# ================== SIMULATION FUNCTION ==================
def run_simulation(time_vec, init_angle, target_angle):
    n_steps = len(time_vec)
    angle = init_angle.copy()
    rate = np.zeros(3)
    integral = np.zeros(3)
    prev_error = np.zeros(3)
    wheel_speed = np.zeros(3)   # flywheel angular velocity (rad/s)

    angle_hist = np.zeros((n_steps, 3))
    rate_hist = np.zeros((n_steps, 3))
    torque_hist = np.zeros((n_steps, 3))
    wheel_speed_hist = np.zeros((n_steps, 3))
    saturated = np.zeros(3, dtype=bool)

    for i, t in enumerate(time_vec):
        angle_hist[i] = angle
        rate_hist[i] = rate
        wheel_speed_hist[i] = wheel_speed

        error = angle - target_angle

        # PID control
        P = Kp * error
        integral += error * dt
        I_term = Ki * integral
        derivative = Kd * (error - prev_error) / dt if i > 0 else np.zeros(3)
        torque_cmd = -(P + I_term + derivative)

        prev_error = error.copy()

        # Disturbance torque (solar radiation)
        torque_dist = T_solar_amp * np.array([
            np.sin(omega_orbit * t),
            np.cos(omega_orbit * t),
            np.sin(2 * omega_orbit * t)
        ])

        total_torque = torque_cmd + torque_dist

        # Spacecraft dynamics
        accel_body = total_torque / I_body
        rate += accel_body * dt
        angle += rate * dt

        # Flywheel dynamics: torque applied on wheel = -torque_cmd
        wheel_accel = -torque_cmd / I_wheel
        wheel_speed += wheel_accel * dt

        # Saturation detection (no actual limiting here, just flag)
        if np.any(np.abs(wheel_speed) >= max_omega_wheel):
            saturated = saturated | (np.abs(wheel_speed) >= max_omega_wheel)

        torque_hist[i] = torque_cmd

    return angle_hist, rate_hist, torque_hist, wheel_speed_hist, saturated, time_vec

# ================== RUN SIMULATIONS ==================
print("Running stabilisation simulation...")
ang_stab, rate_stab, torq_stab, whl_stab, sat_stab, t_stab = run_simulation(
    time_stab, init_stab_rad, target_stab_rad)

print("Running manoeuvre simulation...")
ang_man, rate_man, torq_man, whl_man, sat_man, t_man = run_simulation(
    time_man, init_man_rad, target_man_rad)

# ================== POINTING ACCURACY & BANDWIDTH ==================
def pointing_accuracy(angle_hist, target_rad, t, t_ss_start=500.0):
    """Calculate per‑axis half‑amplitude (deg) and overall P_all from steady state."""
    mask = t >= t_ss_start
    if not np.any(mask):
        mask = slice(None)  # fallback: whole signal
    ang_deg = angle_hist[mask] * 180.0 / np.pi
    target_deg = target_rad * 180.0 / np.pi
    err = ang_deg - target_deg
    P_i = (np.max(err, axis=0) - np.min(err, axis=0)) / 2.0
    P_all = np.sqrt(np.sum(P_i**2))
    return P_i, P_all

def bandwidth_from_rise_time(time, angle, target, tol_start=0.1, tol_end=0.9):

    ang_deg = angle * 180.0 / np.pi
    target_deg = target * 180.0 / np.pi
    # Normalise response
    norm = (ang_deg - ang_deg[0]) / (target_deg - ang_deg[0])
    # Find indices where norm reaches 0.1 and 0.9
    idx10 = np.where(norm >= 0.1)[0]
    idx90 = np.where(norm >= 0.9)[0]
    if len(idx10) == 0 or len(idx90) == 0:
        return np.nan
    t10 = time[idx10[0]]
    t90 = time[idx90[0]]
    rise_time = t90 - t10
    if rise_time <= 0:
        return np.nan
    return 0.35 / rise_time   # Hz

# ================== CALCULATE METRICS ==================
print("\n=== Stabilisation Performance ===")
# Settling time (within 0.05°)
def settling_time(angle_deg, target_deg, t, tol=0.05):
    error = np.abs(angle_deg - target_deg)
    below = error < tol
    for i in range(len(t) - 100):
        if np.all(below[i:i+100]):
            return t[i]
    return np.nan

ang_stab_deg = ang_stab * 180.0 / np.pi
ang_man_deg = ang_man * 180.0 / np.pi

titles = ['Roll', 'Pitch', 'Yaw']
print("Settling time (to within 0.05° of 0):")
for i, name in enumerate(titles):
    ts = settling_time(ang_stab_deg[:, i], 0.0, time_stab)
    print(f"  {name}: {ts:.1f} s" if not np.isnan(ts) else f"  {name}: did not settle")

P_stab_i, P_stab_all = pointing_accuracy(ang_stab, target_stab_rad, time_stab)
print("\nPointing accuracy (steady state):")
for i, name in enumerate(titles):
    print(f"  {name}: {P_stab_i[i]:.4f}°")
print(f"  Overall P_all = {P_stab_all:.4f}°")

print("\nBandwidth estimates:")
for i, name in enumerate(titles):
    bw = bandwidth_from_rise_time(time_stab, ang_stab[:, i], target_stab_rad[i])
    if not np.isnan(bw):
        print(f"  {name}: {bw:.3f} Hz")
    else:
        print(f"  {name}: could not estimate")

print("\nFlywheel max speed (rpm) & saturation:")
for i, name in enumerate(titles):
    max_speed = np.max(np.abs(whl_stab[:, i]))
    sat_flag = sat_stab[i]
    print(f"  {name}: {max_speed * 60 / (2*np.pi):.1f} rpm  {'SATURATED' if sat_flag else 'OK'}")

print("\n=== Manoeuvre Performance ===")
for i, name in enumerate(titles):
    ts = settling_time(ang_man_deg[:, i], target_man_deg[i], time_man)
    print(f"  {name} settling time: {ts:.1f} s" if not np.isnan(ts) else f"  {name}: did not settle")

P_man_i, P_man_all = pointing_accuracy(ang_man, target_man_rad, time_man)
print("\nPointing accuracy (steady state after manoeuvre):")
for i, name in enumerate(titles):
    print(f"  {name}: {P_man_i[i]:.4f}°")
print(f"  Overall P_all = {P_man_all:.4f}°")

print("\nBandwidth estimates during manoeuvre:")
for i, name in enumerate(titles):
    bw = bandwidth_from_rise_time(time_man, ang_man[:, i], target_man_rad[i])
    if not np.isnan(bw):
        print(f"  {name}: {bw:.3f} Hz")
    else:
        print(f"  {name}: could not estimate")

print("\nFlywheel max speed (rpm) & saturation during manoeuvre:")
for i, name in enumerate(titles):
    max_speed = np.max(np.abs(whl_man[:, i]))
    sat_flag = sat_man[i]
    print(f"  {name}: {max_speed * 60 / (2*np.pi):.1f} rpm  {'SATURATED' if sat_flag else 'OK'}")

# ================== SAVE METRICS TO TEXT FILE ==================
results_txt = f"""
EXPERIMENT 3 – REACTION FLYWHEEL CONTROL RESULTS
=================================================
Stabilisation:
  Settling time: Roll={settling_time(ang_stab_deg[:,0],0,time_stab):.1f}s, Pitch={settling_time(ang_stab_deg[:,1],0,time_stab):.1f}s, Yaw={settling_time(ang_stab_deg[:,2],0,time_stab):.1f}s
  Pointing accuracy (deg): Roll={P_stab_i[0]:.4f}, Pitch={P_stab_i[1]:.4f}, Yaw={P_stab_i[2]:.4f}, Overall={P_stab_all:.4f}
  Bandwidth (Hz): Roll={bandwidth_from_rise_time(time_stab,ang_stab[:,0],target_stab_rad[0]):.3f}, Pitch={bandwidth_from_rise_time(time_stab,ang_stab[:,1],target_stab_rad[1]):.3f}, Yaw={bandwidth_from_rise_time(time_stab,ang_stab[:,2],target_stab_rad[2]):.3f}
  Max wheel speed (rpm): Roll={np.max(np.abs(whl_stab[:,0]))*60/(2*np.pi):.1f}, Pitch={np.max(np.abs(whl_stab[:,1]))*60/(2*np.pi):.1f}, Yaw={np.max(np.abs(whl_stab[:,2]))*60/(2*np.pi):.1f}
  Saturation: Roll={'YES' if sat_stab[0] else 'NO'}, Pitch={'YES' if sat_stab[1] else 'NO'}, Yaw={'YES' if sat_stab[2] else 'NO'}

Manoeuvre:
  Settling time: Roll={settling_time(ang_man_deg[:,0],target_man_deg[0],time_man):.1f}s, Pitch={settling_time(ang_man_deg[:,1],target_man_deg[1],time_man):.1f}s, Yaw={settling_time(ang_man_deg[:,2],target_man_deg[2],time_man):.1f}s
  Pointing accuracy (deg): Roll={P_man_i[0]:.4f}, Pitch={P_man_i[1]:.4f}, Yaw={P_man_i[2]:.4f}, Overall={P_man_all:.4f}
  Bandwidth (Hz): Roll={bandwidth_from_rise_time(time_man,ang_man[:,0],target_man_rad[0]):.3f}, Pitch={bandwidth_from_rise_time(time_man,ang_man[:,1],target_man_rad[1]):.3f}, Yaw={bandwidth_from_rise_time(time_man,ang_man[:,2],target_man_rad[2]):.3f}
  Max wheel speed (rpm): Roll={np.max(np.abs(whl_man[:,0]))*60/(2*np.pi):.1f}, Pitch={np.max(np.abs(whl_man[:,1]))*60/(2*np.pi):.1f}, Yaw={np.max(np.abs(whl_man[:,2]))*60/(2*np.pi):.1f}
  Saturation: Roll={'YES' if sat_man[0] else 'NO'}, Pitch={'YES' if sat_man[1] else 'NO'}, Yaw={'YES' if sat_man[2] else 'NO'}
"""
with open('experiment3_results.txt', 'w') as f:
    f.write(results_txt)

# ================== PLOTTING ==================
rad2deg = 180.0 / np.pi
titles = ['Roll', 'Pitch', 'Yaw']

# --- 1. Stabilisation: Angle vs time ---
fig1, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for i, ax in enumerate(axs):
    ax.plot(t_stab, ang_stab_deg[:, i])
    ax.axhline(0, color='r', linestyle='--')
    ax.set_ylabel('Angle (deg)')
    ax.set_title(f'{titles[i]} Stabilisation')
    ax.grid(True)
axs[-1].set_xlabel('Time (s)')
fig1.suptitle('Stabilisation: Reaction Wheels (Target 0°)', fontsize=14)
plt.tight_layout()
plt.savefig('stab_angle.png', dpi=150)
plt.show()
files.download('stab_angle.png')

# --- 2. Stabilisation: Control torque & wheel speed ---
fig2, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
for i in range(3):
    axs[i, 0].plot(t_stab, torq_stab[:, i])
    axs[i, 0].set_ylabel('Torque (N·m)')
    axs[i, 0].grid(True)
    axs[i, 0].set_title(f'{titles[i]} Control Torque')

    axs[i, 1].plot(t_stab, whl_stab[:, i] * 60 / (2*np.pi))
   # axs[i, 1].axhline(max_rpm, color='r', linestyle='--')
    #axs[i, 1].axhline(-max_rpm, color='r', linestyle='--')
    axs[i, 1].set_ylabel('Wheel speed (rpm)')
    axs[i, 1].grid(True)
    axs[i, 1].set_title(f'{titles[i]} Flywheel Speed')
axs[-1, 0].set_xlabel('Time (s)')
axs[-1, 1].set_xlabel('Time (s)')
fig2.suptitle('Stabilisation: Actuator Behaviour', fontsize=14)
plt.tight_layout()
plt.savefig('stab_actuator.png', dpi=150)
plt.show()
files.download('stab_actuator.png')

# --- 3. Stabilisation: Phase plane ---
fig3, axs = plt.subplots(1, 3, figsize=(14, 4))
for i, ax in enumerate(axs):
    ax.plot(ang_stab_deg[:, i], rate_stab[:, i]*rad2deg, linewidth=0.8)
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('Rate (deg/s)')
    ax.set_title(f'{titles[i]} Phase Plane')
    ax.grid(True)
fig3.suptitle('Stabilisation Phase Portraits')
plt.tight_layout()
plt.savefig('stab_phase.png', dpi=150)
plt.show()
files.download('stab_phase.png')

# --- 4. Manoeuvre: Angle vs time ---
fig4, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for i, ax in enumerate(axs):
    ax.plot(t_man, ang_man_deg[:, i])
  #  ax.axhline(target_man_deg[i], color='r', linestyle='--')
    ax.set_ylabel('Angle (deg)')
    ax.set_title(f'{titles[i]} Manoeuvre to {target_man_deg[i]}°')
    ax.grid(True)
axs[-1].set_xlabel('Time (s)')
fig4.suptitle('Manoeuvre: Reaction Wheels', fontsize=14)
plt.tight_layout()
plt.savefig('man_angle.png', dpi=150)
plt.show()
files.download('man_angle.png')

# --- 5. Manoeuvre: Control torque & wheel speed ---
fig5, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
for i in range(3):
    axs[i, 0].plot(t_man, torq_man[:, i])
    axs[i, 0].set_ylabel('Torque (N·m)')
    axs[i, 0].grid(True)
    axs[i, 0].set_title(f'{titles[i]} Control Torque')

    axs[i, 1].plot(t_man, whl_man[:, i] * 60 / (2*np.pi))
#    axs[i, 1].axhline(max_rpm, color='r', linestyle='--')
 #   axs[i, 1].axhline(-max_rpm, color='r', linestyle='--')
    axs[i, 1].set_ylabel('Wheel speed (rpm)')
    axs[i, 1].grid(True)
    axs[i, 1].set_title(f'{titles[i]} Flywheel Speed')
axs[-1, 0].set_xlabel('Time (s)')
axs[-1, 1].set_xlabel('Time (s)')
fig5.suptitle('Manoeuvre: Actuator Behaviour', fontsize=14)
plt.tight_layout()
plt.savefig('man_actuator.png', dpi=150)
plt.show()
files.download('man_actuator.png')

# --- 6. Manoeuvre: Phase plane ---
fig6, axs = plt.subplots(1, 3, figsize=(14, 4))
for i, ax in enumerate(axs):
    ax.plot(ang_man_deg[:, i], rate_man[:, i]*rad2deg, linewidth=0.8)
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('Rate (deg/s)')
    ax.set_title(f'{titles[i]} Phase Plane')
    ax.grid(True)
fig6.suptitle('Manoeuvre Phase Portraits')
plt.tight_layout()
plt.savefig('man_phase.png', dpi=150)
plt.show()
files.download('man_phase.png')

# --- 7. Pointing accuracy bar chart ---
fig7, ax = plt.subplots()
x = np.arange(3)
width = 0.35
bars_stab = ax.bar(x - width/2, P_stab_i, width, label='Stabilisation')
bars_man = ax.bar(x + width/2, P_man_i, width, label='Manoeuvre')
ax.set_xticks(x)
ax.set_xticklabels(titles)
ax.set_ylabel('Pointing Accuracy (deg)')
ax.set_title('Steady‑State Pointing Accuracy')
ax.legend()
ax.grid(axis='y')
plt.tight_layout()
plt.savefig('pointing_accuracy.png', dpi=150)
plt.show()
files.download('pointing_accuracy.png')

# Download results text file
files.download('experiment3_results.txt')

print("\nAll figures and results file have been downloaded.")
