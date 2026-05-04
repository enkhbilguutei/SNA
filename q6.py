# Q6: Simulate diffusion trends for different epidemic models
#     and present results using appropriate visuals.
#     Models: SI, SIS, SIR, SEIR
#
# No CSV needed — epidemic models are simulated mathematically.
# You can adjust the parameters below to match your scenario.

import numpy as np
import matplotlib.pyplot as plt

N     = 10000   # total population
beta  = 0.3     # transmission rate
gamma = 0.05    # recovery rate
sigma = 0.1     # incubation rate (SEIR only)
T     = 160     # time steps
dt    = 0.5
t     = np.arange(0, T + dt, dt)
I0    = 10      # initial infected

def run_SI():
    S, I = N - I0, I0
    S_t, I_t = [S], [I]
    for _ in t[1:]:
        dI = beta * S * I / N * dt
        S -= dI; I += dI
        S_t.append(S); I_t.append(I)
    return np.array(S_t)/N, np.array(I_t)/N

def run_SIS():
    S, I = N - I0, I0
    S_t, I_t = [S], [I]
    for _ in t[1:]:
        new_inf = beta * S * I / N * dt
        new_rec = gamma * I * dt
        S += -new_inf + new_rec; I += new_inf - new_rec
        S_t.append(S); I_t.append(I)
    return np.array(S_t)/N, np.array(I_t)/N

def run_SIR():
    S, I, R = N - I0, I0, 0
    S_t, I_t, R_t = [S], [I], [R]
    for _ in t[1:]:
        new_inf = beta * S * I / N * dt
        new_rec = gamma * I * dt
        S -= new_inf; I += new_inf - new_rec; R += new_rec
        S_t.append(S); I_t.append(I); R_t.append(R)
    return np.array(S_t)/N, np.array(I_t)/N, np.array(R_t)/N

def run_SEIR():
    S, E, I, R = N - I0, 0, I0, 0
    S_t, E_t, I_t, R_t = [S], [E], [I], [R]
    for _ in t[1:]:
        new_exp = beta * S * I / N * dt
        new_inf = sigma * E * dt
        new_rec = gamma * I * dt
        S -= new_exp; E += new_exp - new_inf
        I += new_inf - new_rec; R += new_rec
        S_t.append(S); E_t.append(E); I_t.append(I); R_t.append(R)
    return np.array(S_t)/N, np.array(E_t)/N, np.array(I_t)/N, np.array(R_t)/N

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

S, I = run_SI()
axes[0,0].plot(t, S, label='S', color='steelblue')
axes[0,0].plot(t, I, label='I', color='tomato')
axes[0,0].set_title("SI Model")

S, I = run_SIS()
axes[0,1].plot(t, S, label='S', color='steelblue')
axes[0,1].plot(t, I, label='I', color='tomato')
axes[0,1].set_title("SIS Model")

S, I, R = run_SIR()
axes[1,0].plot(t, S, label='S', color='steelblue')
axes[1,0].plot(t, I, label='I', color='tomato')
axes[1,0].plot(t, R, label='R', color='seagreen')
axes[1,0].set_title("SIR Model")

S, E, I, R = run_SEIR()
axes[1,1].plot(t, S, label='S', color='steelblue')
axes[1,1].plot(t, E, label='E', color='gold')
axes[1,1].plot(t, I, label='I', color='tomato')
axes[1,1].plot(t, R, label='R', color='seagreen')
axes[1,1].set_title("SEIR Model")

for ax in axes.flat:
    ax.set_xlabel("Time")
    ax.set_ylabel("Fraction of Population")
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle(f"Q6: Epidemic Diffusion Models  (N={N}, β={beta}, γ={gamma})", fontsize=13)
plt.tight_layout()
plt.savefig("q6_output.png", dpi=150, bbox_inches='tight')
plt.show()
