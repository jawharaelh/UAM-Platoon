# physical and network parameters
KAPPA = 0.15
KAPPA0 = 0.5 
K_GAIN = 32.0
TAU_C = 2.0 # Nominal time headway
V_REF = 50.0 # Reference velocity (m/s)
N_MAX = 4 # Max followers in platoon
L = 800.0 # Corridor length (m)
S0 = 0.0
DS = 0.1 # Spatial integration step (m)

# Physical Actuator Limits
U_MIN = -1.5          # Max braking capacity (m/s^2)
U_MAX = 1.5           # Max acceleration capacity (m/s^2)

# Safety & CBF Parameters
TAU_MIN = 1.5 # Minimum allowed time headway
DELTA_MIN = TAU_MIN - TAU_C
GAMMA_CBF = 1.0 # Time-headway buffer
ALPHA_CBF = 2.0 # CBF convergence rate
C_BAR_D = 0.3 # Nominal certified disturbance bound