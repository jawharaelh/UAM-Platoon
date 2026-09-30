
# Physical and Controller Parameters
KAPPA = 50.00
KAPPA0 = 0.99
K_GAIN = 8.50=
TAU_C = 2.0 # Nominal time headway (s)
V_REF = 2.00 # Reference velocity (m/s)
N_MAX = 5 # Max followers in platoon
L = 5000.0 # Corridor length (m)
S0 = 0.0
DS = 0.02 # Spatial step

# Disturbances
C_BAR_D = 0.10 # Max disturbance bound (m/s^2)
C_BAR_RHO = 4.00 # Certified state bound (s)
C_BAR = 0.65 # Initial condition bound (s)

# Physical Actuator Limits
U_MIN = -5.0 # Max braking capacity (m/s^2)
U_MAX = 5.0 # Max acceleration capacity (m/s^2)
# Safety & CBF Parameters
TAU_MIN = 1.0 # Minimum allowed time headway (s)
DELTA_MIN = TAU_MIN - TAU_C
GAMMA_CBF = 1.0 # Time-headway buffer
ALPHA_CBF = 2.0 # CBF convergence rate