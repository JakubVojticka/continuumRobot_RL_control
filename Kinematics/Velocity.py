import numpy as np
import matplotlib.pyplot as plt
from forward_kinematics import trans_mat_cc, couple_transformations

# Generovanie hodnôt zakrivení s rozsahom od -1 do 1
kappa1 = np.random.uniform(low=-1, high=1)
kappa2 = np.random.uniform(low=-1, high=1)
kappa3 = np.random.uniform(low=-1, high=1)
l = 0.1000  

# Výpočet transformačnej matice pre prvý segment
T1_cc = trans_mat_cc(kappa1, l)
T1_tip = np.reshape(T1_cc[-1, :], (4, 4), order='F')

# Výpočet transformačnej matice pre druhý segment
T2 = trans_mat_cc(kappa2, l)
T2_cc = couple_transformations(T2, T1_tip)
T2_tip = np.reshape(T2_cc[-1, :], (4, 4), order='F')

# Výpočet transformačnej matice pre tretí segment
T3 = trans_mat_cc(kappa3, l)
T3_cc = couple_transformations(T3, T2_tip)

# Získanie súradníc koncových bodov segmentov
x1, y1 = T1_cc[-1, 12], T1_cc[-1, 13]
x2, y2 = T3_cc[-1, 12], T3_cc[-1, 13]

# Vykreslenie trajektórie
plt.scatter([x1, x2], [y1, y2], color='black', label='Endpoints') 
plt.plot([x1, x2], [y1, y2], color='blue', label='Trajectory')  
plt.xlabel("Position x - [m]", fontsize=12)
plt.ylabel("Position y - [m]", fontsize=12)
plt.title("Trajectory", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()