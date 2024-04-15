import numpy as np
import matplotlib.pyplot as plt
from forward_kinematics import trans_mat_cc, couple_transformations

# Počet bodov v pracovnom priestore
size = 1000

# Generovanie náhodných hodnôt pre zakrivenie s rozsahom od -1 do 1
kappa1 = np.random.uniform(low=-4, high=16, size=(size,)) # 1/m
kappa2 = np.random.uniform(low=-4, high=16, size=(size,))
kappa3 = np.random.uniform(low=-4, high=16, size=(size,))
l = 0.1000  # Dĺžka segmentu robota

x = []
y = []

# Výpočet súradníc bodov v pracovnom priestore pre každú hodnotu zakrivenia
for i in range(size):

    # Výpočet transformačnej matice pre prvý segment
    T1_cc = trans_mat_cc(kappa1[i], l)
    T1_tip = np.reshape(T1_cc[len(T1_cc) - 1, :], (4, 4), order='F')
    
    # Výpočet transformačnej matice pre druhý segment
    T2 = trans_mat_cc(kappa2[i], l)
    T2_cc = couple_transformations(T2, T1_tip)
    T2_tip = np.reshape(T2_cc[len(T2_cc) - 1, :], (4, 4), order='F')
    
    # Výpočet transformačnej matice pre tretí segment
    T3 = trans_mat_cc(kappa3[i], l)
    T3_cc = couple_transformations(T3, T2_tip)
    
    # Získanie súradníc koncových bodov segmentov
    x.append(T3_cc[-1, 12])
    y.append(T3_cc[-1, 13])

# Vykreslenie pracovného priestoru
plt.scatter(x, y, s=5, color='black')  
plt.xlabel("Position x - [m]", fontsize=12)
plt.ylabel("Position y - [m]", fontsize=12)
plt.title("Workspace of Planar Continuum Robot", fontsize=14)
plt.grid(True)
plt.show()

