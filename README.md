# continuumRobot_RL_control
TD3
TEST
priečinok Test - súbor Tetst_2 - súbor na testovanie modelov s využitým Keras
po otvorení súboru - nastavenie ciest súborov (TD3, environment) pre použité triedy (continuumEnv, OUActionNoise, policy) 

Pri teste - 4 možnosti vykreslenia pohybu (4 odmeny)
Po vykonaní hlavného for cyklu - zobrazenie výsledku, potom 3 možnosti ďalšieho vykreslenia (Error, position, curvature)


UČENIE
priečinok Keras - súbor TD3 - súbor na učenie modelov TD3 algoritmu
po otvorení súboru - nastavenie ciest súboru (environment) pre použitú triedu (continuumEnv) 

Pri učení - premenná Train=True
Pri testovaní - premenná Train-False
Čas učenia závisí od počtu epizód (total_episodes) a od počtu krokov v jednej epizóde (for i in range(200))

Hyperparametre - optimalizácia učenia (gamma, tau, critic_lr, actor_lr)
Nastavenie Buffer - buffer = Buffer(int(5e5), 64) pre uchovanie dát

Tak isto pre DDPG
