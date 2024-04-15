import numpy as np               #importovanie knižnice numpy
from math import sin, cos        #importovanie matematickej knižnice

#==================Definovanie transformačnej matice==========================#

def three_section_planar_robot(kappa1, kappa2, kappa3, l):
    cs=cos(kappa1*l[0]+kappa2*l[1]+kappa3*l[2]);
    si=sin(kappa1*l[0]+kappa2*l[1]+kappa3*l[2]);

    a14 = ((np.cos(kappa1*l[0])-1)/kappa1) + ((np.cos((kappa1*l[0])+(kappa2*l[1]))-np.cos(kappa1*l[0]))/kappa2) + ((np.cos((kappa1*l[0])+(kappa2*l[1])+(kappa3*l[2]))-np.cos((kappa1*l[0])+(kappa2*l[1])))/kappa3)
    a24 = ((np.sin(kappa1*l[0]))/kappa1) + ((np.sin((kappa1*l[0])+(kappa2*l[1]))-np.sin(kappa1*l[0]))/kappa2) + ((np.sin((kappa1*l[0])+(kappa2*l[1])+(kappa3*l[2]))-np.sin((kappa1*l[0])+(kappa2*l[1])))/kappa3)   
    T=np.array([[cs,-si,0,a14],[si,cs,0,a24],[0,0,1,0],[0,0,0,1]])

    return T

    
#==================Definovanie Jacobiho matice==========================#

def jacobian_matrix(delta_kappa, kappa1, kappa2, kappa3, l):
    J11 = (three_section_planar_robot(kappa1+delta_kappa,kappa2,kappa3,l)[0,3]-three_section_planar_robot(kappa1-delta_kappa,kappa2,kappa3,l))[0,3]/(2*delta_kappa); # parciálna derivácia funkcie vzhľadom na kappa1 pre prvý riadok T
    J12 = (three_section_planar_robot(kappa1,kappa2+delta_kappa,kappa3,l)[0,3]-three_section_planar_robot(kappa1,kappa2-delta_kappa,kappa3,l))[0,3]/(2*delta_kappa); # parciálna derivácia funkcie vzhľadom na kappa2 pre prvý riadok T
    J13 = (three_section_planar_robot(kappa1,kappa2,kappa3+delta_kappa,l)[0,3]-three_section_planar_robot(kappa1,kappa2,kappa3-delta_kappa,l))[0,3]/(2*delta_kappa); # parciálna derivácia funkcie vzhľadom na kappa3 pre prvý riadok T
    J21 = (three_section_planar_robot(kappa1+delta_kappa,kappa2,kappa3,l)[1,3]-three_section_planar_robot(kappa1-delta_kappa,kappa2,kappa3,l))[1,3]/(2*delta_kappa); # parciálna derivácia funkcie vzhľadom na kappa1 pre druhý riadok T
    J22 = (three_section_planar_robot(kappa1,kappa2+delta_kappa,kappa3,l)[1,3]-three_section_planar_robot(kappa1,kappa2-delta_kappa,kappa3,l))[1,3]/(2*delta_kappa); # parciálna derivácia funkcie vzhľadom na kappa2 pre druhý riadok T
    J23 = (three_section_planar_robot(kappa1,kappa2,kappa3+delta_kappa,l)[1,3]-three_section_planar_robot(kappa1,kappa2,kappa3-delta_kappa,l))[1,3]/(2*delta_kappa); # parciálna derivácia funkcie vzhľadom na kappa3 pre druhý riadok T

    J=np.array([[J11,J12,J13],[J21,J22,J23]])
    return J

#==================Funkcia kinematiky - mapovanie==========================#

def trans_mat_cc (kappa,l):

    si=np.linspace(0,l, num = 50);                #pole hodnôt od 0 po l rozdelených na 50 rovnakých časti 
    T= np.zeros((len(si),16));                    #generovanie transformačnej matice (riadky, stĺpce) 
    
    for i in range(len(si)):                      
        s=si[i];                                  #priradenie aktuálnej hodnoty z "si"           
        c_ks=np.cos(kappa*s);                     #výpočet hodnoty cosínus
        s_ks=np.sin(kappa*s);                     #výpočet hodnoty sínus
        if kappa==0:                              #podmienka ak zakrivenie bude rovné 0
            T[i,:] = np.array([c_ks,s_ks,0,0,-s_ks,c_ks,0,0,0,0,1,0,0,s,0,1]);    #vytvorenie matice s konkrétnymi hodnotami
        else:
            T[i,:] = np.array([c_ks,s_ks,0,0,-s_ks,c_ks,0,0,0,0,1,0,(c_ks-1)/kappa,s_ks/kappa,0,1]);  #vytvorenie matice s inými konkrétnymi hodnotami

    return T  


#==================Funkcia robota - generovanie trajektórií==========================#
'''
Priamu kinematiku pre manipulátor pozostavajúci z n častí dostaneme súčinom n transformačných matíc.
Chceme zistiť orietáciu a polohu koncového bodu robota.
'''
def coupletransformations(T,T_tip):

    Tc=np.zeros((len(T[:,0]),len(T[0,:])));                           #definícia matice Tc, rozmerovo rovnaká ako T, Tc obsahuje nuly výslednej transformačnej matice
    for k in range(len(T[:,0])):                                      #iteruje cez všetky riadky matice T
        p = np.matmul(T_tip,(np.reshape(T[k,:],(4,4),order='F')))     #pre každý riadok matice T sa vykoná maticový súčin medzi T_tip a T, výsledok sa zapíše do p
        Tc[k,:] = np.reshape(p,(16,),order='F');                      #transformácia p na jednorozmerové pole a výledok zapísať do k-teho riadku matice Tc
    return Tc