import numpy as np 


def three_section_robot (kappa1,kappa2,kappa3,l):
    c_ks = np.cos((kappa1*l[0])+(kappa2*l[1])+(kappa3*l[2])); #cos
    s_ks = np.sin((kappa1*l[0])+(kappa2*l[1])+(kappa3*l[2])); #sin

    A_14 = ((np.cos(kappa1*l[0])-1)/kappa1) + ((np.cos((kappa1*l[0])+(kappa2*l[1]))-np.cos(kappa1*l[0]))/kappa2) + ((np.cos((kappa1*l[0])+(kappa2*l[1])+(kappa3*l[2]))-np.cos((kappa1*l[0])+(kappa2*l[1])))/kappa3)
    A_24 = ((np.sin(kappa1*l[0]))/kappa1) + ((np.sin((kappa1*l[0])+(kappa2*l[1]))-np.sin(kappa1*l[0]))/kappa2) + ((np.sin((kappa1*l[0])+(kappa2*l[1])+(kappa3*l[2]))-np.sin((kappa1*l[0])+(kappa2*l[1])))/kappa3)
    
    T = np.array([c_ks,s_ks,0,0,-s_ks,c_ks,0,0,0,0,1,0,A_14,A_24,0,1]); #transformačná matica orientácie a polohy
    T = np.reshape(T,(4,4),order='F');
    return T

def jacobian_matrix(delta_kappa, kappa1, kappa2, kappa3, l):
    J11 = (three_section_robot(kappa1+delta_kappa,kappa2,kappa3,l)[0,3] - three_section_robot(kappa1-delta_kappa,kappa2,kappa3,l))[0,3] / (2*delta_kappa);
    J12 = (three_section_robot(kappa1,kappa2+delta_kappa,kappa3,l)[0,3] - three_section_robot(kappa1,kappa2-delta_kappa,kappa3,l))[0,3] / (2*delta_kappa);
    J13 = (three_section_robot(kappa1,kappa2,kappa3+delta_kappa,l)[0,3] - three_section_robot(kappa1,kappa2,kappa3-delta_kappa,l))[0,3] / (2*delta_kappa);
    J21 = (three_section_robot(kappa1+delta_kappa,kappa2,kappa3,l)[1,3] - three_section_robot(kappa1-delta_kappa,kappa2,kappa3,l))[1,3] / (2*delta_kappa);
    J22 = (three_section_robot(kappa1,kappa2+delta_kappa,kappa3,l)[1,3] - three_section_robot(kappa1,kappa2-delta_kappa,kappa3,l))[1,3] / (2*delta_kappa);
    J23 = (three_section_robot(kappa1,kappa2,kappa3+delta_kappa,l)[1,3] - three_section_robot(kappa1,kappa2,kappa3-delta_kappa,l))[1,3] / (2*delta_kappa);
    
    J = np.array([J11,J12,J13,J21,J22,J23]);
    J = np.reshape(J,(2,3))
    
    return J

def trans_mat_cc(kappa, l):
    
    si=np.linspace(0,l, num = 50);
    T= np.zeros((len(si),16));
    
    for i in range(len(si)):
        s=si[i];
        c_ks=np.cos(kappa*s);
        s_ks=np.sin(kappa*s);
        if kappa==0:
            T[i,:] = np.array([c_ks,s_ks,0,0,-s_ks,c_ks,0,0,0,0,1,0,0,s,0,1]);  
        else:
            T[i,:] = np.array([c_ks,s_ks,0,0,-s_ks,c_ks,0,0,0,0,1,0,(c_ks-1)/kappa,s_ks/kappa,0,1]);

    return T

def couple_transformations(T,T_tip):
    Tc=np.zeros((len(T[:,0]),len(T[0,:])));
    for k in range(len(T[:,0])):
        p = np.matmul(T_tip,(np.reshape(T[k,:],(4,4),order='F')))
        Tc[k,:] = np.reshape(p,(16,),order='F');
    return Tc
