import csv
import pandas as pd
import numpy as np
import math
from variance_analysis import AoV_main

def Tukey(lst, Gro_ave, ms, dof, n_hyp, a_hyp, cv, gro_ele):
    q = []
    n = len(lst[0]) * gro_ele # number of element 要素数
    print(n)
    I_ms = ms[1] # Inner mean square 群内平方平均


    # ASD TD
    dif_ave_1 = abs(float(Gro_ave[0]) - float(Gro_ave[1])) 

    
    sqrt = math.sqrt(float(I_ms)/float(n))
    q1 = dif_ave_1/sqrt
    q.append('{:.3f}'.format(q1))

    # TD 比
    dif_ave_2 = abs(float(Gro_ave[1]) - float(Gro_ave[2])) 
    sqrt = math.sqrt(float(I_ms)/float(n))
    q2 = dif_ave_2/sqrt
    q.append('{:.3f}'.format(q2))

    # ASD 比
    dif_ave_3 = abs(float(Gro_ave[0]) - float(Gro_ave[2]))
    sqrt = math.sqrt(float(I_ms)/float(n))
    q3 = dif_ave_3/sqrt
    q.append('{:.3f}'.format(q3))

    ari = "帰無仮説棄却，有意差あり"
    nasi = "帰無仮説採択，有意差なし"

    print(f"帰無仮説:{n_hyp}")
    print(f"対立仮説:{a_hyp}")

    print(f"臨界値:{cv}")
    print(f"検定統計量:{q}")

    Type = ["ASD-TD", "TD-比", "ASD-比"]

    for i in range(len(q)):
        print(Type[i])
        if float(q[i]) >= cv_1:
            print(f"有意水準1%:{ari}")
        else:
            print(f"有意水準1%:{nasi}")

        if float(q[i]) >= cv_5:
            print(f"有意水準5%:{ari}")
        else:
            print(f"有意水準5%:{nasi}")

        if float(q[i]) >= cv_10:
            print(f"有意水準10%:{ari}")
        else:
            print(f"有意水準10%:{nasi}")
        
        print("~~~~~~~~~~")

    return I_ms, n

def Tukey_HSD(cv, Gro_ave, I_ms, n, n_hyp, a_hyp):
    HSD = [] # 臨界値 # 1% 5% 10%
    Dif_ave = []
    for i in range(len(cv)):
        hsd = float(cv[i]) * math.sqrt(float(I_ms)/float(n))
        HSD.append('{:.3f}'.format(hsd))

    # ASD TD
    dif_ave_1 = abs(float(Gro_ave[0]) - float(Gro_ave[1]))
    Dif_ave.append('{:.3f}'.format(dif_ave_1))
    # TD 比
    dif_ave_2 = abs(float(Gro_ave[1]) - float(Gro_ave[2])) 
    Dif_ave.append('{:.3f}'.format(dif_ave_2))
    # ASD 比
    dif_ave_3 = abs(float(Gro_ave[0]) - float(Gro_ave[2]))
    Dif_ave.append('{:.3f}'.format(dif_ave_3))

    ari = "帰無仮説棄却，有意差あり"
    nasi = "帰無仮説採択，有意差なし"

    print(f"帰無仮説:{n_hyp}")
    print(f"対立仮説:{a_hyp}")

    print(f"HSD:{HSD}")
    print(f"平均値差:{Dif_ave}")

    Type = ["ASD-TD", "TD-比", "ASD-比"]

    for i in range(len(Dif_ave)): # ASD-TD, TD-比, ASD-比
        print(Type[i])
        if float(Dif_ave[i]) >= float(HSD[0]):
            print(f"有意水準1%:{ari}")
        else:
            print(f"有意水準1%:{nasi}")

        if float(Dif_ave[i]) >= float(HSD[1]):
            print(f"有意水準5%:{ari}")
        else:
            print(f"有意水準5%:{nasi}")

        if float(Dif_ave[i]) >= float(HSD[2]):
            print(f"有意水準10%:{ari}")
        else:
            print(f"有意水準10%:{nasi}")
    
        print("----------")

def Fisher_LSD(cv, Gro_ave, I_ms, n, n_hyp, a_hyp):
    LSD = [] # 臨界値 # 1% 5% 10%
    Dif_ave = []
    for i in range(len(cv)):
        lsd = float(cv[i]) * math.sqrt(float(I_ms)/float(n))
        LSD.append('{:.3f}'.format(lsd))

    # ASD TD
    dif_ave_1 = abs(float(Gro_ave[0]) - float(Gro_ave[1]))
    Dif_ave.append('{:.3f}'.format(dif_ave_1))
    # TD 比
    dif_ave_2 = abs(float(Gro_ave[1]) - float(Gro_ave[2])) 
    Dif_ave.append('{:.3f}'.format(dif_ave_2))
    # ASD 比
    dif_ave_3 = abs(float(Gro_ave[0]) - float(Gro_ave[2]))
    Dif_ave.append('{:.3f}'.format(dif_ave_3))

    ari = "帰無仮説棄却，有意差あり"
    nasi = "帰無仮説採択，有意差なし"

    print(f"帰無仮説:{n_hyp}")
    print(f"対立仮説:{a_hyp}")

    print(f"LSD:{LSD}")
    print(f"平均値差:{Dif_ave}")

    Type = ["ASD-TD", "TD-比", "ASD-比"]

    for i in range(len(Dif_ave)): # ASD-TD, TD-比, ASD-比
        print(Type[i])
        if float(Dif_ave[i]) >= float(LSD[0]):
            print(f"有意水準1%:{ari}")
        else:
            print(f"有意水準1%:{nasi}")

        if float(Dif_ave[i]) >= float(LSD[1]):
            print(f"有意水準5%:{ari}")
        else:
            print(f"有意水準5%:{nasi}")

        if float(Dif_ave[i]) >= float(LSD[2]):
            print(f"有意水準10%:{ari}")
        else:
            print(f"有意水準10%:{nasi}")
    
        print("・・・・・・・・・・")


if __name__ == "__main__":
    n_hyp = "グループ間の差はない" # null hypothesis
    a_hyp = "グループ間の差はある" # altenative hypothesis

    lst, W_sos, B_sos, I_sos, Gro_ave, ms, dof = AoV_main()


    print("++++++++++")
    print("多重比較Tukey法")

    # Tukey ----------
    Tcv = []
    ng = 3 # number of group 群数
    I_dof = 417 # Inside degree of freedom 群内自由度 

    # 上を参考にGPT
    cv_1 = 4.02 # critical value 臨界値(1%)
    Tcv.append(cv_1)
    cv_5 = 3.34 # critical value 臨界値(5%)
    Tcv.append(cv_5)
    cv_10 = 2.98 # critical value 臨界値(10%)
    Tcv.append(cv_10)
    # ----------

    gro_ele = 7 # p_rewire_listの各groupの要素数 (7*3 = 21)

    I_ms, n = Tukey(lst, Gro_ave, ms, dof, n_hyp, a_hyp, Tcv, gro_ele)
    print("多重比較Tukey HSD法")
    Tukey_HSD(Tcv, Gro_ave, I_ms, n, n_hyp, a_hyp)
    
    # Fisher ----------
    Fcv = []
    ng = 3 # number of group 群数
    I_dof = 417 # Inside degree of freedom 群内自由度 

    # 上を参考にGPT
    cv_1 = 2.62 # critical value 臨界値(1%)
    Fcv.append(cv_1)
    cv_5 = 1.97 # critical value 臨界値(5%)
    Fcv.append(cv_5)
    cv_10 = 1.65 # critical value 臨界値(10%)
    Fcv.append(cv_10)
    # ----------

    print("多重比較Fisher LSD法")
    Fisher_LSD(Fcv, Gro_ave, I_ms, n, n_hyp, a_hyp)
