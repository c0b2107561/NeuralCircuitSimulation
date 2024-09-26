import csv
import pandas as pd
import numpy as np
# from echo_state_network import main
# from echo_state_network_v2 import main
# from echo_state_network_v3 import main
# from echo_state_network_v4 import main
from echo_state_network_v5 import main

def average(lst, p_rewire_list):
    sum_lst = []
    ave_lst = []

    for i in range(len(lst)):
        sum = 0
        ave = 0
        all_ave = 0
        ac = lst[i]
        # print("*****")
        # print(ac)

        for j in range(len(ac)):
            num = float(ac[j])
            sum += num
        # print("+++++")
        ave = float(sum) / len(ac)
        # print(sum)
        # print(ave)
        # print("~~~~~")
        sum_lst.append('{:.3f}'.format(sum))
        ave_lst.append('{:.3f}'.format(ave))
    print(f"合計{sum_lst}")
    print(f"平均{ave_lst}")
    for i in range(len(ave_lst)):
        all_ave += float(ave_lst[i])
    all_ave = '{:.3f}'.format(all_ave/len(ave_lst))
    print(f"全合計{all_ave}")

    return sum_lst, ave_lst, all_ave

def W_sum_of_squares(allave, lst):
    W_all_sos = []
    sos_lst = []
    for i in range(len(lst)):
        ac = lst[i]
        sumofsquares = 0
        num = 0
        sos = 0
        for j in range(len(ac)): 
            num = float(ac[j])
            sos = (num - float(allave))**2
            sos_lst.append('{:.3f}'.format(sos))
        # print(var_lst)
        for k in range(len(sos_lst)):
            sumofsquares += float(sos_lst[k])
        W_all_sos.append('{:.3f}'.format(sumofsquares))
        sos_lst.clear()
    print("+++++")
    print(f"全体平方和:{W_all_sos}")
    return W_all_sos

def group_separate(ave_lst, gro1, gro2): #gro_ave 要素
    gro1_lst = []
    gro2_lst = []
    gro3_lst = []

    gro1_lst = ave_lst[:gro1]
    gro2_lst = ave_lst[gro1:gro2]
    gro3_lst = ave_lst[gro2:]
    print("*****")
    # print(gro1_lst)
    # print(gro2_lst)
    # print(gro3_lst)

    return gro1_lst, gro2_lst, gro3_lst

def B_sum_of_squares(all_ave, gro1_lst, gro2_lst, gro3_lst):
    gro1_ave = 0
    gro2_ave = 0
    gro3_ave = 0
    
    all_gro_ave = []
    B_all_sos = []
    
    for i in range(len(gro1_lst)):
        gro1_ave += float(gro1_lst[i])
        gro1_ave = float(gro1_ave/len(gro1_lst))
    #print(gro1_ave)
    all_gro_ave.append('{:.3f}'.format(gro1_ave))
    gro1_sos = (float(gro1_ave) - float(all_ave))**2
    B_all_sos.append('{:.3f}'.format(gro1_sos))

    for i in range(len(gro2_lst)):
        gro2_ave += float(gro2_lst[i])
        gro2_ave = float(gro2_ave/len(gro2_lst))
    #print(gro2_ave)
    all_gro_ave.append('{:.3f}'.format(gro2_ave))
    gro2_sos = (float(gro2_ave) - float(all_ave))**2
    B_all_sos.append('{:.3f}'.format(gro2_sos))

    for i in range(len(gro3_lst)):
        gro3_ave += float(gro3_lst[i])
        gro3_ave = float(gro3_ave/len(gro3_lst))
    #print(gro3_ave)
    all_gro_ave.append('{:.3f}'.format(gro3_ave))
    gro3_sos = (float(gro3_ave) - float(all_ave))**2
    B_all_sos.append('{:.3f}'.format(gro3_sos))

    print(f"群平均:{all_gro_ave}")
    print(f"群間平方和:{B_all_sos}")
    return all_gro_ave, B_all_sos

def I_sum_of_squares(gro_ave, lst, gro1, gro2): #gro_ave 群平均
    I_all_sos = []
    sos_lst = []

    # print(lst)

    for i in range(len(lst)): #group毎
        ac = lst[i]
        sumofsquares = 0
        num = 0
        sos = 0
        if i < gro1:
            # print(f"goup1:{ac}")
            for j in range(len(ac)):
                num = float(ac[j])
                sos = (num - float(gro_ave[0]))**2
                sos_lst.append('{:.3f}'.format(sos))
            # print(var_lst)
            for k in range(len(sos_lst)):
                sumofsquares += float(sos_lst[k])

        elif gro1 <= i < gro2:
            # print(f"goup2:{ac}")
            for j in range(len(ac)):
                num = float(ac[j])
                sos = (num - float(gro_ave[1]))**2
                sos_lst.append('{:.3f}'.format(sos))
            # print(var_lst)
            for k in range(len(sos_lst)):
                sumofsquares += float(sos_lst[k])

        else:
            # print(f"goup3:{ac}")
            for j in range(len(ac)):
                num = float(ac[j])
                sos = (num - float(gro_ave[1]))**2
                sos_lst.append('{:.3f}'.format(sos))
            # print(var_lst)
            for k in range(len(sos_lst)):
                sumofsquares += float(sos_lst[k])

        I_all_sos.append('{:.3f}'.format(sumofsquares))
        sos_lst.clear()
    print("+++++")
    print(f"郡内平方和:{I_all_sos}")
    return I_all_sos

def analysis_of_variance_table(B_sos, I_sos, W_sos, lst, n_hyp, a_hyp, cv):
    B_sos_sum = 0
    I_sos_sum = 0
    W_sos_sum = 0
    sos = [] # sum of squares 平方和
    for i in range(len(B_sos)):
        B_sos_sum += float(B_sos[i])
    B_sos_sum = float(B_sos_sum)
    # print(f"群間平方和:{B_sos_sum}")
    sos.append('{:.3f}'.format(B_sos_sum))

    for i in range(len(I_sos)):
        I_sos_sum += float(I_sos[i])
    I_sos_sum = float(I_sos_sum)
    # print(f"群内平方和:{I_sos_sum}")
    sos.append('{:.3f}'.format(I_sos_sum))

    for i in range(len(W_sos)):
        W_sos_sum += float(W_sos[i])
    W_sos_sum = float(W_sos_sum)
    # print(f"全体平方和:{W_sos_sum}")
    sos.append('{:.3f}'.format(W_sos_sum))

    dof = [] #degree of freedom 自由度
    B_dof = len(B_sos) - 1
    dof.append(B_dof)
    
    num = 0
    for i in range(len(lst)):
        lsts = lst[i]
        for j in range(len(lsts)):
            num += 1
    print(num)

    I_dof = num - 3
    dof.append(I_dof)
    W_dof = num - 1
    dof.append(W_dof)

    ms = [] # mean square 平方平均
    B_ms = B_sos_sum / B_dof
    ms.append('{:.3f}'.format(B_ms))
    I_ms = I_sos_sum / I_dof
    ms.append('{:.3f}'.format(I_ms))
    W_ms = W_sos_sum / W_dof
    ms.append('{:.3f}'.format(W_ms))

    F_m = '{:.3f}'.format(B_ms/I_ms) # F_measure F値


    print("群間, 郡内, 全体")
    print(f"平方和:{sos}")
    print(f"自由度:{dof}")
    print(f"平方平均:{ms}")
    print(f"F値:{F_m}")
    print(f"臨界値:{cv}")

    
    ari = "帰無仮説棄却，有意差あり"
    nasi = "帰無仮説採択，有意差なし"

    print(f"帰無仮説:{n_hyp}")
    print(f"対立仮説:{a_hyp}")

    if float(F_m) >= float(cv[0]):
        print(f"有意水準1%:{ari}")
    else:
        print(f"有意水準1%:{nasi}")

    if float(F_m) >= float(cv[1]):
        print(f"有意水準5%:{ari}")
    else:
        print(f"有意水準5%:{nasi}")

    if float(F_m) >= float(cv[2]):
        print(f"有意水準10%:{ari}")
    else:
        print(f"有意水準10%:{nasi}")

    return sos, dof, ms, F_m

def AoV_main():
    p_rewire_list = [0.0, 0.002, 0.005, 0.01, 0.02, 0.35, 0.05, 
                     0.1, 0.13, 0.17, 0.2, 0.23, 0.27,  0.3, 
                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    group1 = 7
    group2 = 14

    # p_rewire_list = [0.0, 0.15, 
    #                  0.3, 0.4, 
    #                  0.9, 1.0]
    # group1 = 2
    # group2 = 4

    # p_rewire_list = [0.0, 
    #                  0.3, 
    #                  1.0]
    # group1 = 1
    # group2 = 2
    
    lst = main()
    print(lst)

    cv = []
    df1 = 2 # 分子の自由度
    df2 = 207 # 分母の自由度
    cv_1 = 4.78 # critical value 臨界値
    cv.append(cv_1)
    cv_5 = 3.05
    cv.append(cv_5)
    cv_10 = 2.31
    cv.append(cv_10)

    n_hyp = "グループ間の差はない" # null hypothesis
    a_hyp = "グループ間の差はある" # altenative hypothesis

    print("分散分析")

    sum, ave, allave = average(lst, p_rewire_list) # 列和lst，列平均lst，全平均
    W_sos = W_sum_of_squares(allave, lst) # 全体平方和lst
    gro1, gro2, gro3 = group_separate(ave, group1, group2) # 組分け
    Gro_ave, B_sos = B_sum_of_squares(allave, gro1, gro2, gro3) # 群平均lst, 群間平方和lst
    I_sos = I_sum_of_squares(Gro_ave, lst, group1, group2) #群内平方和lst
    sos, dof, ms, F_m = analysis_of_variance_table(B_sos, I_sos, W_sos, lst, n_hyp, a_hyp, cv)

    return lst, W_sos, B_sos, I_sos, Gro_ave, ms, dof

if __name__ == "__main__":
    AoV_main()
