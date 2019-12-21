import math
import random
from operator import itemgetter
import copy
import pandas as pd
from ast import literal_eval
from multiprocessing import cpu_count, Process, Pipe
from pulp import *
import csv
import time
from scipy.stats import sem, t
from scipy import mean

node_pos = [(10, 10), (30, 30), (50, 50), (70, 70), (90, 90),
            (10, 30), (30, 10), (30, 50), (50, 30), (50, 70)]
charge_pos = [(10, 50), (90, 50)]
time_move = [1.019803902718557, 1.6, 2.0591260281974]
E = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
e = [0.2, 0.3, 0.3, 0.5, 0.6, 0.2, 0.6, 0.4, 0.6, 0.3]
numNode = len(node_pos)
numCharge = len(charge_pos)
E_mc = 5  # nang luong khoi tao cua MC
e_mc = 1  # cong suat sac moi giay
E_max = 10.0  # nang luong toi da
e_move = 0.1  # nang luong tieu thu moi giay cho viec di chuyen
E_move = [e_move * time_move_i for time_move_i in time_move]  # nang luong tieu thu de di chuyen toi moi charge position
chargeRange = 10 ** 10
velocity = 0.0
alpha = 600
beta = 30
charge = []
delta = [[0 for u, _ in enumerate(charge_pos)] for j, _ in enumerate(node_pos)]
depot = (0, 0)


def getData(file_name="data.csv", index=0):
    global node_pos
    global numNode
    global E
    global e
    global charge_pos
    global numCharge
    global time_move
    global E_mc
    global e_mc
    global E_max
    global e_move
    global E_move
    global alpha
    global beta
    global velocity, depot

    df = pd.read_csv(file_name)
    node_pos = list(literal_eval(df.node_pos[index]))
    numNode = len(node_pos)
    E = [df.energy[index] for _ in node_pos]
    e = map(float, df.e[index].split(","))
    charge_pos = list(literal_eval(df.charge_pos[index]))
    numCharge = len(charge_pos)
    velocity = df.velocity[index]
    E_mc = df.E_mc[index]
    E_max = df.E_max[index]
    e_mc = df.e_mc[index]
    e_move = df.e_move[index]
    alpha = df.alpha[index]
    beta = df.beta[index]
    depot = literal_eval(df.depot[index])

    charge_extend = []
    charge_extend.extend(charge_pos)
    charge_extend.append(depot)
    time_move = [[distance(pos1, pos2) / velocity for pos2 in charge_extend] for pos1 in charge_extend]

    tmp = [time_move[i][i + 1] * e_move for i in range(len(time_move) - 1)]
    E_move = [time_move[-1][0] * e_move]
    E_move.extend(tmp)


def distance(node1, node2):
    return math.sqrt((node1[0] - node2[0]) * (node1[0] - node2[0])
                     + (node1[1] - node2[1]) * (node1[1] - node2[1]))


def charging(node, charge):
    d = distance(node, charge)
    if d > chargeRange:
        return 0
    else:
        return alpha / ((d + beta) ** 2)


def getWeightLinearPrograming():
    model = LpProblem("Charge", LpMinimize)
    x = LpVariable.matrix("x", list(range(numCharge)), 0, None, LpContinuous)
    t = LpVariable.matrix("t", list(range(numNode)), 0, None, LpContinuous)
    for node_index, _ in enumerate(node_pos):
        model += lpSum(
            [x[charge_index] * delta[node_index][charge_index] for charge_index, _ in enumerate(charge_pos)]) - e[
                     node_index] <= t[node_index]
        model += lpSum(
            [x[charge_index] * delta[node_index][charge_index] for charge_index, _ in enumerate(charge_pos)]) - e[
                     node_index] >= -t[node_index]
    model += lpSum(t)
    status = model.solve()
    if status == 1:
        valueX = [value(item) for item in x]
        if sum(valueX):
            weight = [(i, item / sum(valueX)) for i, item in enumerate(valueX)]
        else:
            weight = [1.0 / len(charge_pos) for _ in charge_pos]
    else:
        weight = -1
        print "Can not solve LP"
    return weight


def genRoundUniform(E_mc_now, E_now):
    x = [0 for charge_index, _ in enumerate(charge_pos)]
    # array store energy of sensor at new route
    eNode = [E_now[node_index] / e[node_index] for node_index, _ in enumerate(node_pos) if e[node_index] > 0]
    T_max = min(min(eNode), (E_max - E_mc_now) / e_mc)
    T_min = max(0, (sum(E_move) - E_mc_now) / e_mc)

    if T_max >= T_min:
        # value of T
        T = T_max - 0.2 * abs(T_max - T_min) * random.random()
        # Energy of MC and sensor before u location
        E_mc_new = E_mc_now + T * e_mc - sum(E_move)

        energy_receive = [sum([charge[node_index][charge_index] for node_index, _ in enumerate(node_pos)]) for
                          charge_index, _ in
                          enumerate(charge_pos)]
        tmp = (E_mc_new - sum(E_move)) / sum(energy_receive)
        for charge_index, _ in enumerate(charge_pos):
            x[charge_index] = tmp + 0.5 * tmp * (2 * random.random() - 1)
        a = [charge_index for charge_index, _ in enumerate(charge_pos)]
        random.shuffle(a)
        return [T, a, x]
    else:
        return -1


def genRoundRandom(E_mc_now, E_now):
    x = [0 for charge_index, _ in enumerate(charge_pos)]
    # array store energy of sensor at new route
    eNode = [E_now[node_index] / e[node_index] for node_index, _ in enumerate(node_pos) if e[node_index] > 0]
    T_max = min(min(eNode), (E_max - E_mc_now) / e_mc)
    T_min = max(0, (sum(E_move) - E_mc_now) / e_mc)

    if T_max >= T_min:
        # value of T
        T = T_max - 0.2 * abs(T_max - T_min) * random.random()
        # Energy of MC and sensor before u location
        E_mc_new = E_mc_now + T * e_mc - sum(E_move)

        energy_receive = [sum([charge[node_index][charge_index] for node_index, _ in enumerate(node_pos)]) for
                          charge_index, _ in
                          enumerate(charge_pos)]
        p = [charge_index for charge_index, _ in enumerate(charge_pos)]
        random.shuffle(p)
        for charge_index in p:
            x[charge_index] = random.random() * E_mc_new / energy_receive[charge_index]
            E_mc_new -= energy_receive[charge_index] * x[charge_index]
        a = [charge_index for charge_index, _ in enumerate(charge_pos)]
        random.shuffle(a)
        return [T, a, x]
    else:
        return -1


def genRoundGreedy(E_mc_now, E_now, weight):
    # array store energy of sensor at new route
    eNode = [E_now[node_index] / e[node_index] for node_index, _ in enumerate(node_pos)]
    T_max = min(min(eNode), (E_max - E_mc_now) / e_mc)
    T_min = max(0, (sum(E_move) - E_mc_now) / e_mc)

    if T_max >= T_min:
        T = T_max - 0.2 * abs(T_max - T_min) * random.random()
        E_mc_new = E_mc_now + T * e_mc - sum(E_move)
    else:
        return -1
    print weight
    temp = sorted(weight, key=itemgetter(1), reverse=False)
    minValue = 1.0
    weight = [0.0 for _ in weight]
    for item_index, item in enumerate(temp):
        charge_index, wu = item
        if wu == 0:  # neu wu = 0 thi gan bang gia tri minvalue
            #            print True
            weight[charge_index] = minValue
        else:
            # if wu is first not zero element of w, wu = 10 * min value
            if item_index == 0 or temp[item_index - 1][1] == 0:
                weight[charge_index] = 10.0 * minValue
            else:
                r = random.random()
                # print r
                pre_u, pre_wu = temp[item_index - 1]
                if r <= 0.8:
                    weight[charge_index] = wu / pre_wu * weight[pre_u] + minValue * random.random()
                else:
                    weight[charge_index] = wu / pre_wu * weight[pre_u] - minValue * random.random()
    weight_x = [weight[charge_index] / sum(weight) for charge_index, _ in enumerate(weight)]
    energy_receive = [sum([charge[node_index][charge_index] for node_index, _ in enumerate(node_pos)]) for
                      charge_index, _ in
                      enumerate(charge_pos)]
    #    print weight_x
    t = E_mc_new / sum(
        [weight_x[charge_index] * energy_receive[charge_index] for charge_index, _ in enumerate(charge_pos)])
    x = [weight_x[charge_index] * t for charge_index, _ in enumerate(charge_pos)]
    a = [charge_index for charge_index, _ in enumerate(charge_pos)]
    random.shuffle(a)
    return [T, a, x]


def genRoundZero():
    T = 0.0
    x = [0.0 for _ in charge_pos]
    a = [charge_index for charge_index, _ in enumerate(charge_pos)]
    random.shuffle(a)
    return [T, a, x]


def individual(w, p_ran, p_uni):
    indi = {"T": [], "move": [], "gen": [], "remain": 0.0, "fitness": 0.0, "num_gen": 1}  # so luong chu ki sac
    # so luong chu ki sac
    r = random.random()
    if r <= p_ran:
        T, a, x = genRoundRandom(E_mc, E)
    elif p_ran < r <= p_ran + p_uni:
        T, a, x = genRoundUniform(E_mc, E)
    else:
        T, a, x = genRoundGreedy(E_mc, E, w)
    indi["T"].append(T)
    indi["move"].append(a)
    indi["gen"].append(x)
    indi = injust(indi)
    return indi


def fitness(indi):
    total = 0.0
    for index in range(indi["num_gen"]):
        total += indi["T"][index]
        move = indi["move"][index]
        gen = indi["gen"][index]
        row = [(u, gen[u]) for u in move if gen[u] >= 1]
        if not row:
            continue
        for id, current in enumerate(row):
            u, xu = current
            if id == 0:
                time = distance(depot, charge_pos[u]) / velocity
            else:
                pre = row[id - 1]
                pre_u, pre_xu = pre
                time = distance(charge_pos[pre_u], charge_pos[u]) / velocity
            total += time + xu
        if index != indi["num_gen"] - 1:
            last_u, _ = row[-1]
            total += distance(depot, charge_pos[last_u])
    total += indi["remain"]
    return total


def selectionBest(popu):
    new_list = sorted(popu, key=itemgetter("fitness"), reverse=True)
    return new_list[:population_size]


def selectionRoulette(popu):
    sum_fit = sum([indi["fitness"] for indi in popu])
    weight_fit = [indi["fitness"] / sum_fit for indi in popu]
    range_fit = [sum(weight_fit[:i]) for i in range(len(weight_fit))]
    new_pop = []
    max_fit = 0.0
    indi = {}
    for item in popu:
        if item["fitness"] > max_fit:
            max_fit = indi["fitness"]
            indi = item
    new_pop.append(indi)
    id = 1
    while id < population_size:
        id = 1


def selectionTwoType(popu):
    sorted_gen = sorted(popu, key=itemgetter("num_gen"), reverse=True)
    new_list = copy.copy(sorted_gen[:population_size / 2])
    sorted_fitness = sorted(sorted_gen[population_size / 2:], key=itemgetter("fitness"), reverse=True)
    new_list.extend((sorted_fitness[:population_size / 2]))
    return new_list[:population_size]


# def injust1(indi):
#     E_mc_now = E_mc
#     E_now = [item for item in E]
#
#     off = {}
#     off["T"] = []
#     off["gen"] = []
#     off["remain"] = -1
#
#     isStop = False
#     for index, gen in enumerate(indi["gen"]):
#         T_max = (E_max - E_mc_now) / e_mc
#         T = min(T_max, indi["T"][index])
#         temp_E = [E_now[j] - T * e[j] for j, _ in enumerate(node_pos)]
#         temp_E_mc = E_mc_now + T * e_mc
#         # row chua vi tri va thoi gian sac cua nhung diem sac co thoi gian sac > 0
#         row = [(u, xu) for u, xu in enumerate(gen) if xu > 0]
#         # neu tat ca cac xu = 0 thi bo qua chu ki nay va tinh toan den chu ki tiep theo
#         if not row:
#             isStop = True
#             continue
#         u_first, _ = row[0]
#         eNode = min([temp_E[j] - time_move[-1][u_first] * e[j] for j, _ in enumerate(node_pos)])
#
#         if eNode < 0 or temp_E_mc < sum(E_move):
#             off["remain"] = min([E_now[j] / e[j] for j, _ in enumerate(node_pos)])
#             """if index == 0:
#                 off["remain"] = min([E_now[j] / e[j] for j, _ in enumerate(node_pos)])
#             else:
#                 pre_row = [(u, xu) for u, xu in enumerate(indi["gen"][index-1]) if xu > 0]
#                 pre_u, _ = pre_row[-1]
#                 off["remain"] = min([E_now[j] / e[j] + time_move[-1][pre_u] for j, _ in enumerate(node_pos)])"""
#             break
#         else:
#             E_mc_now = temp_E_mc
#             E_now = temp_E
#             off["T"].append(T)
#             x = [0 for u, _ in enumerate(charge_pos)]
#             for id, current in enumerate(row):
#                 u, xu = current
#                 if id == 0:
#                     time = time_move[-1][u]
#                 else:
#                     pre = row[id - 1]
#                     pre_u, pre_xu = pre
#                     time = time_move[pre_u][u]
#                 p = [min(charge[j][u] * xu, E[j] - E_now[j] + (time + xu) * e[j]) for j, node in enumerate(node_pos)]
#                 temp_E_mc = E_mc_now - sum(p) - time * e_move
#                 temp_E = [E_now[j] + p[j] - (time + xu) * e[j] for j, _ in enumerate(node_pos)]
#
#                 if min(temp_E) < 0 or temp_E_mc < sum(E_move[u + 1:]):
#                     isStop = True
#                     break
#                 else:
#                     x[u] = xu
#                     E_mc_now = temp_E_mc
#                     E_now = temp_E
#             off["gen"].append(x)
#
#             if not isStop:
#                 u_last, _ = row[-1]
#                 E_mc_now = E_mc_now - time_move[-1][u_last] * e_move
#                 E_now = [E_now[j] - time_move[-1][u_last] * e[j] for j, _ in enumerate(node_pos)]
#             else:
#                 break
#
#     """idRound = len(off["gen"]) - 1
#     #    print len(off["gen"]), idRound, indi["num_gen"]
#     while idRound < indi["num_gen"] - 1:
#         off["T"].append(0.0)
#         off["gen"].append([0.0 for _ in charge_pos])
#         idRound += 1"""
#
#     off["num_gen"] = len(off["gen"])
#     if off["remain"] == -1:
#         off["remain"] = min([E_now[j] / e[j] for j, _ in enumerate(node_pos)])
#     off["fitness"] = fitness(off)
#     #   print off["num_gen"]
#     return off


def injust(indi):
    E_mc_now = E_mc
    E_now = [item for item in E]

    off = {"T": [], "move": [], "gen": [], "remain": -1}

    isStop = False
    for index, gen in enumerate(indi["gen"]):
        move = indi["move"][index]
        T_max = (E_max - E_mc_now) / e_mc
        T = min(T_max, indi["T"][index])
        temp_E = [E_now[j] - T * e[j] for j, _ in enumerate(node_pos)]
        temp_E_mc = E_mc_now + T * e_mc
        # row chua vi tri va thoi gian sac cua nhung diem sac co thoi gian sac > 0
        row = [(u, gen[u]) for u in move if gen[u] >= 1]
        # neu tat ca cac xu = 0 thi bo qua chu ki nay va tinh toan den chu ki tiep theo
        if not row:
            isStop = True
            continue
        u_first, _ = row[0]
        eNode = min([temp_E[j] - distance(depot, charge_pos[u_first]) * e[j] / velocity for j, _ in enumerate(node_pos)])

        if eNode < 0 or temp_E_mc < sum(E_move):
            off["remain"] = min([E_now[j] / e[j] for j, _ in enumerate(node_pos) if e[j] > 0])
            break
        else:
            E_mc_now = temp_E_mc
            E_now = temp_E
            off["T"].append(T)
            x = [0 for u, _ in enumerate(charge_pos)]
            for id, current in enumerate(row):
                u, xu = current
                if id == 0:
                    time = distance(depot, charge_pos[u]) / velocity
                else:
                    pre = row[id - 1]
                    pre_u, pre_xu = pre
                    time = distance(charge_pos[pre_u], charge_pos[u]) / velocity

                eNode = min([E_now[j] - time * e[j] for j, _ in enumerate(node_pos)])
                if eNode < 0:
                    isStop = True
                    break

                p = [min(charge[j][u] * xu, E[j] - E_now[j] + (time + xu) * e[j]) for j, node in enumerate(node_pos)]
                temp_E_mc = E_mc_now - sum(p) - time * e_move
                temp_E = [E_now[j] + p[j] - (time + xu) * e[j] for j, _ in enumerate(node_pos)]

                if min(temp_E) < 0 or temp_E_mc < sum(E_move[u + 1:]):
                    isStop = True
                    break
                else:
                    x[u] = xu
                    E_mc_now = temp_E_mc
                    E_now = temp_E
            off["gen"].append(x)
            off["move"].append(move)
            if not isStop:
                u_last, _ = row[-1]
                E_mc_now = E_mc_now - distance(depot, charge_pos[u_last]) * e_move / velocity
                E_now = [E_now[j] - distance(depot, charge_pos[u_last]) * e[j] / velocity for j, _ in enumerate(node_pos)]
            else:
                break

    off["num_gen"] = len(off["gen"])
    if off["remain"] == -1:
        off["remain"] = min([E_now[j] / e[j] for j, _ in enumerate(node_pos) if e[j] > 0])
    off["fitness"] = fitness(off)
    #   print off["num_gen"]
    return off


def BLX(gen1, gen2):
    temp = []
    for x, y in zip(gen1, gen2):
        low = max(min(x, y) - abs(x - y) / 2.0, 0.0)
        upp = max(x, y) + abs(x - y) / 2.0
        temp.append(random.random() * (upp - low) + low)
    return temp


def OX(gen1, gen2):
    a = [i for i, _ in enumerate(charge_pos)]
    random.shuffle(a)
    if gen1 is None:
        temp1 = a
    else:
        temp1 = gen1
    if gen2 is None:
        temp2 = a
    else:
        temp2 = gen2
    off = []
    n = len(temp1)
    cutA = random.randint(1, n - 1)
    cutB = random.randint(1, n - 1)
    while cutB == cutA:
        cutB = random.randint(1, n - 1)
    start = min(cutA, cutB)
    end = max(cutA, cutB)
    # print start, end
    temp = temp2[start:end]
    # print temp
    index = 0
    while index < start:
        for item in temp1:
            if item not in temp and item not in off:
                off.append(item)
                index = index + 1
                break
    off.extend(temp)
    for item in temp1:
        if item not in off:
            off.append(item)
    # print "temp1 =", gen1, "temp2 =", gen2, "Off =", off
    return off


# def OX(gen1, gen2):
#     off = []
#     n = len(gen1)
#     cutA = random.randint(1, n - 1)
#     cutB = random.randint(1, n - 1)
#     start = min(cutA, cutB)
#     end = max(cutA, cutB)
#     temp = gen2[start: end]
#     off = [item for item in gen1 if item not in temp]
#     off.extend(temp)
#     return off


def crossover(father, mother):
    off = {}
    f = father["num_gen"]
    m = mother["num_gen"]

    if f == m:
        off["num_gen"] = f
        off["T"] = BLX(father["T"], mother["T"])
        off["move"] = [OX(father["move"][i], mother["move"][i]) for i, _ in enumerate(father["move"])]
        off["gen"] = [BLX(father["gen"][i], mother["gen"][i]) for i, _ in enumerate(father["gen"])]
    elif f > m:
        off["num_gen"] = f

        tempT = [mother["T"][i] if i < m else 0 for i, _ in enumerate(father["T"])]
        off["T"] = BLX(father["T"], tempT)

        zeroGen = [0 for _ in charge_pos]
        zeroMove = [i for i, _ in enumerate(charge_pos)]
        tempGen = [mother["gen"][i] if i < m else zeroGen for i, _ in enumerate(father["gen"])]
        tempMove = [mother["move"][i] if i < m  and mother["move"][i] is not None else random.shuffle(zeroMove) for i, _ in enumerate(father["move"])]

        off["move"] = [OX(father["move"][i], tempMove[i]) for i, _ in enumerate(father["move"])]
        off["gen"] = [BLX(father["gen"][i], tempGen[i]) for i, _ in enumerate(father["gen"])]
    else:
        off["num_gen"] = m

        tempT = [father["T"][i] if i < f else 0 for i, _ in enumerate(mother["T"])]
        off["T"] = BLX(mother["T"], tempT)

        zeroGen = [0 for _ in charge_pos]
        zeroMove = [i for i, _ in enumerate(charge_pos)]
        tempGen = [father["gen"][i] if i < f else zeroGen for i, _ in enumerate(mother["gen"])]
        tempMove = [father["move"][i] if i < f and father["move"][i] is not None else random.shuffle(zeroMove) for i, _ in enumerate(mother["move"])]

        off["move"] = [OX(mother["move"][i], tempMove[i]) for i, _ in enumerate(mother["move"])]
        off["gen"] = [BLX(mother["gen"][i], tempGen[i]) for i, _ in enumerate(mother["gen"])]
    off = injust(off)
    return off


# def crossover(father, mother):
#     off = {}
#     f = father["num_gen"]
#     m = mother["num_gen"]
#     zeroGen = [0 for _ in charge_pos]
#     zeroMove = [i for i, _ in enumerate(charge_pos)]
#
#     if f == m:
#         off["num_gen"] = f
#         off["T"] = BLX(father["T"], mother["T"])
#         off["gen"] = [BLX(father["gen"][i], mother["gen"][i]) for i, _ in enumerate(father["gen"])]
#         off["move"] = [OX(father["move"][i], mother["move"][i]) for i, _ in enumerate(father["move"])]
#     elif f > m:
#         off["num_gen"] = f
#
#         tempT = [mother["T"][i] if i < m else 0 for i, _ in enumerate(father["T"])]
#         off["T"] = BLX(father["T"], tempT)
#
#         tempGen = [mother["gen"][i] if i < m else zeroGen for i, _ in enumerate(father["gen"])]
#         tempMove = [mother["move"][i] if i < m else random.shuffle(zeroMove) for i, _ in enumerate(father["move"])]
#         off["gen"] = [BLX(father["gen"][i], tempGen[i]) for i, _ in enumerate(father["gen"])]
#         off["move"] = [OX(father["move"][i], tempMove[i]) for i, _ in enumerate(father["move"])]
#     else:
#         off["num_gen"] = m
#
#         tempT = [father["T"][i] if i < f else 0 for i, _ in enumerate(mother["T"])]
#         off["T"] = BLX(mother["T"], tempT)
#
#         tempGen = [father["gen"][i] if i < f else zeroGen for i, _ in enumerate(mother["gen"])]
#         tempMove = [father["move"][i] if i < f else random.shuffle(zeroMove) for i, _ in enumerate(mother["move"])]
#         off["gen"] = [BLX(mother["gen"][i], tempGen[i]) for i, _ in enumerate(mother["gen"])]
#         off["move"] = [OX(mother["gen"][i], tempMove[i]) for i, _ in enumerate(mother["move"])]
#     off = injust(off)
#     return off


def mutation(indi, m_ran, m_uni, w):
    off = copy.copy(indi)

    E_mc_now = E_mc
    E_now = [E[j] for j, _ in enumerate(node_pos)]
    energy_add = [0 for k, _ in enumerate(node_pos)]
    for k, _ in enumerate(off["gen"]):
        E_mc_now = E_mc_now + off["T"][k] * e_mc
        E_now = [E_now[j] - off["T"][k] * e[j] for j, _ in enumerate(node_pos)]
        gen = indi["gen"][k]
        move = indi["move"][k]
        row = [(u, gen[u]) for u in move if gen[u] > 0]
        if not row:
            continue
        for id, current in enumerate(row):
            u, xu = current
            if id == 0:
                time = distance(depot, charge_pos[u]) / velocity
            else:
                pre = row[id - 1]
                pre_u, pre_xu = pre
                time = distance(charge_pos[pre_u], charge_pos[u]) / velocity
            p = [min(charge[j][u] * xu, E[j] - E_now[j] + (time + xu) * e[j]) for j, node in enumerate(node_pos)]
            E_mc_now = E_mc_now - sum(p) - time * e_move
            E_now = [E_now[j] + p[j] - (time + xu) * e[j] for j, _ in enumerate(node_pos)]
        last_u, _ = row[-1]
        E_mc_now -= distance(depot, charge_pos[last_u]) * e_move / velocity
        E_now = [E_now[j] - distance(depot, charge_pos[last_u]) * e[j] / velocity for j, _ in enumerate(node_pos)]

    if min(E_now) < 0 or E_mc_now < 0:
        # mang khong du nang luong de sinh round moi
        return -1
    else:
        r = random.random()
        if r <= m_ran:
            tmp = genRoundRandom(E_mc_now, E_now)
        elif m_ran < r <= m_ran + m_uni:
            tmp = genRoundUniform(E_mc_now, E_now)
        else:
            tmp = genRoundGreedy(E_mc_now, E_now, w)

        if tmp != -1:
            T, a, x = tmp
            off["T"].append(T)
            off["move"].append(a)
            off["gen"].append(x)
            off = injust(off)
            return off
        else:
            # mang khong du nang luong de sinh round moi
            return -1


def genetic(start, end, pc, pm, m_ran, m_uni, w, connection):
    global population
    sub_pop = []
    count = 0
    i = start
    while i < end:
        rc = random.random()
        rm = random.random()
        if rc <= pc:
            j = random.randint(0, population_size - 1)
            while j == i:
                j = random.randint(0, population_size - 1)
            child = crossover(population[i], population[j])
            mutated_child = mutation(child, m_ran, m_uni, w)
            if mutated_child != -1:
                count += 1
                sub_pop.append(mutated_child)
            else:
                sub_pop.append(child)
        if rm <= pm:
            mutated_child = mutation(population[i], m_ran, m_uni, w)
            if mutated_child != -1:
                #  print True
                count += 1
                sub_pop.append(mutated_child)
        i += 1
    connection.send([count, sub_pop])
    connection.close()


def evolution(maxIterator, pc, pm, m_ran, m_uni, w):
    global population
    bestFitness = 0.0
    nbIte = 0
    t = 0
    conv = []
    while t < maxIterator and nbIte < 200:
        # print "t = ", t, "Fitness = ", population[0]["fitness"]
        count = 0  # dem so lan mutation
        nproc = cpu_count()
        process = []
        connection = []

        for pid in range(nproc):
            connection.append(Pipe())
        for pid in range(nproc):
            pro = Process(target=genetic, args=(10 * pid, 10 * (pid + 1), pc, pm, m_ran, m_uni, w, connection[pid][1]))
            process.append(pro)
            pro.start()
        for pid in range(nproc):
            nbMutation, sub_pop = connection[pid][0].recv()
            count += nbMutation
            population.extend(sub_pop)
            process[pid].join()
        try:
            population = selectionBest(population)
            if population[0]["fitness"] - bestFitness >= 1:
                bestFitness = population[0]["fitness"]
                nbIte = 0
            else:
                nbIte = nbIte + 1

            if t % 10 == 0:
                conv.append((t, population[0]["fitness"]))
        except:
            print population
            break
        # max_gen = population[0]["num_gen"]
        # population = selectionBest(population)
        if t % 10 == 0:
            print t, count, round(population[0]["fitness"], 1), population[0]["num_gen"]
        t += 1
    # population = selectionBest(population)
    return population[0], conv


def test(indi):
    E_now = E
    E_mc_now = E_mc
    for index in range(len(indi["T"])):
        T = indi["T"][index]
        gen = indi["gen"][index]
        E_now = [E_now[j] - T * e[j] for j, _ in enumerate(node_pos)]
        E_mc_now = E_mc_now + T * e_mc
        print "E_mc = ", round(E_mc_now, 2), "min E = ", round(min(E_now), 2), "max E = ", round(max(E_now),
                                                                                                 2), "T = ", T
        row = [(u, xu) for u, xu in enumerate(gen) if xu > 0]
        if not row:
            break
        for id, current in enumerate(row):
            u, xu = current
            if id == 0:
                time = time_move[-1][u]
            else:
                pre_u, pre_xu = row[id - 1]
                time = time_move[pre_u][u]
            p = [min(charge[j][u] * xu, E[j] - E_now[j] + (time + xu) * e[j]) for j, _ in enumerate(node_pos)]
            E_mc_now = E_mc_now - sum(p) - time * e_move
            E_now = [E_now[j] + p[j] - (time + xu) * e[j] for j, _ in enumerate(node_pos)]
            print "E_mc = ", round(E_mc_now, 2), "min E = ", round(min(E_now), 2), "max E = ", round(max(E_now),
                                                                                                     2), "xu = ", xu, "max_charge = ", max(
                charge[:][u])
        last_u, _ = row[-1]
        E_mc_now -= time_move[-1][last_u] * e_move
        E_now = [E_now[j] - time_move[-1][last_u] * e[j] for j, _ in enumerate(node_pos)]
        print "E_mc = ", round(E_mc_now, 2), "min E = ", round(min(E_now), 2), "max E = ", round(max(E_now), 2)


# main task
index = 0

while index < 1:
    print "Data Set ", index
    if index == 2:
        index = index + 1
        continue
    file_name = "GA/DataSet" + str(index) + ".csv"
    f = open(file_name, mode="w")
    header = ["Lan Chay", "Time", "Co Sac", "Khong Sac"]
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()

    sum_lifetime = 0.0
    sum_time = 0.0
    nbRun = 5
    conv = []
    confidence_interval = []
    for idRun in range(nbRun):
        start_time = time.time()

        random.seed(idRun)
        getData(file_name="thaydoisonode.csv", index=index)
        population_size = 10 * cpu_count()
        charge = [[charging(node, pos) for u, pos in enumerate(charge_pos)] for j, node in enumerate(node_pos)]
        delta = [[charge[j][u] - e[j] for u, _ in enumerate(charge_pos)] for j, _ in enumerate(node_pos)]
        # print max(charge)
        # print max(e)
        w = getWeightLinearPrograming()
        population = [individual(w=w, p_ran=0.5, p_uni=0.5) for _ in range(population_size)]
        # for indi in population:
        #     print indi["num_gen"], len(indi["T"]), len(indi["move"]), len(indi["gen"])
        # for indi in population:
        #     print indi
        indi, conv = evolution(maxIterator=5000, pc=0.8, pm=0.5, m_ran=0.5, m_uni=0.5, w=w)

        end_time = time.time()
        sum_lifetime = sum_lifetime + indi["fitness"]
        sum_time = sum_time + end_time - start_time
        # write to file
        row = {"Lan Chay": "No." + str(idRun), "Time": end_time - start_time, "Co Sac": indi["fitness"],
               "Khong Sac": min([E[j] / e[j] for j, _ in enumerate(node_pos) if e[j] > 0])}
        writer.writerow(row)
        confidence_interval.append(indi["fitness"])
        idRun = idRun + 1

    row = {"Lan Chay": "Average", "Time": sum_time / nbRun, "Co Sac": sum_lifetime / nbRun,
           "Khong Sac": min([E[j] / e[j] for j, _ in enumerate(node_pos) if e[j] > 0])}
    writer.writerow(row)

    confidence = 0.95
    n = len(confidence_interval)
    m = mean(confidence_interval)
    std_err = sem(confidence_interval)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    row = {"Co Sac": h}
    writer.writerow(row)
    f.close()

    print "Done Data Set ", index
    index = index + 1

print "Done All"
