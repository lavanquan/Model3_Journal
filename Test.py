import random

a = [1, 2, 3, 4, 5, 6, 7]
b = [2, 7, 4, 1, 6, 5, 3]


def OX(gen1, gen2):
    off = []
    n = len(gen1)
    cutA = random.randint(1, n - 1)
    cutB = random.randint(1, n - 1)
    start = min(cutA, cutB)
    end = max(cutA, cutB)
    print start, end
    temp = gen2[start: end]
    print temp
    index = 0
    while index < start:
        for item in gen1:
            if item not in temp and item not in off:
                off.append(item)
                index = index + 1
                break
    print off
    off.extend(temp)
    print off
    for item in gen1:
        if item not in off:
            off.append(item)
    return off


print OX(a, b)
