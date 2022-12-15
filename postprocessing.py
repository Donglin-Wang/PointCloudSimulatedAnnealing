import pickle

from tabulate import tabulate

CATS = {'Motorbike': 6, 'Guitar': 3, 'Rocket': 3, 'Cap': 2, 'Bag': 2, 'Airplane': 4, 'Lamp': 4,
                    'Car': 4, 'Skateboard': 3, 'Table': 3, 'Mug': 2, 'Knife': 2, 'Chair': 4, 'Laptop': 2, 'Pistol': 3, 'Earphone': 3}

logs = ['r10s20k32o2i1000_acc_2022-12-06 13:20:51.818979.pkl',
        'r10s20k32o4i500_acc_2022-12-06 15:31:01.887125.pkl',
        'r10s20k32o10i200_acc_2022-12-06 19:01:15.552147.pkl']

# logs = ['r10s20k8o10i200_acc_2022-12-09 13:31:02.684263.pkl', 'r10s20k16o10i200_acc_2022-12-08 18:36:43.797016.pkl',
# 'r10s20k32o10i200_acc_2022-12-06 19:01:15.552147.pkl', 'r10s20k64o10i200_acc_2022-12-08 23:39:18.752411.pkl']

n = len(logs) * 2
n_cats = len(CATS)

print("&".join(list(CATS.keys())))

table = {k:[] for k in CATS.keys()}

for i, log in enumerate(logs):
    log = pickle.load(open('./logs/' + log, 'rb'))
    for k, v in log.items():
        table[k] += [round(ele, 6) for ele in v]

avg = [0] * n
for k, v in table.items():
    for i, ele in enumerate(v):
        avg[i] += ele

avg = [round(ele / n_cats, 6) for ele in avg]
table['Average'] = avg

for k, v in table.items():
    print(len(v))
    befores_avg = round(sum(v[::2]) / len(v[::2]), 6)
    del v[::2]
    table[k] = [befores_avg] + v

for k, v in table.items():
    max_val = float('-inf')
    max_id = 0
    for i, ele in enumerate(v):
        if ele > max_val:
            max_val = ele
            max_id = i
            
    table[k][max_id] = "\\textbf{" + str(table[k][max_id]) + "}"

table = [[k] + v for k, v in table.items()]

print(tabulate(table, tablefmt='latex_raw'))
        

