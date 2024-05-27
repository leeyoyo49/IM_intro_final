import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from copy import deepcopy
from gurobipy import *
import os

specific_item_id = 0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def OR_scheduler(menu):
    # display(menu)
    machine = pd.read_excel(os.path.join(BASE_DIR,'waiting.xlsx'), sheet_name='station_id')    

    job_name = menu['name_zh']
    jobs = len(job_name)

    stage1Time = menu['S1_time']
    stage2Time = menu['S2_time']

    D = menu['due_time']

    p = [[0] * jobs for i in range(2)]
    for i in range(jobs):
        p[0][i] = stage1Time[i]
    for i in range(jobs):
        p[1][i] = stage2Time[i]

    available1 = menu['S1_machine']
    available2 = menu['S2_machine']
    available2 = available2.fillna(0)

    machine_name = machine['machine']
    machines = len(machine_name)

    available_machine = np.zeros((2,jobs,machines))

    for i in range(jobs):
        for item in available1[i]:
            if(item.isdigit()):
                item = int(item)
                available_machine[0][i][item-1] = 1
    for i in range(jobs):
        if(available2[i] == 0):
            continue
        for item in available2[i]:
            if(item.isdigit()):
                item = int(item)
                available_machine[1][i][item-1] = 1
                
    env = Env("gurobi.log")
    eg1 = Model("eg1", env = env)

    y = []
    for i in range(2):
        y.append([])
        for j in range(jobs):
            y[i].append([])
            for k in range(machines):
                y[i][j].append(eg1.addVar(lb = 0, vtype = GRB.BINARY, name = "y" + str(i+1) + str(j+1) + str(k+1)))

    c = []
    for i in range(2):
        c.append([])
        for j in range(jobs):
            c[i].append(eg1.addVar(lb = 0, vtype = GRB.CONTINUOUS, name = "c" + str(i+1) + str(j+1)))

    T = []
    for j in range(jobs):
        T.append(eg1.addVar(lb = 0, vtype = GRB.CONTINUOUS, name = "D" + str(j+1)))

    z = []
    for i in range(2):
        z.append([])
        for j in range(jobs):
            z[i].append([])
            for a in range(2):
                z[i][j].append([])
                for b in range(jobs):
                    z[i][j][a].append([])
                    for k in range(machines):
                        z[i][j][a][b].append(eg1.addVar(lb = 0, vtype = GRB.BINARY, name = "z" + str(i+1) + str(j+1) + str(a+1) + str(b+1) + str(k+1)))

    eg1.setObjective(
        quicksum(T[j] for j in range(jobs)), GRB.MINIMIZE)

    M = 0
    for i in range(2):
        for j in range(jobs):
            M += p[i][j]

    for j in range(jobs):
        eg1.addConstr(c[0][j] <= c[1][j] - p[1][j])
        eg1.addConstr(c[0][j] + 1 + p[1][j] >= c[1][j])
        eg1.addConstr(c[0][j] >= p[0][j])
        eg1.addConstr(T[j] >= c[1][j] - D[j])
        eg1.addConstr(T[j] >= 0)
    for i in range(2):
        for j in range(jobs):
            if all(k == 0 for k in available_machine[i][j]):
                continue
            eg1.addConstr(quicksum(y[i][j][k] for k in range(machines)) == 1)
    for i in range(2):
        for j in range(jobs):
            for k in range(machines):
                eg1.addConstr(available_machine[i][j][k] - y[i][j][k] >= 0)
                
    for i in range(2):
        for j in range(jobs):
            for a in range(2):
                for b in range(jobs):
                        if(i == a and j == b):
                            continue
                        for k in range(machines):
                            eg1.addConstr(y[i][j][k] + y[a][b][k] <= z[i][j][a][b][k] + z[a][b][i][j][k] + 1)
                            eg1.addConstr(c[i][j] + p[a][b] - c[a][b] <= M * (1 - z[i][j][a][b][k]))
    eg1.optimize()
    # print("total tardiness = ", eg1.objVal)

    # print("whether stage i of job j is processed in machine k:")
    # for j in range(jobs):
    #     for i in range(2):
    #         for k in range(machines):
    #             print(y[i][j][k].x, "\t", end = "")
    #         print("")
    #     print("")

    # print("the completion time of stage i of job j:")
    # for j in range(jobs):
    #     for i in range(2):
    #         print(c[i][j].x, "\t", end = "")
    #     print("")

    # print("the tadiness of job j:")
    # for j in range(jobs):
    #     print(T[j].x, "\t", end = "")

    c_seq = {k: [] for k in range(machines)}
    # print("the sequence of stage i of job j in machine k:")
    for k in range(machines):
        for j in range(jobs):
            for i in range(2):  # Assuming there are 2 stages
                if y[i][j][k].x:  # if y[i][j][k].x is true
                    c_seq[k].append((c[i][j].x, i, j))

    return_df = pd.DataFrame(columns=["Machine_assigned", "distinct_id", "Priority"])

    for k in range(machines):
        priority = 0
        # Sort the list by completion time
        c_seq[k].sort(key=lambda x: x[0])   
        # print(f"Machine {k+1}:")
        for completion_time, i, j in c_seq[k]:
            distinct_id = menu.loc[j, "distinct_id"]
            new_row = {"Machine_assigned": k, "distinct_id": distinct_id, "Priority": priority}
            return_df = pd.concat([return_df, pd.DataFrame([new_row])], ignore_index=True)
            priority += 1
        #     print(f" Stage {i+1} of Job {j+1}")
        # print("")
        # Print a new line for better readability between machines

    output_data = []
    output_data.append({'Tardiness': eg1.objVal})
    for k in range(machines):
    # Initialize list to hold job sequences for this machine
        job_sequence = []
        
        # Populate job sequences
        for completion_time, i, j in c_seq[k]:
            job_sequence.append(f"stage {i + 1} of job {j + 1}")
        
        # Add machine and job sequences to output data
        output_data.append({'Machine': f'machine{k + 1}', 'Job Sequence': ', '.join(job_sequence)})

    # Convert the output data to a DataFrame
    df = pd.DataFrame(output_data)

    # Write the DataFrame to an Excel file
    output_file = 'job_sequence.xlsx'
    df.to_excel(output_file, index = False)

    print(f"Data has been written to {output_file}\n\n\n")
    return return_df

class Item:
    def __init__(self, id: int, name: str, price: float, type: str, S1_time: float, S2_time: float, S1_machine: list, S2_machine: list):
        self.Item_id = id
        self.name = name
        self.price = price
        self.type = type
        self.S1_time = S1_time
        self.S2_time = S2_time
        self.S1_machine = S1_machine
        self.S2_machine = S2_machine

        self.S1_machine_finished = False
        self.Item_completed = False

        self.specific_id = 0
        self.priority = 0
        self.time_created = 0
        self.Order_id = 0



# read_from csv to update Item_dict 
# id : init Item object
Item_dict = {}
id_to_item_df = pd.read_excel(os.path.join(BASE_DIR,'waiting.xlsx'), sheet_name='menu')
for i in range(len(id_to_item_df)):
    Item_dict[id_to_item_df['id'][i]] = Item(id_to_item_df['id'][i], id_to_item_df['name_zh'][i], id_to_item_df['price'][i], id_to_item_df['type'][i], id_to_item_df['S1_time'][i], id_to_item_df['S2_time'][i], id_to_item_df['S1_machine'][i], id_to_item_df['S2_machine'][i])


# order_id depends on the time it was created
class Order:
    # input a list of item ids
    def __init__(self, ids: list):
        global specific_item_id
        self.item_list = []
        self.total_price = 0
        self.order_id = int(time.time())

        for id in ids:
            tmp = deepcopy(Item_dict[id])
            tmp.time_created = time.time()
            tmp.Order_id = self.order_id
            tmp.specific_id = specific_item_id
            # print(tmp.specific_id)

            specific_item_id += 1
            self.total_price += tmp.price

            self.item_list.append(tmp)

        # for item in self.item_list:
            # print(item.specific_id)
        self.time_created = time.time()

    # add a list of item ids to the order
    def add_item(self, ids: list):
        for id in ids:
            tmp = deepcopy(Item_dict[id])
            tmp.time_created = time.time()
            tmp.Order_id = self.order_id
            tmp.specific_id = specific_item_id
            specific_item_id += 1
            self.total_price += tmp.price

            self.item_list.append(tmp)

        self.time_created = time.time()


class ItemQueue:
    def __init__(self):
        # everything
        self.all_waiting = []

        # currently doing
        self.all_main = [[] for i in range(7)]

        # waiting
        self.all_waiting_arranged = [[] for i in range(7)]


    # throw the order into the queue then schedule
    def add_order(self, order) -> Order:
        self.all_waiting_arranged = [[] for i in range(7)]

        for item in order.item_list:
            self.all_waiting.append(item)
        scheduled_df = self.schedule()


        for i in range(len(scheduled_df)):
            # find the item with its distinct_id
            for item in self.all_waiting:
                if item.specific_id == scheduled_df["distinct_id"][i]:
                    item.priority = scheduled_df["Priority"][i]
                    self.all_waiting_arranged[scheduled_df["Machine_assigned"][i]].append(item)
                    break
        
        print("品項, distinct_id")
        for i in range(7):
            print("Station", i)
            for item in self.all_waiting_arranged[i]:
                print(item.name, item.specific_id)

    # 0 - 6, 0-2 is cooking, 3-4 is packing, 5-6 is drinking
    # when that item is clicked on the screen, run pop_wait, it will move the item from waiting to main
    def pop_wait(self, Q_id: int):
        tmp = self.all_waiting_arranged[Q_id].pop(0) 
        self.all_main[Q_id].append(tmp)

        # find id in all_waiting and remove it
        for item in self.all_waiting:
            if item.specific_id == tmp.specific_id:
                item.S1_machine_finished = True
                item.S1_time = 0
                item.S1_machine = math.nan

        return tmp
    
    # 0 - 6, 0-2 is cooking, 3-4 is packing, 5-6 is drinking
    # return the item that is completed, need to go back to Order list to find the order, then update that order
    def pop_main(self, Q_id: int):
        tmp = self.all_main[Q_id].pop(0)
        
        for item in self.all_waiting:
            if item.specific_id == tmp.specific_id:
                if (item.S1_time == 0) and (item.S2_time == 0):
                    item.Item_completed = True
                    # remove item from all_waiting
                    self.all_waiting.remove(item)

        return tmp

    def schedule(self):
        # set df columns
        column = ["distinct_id", "item_id", "name_zh", "type", "S1_time", "S2_time", "S1_machine", "S2_machine", "due_time", "price"]
        df = pd.DataFrame(columns=column)
        for item in self.all_waiting:     
            new_row = pd.DataFrame([{"distinct_id": item.specific_id, "item_id": item.Item_id, "name_zh": item.name, "type": item.type, "S1_time": item.S1_time, "S2_time": item.S2_time, "S1_machine": item.S1_machine, "S2_machine": item.S2_machine, "due_time": (900-time.time()+item.time_created)/60, "price": item.price}])
            df = pd.concat([df, new_row], ignore_index=True) 

        display(df)
        scheduled_df = OR_scheduler(df)
        # display(scheduled_df)
        return scheduled_df        
                
        