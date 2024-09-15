import numpy as np
import sys, math

transit_A = []
emit_B = []

states = []
observations = []

lat_obs = []
len_latobs = 0

code_stat = {}
code_lat = {}

def read_input():
    global len_latobs
    with open(sys.argv[1], "r", encoding="utf-8", errors="ignore") as in_f :
        for line in in_f:
            splits = line.strip().split(';')
            len_latobs += len(splits)
            states.append("<s>")
            for trans in splits:
                lat_obs.append(trans)
                state, observation = trans.split()
                states.append(state)
                observations.append(observation)
            states.append("</s>") 

def init_A():
    global transit_A
    trans_counts = {}
    prev = 0
    next = 1
    while prev < len(states) - 1 and next < len(states): 
        if states[prev] not in trans_counts.keys():
            trans_counts[states[prev]] = {}
            trans_counts[states[prev]][states[next]] = 1
            prev += 1
            next += 1
        else: 
            if states[next] not in trans_counts[states[prev]].keys():
                trans_counts[states[prev]][states[next]] = 1
                prev += 1
                next += 1
            else: 
                trans_counts[states[prev]][states[next]] += 1
                prev += 1
                next += 1
    
    ns = len(list(set(states)))
    nodupesns = list(set(states))
    transit_A = np.zeros((ns, ns))
    index = 0

    for a in nodupesns:
        code_stat[index] = a
        index += 1

    
    for i in range(ns):
        for j in range(ns):
            try:
                if code_stat[i] == "</s>": 
                    continue
                elif code_stat[j] == "<s>": 
                    continue
                else:
                    count =  trans_counts[code_stat[i]][code_stat[j]]
                    transit_A[i, j] = float(count + 1)
            except KeyError:
                transit_A[i, j] = float(1)
                

    total_sum = np.sum(transit_A, axis=1)
    transit_A = transit_A / total_sum[:, None]
    
def print_matrix(matrix, row_labels, col_labels):
     
    max_row_label_length = max(len(label) for label in row_labels)
    max_col_label_length = max(len(label) for label in col_labels)

    
    print(" " * (max_row_label_length + 2), end="")  
    for label in col_labels:
        print(f"{label:^{max_col_label_length + 6}}", end="")
    print() 

    for i, row_label in enumerate(row_labels):
        print(f"{row_label:<{max_row_label_length + 2}}", end="")  
        for value in matrix[i]:
            formatted_value = f"{value:.6f}"
            print(f"{formatted_value:>{max_col_label_length + 6}}", end="")
        print() 
            
    

def init_B():
    global emit_B
    nodupeno = list(set(observations))
    nodupeno.append("<unk>")
    ns = len(list(set(states)))
    no = len(nodupeno)
    emit_B = np.zeros((ns, no))

    emit_counts = {}

    for elem in lat_obs:
        #print(elem)
        stat, lat = elem.split()
        if stat not in emit_counts.keys():
            emit_counts[stat] = {}
            emit_counts[stat][lat] = 1
            emit_counts[stat]["<unk>"] = 1 
        else:
            if lat not in emit_counts[stat].keys():
                emit_counts[stat][lat] = 1
            else:
                emit_counts[stat][lat] += 1

    i = 0
    for b in nodupeno:
        code_lat[i] = b
        i += 1

    for i in range(ns):
        for j in range(no):
            if code_stat[i] in emit_counts.keys():
                sum = 0
                for a in emit_counts[code_stat[i]].values():
                    sum += a
                if code_lat[j] in emit_counts[code_stat[i]].keys():
                    count = emit_counts[code_stat[i]][code_lat[j]]
                    emit_B[i, j] = float(count) / float(sum)


def write_to_file():
    num_e_rows = emit_B.shape[0]
    num_e_cols = emit_B.shape[1]

    num_t_rows = transit_A.shape[0]
    num_t_cols = transit_A.shape[1]

    with open("model.hmm", "w", encoding="utf-8", errors="ignore") as file:
        for i in range(num_e_rows):
            for j in range(num_e_cols):
                if code_stat[i] != "</s>" and code_stat[i] != "<s>":
                    if emit_B[i, j] != 0.0:
                        file.write("E " + str(code_stat[i]) + " " + str(code_lat[j]) + " : " + str(emit_B[i, j]) + "\n")
                    else:
                        file.write("E " + str(code_stat[i]) + " " + str(code_lat[j]) + " : " + str(0) + "\n")

        for i in range(num_t_rows):
            for j in range(num_t_cols):
                if code_stat[j] != "<s>" and code_stat[i] != "</s>":
                    if transit_A[i, j] != 0.0:
                        file.write("T " + str(code_stat[i]) + " " + str(code_stat[j]) + " : " + str(transit_A[i, j]) + "\n")
                    else:
                        file.write("T " + str(code_stat[i]) + " " + str(code_stat[j]) + " : " + str(0) + "\n")
        

if __name__ == "__main__":
    read_input()
    init_A()
    init_B()
    write_to_file()
   
    
 

