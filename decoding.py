import numpy as np
import sys
import math

transit_A = {}
emit_B = {}

A = []
B = []

unique_states = []
unique_emit = []

state_idx = {}
idx_state = {}

idx_emit = {}
emit_idx = {}

vt = []
vb = []

obseq = []
finseq = ""

def read_model():
    global A
    global B

    with open (sys.argv[1], "r", encoding="utf-8", errors="ignore") as model:
        for line in model:
            clean = line.strip().split()
            if clean[0] == "E":
                state, emission = clean[1], clean[2]
                if state not in unique_states:
                    unique_states.append(state)
                if emission not in unique_emit:
                    unique_emit.append(emission)

                if state not in emit_B.keys():
                    emit_B[state] = {}
                    emit_B[state][emission] = clean[4]
                else:
                    emit_B[state][emission] = clean[4]


            elif clean[0] == "T":
                state1, state2  = clean[1], clean[2]
                if state1 not in unique_states:
                    unique_states.append(state1)
                if state2 not in unique_states:
                    unique_states.append(state2)

                if state1 not in transit_A.keys():
                    transit_A[state1] = {}
                    transit_A[state1][state2] = clean[4]
                else:
                    transit_A[state1][state2] = clean[4]

    n_states = len(unique_states)
    n_emits = len(unique_emit)

    #unique_states.sort()
    #unique_emit.sort()

    i = 0
    for b in unique_states:
        idx_state[i] = b
        state_idx[b] = i
        i += 1

    j = 0
    for b in unique_emit:
        idx_emit[j] = b
        emit_idx[b] = j
        j += 1
      
    A = np.zeros((n_states, n_states))
    B = np.zeros((n_states, n_emits))

    for i in range(n_states):
        for j in range(n_states):
            state1, state2 = unique_states[i], unique_states[j]
            if (state2 != "<s>") and (state1 != "</s>"):
                A[state_idx[state1], state_idx[state2]] = transit_A[state1][state2]

    for i in range(n_states):
        for j in range(n_emits):
            state, emit = unique_states[i], unique_emit[j]
            if (state != "</s>") and (state != "<s>"):
                B[state_idx[state], emit_idx[emit]] = emit_B[state][emit]

def read_test_seq():
    global obseq
    with open(sys.argv[2], "r", encoding="utf-8", errors="ignore") as sequence:
        obseq = sequence.readline().strip().split()
        for a in range(len(obseq)):
            if obseq[a] not in unique_emit:
                obseq[a] = "<unk>"

def viterbi_algo():
    global vt
    global vb
    global finseq

    nseq = len(obseq)
    ns = len(unique_states)

    vt = np.zeros((ns, nseq))
    vb = np.zeros((ns, nseq+1))

    #initialization
    for i in range(ns):
        vt[i, 0] = float(A[state_idx["<s>"], i]) * float(B[i, emit_idx[obseq[0]]])
        vb[i, 0] = "-1"

    #recursion
    for t in range(1, nseq):
        for j in range(ns):
            comp = np.zeros(ns)
            for i in range(ns):
                comp[i] = (float(vt[i, (t-1)]) * float(A[i, j]) * float(B[j, emit_idx[obseq[t]]]))
            vt[j, t] = np.max(comp)
            vb[j, t] = np.argmax(comp)

    #termination
    end = np.zeros(ns)
    for i in range(ns):
        last = vt[i, (nseq - 1)]
        ei = A[i, state_idx["</s>"]]
        end[i] = float(last) * float(ei)

    probability = math.log(np.max(end))
    vb[:, nseq] = np.argmax(end)

    arr = []
    arr.append(idx_state[np.argmax(end)])
    start = vb[np.argmax(end), nseq-1]
    #backtracking
    for t in range((nseq-2), -1, -1): 
        arr.append(idx_state[start])
        start = vb[state_idx[idx_state[start]], t]

    arr.reverse()
    finseq = ' '.join(arr)
    

    return probability

def write_to_file():
    with open("hmm.decoding", "w", encoding="utf-8", errors="ignore") as file:
        file.write("Best Sequence: " + finseq
                   + "\n")
        file.write("Best LL: " + str(viterbi_algo())
                   + "\n")
        
if __name__ == "__main__":
    read_model()
    read_test_seq()
    viterbi_algo()
    write_to_file()

