import numpy as np

from pyquil import Program
from pyquil.paulis import *
from pyquil.api import WavefunctionSimulator
from pyquil.gates import *
from pyquil.quil import DefGate

from scipy.optimize import minimize

num_cities = 3

distance_matrix = np.matrix([[0.0, 4.0, 5.0],[4.0, 0.0, 3.0],[5.0, 3.0, 0.0]])

multiple_loc_penalty = 100

def penalize_distances():
	ps = sI()
	for i in range(0, num_cities):
		for j in range(i, num_cities):
			for t in range(0, num_cities - 1):
				wij =  - (distance_matrix.item((i, j))) / 2
				q1 = t * num_cities + i
				q2 = t * num_cities + j
				ps += (wij * sI() - wij * sZ(q1) * sZ(q2))
	return ps

def penalize_multiple_locations():
	ps = sI()
	for t in range(num_cities):
		binary_representation = list(range(t * num_cities, (t + 1) * num_cities))
		ps += penalize_range(binary_representation)
	return ps

def penalize_repeated_locations():
	ps = sI()
	for i in range(num_cities):
		occurances_of_i = list(range(i, num_cities**2, num_cities))
		ps += penalize_range(occurances_of_i)
	return ps


def penalize_range(must_be_unique):
	weight = (- multiple_loc_penalty * np.max(distance_matrix)).item()
	z_terms = weight * sZ(must_be_unique[0])
	one_terms = 0.5 * weight * (sI(must_be_unique[0]) - sZ(must_be_unique[0]))
	for i in range(1, num_cities):
		z_terms = z_terms * sZ(must_be_unique[i])
		one_terms = one_terms * 0.5 *(sI(must_be_unique[0]) - sZ(must_be_unique[i]))
	return weight * sI(0) - z_terms - one_terms

def prepare_cost():
	return penalize_distances() + penalize_repeated_locations() + penalize_multiple_locations()

# 0 1 2 | 3 4 5 | 6 7 8
def prepare_qubits(num_issues: int):
	pq = Program()
	pq += Program(X(0))
	if num_issues < 0:
		pq += Program(X(0))
	if num_issues > 0:
		pq += Program(X(1))
	if num_issues > 1:
		pq += Program(X(2))
	return pq

def test_penalties():
	sim = WavefunctionSimulator()
	pq = prepare_qubits(1)
	initial_wf = sim.wavefunction(pq)
	print("Initial WF is {}".format(initial_wf))
	#000 Test
	expectation = sim.expectation(pq, prepare_cost())
	print("Expectation value is {}".format(expectation))

def test_z_gates(errors):
	# pq = prepare_qubits(0)
	# Z = np.array([[1,0],[0,-1]])
	# Z1Z2Z3 = np.kron(Z, np.kron(Z,Z))
	# first_penalty = 0.5 * (np.eye(8) + Z1Z2Z3)
	# helper_op = 0.5 * (np.eye(2) - Z)
	# penalty_111 = np.kron(helper_op, np.kron(helper_op, helper_op))
	# final_penalty = first_penalty

	# penalty_dfn = DefGate("PENALTY", final_penalty)
	# PENALTY = penalty_dfn.get_constructor()

	sim = WavefunctionSimulator()

	pq = prepare_qubits(errors)
	initial_wf = sim.wavefunction(pq)
	print("Initial is {}".format(initial_wf))
	#ps = (sI(0) - sZ(0))*(sI(1) - sZ(1))*(sI(2) - sZ(2))
	ps = sI(0) + (sZ(0) * sZ(1) * sZ(2))
	expectation = sim.expectation(pq, ps)
	print("Expectation value for {} errors is {}".format(errors, expectation))

	# pq = prepare_qubits(2)
	# initial_wf = sim.wavefunction(pq)
	# print("Initial is {}".format(initial_wf))
	# ps = sI(0)+ sI(1)+sI(2)
	# expectation = sim.expectation(pq, ps)
	# print("Expectation value for 111 is {}".format(expectation))

test_z_gates(0)
test_z_gates(1)
test_z_gates(2)
test_z_gates(-1)