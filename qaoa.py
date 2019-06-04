import numpy as np

from pyquil import Program
from pyquil.paulis import *
from pyquil.api import WavefunctionSimulator
from pyquil.gates import *

from scipy.optimize import minimize
import matplotlib
import matplotlib
import matplotlib.pyplot as plt

num_cities = 3

distance_matrix = np.matrix([[0.0, 3.0, 5.0],
 [3.0, 0.0, 4.0],
 [5.0, 4.0, 0.0]
 ])

p = 3

weight1 = 200
weight2 = 80


sim = WavefunctionSimulator()

def prepare_initial_state():
    pq = Program()
    for i in range(0, num_cities**2):
        pq += Program(H(i))
    return pq

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

def penalize_range(must_be_single_one: list):
	print("Must have a single 1: {}".format(must_be_single_one))
	i_terms = sI()
	z_terms = sI()
	mixed_terms = sI()
	for qubit in must_be_single_one:
		i_terms = i_terms * sI(qubit)
		z_terms = z_terms * sZ(qubit)
		mixed_terms = mixed_terms * (sI(qubit) - sZ(qubit))
	punishment = weight1 * (i_terms + z_terms) + weight2 * mixed_terms

	return punishment

def prepare_cost():
	return penalize_distances() + penalize_repeated_locations() + penalize_multiple_locations()


#could not be right
# PauliSum Exponentiation from https://github.com/rigetti/pyquil/commit/785a5a549ce45da054369351032fb9ec826fdf70
def prepare_exponential(ps: PauliSum):
	fns = [exponential_map(term) for term in ps]
	def exp(param):
		return sum([f(param) for f in fns], Program())
	return exp

def parameterized_quantum_state(h_cost, h_driver, gammas, betas):
	pq = prepare_initial_state()
	for t in range(p):
		pq += (prepare_exponential(h_cost))(gammas[t])
		pq += (prepare_exponential(h_driver))(betas[t])
	return pq


def prepare_driver():
	ps = sI(0)
	for i in range(num_cities**2):
		ps -= sX(i)
	return ps

def objective(params, cost_hamiltonian, driver_hamiltonian):
	gammas = params[0:p]
	betas = params[p:2 * p]
	parameterized_pq = parameterized_quantum_state(cost_hamiltonian, driver_hamiltonian, gammas, betas)
	expectation = sim.expectation(parameterized_pq, cost_hamiltonian)
	print("Expectation is {}".format(expectation.real))
	return expectation.real

def second(elem):
	return elem[1]

def solve_qaoa():
	initial_params = np.random.uniform(0.0, 2*np.pi, size=2*p)
	hamiltonians = (prepare_cost(), prepare_driver())
	optimal = minimize(objective, initial_params, hamiltonians, method='Nelder-Mead')
	angles = optimal.x
	gammas = angles[0:p]
	betas = angles[p:2*p]
	soln = parameterized_quantum_state(*hamiltonians, gammas, betas)
	prob_dict = sim.wavefunction(soln).get_outcome_probs()
	sorted_strings = []
	sorted_counts = []
	for key, value in sorted(prob_dict.items(), key=second, reverse=True):
		sorted_strings.append(key)
		sorted_counts.append(value)
	length = len(sorted_strings)
	if length > 10:
		length = 10
		sorted_strings = sorted_strings[:10]
		sorted_counts = sorted_counts[:10]
	width = 2/length
	plt.bar(sorted_strings, sorted_counts, width, color='g')
	plt.xticks(rotation=90)
	print(sorted_strings)
	print(sorted_counts)
	plt.show()


solve_qaoa()
