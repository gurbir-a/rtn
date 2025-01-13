import numpy as np
from scipy.stats import unitary_group
import matplotlib.pyplot as plt
import functools as ft
from collections import deque
import pprint
import scipy
import sympy as sym
from sympy.interactive import printing
printing.init_printing(use_latex=True)

# np.random.seed(222332)
np.set_printoptions(formatter={'float': lambda x: f"{x:10.4g}"})

def generate_random_unitary(N, bits=2):
    unitary_matrix = unitary_group.rvs(N**(bits))
    return unitary_matrix

def generateGuassRandomMatrix(n):
	A = np.random.normal(0,1/2, (n,n)) + 1j *np.random.normal(0,1/2, (n,n))
	# A = (A + A.conj().T) / 2 
	return A

def createIsometry(n, m, dim=2):
	unitary = generate_random_unitary(dim, m)
	isometry = np.array([row[::dim**(m-n)] for row in unitary])
	return isometry

def createApproxIsometry(n, m, dim=2):
	unitary = generate_random_unitary(dim, m)
	isometry = np.array([row[::dim**(m-n)] for row in unitary])
	return isometry

def isIsometry(matrix, tol=1e-10):
	matrix_dagger = np.conjugate(matrix.T)
	product_correct = np.dot(matrix_dagger, matrix)
	product_incorrect = np.dot(matrix, matrix_dagger)
	identity_correct = np.eye(matrix.shape[1])
	identity_incorrect = np.eye(matrix.shape[0])
	return (np.allclose(product_correct, identity_correct, atol=tol) and not np.allclose(product_incorrect, identity_incorrect, atol=tol))

def postSelect(matrix, N, bits=2):
	postMatrix = np.array([row[:N**bits] for row in matrix[::N]])
	# postMatrix = np.array([row[:N**bits] for row in matrix[:N**bits]])
	return postMatrix

def createCircuit(N, bits=2):
	circuitList = []
	for j in range(bits):
		matrixList = deque([])
		for i in range(bits+1):
			if i == j+1:
				continue
			if i != j:
				matrixList.appendleft(np.eye(N))
			else:
				matrixList.appendleft(generate_random_unitary(N,2))
		circuitList.append(ft.reduce(np.kron, matrixList))
	circuit = ft.reduce(np.matmul, circuitList)
	return circuit


def is_unitary(matrix, tol=1e-10):
    matrix_dagger = np.conjugate(matrix.T)
    product = np.dot(matrix, matrix_dagger)
    identity = np.eye(matrix.shape[0])
    return np.allclose(product, identity, atol=tol)

maxDim = 9
dims = range(2,maxDim+1)
maxBits = 9
bits = range(2,maxBits+1)
fixedBit = 8
fixedD = 2
r = []
error = []
unitary = []
beforePost = []
singularValues = []
overlapValues = []
numVectorSamples = 10000
numCircuits = 1
startCircuit = 1
numPostSelect = np.arange(1)

def frobeniusNorm():
	fNorm = []
	for dimension in dims:
		u = createCircuit(fixedD, dimension)
		pU = postSelect(u, fixedD, dimension)
		pU_dagger = np.conjugate(pU.T)
		target = pU_dagger @ pU - np.eye(2**dimension)
		fNorm.append(np.linalg.norm(target, 'fro')/2**(dimension/2))
	return fNorm

def concatenation():
	for i in range(numCircuits):
		us = []
		for j in range(i+startCircuit):
			us.append(createCircuit(fixedD,8))
		u = ft.reduce(np.matmul, us)
		pU = postSelect(u, fixedD, 8)
		pU_dagger = np.conjugate(pU.T)
		for vecSample in range(numVectorSamples):
			vec = np.random.rand(2**8) + 1j * np.random.rand(2**8)
			vec /= np.linalg.norm(vec)
			newVec = pU @ vec
			overlapValues.append(np.linalg.norm(newVec)**2)
		# gpU, right = np.linalg.qr(pU)
		# gpU_dagger = np.conjugate(gpU.T)
		pUs.append(pU)
		singularValues.append(scipy.linalg.svdvals(pU))

	randomU = generate_random_unitary(fixedD, 9)
	pU = postSelect(randomU, fixedD, 8)
	singularValues.append(scipy.linalg.svdvals(pU))

	# Print the SVD

	fig, ax = plt.subplots(numCircuits+1)

	for i in range(numCircuits):
		ax[i].plot(range(2**(8)), singularValues[i])
		print("i = ", i+startCircuit)
		print("mean = ", np.mean(singularValues[i]))
		print("var = ", np.var(singularValues[i]))
		print(" ")

	ax[-1].hist(overlapValues, bins=np.arange(0,1,0.005), color='blue', edgecolor='black', alpha=0.7)
	print("overlaps")
	print("mean = ", np.mean(overlapValues))
	print("var = ", np.var(overlapValues))
	print(" ")

	# ax[-1].plot(range(2**8))
	# print("i = random")
	# print("mean = ", np.mean(singularValues[-1]))
	# print("var = ", np.var(singularValues[-1]))
	# print(" ")

	plt.show()


def concatenationWithPost():
	for i in range(numCircuits):
		pUs = []
		for j in range(i+startCircuit):
			u = createCircuit(fixedD,fixedBit)
			pUs.append(postSelect(u, fixedD, fixedBit))
		pU = ft.reduce(np.matmul, pUs)
		singularValues.append(scipy.linalg.svdvals(pU))

	pUs = []
	for j in range(startCircuit+numCircuits-1):
		randomU = generate_random_unitary(fixedD, fixedBit+1)
		pUs.append(postSelect(randomU, fixedD, fixedBit))
	pU = ft.reduce(np.matmul,pUs)
	singularValues.append(scipy.linalg.svdvals(pU))

	numPlots = len(bits)
	fig, ax = plt.subplots(numCircuits+1)

	for i in range(numCircuits):
		ax[i].plot(range(2**(fixedBit)), singularValues[i])
		print("i = ", i+startCircuit)
		print("mean = ", np.mean(singularValues[i]))
		print("var = ", np.var(singularValues[i]))
		print(" ")

	ax[-1].plot(range(2**(fixedBit)), singularValues[-1])
	print("i = random")
	print("mean = ", np.mean(singularValues[-1]))
	print("var = ", np.var(singularValues[-1]))
	print(" ")
	plt.show()


def probabilityDistribution():
	u = createCircuit(fixedD, fixedBit)

def svdIsometry():
	# iso = createIsometry(8, 9, 2)
	iso = generate_random_unitary(fixedD, 9)
	# iso = generateGuassRandomMatrix(fixedD**8)
	iso = postSelect(iso, fixedD, 8)
	iso_dagger = np.conjugate(iso.T)
	u, s, vh = np.linalg.svd(iso, full_matrices=False)
	isoConstruct = u @ vh
	cRange = np.linspace(0,2,num=50)
	mean = []
	var = []
	print(np.sqrt(np.mean(np.power(s,2)-2*s + 1)))
	plt.plot(s)
	plt.show()
	return
	# print(is_unitary(iso))
	# return
	# for c in cRange:
	# 	v = []
	# 	for vecIndex in range(numVectorSamples):
	# 		vec = np.random.rand(2**8) + 1j * np.random.rand(2**8)
	# 		vec /= np.linalg.norm(vec)
	# 		uTransformedVec = isoConstruct @ vec
	# 		isoTransformedVec = c * (iso @ vec)
	# 		difference = uTransformedVec - isoTransformedVec
	# 		v.append(np.sqrt(np.linalg.norm(difference)))
	# 	mean.append(np.mean(v))
	# 	var.append(np.sqrt(np.var(v)))

	for vecIndex in range(numVectorSamples):
		vec = np.random.rand(2**8) + 1j * np.random.rand(2**8)
		vec /= np.linalg.norm(vec)
		uTransformedVec = isoConstruct @ vec
		isoTransformedVec =  (iso @ vec)
		difference = uTransformedVec - isoTransformedVec
		overlapValues.append(np.linalg.norm(difference))


	# fig, ax = plt.subplots(2)
	# ax[0].plot(cRange, mean)
	# ax[1].plot(cRange, var)

	plt.hist(overlapValues, bins=np.arange(0,1,0.005), color='blue', edgecolor='black', alpha=0.7)
	print("overlaps")
	print("mean = ", np.mean(overlapValues))
	print("mean of singularValues = ", np.mean(s))
	print("var = ", np.var(overlapValues))
	print("var of singularValues = ", np.var(s))
	print("std = ", np.sqrt(np.var(overlapValues)))
	print("1/2^(8/2) = ", 1/2**(8/2))
	print(" ")
	plt.show()

def polarUnitary():
	# iso = createIsometry(8, 9, 2)
	iso = generate_random_unitary(fixedD, 9)
	# iso = generateGuassRandomMatrix(fixedD**8)
	iso = postSelect(iso, fixedD, 8)
	iso_dagger = np.conjugate(iso.T)
	u, p = scipy.linalg.polar(iso, side='left')
	isoConstruct = u
	# plt.plot(s)
	# plt.show()
	# return
	# print(is_unitary(isoConstruct))
	# return
	for vecIndex in range(numVectorSamples):
		vec = np.random.rand(2**8) + 1j * np.random.rand(2**8)
		vec /= np.linalg.norm(vec)
		uTransformedVec = isoConstruct @ vec
		isoTransformedVec = (iso @ vec)
		difference = uTransformedVec - isoTransformedVec
		overlapValues.append(np.linalg.norm(difference))



	# fig, ax = plt.subplots(2)
	# ax[0].plot(cRange, mean)
	# ax[1].plot(cRange, var)

	plt.hist(overlapValues, bins=np.arange(0,1,0.005), color='blue', edgecolor='black', alpha=0.7)
	print("overlaps")
	print("mean = ", np.mean(overlapValues))
	# print("mean of singularValues = ", np.mean(s))
	# print("var = ", np.var(overlapValues))
	# print("var of singularValues = ", np.var(s))
	print("std = ", np.sqrt(np.var(overlapValues)))
	print("1/2^(8/2) = ", 1/2**(8/2))
	print(" ")
	plt.show()


def proof():
	# x = createIsometry(1,2)
	x = sym.MatrixSymbol('x', 4,2)
	m = createIsometry(1,2)
	w, s, vd = np.linalg.svd(m, full_matrices=False)
	u = w @ vd

	v = np.conjugate(vd.T)
	ud = np.conjugate(u.T)


	ansatz = vd @ ud @ x @ v

	constraints = []
	constraints.append(ansatz[0,0]-1)
	constraints.append(ansatz[1,1]-1)
	uContraint = (np.conjugate(x.T) @ x - np.eye(2)).flatten()
	for i in uContraint:
		constraints.append(i)
	print(constraints)
	return ansatz

# proof()
svdIsometry()
# polarUnitary()

# plt.plot(frobeniusNorm())
# plt.show()















