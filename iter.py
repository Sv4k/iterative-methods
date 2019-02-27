import numpy as np
from contextlib import redirect_stdout

def simple_iteration(inp, eps):
	with open(inp, 'r') as in_f, open("simple_iterations_out.txt", 'w') as out_f, redirect_stdout(out_f): # open file with given data and file to print out the solution
		dimention = int(in_f.readline())
		A = np.array([list(map(float, in_f.readline().split())) for i in range(dimention)]) # read a matrix of system
		b = np.array([float(in_f.readline()) for i in range(dimention)]) # read a vector of ...
		x = np.zeros(dimention) # create an initial guess of solution

		
		print("the matrix of system A:\n", A)
		print("the vector of ...:\n", b)
		print("the initial guess of solution:\n", x)
		print("the required accuracy of approximation:\n", eps)

		#e = list(i.real for i in np.linalg.eig(A)[0]) # get eigen values of A
		min_e, max_e = A[0][0] - sum(map(abs, A[0][1:])), A[0][0] + sum(map(abs, A[0][1:]))
		for i in range(1, dimention):
			 tmp = sum(map(abs, A[i])) + abs(A[i][i])
			if A[i][i] - tmp < min_e:
				min_e = A[i][i] - tmp
			if A[i][i] + tmp > max_e:
				max_e = A[i][i] + tmp	

		t_opt = 2 / (min_e + max_e) # count optimal t
		
		cnt = 0 # iterations counter
		# x(k + 1) = x(k) - t * (A * x(k) - b)
		while np.linalg.norm(np.dot(A, x) - b) > eps:
			x = x - t_opt * (np.dot(A, x) - b)
			cnt += 1

		print("the vector of solution:\n", x)
		print("iterations number:\n", cnt)

	return x


simple_iteration("s.txt", 1e-4)	
