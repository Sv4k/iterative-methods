import numpy as np
from contextlib import redirect_stdout


def jacobi(inp, eps):
    # open file with given data and file to print out the solution
    with open(inp, 'r') as in_f, open("jacobi_output.txt", 'w') as out_f, redirect_stdout(out_f):
        dimention = int(in_f.readline())
        A = np.array([list(map(float, in_f.readline().split()))
                      for i in range(dimention)])  # read a matrix of system
        # read a vector of ...
        b = np.array([float(in_f.readline()) for i in range(dimention)])
        x = np.zeros(dimention)  # create an initial guess of solution

        print("the matrix of system A:\n", A)
        print("the vector of ...:\n", b)
        print("the initial guess of solution:\n", x)
        print("the required accuracy of approximation:\n", eps)

        # check if A[i][i] = 0 swap with string which i,j element != 0
        for i in range(dimention):
            if not A[i][i]:
                for j in range(i, dimention):
                    if A[j][i] and abs(A[j][i]) < sum(map(abs, A[i])) - abs(A[j][i]):
                        A[i], A[j] = A[j], A[i]
            if abs(A[i][i]) < sum(map(abs, A[i])) - abs(A[i][i]):
                print("Matrix hasn't diagonal dominance")
                return

        # create a matrix of the diagonal elements of A (Dij = 1 if i == j else 0)
        D = np.diag(A)
        # create a matrix which is equals to matrix A without A's diagonal elements
        R = A - np.diagflat(D)

        # process x(k + 1) = D^(-1) * (b - R * x(k))
        cnt = 0  # an iteration counter
        while np.linalg.norm(np.dot(A, x) - b) > eps:
            x = (b - np.dot(R, x)) / D
            cnt += 1

        print("the vector of solution:\n", x)
        print("iterations number:\n", cnt)

    return x


def simple_iteration(inp, eps):
    # open file with given data and file to print out the solution
    with open(inp, 'r') as in_f, open("simple_iterations_out.txt", 'w') as out_f, redirect_stdout(out_f):
        dimention = int(in_f.readline())
        A = np.array([list(map(float, in_f.readline().split()))
                      for i in range(dimention)])  # read a matrix of system
        # read a vector of ...
        b = np.array([float(in_f.readline()) for i in range(dimention)])
        x = np.zeros(dimention)  # create an initial guess of solution

        print("the matrix of system A:\n", A)
        print("the vector of ...:\n", b)
        print("the initial guess of solution:\n", x)
        print("the required accuracy of approximation:\n", eps)

        # e = list(i.real for i in np.linalg.eig(A)[0]) # get eigen values of A
        min_e, max_e = A[0][0] - \
            sum(map(abs, A[0][1:])), A[0][0] + sum(map(abs, A[0][1:]))
        for i in range(1, dimention):
            tmp = sum(map(abs, A[i])) + abs(A[i][i])
            if A[i][i] - tmp < min_e:
                min_e = A[i][i] - tmp
            if A[i][i] + tmp > max_e:
                max_e = A[i][i] + tmp

        t_opt = 2 / (min_e + max_e)  # count optimal t

        cnt = 0  # iterations counter
        # x(k + 1) = x(k) - t * (A * x(k) - b)
        while np.linalg.norm(np.dot(A, x) - b) > eps:
            x = x - t_opt * (np.dot(A, x) - b)
            cnt += 1

        print("the vector of solution:\n", x)
        print("iterations number:\n", cnt)

    return x


simple_iteration("s.txt", 1e-4)
jacobi("j.txt", 1e-4)
