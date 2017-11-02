# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries

import numpy as np
from cmath import pi, exp, cos

class DFT:
    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""

        N = matrix.shape[0]

        # Create an empty array to store the transform
        tempArr = np.zeros(shape=(N, N), dtype=complex)

        for u in range(N):
            for v in range(N):

                for i in range(N):
                    for j in range(N):
                        tempArr[u, v] += (matrix[i][j]) * exp(-1 * 1j * (2 * pi / N) * (u * i + v * j))

        return tempArr


    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        matrix: a 2d matrix (DFT) usually complex
        takes as input:
        returns a complex matrix representing the inverse fourier transform"""

        N = matrix.shape[0]

        # Create an empty array to store the transform
        tempArr = np.zeros(shape=(N, N), dtype=complex)

        for i in range(N):
            for j in range(N):

                for u in range(N):
                    for v in range(N):
                        tempArr[i, j] += (matrix[u][v]) * exp(1j * (2 * pi / N) * (u * i + v * j))

        return tempArr


    def discrete_cosine_transform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""

        N = matrix.shape[0]

        # Create an empty array to store the transform
        tempArr = np.zeros(shape=(N, N), dtype=complex)

        for u in range(N):
            for v in range(N):

                for i in range(N):
                    for j in range(N):
                        tempArr[u, v] += (matrix[i][j]) * (cos((2 * pi / N) * (u * i + v * j)))
        return tempArr



    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""

        N = matrix.shape[0]

        # Create an empty array to store the transform
        tempArr = np.zeros(shape=(N, N), dtype=float)

        for u in range(N):
            for v in range(N):
                tempArr[u][v] = abs(matrix[u][v])

        return tempArr