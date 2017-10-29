# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries
from cmath import pi, exp, cos
import numpy as np

class DFT:

    # Function to calculate complex number
    def dft_calculator(self, matrix, u, v, type):

        result = 0.0

        #Forward transform
        if (type == 0):

            for i in range(15):
                for j in range(15):
                    result += (matrix[i][j]) * exp(-1 * 1j * (2 * pi / 15) * (u * i + v * j))
        #Inverse transform
        elif (type == 1):

            for i in range(15):
                for j in range(15):
                    result += (matrix[i][j]) * exp(1j * (2 * pi / 15) * (u * i + v * j))
        #Discrete cosine transform
        elif (type == 2):

            for i in range(15):
                for j in range(15):
                    result += (matrix[i][j]) * (cos((2 * pi / 15) * (u * i + v * j)))

        return result

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""

        # Create an empty array to store the transform
        tempArr = np.zeros(shape=(15, 15), dtype=complex)

        for u in range(15):
            for v in range(15):
                tempArr[u][v] = self.dft_calculator(matrix, u, v, 0)


        return tempArr

    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        matrix: a 2d matrix (DFT) usually complex
        takes as input:
        returns a complex matrix representing the inverse fourier transform"""

        # Create an empty array to store the transform
        tempArr = np.zeros(shape=(15, 15), dtype=complex)

        for u in range(15):
            for v in range(15):
                tempArr[u][v] = self.dft_calculator(matrix, u, v, 1)

        return tempArr


    def discrete_cosine_tranform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""

        # Create an empty array to store the transform
        tempArr = np.zeros(shape=(15, 15), dtype=complex)

        for u in range(15):
            for v in range(15):
                tempArr[u][v] = self.dft_calculator(matrix, u, v, 2)

        return tempArr


    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""

        # Create an empty array to store the transform
        tempArr = np.zeros(shape=(15, 15), dtype=complex)

        for u in range(15):
            for v in range(15):
                tempArr[u][v] = abs(matrix[u][v])

        return tempArr