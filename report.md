# Report
DFT.py
**************************
def forward_transform():
    First, create an empty array to store the fft
    Second, use 4 nested for-loop:
            The 2 outer nested for-loop (u, v) is used to go through all the empty spot of the empty array to fill in the calculate value
            The 2 inner nested for-loop (i, j) is used to go through the input matrix and get the current value to compute the fft using the formula:
                fft_value = (matrix[i][j]) * exp(-1 * 1j * (2 * pi / N) * (u * i + v * j))
    Finally return the computed fft matrix

**************************
def inverse_transform():
    First, create an empty array to store the fft
    Second, use 4 nested for-loop:
            The 2 outer nested for-loop (i, j) is used to go through all the empty spot of the empty array to fill in the calculate value
            The 2 inner nested for-loop (u, v) is used to go through the input matrix (fft) and get the current value to compute the fft using the formula:
                ifft_value = (matrix[u][v]) * exp(1j * (2 * pi / N) * (u * i + v * j))
    Finally return the computed ifft matrix

**************************
def discrete_cosine_transform():
    First, create an empty array to store the fft
    Second, use 4 nested for-loop:
            The 2 outer nested for-loop (u, v) is used to go through all the empty spot of the empty array to fill in the calculate value
            The 2 inner nested for-loop (i, j) is used to go through the input matrix and get the current value to compute the fft using the formula:
                dct_value = (matrix[i][j]) * (cos((2 * pi / N) * (u * i + v * j)))
    Finally return the computed dct matrix

**************************
def magnitude():
    First, create an empty array to store the fft
    Second, use 2 nested for-loop (u, v) to go through all the pixel and use abs() to the get absolute value
            absolute_value = abs(matrix[u][v])
    Finally return the computed magnitude matrix