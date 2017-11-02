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
    
    
    
    
    
    
Filtering.py

This only works for gray-scale image (1 channel).
The input image must be gray-scale image that has only 1 channel
Assume the input image is of size NxN (square)

*************************
def get_ideal_low_pass_filter():

    1. Get the size of the input image - N
    2. Create an empty array with the same size of the input image, and with type of float
    3. Use 2 nested for-loop (u, v) to go all the position in the empty array
            Inside the nested for-loop, use the formula to calculate the distance between current position (u ,v) and the position of the center of the array
            Sqrt( (u - N/2)^2   +   (v - N/2)^2   )
            
            Check the distance with the cutoff frequency: 
                   Assign 1 for current position in the array if distance is less or equal to the cutoff frequency
                   Assign 0 if the distance is greater the cutoff frequency
    4. Return the array - the mask

*************************
def get_ideal_high_pass_filter():

    1. Get the size of the input image - N
    2. Create an empty array with the same size of the input image, and with type of float
    3. Use 2 nested for-loop (u, v) to go all the position in the empty array
            Inside the nested for-loop, use the formula to calculate the distance between current position (u ,v) and the position of the center of the array
            Sqrt( (u - N/2)^2   +   (v - N/2)^2   )
            
            Check the distance with the cutoff frequency: 
                   Assign 0 for current position in the array if distance is less or equal to the cutoff frequency
                   Assign 1 if the distance is greater the cutoff frequency
    4. Return the array - the mask

*************************
def get_butterworth_low_pass_filter():

    1. Get the size of the input image - N
    2. Create an empty array with the same size of the input image, and with type of float
    3. Use 2 nested for-loop (u, v) to go all the position in the empty array            
            1. Use the formula to calculate the distance between current position (u ,v) and the position of the center of the array
                    Sqrt( (u - N/2)^2   +   (v - N/2)^2   )
            2. Use the distance and the cutoff frequency to calculate the value of the current position on the mask
                    1 / ( 1 +  (distance / cutoff)^2 )
            3. Assign the value computed from step 2 to the current position on the mask
    4. Return the array - the mask

*************************
def get_butterworth_high_pass_filter():

    1. Get the size of the input image - N
    2. Create an empty array with the same size of the input image, and with type of float
    3. Use 2 nested for-loop (u, v) to go all the position in the empty array            
            1. Use the formula to calculate the distance between current position (u ,v) and the position of the center of the array
                    Sqrt( (u - N/2)^2   +   (v - N/2)^2   )
            2. Use the distance and the cutoff frequency to calculate the value of the current position on the mask
                    1 / ( 1 +  (cutoff / distance)^2 )
            3. Assign the value computed from step 2 to the current position on the mask
    4. Return the array - the mask

*************************
def get_gaussian_low_pass_filter():

    1. Get the size of the input image - N
    2. Create an empty array with the same size of the input image, and with type of float
    3. Use 2 nested for-loop (u, v) to go all the position in the empty array            
            1. Use the formula to calculate the distance between current position (u ,v) and the position of the center of the array
                    Sqrt( (u - N/2)^2   +   (v - N/2)^2   )
            2. Use the distance and the cutoff frequency to calculate the value of the current position on the mask
                    exp( (-1) * (distance^2) / (2 * cutoff^2) )
            3. Assign the value computed from step 2 to the current position on the mask
    4. Return the array - the mask

*************************
def get_gaussian_high_pass_filter():

    1. Get the size of the input image - N
    2. Create an empty array with the same size of the input image, and with type of float
    3. Use 2 nested for-loop (u, v) to go all the position in the empty array            
            1. Use the formula to calculate the distance between current position (u ,v) and the position of the center of the array
                    Sqrt( (u - N/2)^2   +   (v - N/2)^2   )
            2. Use the distance and the cutoff frequency to calculate the value of the current position on the mask
                   1 - (exp( (-1) * (distance^2) / (2 * cutoff^2) ))
            3. Assign the value computed from step 2 to the current position on the mask
    4. Return the array - the mask

*************************
def post_process_image():

    1. Get the size of the input image - N
    2. Get min value of the input array - image.min()
    3. Get max value of the input array - image.max()
    2. Create an empty array with the same size of the input image, and with type of float
    3. Use 2 nested for-loop (u, v) to go all the position in the empty array            
            ( (pixel - min) / (max - min) ) * 255
            Use the above function to perform full contrast stretch for the input image, where pixel is the current value of the pixel of the input image at (u, v)
            Assign the computed value from above to the current position on the empty array            
    4. Return the array - full contrast stretched image

*************************
def filtering():

    1. Compute the fft of the image using built-in function np.fft.fft2()
    2. Shift the fft using the built-in function np.fft.fftshift()
    3. Get the magnitude of the shifted fft and compress it using built-in functions: np.abs() and np.log()
    4. Convert result from step 3 to type of uint8 and save the image for Magnitude of DFT of the Image
    5. Creating the mask according to the filter the user choose and call the filter to create the mask accordingly
    6. After creating the mask, filter the image by multiply the mask with the shifted fft
    7. Compute the inverse shift of the filtered image from step 6 by using built-in function np.fft.ifftshift()
    8. Compute the inverse fft of the result from step 7 using built-in function np.fft.ifft2()
    9. Compute the magnitude of the result from step 8 by taking the absolute value of the result from step 8
    10. Convert the result from step 9 to type uint8 and save it as the image of the Magnitude of Filtered DFT
    11. Using the result from step 9, perform a full contrast stretch by calling the post_process_image() function
    12. Convert the result from step 11 to type uint8 and save it as the image of the Filtered Image
    
    Finally, return images from step: 4, 10, and 12
    