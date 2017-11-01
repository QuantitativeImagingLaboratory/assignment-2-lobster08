# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv
import numpy as np
from cmath import exp

class Filtering:
    image = None
    filter = None
    cutoff = None
    order = None

    def __init__(self, image, filter_name, cutoff, order = 0):
        """initializes the variables frequency filtering on an input image
        takes as input:
        image: the input image
        filter_name: the name of the mask to use
        cutoff: the cutoff frequency of the filter
        order: the order of the filter (only for butterworth
        returns"""
        self.image = image
        if filter_name == 'ideal_l':
            self.filter = self.get_ideal_low_pass_filter
        elif filter_name == 'ideal_h':
            self.filter = self.get_ideal_high_pass_filter
        elif filter_name == 'butterworth_l':
            self.filter = self.get_butterworth_low_pass_filter
        elif filter_name == 'butterworth_h':
            self.filter = self.get_butterworth_high_pass_filter
        elif filter_name == 'gaussian_l':
            self.filter = self.get_gaussian_low_pass_filter
        elif filter_name == 'gaussian_h':
            self.filter = self.get_gaussian_high_pass_filter

        self.cutoff = cutoff
        self.order = order


    def get_ideal_low_pass_filter(self, shape, cutoff):
        """Computes a Ideal low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal low pass mask"""

        # Get the N size of the square image
        N = shape.shape[0]

        # Create empty mask array
        mask = np.zeros(shape=(N, N), dtype=float)

        for row in range(N):
            for col in range(N):
                distance = np.sqrt((np.power((row - N / 2), 2)) + (np.power((col - N / 2), 2)))

                # Check distance with cutoff
                if (distance <= cutoff):
                    mask[row, col] = 1
                elif (distance > cutoff):
                    mask[row, col] = 0

        return mask


    def get_ideal_high_pass_filter(self, shape, cutoff):
        """Computes a Ideal high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal high pass mask"""

        # Get the N size of the square image
        N = shape.shape[0]

        # Create empty mask array
        mask = np.zeros(shape=(N, N), dtype=float)

        for row in range(N):
            for col in range(N):
                distance = np.sqrt((np.power((row - N / 2), 2)) + (np.power((col - N / 2), 2)))

                # Check distance with cutoff
                if (distance <= cutoff):
                    mask[row, col] = 0
                elif (distance > cutoff):
                    mask[row, col] = 1

        return mask

    def get_butterworth_low_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth low pass mask"""

        # Get the N size of the square image
        N = shape.shape[0]

        # Create empty mask array
        mask = np.zeros(shape=(N, N), dtype=float)

        for row in range(N):
            for col in range(N):
                distance = np.sqrt((np.power((row - N / 2), 2)) + (np.power((col - N / 2), 2)))
                sum = 1 + np.power((distance / cutoff), (2 * order))
                value = 1 / sum

                mask[row, col] = value

        return mask


    def get_butterworth_high_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth high pass mask"""

        # Get the N size of the square image
        N = shape.shape[0]

        # Create empty mask array
        mask = np.zeros(shape=(N, N), dtype=float)

        for row in range(N):
            for col in range(N):
                distance = np.sqrt((np.power((row - N / 2), 2)) + (np.power((col - N / 2), 2)))
                sum = 1 + np.power((cutoff / distance), (2 * order))
                value = 1 / sum

                mask[row, col] = value

        return mask

    def get_gaussian_low_pass_filter(self, shape, cutoff):
        """Computes a gaussian low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian low pass mask"""

        # Get the N size of the square image
        N = shape.shape[0]

        # Create empty mask array
        mask = np.zeros(shape=(N, N), dtype=float)

        for row in range(N):
            for col in range(N):
                distance = np.sqrt((np.power((row - N / 2), 2)) + (np.power((col - N / 2), 2)))
                value = exp(-1 * np.power(distance, 2) / (2 * np.power(cutoff)))

                mask[row, col] = value

        return mask

    def get_gaussian_high_pass_filter(self, shape, cutoff):
        """Computes a gaussian high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian high pass mask"""

        # Get the N size of the square image
        N = shape.shape[0]

        # Create empty mask array
        mask = np.zeros(shape=(N, N), dtype=float)

        for row in range(N):
            for col in range(N):
                distance = np.sqrt((np.power((row - N / 2), 2)) + (np.power((col - N / 2), 2)))
                value = 1 - (exp(-1 * np.power(distance, 2) / (2 * np.power(cutoff))))

                mask[row, col] = value

        return mask

    def post_process_image(self, image):
        """Post process the image to create a full contrast stretch of the image
        takes as input:
        image: the image obtained from the inverse fourier transform
        return an image with full contrast stretch
        -----------------------------------------------------
        1. Full contrast stretch (fsimage)
        2. take negative (255 - fsimage)
        """
        N = image.shape[0]

        # 1. Full contrast stretch (fsimage)
        processed = np.copy(image)

        for row in range(N):
            for col in range(N):
                value = ((image[row, col] - image.min()) / (image.max() - image.min())) * 255
                processed[row, col] = value

        # 2. take negative (255 - fsimage)
        for row in range(N):
            for col in range(N):
                processed[row, col] = (255 - processed[row, col])

        # Convert to uint8 type
        processed = processed.astype(np.uint8)

        return processed


    def filtering(self):
        """Performs frequency filtering on an input image
        returns a filtered image, magnitude of DFT, magnitude of filtered DFT        
        ----------------------------------------------------------
        You are allowed to used inbuilt functions to compute fft
        There are packages available in numpy as well as in opencv
        Steps:
        """
        #Steps:
        #1. Compute the fft of the image
        fftImage = np.fft.fft2(self.image)

        #2. shift the fft to center the low frequencies
        shiftedImage = np.fft.fftshift(fftImage)

        #3. get the mask (write your code in functions provided above) the functions can be called by self.filter(shape, cutoff, order)
        mask = self.get_ideal_low_pass_filter(self.image, self.cutoff)

        #4. filter the image frequency based on the mask (Convolution theorem)
        filtered = np.dot(mask, shiftedImage)

        #5. compute the inverse shift
        inverse_shift = np.fft.ifftshift(filtered)

        #6. compute the inverse fourier transform
        inverse_fft = np.fft.ifft2(inverse_shift)

        #7. compute the magnitude
        computed_magnitude = np.abs(inverse_fft)

        #8. You will need to do a full contrast stretch on the magnitude and depending on the algorithm you may also need to
        processed = self.post_process_image(computed_magnitude)

        #take negative of the image to be able to view it (use post_process_image to write this code)
        #Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        #filtered image, magnitude of DFT, magnitude of filtered DFT: Make sure all images being returned have grey scale full contrast stretch and dtype=uint8


        return processed
