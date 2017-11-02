# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv
import numpy as np
import cv2

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
        mask = np.array(shape, dtype=float)

        for u in range(N):
            for v in range(N):
                distance = np.sqrt((np.power((u - (N / 2)), 2)) + (np.power((v - (N / 2)), 2)))

                # Check distance with cutoff
                if (distance <= cutoff):
                    mask[u, v] = 1
                elif (distance > cutoff):
                    mask[u, v] = 0

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

        for u in range(N):
            for v in range(N):
                distance = np.sqrt((np.power((u - (N / 2)), 2)) + (np.power((v - (N / 2)), 2)))

                # Check distance with cutoff
                if (distance <= cutoff):
                    mask[u, v] = 0
                elif (distance > cutoff):
                    mask[u, v] = 1

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

        for u in range(N):
            for v in range(N):
                distance = np.sqrt((np.power((u - (N / 2)), 2)) + (np.power((v - (N / 2)), 2)))
                sum = 1 + np.power((distance / cutoff), (2 * order))
                value = 1 / sum

                mask[u, v] = value

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

        for u in range(N):
            for v in range(N):
                distance = np.sqrt((np.power((u - (N / 2)), 2)) + (np.power((v - (N / 2)), 2)))

                # Check if distance is less than 0
                if (distance <= 0):
                    distance = 1

                sum = 1 + np.power((cutoff / distance), (2 * order))
                value = 1 / sum

                mask[u, v] = value

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

        for u in range(N):
            for v in range(N):
                distance = np.sqrt((np.power((u - (N / 2)), 2)) + (np.power((v - (N / 2)), 2)))
                value = np.exp((-1 * np.power(distance, 2)) / (2 * np.power(cutoff, 2)))

                mask[u, v] = value

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

        for u in range(N):
            for v in range(N):
                distance = np.sqrt((np.power((u - (N / 2)), 2)) + (np.power((v - (N / 2)), 2)))
                value = 1 - (np.exp(-1 * np.power(distance, 2) / (2 * np.power(cutoff, 2))))

                mask[u, v] = value

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
        processed = np.zeros(shape=(N, N), dtype=float)

        min = image.min()
        max = image.max()

        for row in range(N):
            for col in range(N):
                pixel = image[row, col]
                value = ((pixel - min) / (max - min)) * 255

                processed[row, col] = value

        return processed

    def filtering(self):
        """Performs frequency filtering on an input image
        returns a filtered image, magnitude of DFT, magnitude of filtered DFT
        ----------------------------------------------------------
        You are allowed to used inbuilt functions to compute fft
        There are packages available in numpy as well as in opencv
        Steps:
        """
        # Steps:

        # 1. Compute the fft of the image
        fftImage = np.fft.fft2(self.image)

        # 2. shift the fft to center the low frequencies
        shifted_fft = np.fft.fftshift(fftImage)

        # Save magnitude of DFT
        dft_magnitude = 10 * np.log(np.abs(shifted_fft))
        dft_image = dft_magnitude.astype('uint8')
        cv2.imwrite('output\Magnitude_of_DFT.jpg', dft_image)

        # 3. get the mask (write your code in functions provided above) the functions can be called by self.filter(shape, cutoff, order)
        # Check which mask is using
        if (self.filter == self.get_ideal_low_pass_filter):
            mask = self.filter(self.image, self.cutoff)

        elif (self.filter == self.get_ideal_high_pass_filter):
            mask = self.filter(self.image, self.cutoff)

        elif (self.filter == self.get_butterworth_low_pass_filter):
            mask = self.filter(self.image, self.cutoff, self.order)

        elif (self.filter == self.get_butterworth_high_pass_filter):
            mask = self.filter(self.image, self.cutoff, self.order)

        elif (self.filter == self.get_gaussian_low_pass_filter):
            mask = self.filter(self.image, self.cutoff)

        elif (self.filter == self.get_gaussian_high_pass_filter):
            mask = self.filter(self.image, self.cutoff)


        # 4. filter the image frequency based on the mask (Convolution theorem)
        filtered = mask * shifted_fft

        # 5. compute the inverse shift
        inverse_shift = np.fft.ifftshift(filtered)

        # 6. compute the inverse fourier transform
        inverse_fft = np.fft.ifft2(inverse_shift)

        # 7. compute the magnitude
        computed_magnitude = np.abs(inverse_fft)

        # Save magnitude of filtered dft
        filtered_magnitude = computed_magnitude.astype('uint8')
        cv2.imwrite('output\Magnitude_of_Filtered_DFT.jpg', filtered_magnitude)

        # 8. You will need to do a full contrast stretch on the magnitude and depending on the algorithm you may also need to
        # take negative of the image to be able to view it (use post_process_image to write this code)
        # Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        # filtered image, magnitude of DFT, magnitude of filtered DFT: Make sure all images being returned have grey scale full contrast stretch and dtype=uint8
        processed = self.post_process_image(computed_magnitude)

        # Save filtered image
        processed = processed.astype('uint8')
        cv2.imwrite('output\Filtered_Image.jpg', processed)



        # filtered image, magnitude of the DFT and magnitude of filtered dft
        return [processed, dft_image, filtered_magnitude]
