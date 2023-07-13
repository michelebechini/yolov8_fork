import numpy as np
import random
import cv2
from scipy.stats import tukeylambda
import albumentations as Alb


def get_image_bit_depth(image):
    if image is None:
        raise ValueError("Failed to read the image.")
    bit_depth = image.dtype.itemsize * 8
    return bit_depth


def apply_random_flare(image, char_dim=None, roi=None):

    bit_depth = get_image_bit_depth(image)

    flag_GRAYtoRGB = False
    flag_SCALE = False

    if bit_depth != 8:
        # scale to 8 bit
        flag_SCALE = True
        scaling = ((2 ** 8) / (2 ** bit_depth))
        image = image * scaling

    elif len(image.shape) == 2:
        flag_GRAYtoRGB = True
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if roi is None:
        roi=np.array([0., 0., image.shape[1], image.shape[0]])
        img_roi_cropped = image.copy()
    else:
        img_roi_cropped = image[int(roi[1]):int(roi[3]), int(roi[0]):int(roi[2])]

    # get flare center point
    flare_center = get_flare_center(img_roi_cropped, thresh=240)
    
    if not np.isnan(flare_center).any():

        flare_center = flare_center + roi[:2]

        # normalize flare center coordinate
        center_norm_x = flare_center[0]/image.shape[1]
        center_norm_y = flare_center[1] / image.shape[0]

        x_min = np.max([0, center_norm_x-0.01])
        y_min = np.max([0, center_norm_y-0.01])
        x_max = np.min([1, center_norm_x+0.01])
        y_max = np.min([1, center_norm_y+0.01])

        if char_dim is None:
            diag_size = np.hypot(image.shape[0], image.shape[1])
        else:
            diag_size = char_dim
        radius_limit_lower = int(diag_size*0.05)
        radius_limit_upper = int(diag_size*0.35)

        # apply random flare
        transf_flare = Alb.RandomSunFlare(
            flare_roi=(x_min, y_min, x_max, y_max),
            angle_lower=0,
            angle_upper=1,
            num_flare_circles_lower=0,
            num_flare_circles_upper=3,
            src_radius=np.random.randint(radius_limit_lower, radius_limit_upper),
            always_apply=True,
            p=0.5
        )

        output_transf = transf_flare(image=image)

        flare_img = output_transf['image']
    
    else:
        # case in which flare should not be applied
        flare_img = image.copy()

    # blur the output image to smooth the flare edges
    transf_blur = Alb.OneOf([
        Alb.MotionBlur(blur_limit=11, p=0.3),
        Alb.GaussianBlur(blur_limit=11, sigma_limit=(10, 150), p=0.4),
        Alb.GlassBlur(sigma=1., p=0.3)
    ], p=1)

    output_blur = transf_blur(image=flare_img)

    output_img = output_blur['image']

    if bit_depth == 8:
        output_img = output_img.astype('uint8')
    else:
        output_img = output_img.astype('uint')

    if flag_GRAYtoRGB is True:
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2GRAY)

    if flag_SCALE is True:
        output_img = output_img/scaling

    return output_img


def get_flare_center(image, thresh=240):

    bit_depth = get_image_bit_depth(image)

    if bit_depth != 8:
        # scale to 8 bit
        image = image * ((2 ** 8) / (2 ** bit_depth))

    # get the grayscale image if the image is RGB
    if len(image.shape) == 3:
        image_gs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gs = image.copy()

    # threshold the image to get a binary mask to extract only brightest portions
    _, mask = cv2.threshold(src=image_gs.astype('float64'), thresh=thresh, maxval=1, type=cv2.THRESH_BINARY)

    flare_center = find_random_extpoint_from_mask(image_gs, mask)

    return flare_center


def find_random_extpoint_from_mask(image, mask):

    if len(image.shape) == 3:
        image_gs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gs = image.copy()

    # Get the contours of the binary image
    contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize the variable that will contain the blob with the highest amount of high intensity pixels
    max_blob = None
    max_blob_area = 0

    # Get the blob with the highest number of high intensity pixels
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_blob_area:
            max_blob_area = area
            max_blob = contour

    if max_blob is None:
        return np.array([np.nan, np.nan])

    # Get the perimeter of the blob
    perimeter = cv2.arcLength(max_blob, True)

    # Get extreme points on the perimeter of the blob 
    perimeter_points = cv2.approxPolyDP(max_blob, 0.02 * perimeter, True)

    iter_num = 0
    flag_WHITE = False

    while flag_WHITE is False and iter_num < 1000:

        iter_num += 1

        # Get the indexes of two random points
        idxs = random.sample(list(np.arange(len(perimeter_points))), 2)

        # Compute point to point distance and pointing unit vector 
        dist = np.linalg.norm(perimeter_points[idxs[0]] - perimeter_points[idxs[1]])
        if dist == 0:
            continue

        dir = (perimeter_points[idxs[0]] - perimeter_points[idxs[1]]) / dist

        # Define a random scale for the displacement
        random_scale = np.random.randint(0, dist + 0.1)

        # Extract a random point
        random_point = perimeter_points[idxs[1]] + random_scale * dir
        random_point = np.squeeze(random_point)

        # Check that the selected point is over a white region
        if image_gs[random_point.astype('int')[1], random_point.astype('int')[0]] > 200:
            flag_WHITE = True
            break

    return random_point


def sensor_noise(image, sensor, integration_time=0.1, use_PNRU=False, use_blur=False, use_bloom=False, row_noise=False):
    """
    Function to apply sensor noise to the input image

    Parameters
    ----------
    image
    sensor
    bit_depth
    integration_time
    use_PNRU
    use_blur

    Returns
    -------

    """

    bit_depth = get_image_bit_depth(image)

    flag_RGBtoGRAY = False
    flag_GRAYtoRGB = False

    if len(image.shape) == 3 and sensor.is_RGB is False:
        flag_RGBtoGRAY = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    elif len(image.shape) == 2 and sensor.is_RGB is True:
        flag_GRAYtoRGB = True
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # apply bloom if required
    if use_bloom:
        image = do_blooming(image, bloom_thresh=sensor.bloom_thresh)

    # Apply blur to the input image if required
    if use_blur:
        if image.max() > 1:
            img_norm = image / (2 ** bit_depth - 1)
            img_blurred = cv2.GaussianBlur(img_norm.astype('float64'), ksize=(0, 0), sigmaX=1., sigmaY=0.)
            img_blurred = np.clip(img_blurred * (2 ** bit_depth - 1), 0, (2 ** bit_depth - 1))
        else:
            img_norm = image.copy()
            img_blurred = cv2.GaussianBlur(img_norm.astype('float64'), ksize=(0, 0), sigmaX=1., sigmaY=0.)
            img_blurred = np.clip(img_blurred, 0, 1)

    else:
        img_blurred = image.astype('float64')

    # Step 1) Get input photons from noiseless image
    if sensor.is_RGB:
        # split in RGB channels and apply the specified transformation considering the per-channel QE
        R_ch, G_ch, B_ch = np.split(img_blurred, 3, axis=2)
        photons_R_ch = R_ch * sensor.G / sensor.n_tot[0]
        photons_G_ch = G_ch * sensor.G / sensor.n_tot[1]
        photons_B_ch = B_ch * sensor.G / sensor.n_tot[2]
        photons = np.concatenate((photons_R_ch,photons_G_ch,photons_B_ch), axis=2)
    else:
        photons = img_blurred * sensor.G / sensor.n_tot

    # Step 2) Apply the photon shot noise
    I_ph = np.random.poisson(photons).astype('float64')

    # Step 3) Convert Photons to electrons
    if sensor.is_RGB:
        # split in RGB channels and apply the specified transformation considering the per-channel QE
        I_ph_R_ch, I_ph_G_ch, I_ph_B_ch = np.split(I_ph, 3, axis=2)
        I_e_R_ch = I_ph_R_ch * sensor.n_tot[0]
        I_e_G_ch = I_ph_G_ch * sensor.n_tot[1]
        I_e_B_ch = I_ph_B_ch * sensor.n_tot[2]
        I_e = np.concatenate((I_e_R_ch, I_e_G_ch, I_e_B_ch), axis=2)
    else:
        I_e = I_ph * sensor.n_tot

    # Step 4) Apply PRNU if required
    if use_PNRU:
        I_light_e = I_e + I_e * sensor.PRNU_noise  # note that PRNU_noise is constant for all the images
    else:
        I_light_e = I_e.copy()

    # Step 5) Compute the dark current shot noise
    I_dc_e = np.random.poisson(integration_time * sensor.dark_current, size=I_e.shape).astype('float64')

    # Step 6) Compute the read out noise using a Tukey-Lambda distribution
    if sensor.is_RGB:
        I_read_R_ch_e = tukeylambda.rvs(lam=sensor.read_shape,
                                        loc=sensor.read_mean[0], scale=sensor.std_read,
                                        size=(I_e.shape[0], I_e.shape[1], 1))
        I_read_G_ch_e = tukeylambda.rvs(lam=sensor.read_shape,
                                        loc=sensor.read_mean[1], scale=sensor.std_read,
                                        size=(I_e.shape[0], I_e.shape[1], 1))
        I_read_B_ch_e = tukeylambda.rvs(lam=sensor.read_shape,
                                        loc=sensor.read_mean[2], scale=sensor.std_read,
                                        size=(I_e.shape[0], I_e.shape[1], 1))
        I_read_e = np.concatenate((I_read_R_ch_e, I_read_G_ch_e, I_read_B_ch_e), axis=2)
    else:
        I_read_e = tukeylambda.rvs(lam=sensor.read_shape, loc=sensor.read_mean, scale=sensor.std_read, size=I_e.shape)

    # Step 7) Get the row noise if required
    if row_noise is True:
        I_row = np.random.normal(loc=0, scale=sensor.row_std, size=(I_light_e.shape[0], 1))
        I_row_mat = np.zeros(I_light_e.shape[:2]) + I_row
    else:
        I_row_mat = np.zeros(I_light_e.shape)

    # Step 8) Get the total electrons to form the noised image and convert to DN
    I_total = np.rint(I_light_e + I_dc_e + I_read_e + I_row_mat)
    I_total_e = np.clip(I_total, 0, sensor.full_well)
    I_dn = I_total_e / sensor.G

    if I_total.max() >= sensor.full_well:
        I_dn_norm = np.zeros((I_dn.shape))
        I_dn_norm = cv2.normalize(src=I_dn, dst=I_dn_norm, alpha=0, beta=2 ** bit_depth - 1, norm_type=cv2.NORM_MINMAX)
    else:
        I_dn_norm = I_dn.copy()

    # Step 9) Apply quantization error
    I_q = np.random.uniform(low=-0.5, high=0.5, size=I_dn_norm.shape)
    I_dn_q = np.clip(np.rint(I_dn_norm + I_q), 0, (2 ** bit_depth - 1))

    if bit_depth == 8:
        I_dn_q = I_dn_q.astype('uint8')
    else:
        I_dn_q = I_dn_q.astype('uint')

    if flag_GRAYtoRGB is True:
        I_dn_q = cv2.cvtColor(I_dn_q, cv2.COLOR_RGB2GRAY)
    elif flag_RGBtoGRAY is True:
        I_dn_q = cv2.cvtColor(I_dn_q, cv2.COLOR_GRAY2RGB)

    return I_dn_q


def do_blooming(image, bloom_thresh=200):
    """
    Function to add blooming to an image in post processing.

    :param image:
    :param bloom_thresh:
    :return:
    """

    bit_depth = get_image_bit_depth(image)
    flag_scale = False

    if bit_depth != 8:
        # scale to 8 bit
        flag_scale = True
        image = image * ((2**8)/(2**bit_depth))

    flag_RGBtoGRAY = False

    # get the grayscale image if the image is RGB
    if len(image.shape) == 3:
        image_gs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        flag_RGBtoGRAY = True
    else:
        image_gs = image.copy()

    # threshold the image to get a binary mask to extract only brightest portions
    _, mask = cv2.threshold(src=image_gs.astype('float64'), thresh=bloom_thresh, maxval=1, type=cv2.THRESH_BINARY)

    # get the masked image with only brightest pixels
    bright_image = np.where(mask, image_gs, 0)

    # blur the mask and the bright image
    mask_blurred = cv2.GaussianBlur(mask, ksize=(31, 31), sigmaX=40., sigmaY=0.)

    bright_image_blurred = cv2.GaussianBlur(bright_image.astype('float64'), ksize=(31, 31), sigmaX=40., sigmaY=0.)

    for i in range(4):
        # blur the mask and the bright image
        mask_blurred = cv2.GaussianBlur(mask_blurred, ksize=(31, 31), sigmaX=40., sigmaY=0.)

        bright_image_blurred = cv2.GaussianBlur(bright_image_blurred.astype('float64'), ksize=(31, 31), sigmaX=40., sigmaY=0.)

    # weights for blending
    blend_weight = np.where(bright_image_blurred > image_gs, 1, 0)
    blend_weight = cv2.GaussianBlur(blend_weight.astype('float64'), ksize=(31, 31), sigmaX=40., sigmaY=0.)

    # get the bloomed image by using the normed mask as weight
    bloomed_img = (1-blend_weight)*image_gs + blend_weight*bright_image_blurred

    # clip the bloomed_image to the range
    bloomed_img = np.clip(bloomed_img, 0, 2**bit_depth - 1)

    if flag_RGBtoGRAY is True:
        # use uint8 because it is the only format available for grayscale images color conversion
        bloomed_img = cv2.cvtColor(bloomed_img.astype('uint8'), cv2.COLOR_GRAY2RGB)

    if flag_scale is True:
        bloomed_img = bloomed_img * ((2**bit_depth)/(2**8))

    return bloomed_img


class Sensor_VIS:
    """Simple vis sensor model

    Example of usage:

    TODO

    """

    def __init__(self, G=40.0, full_well=6060, QE=0.7, n_eo=0.65, FF=0.99, is_RGB=False):
        """
        Sensor Parameters initialization.

        :param G: [e-/ADU] - electron to ADU gain.
        :param full_well: [e-] - Amount of charge that a pixel can hold.
        :param QE: [-] - quantum efficiency in percentage. Typical range [0.45; 0.7]
        :param n_eo: [-] - optical throughput. Typical range [0.6; 0.7]
        :param FF: [-] - fill factor. Typical range [0.8; 1.0]

        :return: None.
        """
        self.G = G
        self.full_well = full_well
        self.is_RGB = is_RGB

        if is_RGB is True:
            if np.asarray(QE).size == 1:
                self.QE = [QE, QE, QE]
            else:
                self.QE = QE
        else:
            if np.asarray(QE).size == 3:
                self.QE = QE[0]
            else:
                self.QE = QE

        self.n_eo = n_eo
        self.FF = FF

        self.n_tot = np.asarray(self.QE) * n_eo * FF # [-] overall sensor efficiency

        return

    def noise_parameters(self, PRNU=0.01, dark_current=750.0, read_shape=0.14, read_mean=0.0, std_read=40.0,
                         bloom_thresh=200, row_std=20.):
        """
        Sensor Noise Parameters initialization.

        :param PRNU: [-] - Photo Response Non Uniformity as percentage of input signal. Typical values [0.01; 0.02]
        :param dark_current: [e-/sec] - Thermally generated electrons per seconds (depends on temperature).
        :param read_shape: [-] - Tukey-Lambda shape parameter. If equal to 0.14 it approximates a Normal distribution.
        :param read_mean: [-] - Mean (or "loc") value for Tukey-Lambda distribution.
        :param std_read: [-] - Standard deviation (or "scale") value for Tukey-Lambda distribution.
        :param bloom_thresh: [-] - Threshold intensity value over which the pixel can generate blooming effect.
        :param row_std: [-] - Standard deviation (or "scale") value for row noise intensity fluctuation.


        Note: to have a given offset X in pixels values due to read noise, set read_mean as X*sensor_Gain.

        :return: None.
        """

        self.std_PRNU = PRNU
        self.dark_current = dark_current
        self.read_shape = read_shape

        if self.is_RGB is True:
            if np.asarray(read_mean).size == 1:
                self.read_mean = [read_mean, read_mean, read_mean]
            else:
                self.read_mean = read_mean
        else:
            if np.asarray(read_mean).size == 3:
                self.read_mean = read_mean[0]
            else:
                self.read_mean = read_mean

        self.std_read = std_read

        self.bloom_thresh = bloom_thresh
        self.row_std = row_std

        return

    def compute_PRNU_noise(self, img_shape):
        """
        Compute the PRNU noise matrix that is constant for all the images.

        Parameters
        ----------
        img_shape

        Returns
        -------

        """

        self.PRNU_noise = np.random.normal(loc=0, scale=self.std_PRNU, size=img_shape)

        return