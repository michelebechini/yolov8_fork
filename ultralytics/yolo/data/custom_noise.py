import numpy as np
import cv2
from scipy.stats import tukeylambda


def sensor_noise(image, sensor, bit_depth=8, integration_time=0.1, use_PNRU=False, use_blur=False):
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

    flag_RGBtoGRAY = False
    flag_GRAYtoRGB = False

    if len(image.shape) == 3 and sensor.is_RGB is False:
        flag_RGBtoGRAY = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    elif len(image.shape) == 2 and sensor.is_RGB is True:
        flag_GRAYtoRGB = True
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

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

    # Step 7) Get the total electrons to form the noised image and convert to DN
    I_total = np.rint(I_light_e + I_dc_e + I_read_e)
    I_total_e = np.clip(I_total, 0, sensor.full_well)
    I_dn = I_total_e / sensor.G

    if I_total.max() >= sensor.full_well:
        I_dn_norm = np.zeros((I_dn.shape))
        I_dn_norm = cv2.normalize(src=I_dn, dst=I_dn_norm, alpha=0, beta=2 ** bit_depth - 1, norm_type=cv2.NORM_MINMAX)
    else:
        I_dn_norm = I_dn.copy()

    # Step 8) Apply quantization error
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

    def noise_parameters(self, PRNU=0.01, dark_current=750.0, read_shape=0.14, read_mean=0.0, std_read=40.0):
        """
        Sensor Noise Parameters initialization.

        :param PRNU: [-] - Photo Response Non Uniformity as percentage of input signal. Typical values [0.01; 0.02]
        :param dark_current: [e-/sec] - Thermally generated electrons per seconds (depends on temperature).
        :param read_shape: [-] - Tukey-Lambda shape parameter. If equal to 0.14 it approximates a Normal distribution.
        :param read_mean: [-] - Mean (or "loc") value for Tukey-Lambda distribution.
        :param std_read: [-] - Standard deviation (or "scale") value for Tukey-Lambda distribution.

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