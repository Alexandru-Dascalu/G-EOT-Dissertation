import skimage.color

def get_normalised_lab_image(rgb_images):
    """
    Args : 4D tensor of size batch_size x 299 x 299 x 3
    Returns : A 4-D numpy array with shape [batch_size, 299, 299, 3] and with values between 0 and 1.
    """
    assert rgb_images.shape[1] == 299
    assert rgb_images.shape[2] == 299
    assert rgb_images.shape[3] == 3

    lab_images = skimage.color.rgb2lab(rgb_images)

    # normalise the lightness channel, which has values between 0 and 100
    lab_images[..., 0] = lab_images[..., 0] / 100
    # normalise the greeness-redness and blueness-yellowness channels, which normally are between -128 and 127
    lab_images[..., 1] = (lab_images[..., 1] + 128) / 255
    lab_images[..., 2] = (lab_images[..., 2] + 128) / 255

    return lab_images