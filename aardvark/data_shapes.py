"""
Data is stored in memmaps for access speed. Here we give the shapes required to load these files
"""

CLIMATOLOGY_SHAPE = (4, 366, 24, 240, 121)

ICOADS_Y_SHAPE = (33601, 5, 12000)
ICOADS_X_SHAPE = (33601, 2, 12000)

IGRA_Y_SHAPE = (33604, 24, 1375)
IGRA_X_SHAPE = (1375, 2)

AMSUA_Y_SHAPE = (21916, 180, 360, 13)
AMSUB_Y_SHAPE = (21916, 360, 181, 12)
ASCAT_Y_SHAPE = (21913, 360, 181, 17)
HIRS_Y_SHAPE = (21913, 360, 181, 26)
GRIDSAT_Y_SHAPE = (48211, 2, 514, 200)
IASI_Y_SHAPE = (23373, 360, 181, 52)


def get_hadisd_shape(mode):
    """
    Return the shape of the HadISD array depending on variable
    """

    if mode != "train":
        dim_1 = 415
    else:
        var_dict = {"tas": 8719, "tds": 8617, "psl": 8016, "u": 8721, "v": 8721}
        dim_1 = var_dict[var]
    return (106652, dim_1)
