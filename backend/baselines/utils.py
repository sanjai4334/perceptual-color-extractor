def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(rgb[0], rgb[1], rgb[2])


def wrap_colors(rgb_colors):
    return {
        "rgb": rgb_colors,
        "hex": [rgb_to_hex(c) for c in rgb_colors],
    }
