from io import BytesIO


def fig_to_image(fig):
    """Converts a Matplotlib figure to an image in memory with BytesIO."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf


def inches_to_meters(inches):
    return inches * 0.0254


def meters_to_inches(meters):
    return meters / 0.0254


def grams_to_newtons(grams):
    return grams * 0.00981


def newtons_to_grams(newtons):
    return newtons / 0.00981
