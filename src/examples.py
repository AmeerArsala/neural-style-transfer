from PIL import Image


def get_louvre_img():
    content_image = Image.open("../data/louvre.jpg")
    print("The content image (C) shows the Louvre museum's pyramid surrounded by old Paris buildings, against a sunny sky with a few clouds.")
    return content_image


def get_claude_monet_img():
    example = Image.open("../data/monet_800600.jpg")
    return example