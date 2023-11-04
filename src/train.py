import torch
import imageio
from PIL import Image
from torchvision import transforms
import config
from data_processing import get_test_data, get_train_data

# train_data = get_train_data()
# test_data = get_test_data()
train_data = get_train_data().head(1)

train_filenames = train_data['file']
print(len(train_filenames))


processed_images = []
transform = transforms.Compose([
    # Replace new_width and new_height with desired dimensions
    transforms.Resize((config.WIDTH, config.HEIGHT)),
    transforms.ToTensor(),
])

for image_path in train_filenames:
    image = Image.open(image_path)
    processed_images.append(transform(image))

print(type(processed_images[0]))
