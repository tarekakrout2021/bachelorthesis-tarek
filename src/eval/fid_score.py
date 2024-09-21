from torcheval.metrics import FrechetInceptionDistance
from torchvision import transforms
from torchvision.datasets import MNIST

from src.utils.helpers import load_model

# Initialize the FID metric
fid_metric = FrechetInceptionDistance()  # device="cuda"

# Define the transform to preprocess images
transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),  # InceptionV3 input size
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ),  # ImageNet normalization
    ]
)

# Load real images (MNIST for this example)
real_dataset = MNIST(root="./data", train=True, download=True, transform=transform)

n_samples = 250

# Load the model
model = load_model("baseline_mnist")
generated_samples = model.sample(n_samples=n_samples)
generated_samples = generated_samples.reshape(n_samples, 1, 28, 28)
generated_samples = generated_samples.repeat(1, 3, 1, 1)

real_dataset = real_dataset.data.unsqueeze(1)
real_dataset = real_dataset.repeat(1, 3, 1, 1)

res = real_dataset / 255.0
res = res[:n_samples, :, :, :]
fid_metric.update(res, is_real=True)
# Add generated images to the FID metric
fid_metric.update(generated_samples, is_real=False)  # Generated images

# Compute the FID score
fid_score = fid_metric.compute()
print(f"FID Score: {fid_score}")
