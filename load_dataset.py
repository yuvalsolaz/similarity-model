
'''
collect all text files in given directory with metadata fields
cmd ine
find . -type f -exec file {} \; | grep ":.* ASCII text"

if we want to find all text files in the current directory, including its sub-directories, then we have to augment
the command using the Linux find command:
find . -type f -exec file -i {} \; | grep " text/plain;" | wc

'''

import torch
from datasets import load_dataset
from datasets import Dataset
from transformers import AutoImageProcessor, ConvNextModel
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from transformers import AutoFeatureExtractor, AutoModel

import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
    load images and labels from disk 
'''

def load_images(root_directory):
    target_size = (512, 512)
    # Define transformations to apply to the images (resizing, normalization, etc.)
    transform = transforms.Compose([
        transforms.Resize(target_size),  # Resize images to target size pixels
        transforms.ToTensor(),           # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize using ImageNet statistics
    ])

    # Create an ImageFolder dataset
    return ImageFolder(root=root_directory, transform=transform)


def add_embeddings(dataset:Dataset):
    model_checkpoint = 'facebook/convnext-tiny-224'
    print(f'loading model from : {model_checkpoint}')
    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    model = ConvNextModel.from_pretrained(model_checkpoint)
    # print(f'copy model to {device} device...')
    # model.to_device(device)

    def get_embeddings(image):
        inputs = image_processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            return outputs.last_hidden_state

    image_embedding_field = 'image_embedding'
    print(f'map image embeddings to {image_embedding_field}')
    dataset = dataset.map(lambda x: {image_embedding_field: get_embeddings(x).detach().cpu().numpy()[0]})
    return dataset


if __name__ == '__main__':
    # TODO parameters
    root_directory = "./data/docs-sm"  # TODO args
    model_checkpoint = 'facebook/convnext-tiny-224'

    dataset = load_dataset('imagefolder', data_dir="data/docs-sm", drop_labels=False)

    print(f'loading model from : {model_checkpoint}')
    # image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    # model = ConvNextModel.from_pretrained(model_checkpoint)

    extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    model = AutoModel.from_pretrained(model_checkpoint)
    hidden_dim = model.config.hidden_sizes
    # Data transformation chain.
    transformation_chain = T.Compose(
        [
            # We first resize the input image to 256x256 and then we take center crop.
            T.Resize(int((256 / 224) * extractor.size["shortest_edge"])),
            T.CenterCrop(extractor.size["shortest_edge"]),
            T.ToTensor(),
            T.Lambda(lambda x: torch.cat([x, x, x], 0)),
            T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
        ]
    )


    def extract_embeddings(model: torch.nn.Module):
        """Utility to compute embeddings."""
        device = model.device

        def pp(batch):
            images = batch["image"]
            image_batch_transformed = torch.stack(
                [transformation_chain(image) for image in images]
            )
            new_batch = {"pixel_values": image_batch_transformed.to(device)}
            with torch.no_grad():
                embeddings = model(**new_batch).pooler_output.cpu()
            return {"embeddings": embeddings}

        return pp

    # Here, we map embedding extraction utility on our subset of candidate images.
    batch_size = 24
    extract_fn = extract_embeddings(model.to(device))
    dataset_emb = dataset.map(extract_fn, batched=True, batch_size=24)
    print(f'{dataset_emb.shape} embeddings dataset' )












