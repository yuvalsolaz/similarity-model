from datasets import Dataset
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as transforms
from PIL import Image

thumbnail_width = 64
thumbnail_height = 64


def load_image(image_file_name):
    return Image.open(image_file_name)


def write_embedding(log_dir, ds: Dataset, embedding_field, tag):
    # embedding tensor:
    embedding_tensor = torch.Tensor(ds[:][embedding_field])

    # metadata:
    metadata_df = ds.with_format('pandas')
    metadata_header = ['image_path', 'label']
    metadata = metadata_df[:][metadata_header].values.tolist()

    print(f'writes {embedding_tensor.shape[0]} points with {embedding_tensor.shape[1]} embedding'
          f'and {metadata_header} metadata to {log_dir}')
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_embedding(embedding_tensor, tag=tag, metadata=metadata,
                         metadata_header=metadata_header if len(metadata_header) > 1 else None)
    writer.flush()
    writer.close()


def write_embedding_preview(log_dir, ds: Dataset, embedding_field, tag):
    # embedding tensor:
    embedding_tensor = torch.Tensor(ds[:][embedding_field])

    # metadata:
    metadata_df = ds.with_format('pandas')
    metadata_header = ['image_path', 'label', 'label_name']
    metadata = metadata_df[:][metadata_header].values.tolist()

    # image labels (thumbnails)
    transform = transforms.Compose([transforms.PILToTensor(), transforms.Resize([thumbnail_width, thumbnail_height],
                                                                                interpolation=transforms.InterpolationMode.BICUBIC,
                                                                                antialias=True),
                                    transforms.Lambda(lambda x: x / 255.0)])
    def thumbnail(image_file_name):
        image = load_image(image_file_name=image_file_name)
        return transform(image)

    print('map image files to thumbnails ')
    image_label_field = 'thumbnail'
    img_label_dataset = ds.map(lambda x: {image_label_field: thumbnail(x['image_path'])})
    label_images = torch.Tensor(img_label_dataset[:][image_label_field])

    print(f'writes {embedding_tensor.shape[0]} points with {embedding_tensor.shape[1]} embedding'
          f'and {metadata_header} metadata to {log_dir}')
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_embedding(embedding_tensor, tag=tag, metadata=metadata,
                         metadata_header=metadata_header if len(metadata_header) > 1 else None,
                         label_img=label_images)
    writer.flush()
    writer.close()
