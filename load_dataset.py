import torch
from tensorboard_handler import write_embedding_preview
from datasets import load_dataset
import torchvision.transforms as T
from transformers import AutoFeatureExtractor, AutoModel
from transformers import CLIPProcessor, CLIPModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # TODO parameters
    root_directory = "./data/docs-sm"  # TODO args
    dataset = load_dataset('imagefolder', data_dir="data/docs-sm", drop_labels=False)
    label_ids2label_names = dataset['train'].features['label'].names
    # model_checkpoint = 'facebook/convnext-tiny-224'
    model_checkpoint = 'openai/clip-vit-base-patch32'

    print(f'loading model from : {model_checkpoint}')
    # extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    extractor = CLIPProcessor.from_pretrained(model_checkpoint).feature_extractor
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
            image_path = [image.filename for image in images]
            labels = batch["label"]
            label_names = [label_ids2label_names[label] for label in labels]
            image_batch_transformed = torch.stack(
                [transformation_chain(image) for image in images]
            )
            new_batch = {"pixel_values": image_batch_transformed.to(device)}
            with torch.no_grad():
                # embeddings = model(**new_batch).pooler_output.cpu()
                embeddings = model.get_image_features(**new_batch).cpu()

            # add image path
            return {"image_path": image_path, "label_name": label_names, "embeddings": embeddings}

        return pp


    # Here, we map embedding extraction utility on our subset of candidate images.
    # model = AutoModel.from_pretrained(model_checkpoint)
    model = CLIPModel.from_pretrained(model_checkpoint)
    batch_size = 24
    extract_fn = extract_embeddings(model.to(device))
    dataset_emb = dataset.map(extract_fn, batched=True, batch_size=24)
    print(f'embeddings dataset created with {dataset_emb.shape} samples...')

    log_dir = f'embeddings/{model_checkpoint}'
    print(f'write tensorboard logs to {log_dir}')
    write_embedding_preview(log_dir=log_dir, ds=dataset_emb['train'], embedding_field='embeddings', tag='vision')
    print(f'run: tensorboard --logdir {log_dir} --bind_all')
