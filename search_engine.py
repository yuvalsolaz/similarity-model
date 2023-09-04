import os.path

import torch
import faiss
from datasets import load_dataset, Dataset
import torchvision.transforms as T
from transformers import AutoFeatureExtractor, AutoModel
from transformers import CLIPProcessor, CLIPModel

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


class SearchEngine(object):

    def __init__(self,
                 dataset_path,
                 similarity_model,
                 index_column='embeddings',
                 metric_type=faiss.METRIC_INNER_PRODUCT):
        self._similarity_model = similarity_model
        self._index_column = index_column
        self._metric_type = metric_type
        print(f'loading similarity model {self._similarity_model}...')
        # self._model = AutoModel.from_pretrained(model_checkpoint)
        self._model = CLIPModel.from_pretrained(self._similarity_model)
        self._extractor = CLIPProcessor.from_pretrained(self._similarity_model).feature_extractor
        self._transformation_chain = T.Compose(
            [
                # We first resize the input image to 256x256 and then we take center crop.
                T.Resize(int((256 / 224) * self._extractor.size["shortest_edge"])),
                T.CenterCrop(self._extractor.size["shortest_edge"]),
                T.ToTensor(),
                T.Lambda(lambda x: torch.cat([x, x, x], 0)),
                T.Normalize(mean=self._extractor.image_mean, std=self._extractor.image_std),
            ]
        )

        print(f'copy model to {device} device')
        self._model.to(device)

        print(f'load search dataset from {dataset_path}')
        self._dataset = load_dataset('imagefolder', data_dir=dataset_path, drop_labels=False)['train']
                                     # , download_mode='force_redownload')['train']
        self._label_ids2label_names = self._dataset.features['label'].names

        extract_fn = self.__batch_embeddings(self._model.to(device))
        self._dataset = self._dataset.map(extract_fn, batched=True, batch_size=24)
        print(f'embeddings dataset created with {self._dataset.shape} samples...')

        self._transformation_chain = T.Compose(
            [
                # We first resize the input image to 256x256 and then we take center crop.
                T.Resize(int((256 / 224) * self._extractor.size["shortest_edge"])),
                T.CenterCrop(self._extractor.size["shortest_edge"]),
                T.ToTensor(),
                T.Normalize(mean=self._extractor.image_mean, std=self._extractor.image_std),
            ]
        )

        print(f'create faiss index on {self._index_column}...')
        self._dataset.add_faiss_index(column=self._index_column, metric_type=metric_type)

    # map embedding on dataset images.
    def __batch_embeddings(self, model: torch.nn.Module):
        """Utility to compute embeddings."""
        device = model.device

        def pp(batch):
            images = batch["image"]
            image_path = [image.filename for image in images]
            labels = batch["label"]
            label_names = [self._label_ids2label_names[label] for label in labels]
            image_batch_transformed = torch.stack(
                [self._transformation_chain(image) for image in images]
            )
            new_batch = {"pixel_values": image_batch_transformed.to(device)}
            with torch.no_grad():
                # embeddings = model(**new_batch).pooler_output.cpu()
                embeddings = model.get_image_features(**new_batch).cpu()

            # add image path
            return {"image_path": image_path, "label_name": label_names, "embeddings": embeddings}

        return pp

    def get_embeddings(self, image_path):
        single_dataset = load_dataset('imagefolder', data_dir=os.path.dirname(image_path), drop_labels=False)['train']
        extract_fn = self.__batch_embeddings(self._model.to(device))
        single_dataset = single_dataset.map(extract_fn, batched=True)
        return torch.Tensor(single_dataset['embeddings'].__iter__().__next__())
        # transformation_chain = T.Compose(
        #     [
        #         # We first resize the input image to 256x256 and then we take center crop.
        #         T.Resize(int((256 / 224) * self._extractor.size["shortest_edge"])),
        #         T.CenterCrop(self._extractor.size["shortest_edge"]),
        #         T.ToTensor(),
        #         T.Normalize(mean=self._extractor.image_mean, std=self._extractor.image_std),
        #     ]
        # )
        #
        # new_image = transformation_chain(image)
        # with torch.no_grad():
        #     # embeddings = model(**new_batch).pooler_output.cpu()
        #     embeddings = self._model.get_image_features(**new_image).cpu()
        # return {"embeddings": embeddings}

    def search(self, query_image, k=5):
        query_embedding = self.get_embeddings(image_path=query_image).cpu().detach().numpy()
        scores, images = self._dataset.get_nearest_examples(index_name=self._index_column, query=query_embedding, k=k)
        # normalize scores
        query_norm = torch.norm(torch.Tensor(query_embedding))
        for i in range(len(scores)):
            image_norm = torch.norm(torch.Tensor(images[self._index_column][i]))
            scores[i] /= (query_norm * image_norm)

        return scores, images
