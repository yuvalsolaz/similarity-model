import torch
import faiss
from datasets import load_dataset
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

        print(f'loading similarity model {self._similarity_model}...')
        # self._model = AutoModel.from_pretrained(model_checkpoint)
        self._model = CLIPModel.from_pretrained(self._similarity_model)
        self._extractor = CLIPProcessor.from_pretrained(self._similarity_model).feature_extractor
        self._transformation_chain = T.Compose(
            [
                # We first resize the input image to 256x256 and then we take center crop.
                T.Resize(int((256 / 224) * extractor.size["shortest_edge"])),
                T.CenterCrop(extractor.size["shortest_edge"]),
                T.ToTensor(),
                T.Lambda(lambda x: torch.cat([x, x, x], 0)),
                T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
            ]
        )

        print(f'TODO : copy model to {device} device')
        self._model.to(device)

        print(f'load search dataset from {dataset_path}')
        self._dataset = load_dataset('imagefolder', data_dir=dataset_path, drop_labels=False)
        self._label_ids2label_names = self._dataset['train'].features['label'].names

        print(f'create faiss index on {self._index_column}...')
        self._dataset.add_faiss_index(column=self._index_column, metric_type=metric_type)

    def __extract_embeddings(self, image):
        device = self.__model.device
        image_transformed = transformation_chain(image)
        new_image = {"pixel_values": image_transformed.to(device)}
        with torch.no_grad():
            # embeddings = model(**new_batch).pooler_output.cpu()
            embeddings = self._model.get_image_features(**new_image).cpu()
        return {"embeddings": embeddings}

    def get_embeddings(self, image):
        return self.__extract_embeddings(image)

    def search(self, query_image, k=5):
        query_embedding = self.get_embeddings(query_image=query_image).cpu().detach().numpy()
        scores, images = self._dataset.get_nearest_examples(index_name=self.index_name, query=query_embedding, k=k)
        # normalie distances
        query_norm = torch.norm(torch.Tensor(query_embedding))
        for idx in range(scores.shape[0]):
            image_norm = torch.norm(torch.Tensor(images[self._index_column][idx]))
            scores[idx] /= (query_norm * image_norm)
        return scores, images
