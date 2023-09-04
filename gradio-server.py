import gradio as gr
import torch
import faiss

from search_engine import SearchEngine

dataset_path = r'data/docs-sm'
similarity_model = 'openai/clip-vit-base-patch32'
metric_type = faiss.METRIC_L2

search_engine = SearchEngine(dataset_path=dataset_path,
                             similarity_model=similarity_model,
                             index_column='embeddings',
                             metric_type=metric_type)

title = "search similar documents"
description = 'Select document for search'


def run_query(input_image):
    top_k = 5
    scores, images = search_engine.search(query_image=input_image, k=top_k)
    results = []
    for i in range(len(scores)):
        image_path = images['image_path'][i]
        metric_result = 'score' if metric_type == faiss.METRIC_INNER_PRODUCT else 'distance'
        results.append((image_path, f'{metric_result}:{format(scores[i],".3f")}'))
    return results


if __name__ == '__main__':
    inputs_image = [
        gr.Image(
            type='filepath',
            label='Input Image'
        )
    ]
    outputs_image = [
        gr.Gallery(label='Search Results', ).style(columns=[3], rowa=[3], object_fit='contain', height='auto')]
    demo = gr.Interface(
        fn=run_query,
        inputs=inputs_image,
        outputs=outputs_image,
        title=title,
        description=description,
    )

    demo.launch(server_name='0.0.0.0')

# with gr.Blocks() as demo:
#     with gr.Row():
#         gr.Markdown('query document')
#         inputs_image = [
#             gr.Image(
#                 type='filepath',
#                 label='Input Image'
#             )
#         ]
#         search_btn = gr.Button('search')
#         top_k_input = gr.Slider(label='top_k', show_label=True, value=5, minimum=1, maximum=10, step=1)
#     with gr.Column():
#         gr.Markdown('results documents')
#         search_inputs = [inputs_image, top_k_input]
#         results_output = gr.Gallery(columns=[3], rows=[3], height='auto')
#         search_btn.click(fn=run_query, inputs=search_inputs, outputs=results_output)
