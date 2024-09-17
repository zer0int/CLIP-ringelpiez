### Plotting a CLIP (pre-trained) vs a CLIP (fine-tuned) in rather arbitrary ways. 🤖🦿💥🦾🤖

- ⚠️ If you're looking for a legitimate eval of your model, see [LAION-AI/CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark) instead.
- ⚠️ If you're looking for a legitimate exploration of your model, see [yossigandelsman/second_order_lens](https://github.com/yossigandelsman/second_order_lens) instead.
-----
- Models used: [openai/CLIP](https://github.com/openai/CLIP) ViT-L/14 vs. my [huggingface.co/zer0int/CLIP-GmP-ViT-L-14](https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14) / [direct link](https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14/blob/main/ViT-L-14-BEST-smooth-GmP-ft-state_dict.pt)

-----
`clipvsclip-flickr8k-ringelpiez.py`
- 👀 Eval for something CLIP is notoriously bad at, on a hard(ish) dataset: Cross-modal retrieval.
- ♻️ Roundhouse swaps text encoders and vision encoders of two CLIP models to see which one is the true combo breaker.
- ℹ️ Prerequisite: Requires flickr8k (dataset images).

![ringelpiez](https://github.com/user-attachments/assets/34afb7ef-51e7-434b-92ed-443473cc8a77)
-----
`compare-CLS-cossim-by-layer.py`
- 🪙 CLS token diff by layer

![cls-cat](https://github.com/user-attachments/assets/bdefe4cb-0f16-4e7d-a2ba-b1930eb4556b)
-----
`activations-text-vision-KL-PCA.py`
- 🧮 Kullback-Leibler divergence of activations

![text-and-vision](https://github.com/user-attachments/assets/8ac69dfd-1eca-42be-882d-d91b53b6fcb2)
-----
`activation-vision-KL_reversed-KL.py`
- ❗ Like previous, but just vision - and with a second, reversed KL. Because KL(p||q)≠KL(p||q).

![kldiv](https://github.com/user-attachments/assets/2617fd9c-3e5a-499e-a17d-cfc7447b26ef)
