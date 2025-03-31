# GPT-model

#### The model classifies dog breeds from images using ViT architecture trained on a dataset of dog breeds.

[Transformers Library Installation](https://huggingface.co/docs/transformers/en/installation)  
[Hugging Face - Dog Breed Classifier](https://huggingface.co/skyau/dog-breed-classifier-vit/tree/main)  
[Vision Transformer (ViT)](https://huggingface.co/docs/transformers/en/model_doc/vit)  

`du -sh ~/.cache/huggingface/hub`

#### PyTorch:  
`python dog_breed_pytorch.py`

#### TensorFlow:  
`python dog_breed_tensorflow.py`  

#### Output example:
```
Model is loaded: vit
torch.Size([1, 3, 224, 224])
Predicted dog breed: Cardigan
```




