# ViT-GoogleQuickDraw
Using CLIP (Contrastive Language-Image Pre-Training) and ViT (Visual Transformers) to predict user drawings from set classes. In this case, I used the labels ("a drawing of a crab", "a drawing of a crocodile", "a drawing of a lion", "a drawing of a lobster", "a drawing of a monkey", "a drawing of an octopus", "a drawing of a panda", "a drawing of a swan"). 

Drawing pad is created using Pygame, PIL is used to take a screenshot when the user hits finishes doodling and presses enter. To clear the current drawing, the user can press the spacebar or backspace. The image is then resized to 28x28 and fed is processed by the processor provided by the CLIP model. These tensors are then fed into  the real model, and the logits provide the similarities between the image and each of the labels. Finally, we take the softmax of this tensor to get the probabilities that the image is represented by each class. 

An interesting thing to note is that using OpenAI's clip-vit-large-patch14 model performs extremely well when working with 4 classes, but when more labels are added, it begins to perform relatively poorly. To combat this, I finetuned this model using Google's Quick, Draw! dataset. When testing on a sample set of 1000 images, the model had an overall accuracy of 55.6%, with the worst group only having an accuracy of 39.35%. After finetuning, the model had an overall accuracy of 87.3% and the worst-performing group still had an accuracy of 76%! 

If you want to run this yourself, I recommend having an Nvidia GPU. You should also have the proper CUDA version that matches your GPU. IDK how to make a requirements.txt file, so just yeah. 


You can download the finetuned weights here: https://drive.google.com/file/d/1CvMXjFhXSgRipkkGD0E-vmviKmP-Ew17/view?usp=sharing
Download the npy files here: https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap;tab=objects?prefix=&forceOnObjectsSortingFiltering=false
