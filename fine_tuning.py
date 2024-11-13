from transformers import CLIPProcessor, CLIPModel
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np
from PIL import Image
import random

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

dataset2 = []
files = ["full_numpy_bitmap_crab.npy", "full_numpy_bitmap_crocodile.npy", "full_numpy_bitmap_lion.npy",
         "full_numpy_bitmap_lobster.npy", "full_numpy_bitmap_monkey.npy", "full_numpy_bitmap_octopus.npy",
         "full_numpy_bitmap_panda.npy", "full_numpy_bitmap_swan.npy"]


# For each class, we are giving them a corresponding label: 0 to 5
class_labels = {
    "full_numpy_bitmap_crab.npy": 0,
    "full_numpy_bitmap_crocodile.npy": 1,
    "full_numpy_bitmap_lion.npy": 2,
    "full_numpy_bitmap_lobster.npy": 3,
    "full_numpy_bitmap_monkey.npy": 4,
    "full_numpy_bitmap_octopus.npy": 5,
    "full_numpy_bitmap_panda.npy": 6,
    "full_numpy_bitmap_swan.npy": 7
}

for filename in files:
    images = np.load(filename)
    print(f"Loaded {filename} with shape: {images.shape}")

    t_0 = time.perf_counter()
    count = 0

    # Loop through each image in the file
    for i in range(len(images)):
        # Only process the first 1000 images from each class

        image = images[i]  # Provides (728,) array
        reshape = image.reshape(28, 28)  # Reshapes to (28, 28) numpy array
        image = Image.fromarray(reshape)
        grayscale_image = image.convert("RGB")

        # Assign the label based on the class of the file
        label = class_labels[filename]  # Get the label for the class

        data = {
            'image': grayscale_image,  # The image tensor
            'label': label  # The corresponding label (class)
        }

        dataset2.append(data)
        count += 1

    t_1 = time.perf_counter()
    print(f"Successfully processed {filename} in {t_1 - t_0:.2f} seconds")



labels = ["a drawing of a crab", "a drawing of a crocodile", "a drawing of a lion", "a drawing of a lobster",
          "a drawing of a monkey", "a drawing of a octopus", "a drawing of a panda", "a drawing of a swan"]
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

print(label2id)


random.shuffle(dataset2)

def homemade_batch(num_img, batch_size=10, start_img=0, data_type = "test"):
    # Initialize empty set to store predicted values and their probabilities
    homemade = []
    num_batches = num_img // batch_size
    extra = num_img % batch_size # Not implemented yet

    # Allows computations to be run on GPU instead of CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    t_0 = time.perf_counter()

    for i in range(num_batches):
        t1 = time.perf_counter()

        # Create a temporary batch of data
        start = i * batch_size + start_img
        end = (i + 1) * batch_size + start_img

        batch = dataset2[start:end]
        images = [item["image"] for item in batch]


        subset = dataset2[start:end]
        input = processor(text=labels, images=images, return_tensors="pt", padding=False).to(device)
        output = model(**input)

        # Access logits of the input images, apply softmax function
        logits = output.logits_per_image
        probs = logits.softmax(dim=1)

        # Find maximum of the probabilities, as well as their corresponding index, append them to list
        max_prob, max_id = probs.max(dim=1)
        homemade.append([max_prob.cpu().detach(), max_id.cpu().detach()])
        torch.cuda.empty_cache()

        if i % 50 == 0:
          t2 = time.perf_counter()
          print(f"Finished batch {i + 1} of {num_batches} in {t2 - t1} seconds")

    t_3 = time.perf_counter()
    print(f"Finished entire dataset in {t_3 - t_0} seconds")

    # Returns list of tensors, structure is [[tensor([first batch maximum probabilities]), tensor([corresponding indices/labels])],
    #                                         [tensor([second batch maximum probabilities]), tensor([corresponding indices/labels])],
    #                                         [tensor([third batch maximum probabilities]), tensor([corresponding indices/labels])]]
    return homemade


# Takes output of homemade_batch as input and returns clean data
def prediction_reformat(subset):
    # Initialize empty list to store new reformatted data
    predicted = []
    count = 0

    # len(subset) = number of batches
    for i in range(len(subset)):
        for k in range(len(subset[0][0])):
            prob = subset[i][0][k].item()
            id = subset[i][1][k].item()

            label = id2label[id]
            predicted.append([count, label, prob, id])

            count += 1

    # Returns nested list with form [[index, "label", probability, id],
    #                                [index, "label", probability, id]]
    return predicted

def accuracy(result, data_type = "test"):
    correct = 0
    total = 0

    # Create dictionary to count how many of each label occurs in the subset, all labels initialized to 0
    all_labels = {}
    for label in labels:
        all_labels[label] = 0

    # Dictionary to keep track of which classes were incorrectly predicted
    incorrect = {}
    for label in labels:
        incorrect[label] = 0

    # Iterate through the results for each image in the subset
    for i in range(len(result)):
        # Automatically increase count of label in dictionary for appearing
        all_labels[result[i][1]] += 1

        # If the actual id/label aligns with the predicted one, add to correct count
        if dataset2[i]["label"] == result[i][3]:
            correct += 1
            total += 1
            if total % 50 == 0:
              print(f"Model accurately predicted {result[i][1]} with {result[i][2] * 100}% confidence.")
        else:
            # If they do not align, increase count of predicted id/label in incorrect dictionary
            total += 1
            if total % 50 == 0:
              print(f"Model inaccurately predicted {result[i][1]} with {result[i][2] * 100}% confidence.")
            incorrect[result[i][1]] += 1

    print(f"Accuracy: {(correct/total) * 100}%")

    worst_accuracy = []
    # For every label, calculate percentage predicted correctly by subtracting total by incorrect
    for label in all_labels:
        correct =  all_labels[label] - incorrect[label]
        total = all_labels[label]
        if total != 0:
          print(f"For {label}: Predicted {correct} out of {total} correct. {(correct) / total * 100}% Accuracy")
          worst_accuracy.append([label, correct/total])
        else:
          print(f"{label} was not in this analysis set")

    worst_group = min(worst_accuracy, key=lambda x: x[1])
    print(f"The worst performing group is '{worst_group[0]}' with an accuracy of {worst_group[1] * 100}%")


def data_analysis(predictions, data_type = "test"):
    cleaned = prediction_reformat(predictions)
    final_results = accuracy(cleaned, data_type)

    return final_results


def train_shuffled(num_img, batch_size=10, num_epoch=2):
    # Set up training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    model.to(device)
    model.train()
    best_loss = float("inf")
    total_loss = 0

    for epoch in range(num_epoch):
        # Shuffle the dataset at the beginning of each epoch
        random.shuffle(dataset2)
        epoch_loss = 0
        t0 = time.perf_counter()

        # Process the dataset in batches
        for i in range(num_img // batch_size):
            start = i * batch_size
            end = (i + 1) * batch_size

            # Define the batch
            batch = dataset2[start:end]
            images = [item["image"] for item in batch]
            label = [item["label"] for item in batch]

            # Convert images and labels to tensors and move them to device
            inputs = processor(text=labels, images=images, return_tensors="pt", padding=False).to(device)


            # Model forward pass
            output = model(**inputs)
            logits_per_image = output.logits_per_image
            logits_per_text = output.logits_per_text.squeeze()

            # Convert labels to tensor for calculating loss
            targets = torch.tensor(label).to(device)

            # Calculate loss
            loss_img = F.cross_entropy(logits_per_image, targets)
            loss_text = F.cross_entropy(logits_per_text, targets)
            loss = (loss_img + loss_text) / 2

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss and time
            epoch_loss += loss.item()
            print(f"Finished batch {i + 1}/{num_img // batch_size} with loss {loss.item():.4f}")

        # Epoch completion tracking
        avg_loss = epoch_loss / (num_img // batch_size)
        total_loss += epoch_loss
        t1 = time.perf_counter()

        # Save model if the current epoch's loss is the best so far
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), '/content/clip.pt')

        print(f"Epoch {epoch+1}/{num_epoch} completed in {t1 - t0:.2f} seconds, Loss: {avg_loss:.4f}")

    return total_loss / (num_epoch * (num_img // batch_size))
