import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

class PositionalEmbedding(nn.Module):
  def __init__(self, width, max_seq_length):
    super().__init__()

    # Creating positional encoding
    pe = torch.zeros(max_seq_length, width)

    for pos in range(max_seq_length):
      for i in range(width):
        if i % 2 == 0:
          pe[pos][i] = np.sin(pos/(10000 ** (i/width)))
        else:
          pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/width)))

    self.register_buffer('pe', pe.unsqueeze(0))

  def forward(self, x):
    # Add positional encoding to embeddings
    x = x + self.pe

    return x

class AttentionHead(nn.Module):
  def __init__(self, width, head_size):
    super().__init__()
    self.head_size = head_size

    self.query = nn.Linear(width, head_size)
    self.key = nn.Linear(width, head_size)
    self.value = nn.Linear(width, head_size)

  def forward(self, x, mask=None):
    # Obtaining Queries, Keys, and Values
    Q = self.query(x)
    K = self.key(x)
    V = self.value(x)

    # Dot Product of Queries and Keys
    attention = Q @ K.transpose(-2,-1)

    # Scaling
    attention = attention / (self.head_size ** 0.5)

    # Applying Attention Mask
    if mask is not None:
        attention = attention.masked_fill(mask == 0, float("-inf"))

    attention = torch.softmax(attention, dim=-1)

    attention = attention @ V

    return attention

class MultiHeadAttention(nn.Module):
  def __init__(self, width, n_heads):
    super().__init__()
    self.head_size = width // n_heads

    self.W_o = nn.Linear(width, width)

    self.heads = nn.ModuleList([AttentionHead(width, self.head_size) for _ in range(n_heads)])

  def forward(self, x, mask=None):
    # Combine attention heads
    out = torch.cat([head(x, mask=mask) for head in self.heads], dim=-1)

    out = self.W_o(out)

    return out

class TransformerEncoder(nn.Module):
    def __init__(self, width, n_heads, r_mlp=4):
        super().__init__()
        self.width = width
        self.n_heads = n_heads

        # Sub-Layer 1 Normalization
        self.ln1 = nn.LayerNorm(width)

        # Multi-Head Attention
        self.mha = MultiHeadAttention(width, n_heads)

        # Sub-Layer 2 Normalization
        self.ln2 = nn.LayerNorm(width)

        # Multilayer Perception
        self.mlp = nn.Sequential(
            nn.Linear(self.width, self.width*r_mlp),
            nn.GELU(),
            nn.Linear(self.width*r_mlp, self.width)
        )


    def forward(self, x, mask=None):
        # Residual Connection After Sub-Layer 1
        x = x + self.mha(self.ln1(x), mask=mask)

        # Residual Connection After Sub-Layer 2
        x = x + self.mlp(self.ln2(x))

        return x

def tokenizer(text, encode=True, mask=None, max_seq_length=32):
    if encode:
        out = chr(2) + text + chr(3) # Adding SOT and EOT tokens
        out = out + "".join([chr(0) for _ in range(max_seq_length-len(out))]) # Adding Padding
        out = torch.IntTensor(list(out.encode("utf-8"))) # Encoding Text
        mask = torch.ones(len(out.nonzero()))
        mask = torch.cat((mask,torch.zeros(max_seq_length-len(mask)))).type(torch.IntTensor)
    else:
        out = [chr(x) for x in text[1:len(mask.nonzero())-1]]
        out = "".join(out)
        mask = None

    return out, mask

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, width, max_seq_length, n_heads, n_layers, emb_dim):
        super().__init__()

        self.max_seq_length = max_seq_length  # Maximum length of input sequence

        self.encoder_embedding = nn.Embedding(vocab_size, width) # Embedding Table

        self.positional_embedding = PositionalEmbedding(width, max_seq_length)

        self.encoder = nn.ModuleList([TransformerEncoder(width,n_heads) for _ in range(n_layers)])

        # learned proj of image to embed
        self.projection = nn.Parameter(torch.randn(width, emb_dim))

    def forward(self, text, mask=None):
        # Text Embedding
        x = self.encoder_embedding(text)

        # Positional Embedding
        x = self.positional_embedding(x)

        # Transformer Encoder
        for encoder_layer in self.encoder:
            x = encoder_layer(x, mask=mask)

        # Takes features from the EOT Embedding
        x = x[torch.arange(text.shape[0]),torch.sub(torch.sum(mask[:,0],dim=1),1)]

        # joint multimodal embedding
        if self.projection is not None:
            x = x @ self.projection

        x = x / torch.norm(x, dim=-1, keepdim=True)

        return x

class ImageEncoder(nn.Module):
    def __init__(self, width, img_size, patch_size, n_channels, n_layers, n_heads, emb_dim):
        super().__init__()

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert width % n_heads == 0, "width must be divisible by n_heads"

        self.n_patches = (img_size[0] * img_size[1]) // (patch_size[0] * patch_size[1])

        self.max_seq_length = self.n_patches + 1

        # Patch Embedding
        self.linear_project = nn.Conv2d(n_channels, width, kernel_size=patch_size, stride=patch_size)

        # Classification Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, width))

        self.positional_embedding = PositionalEmbedding(width,self.max_seq_length)

        self.encoder = nn.ModuleList([TransformerEncoder(width,n_heads) for _ in range(n_layers)])

        # learned proj of image to embed
        self.projection = nn.Parameter(torch.randn(width, emb_dim))


    def forward(self,x):
        # Patch Embedding
        x = self.linear_project(x)
        x = x.flatten(2).transpose(1, 2)

        # Positional Embedding
        x = torch.cat((self.cls_token.expand(x.size()[0], -1, -1),x), dim=1)
        x = self.positional_embedding(x)

        # Transformer Encoder
        for encoder_layer in self.encoder:
            x = encoder_layer(x)

        # Takes Class Tokens
        x = x[:, 0, :]

        # joint multimodal embedding
        if self.projection is not None:
            x = x @ self.projection

        x = x / torch.norm(x, dim=-1, keepdim=True)

        return x

class CLIP(nn.Module):
    def __init__(self, emb_dim, vit_width, img_size, patch_size, n_channels, vit_layers, vit_heads, vocab_size, text_width, max_seq_length, text_heads, text_layers):
        super().__init__()

        self.image_encoder = ImageEncoder(vit_width, img_size, patch_size, n_channels, vit_layers, vit_heads, emb_dim)

        self.text_encoder = TextEncoder(vocab_size, text_width, max_seq_length, text_heads, text_layers, emb_dim)

        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def forward(self,image,text, mask=None):
        I_e = self.image_encoder(image)
        T_e = self.text_encoder(text, mask=mask)

        # scaled pairwise cosine similarities [n, n]
        logits = (I_e @ T_e.transpose(-2,-1)) * torch.exp(self.temperature)

        # symmetric loss function
        labels = torch.arange(logits.shape[0]).to(self.device)

        loss_i = nn.functional.cross_entropy(logits.transpose(-2,-1), labels)
        loss_t = nn.functional.cross_entropy(logits, labels)

        loss = (loss_i + loss_t) / 2

        return loss

import time
from PIL import Image

dataset2 = []
files = ["classes/full_numpy_bitmap_crab.npy", "classes/full_numpy_bitmap_crocodile.npy", "classes/full_numpy_bitmap_lion.npy",
         "classes/full_numpy_bitmap_lobster.npy", "classes/full_numpy_bitmap_monkey.npy", "classes/full_numpy_bitmap_octopus.npy",
         "classes/full_numpy_bitmap_panda.npy", "classes/full_numpy_bitmap_swan.npy"]


# For each class, we are giving them a corresponding label: 0 to 5
class_labels = {
    "classes/full_numpy_bitmap_crab.npy": 0,
    "classes/full_numpy_bitmap_crocodile.npy": 1,
    "classes/full_numpy_bitmap_lion.npy": 2,
    "classes/full_numpy_bitmap_lobster.npy": 3,
    "classes/full_numpy_bitmap_monkey.npy": 4,
    "classes/full_numpy_bitmap_octopus.npy": 5,
    "classes/full_numpy_bitmap_panda.npy": 6,
    "classes/full_numpy_bitmap_swan.npy": 7
}

for filename in files:
    images = np.load(rf"{filename}")
    print(f"Loaded {filename} with shape: {images.shape}")

    t_0 = time.perf_counter()
    count = 0

    # Loop through each image in the file
    for i in range(len(images)):
        # Only process the first 1000 images from each class

        image = images[i]  # Provides (728,) array
        reshape = image.reshape(28, 28)  # Reshapes to (28, 28) numpy array
        image = Image.fromarray(reshape)
        grayscale_image = image.convert("L")

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


random.shuffle(dataset2)


class MyCustomDataset(Dataset):
    def __init__(self):
        self.dataset = dataset2

        self.transform = T.ToTensor()

        self.captions = {0: "a drawing of a crab",
                         1: "a drawing of a crocodile",
                         2: "a drawing of a lion",
                         3: "a drawing of a lobster",
                         4: "a drawing of a monkey",
                         5: "a drawing of a octopus",
                         6: "a drawing of a panda",
                         7: "a drawing of a swan"}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        img = self.dataset[i]["image"]
        img = self.transform(img)

        cap, mask = tokenizer(self.captions[self.dataset[i]["label"]])

        mask = mask.repeat(len(mask), 1)

        return {"image": img, "caption": cap, "mask": mask}


emb_dim = 32
vit_width = 9
img_size = (28,28)
patch_size = (14,14)
n_channels = 1
vit_layers = 3
vit_heads = 3
vocab_size = 256
text_width = 32
max_seq_length = 32
text_heads = 8
text_layers = 4
lr = 1e-3
epochs = 10
batch_size = 128


train_set = MyCustomDataset()
test_set = MyCustomDataset()
train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size)


while True:
    idx = random.randint(0, len(train_set) - 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    text = torch.stack([tokenizer(x)[0] for x in test_set.captions.values()]).to(device)
    print(text)
    mask = torch.stack([tokenizer(x)[1] for x in test_set.captions.values()])
    mask = mask.repeat(1,len(mask[0])).reshape(len(mask),len(mask[0]),len(mask[0])).to(device)
    print(mask)

    class_names = ["a drawing of a crab", "a drawing of a crocodile", "a drawing of a lion", "a drawing of a lobster",
              "a drawing of a monkey", "a drawing of a octopus", "a drawing of a panda", "a drawing of a swan"]

    model = CLIP(emb_dim, vit_width, img_size, patch_size, n_channels, vit_layers, vit_heads, vocab_size, text_width, max_seq_length, text_heads, text_layers).to(device)
    model.load_state_dict(torch.load("clip2.pt", map_location=device))

    img = test_set[idx]["image"]
    print(img)
    plt.imshow(img[0]  ,cmap="gray")
    plt.title("Ground Truth: "+ tokenizer(test_set[idx]["caption"], encode=False, mask=test_set[idx]["mask"][0])[0])
    plt.show()
    img = img.to(device)
    print(img.shape)
    img = img.unsqueeze(0)
    with torch.no_grad():
      image_features = model.image_encoder(img)
      text_features = model.text_encoder(text, mask=mask)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    probs = similarity
    guesses = []

    for i in range(probs.size(1)):
        guesses.append((probs[0, i].item()))

    for i in range(len(guesses)):
        print(f"{guesses[i] * 100:.4f}% likelihood of being {class_names[i]}")

    best = guesses.index(max(guesses))
    print(f"I think this is {class_names[best]}. I am {guesses[best] * 100:.4f}% sure!")

    input("Go again? y/n ")


