import pygame
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import ImageGrab, Image
import time
import numpy as np


pygame.init()
screen = pygame.display.set_mode((720, 720))
running = True
screen.fill("black")
clock = pygame.time.Clock()

def guess_doodle():
    # Take a 720x720 screenshot at specific coordinates
    screenshot = ImageGrab.grab(bbox=(740, 260, 1820, 1340))
    resize = screenshot.resize((28, 28), Image.Resampling.LANCZOS)
    gray = resize.convert('L')
    save_path = r"C:\Users\allan\PycharmProjects\MyOwnDoodleGuesser\screenshot\screenshot.png"
    gray.save(save_path)
    image_matrix = np.array(gray)
    # Display the screenshot (optional)
    #resize.show()
    return gray



while running:
    mouse_buttons = pygame.mouse.get_pressed()
    (x, y) = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE or event.key == pygame.K_BACKSPACE:
                screen.fill("black")
            elif event.key == pygame.K_RETURN:
                image_object = guess_doodle()
                time.sleep(1)
                running = False


    if mouse_buttons[0] == 1:
        pygame.draw.circle(screen, "white", (x, y), 5)

    clock.tick(240)
    pygame.display.flip()

pygame.quit()

labels = ["a drawing of a crab", "a drawing of a crocodile", "a drawing of a lion", "a drawing of a lobster",
          "a drawing of a monkey", "a drawing of a octopus", "a drawing of a panda", "a drawing of a swan"]
#crab, monkey

#Load model and processor from Hugging Face's transformers library
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

inputs = processor(text=labels, images=image_object, return_tensors="pt", padding=True).to(device)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

guesses = []

for i in range(probs.size(1)):
    guesses.append((probs[0, i].item()))

for i in range(len(guesses)):
    print(f"{guesses[i] * 100:.4f}% likelihood of being {labels[i]}")

best = guesses.index(max(guesses))
print(f"I think this is {labels[best]}. I am {guesses[best] * 100:.4f}% sure!")
