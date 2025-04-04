import pygame, sys
from pygame.locals import *
import numpy as np 
from keras.models import load_model
import cv2

# Constants
WINDOW_SIZE_X = 640
WINDOW_SIZE_Y = 480
BOUNDARY_INC = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BRUSH_SIZE = 4

# Load model and labels
MODEL = load_model("mnist_model.keras")
LABELS = {
    0: "ZERO", 1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR",
    5: "FIVE", 6: "SIX", 7: "SEVEN", 8: "EIGHT", 9: "NINE"
}

# Initialize pygame
pygame.init()
FONT = pygame.font.Font("freesansbold.ttf", 18)
DISPLAY_SURF = pygame.display.set_mode((WINDOW_SIZE_X, WINDOW_SIZE_Y))
pygame.display.set_caption("Digit Recognition Board")

def clear_screen():
    """Clear the drawing surface."""
    DISPLAY_SURF.fill(BLACK)

def predict_digit(img_array):
    """Process and predict digit from drawn image."""
    image = cv2.resize(img_array, (28, 28))
    image = np.pad(image, (10, 10), 'constant', constant_values=0)
    image = cv2.resize(image, (28, 28))/255
    
    prediction = MODEL.predict(image.reshape(1, 28, 28, 1))
    return LABELS[np.argmax(prediction)]

def main():
    is_writing = False
    number_xcord = []
    number_ycord = []
    predict_mode = True
    save_image = False
    image_cnt = 1
    
    clear_screen()
    
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
                
            # Handle drawing
            if event.type == MOUSEMOTION and is_writing:
                xcord, ycord = event.pos
                pygame.draw.circle(DISPLAY_SURF, WHITE, (xcord, ycord), BRUSH_SIZE, 0)
                number_xcord.append(xcord)
                number_ycord.append(ycord)
                
            if event.type == MOUSEBUTTONDOWN:
                is_writing = True
                
            if event.type == MOUSEBUTTONUP and number_xcord and number_ycord:
                is_writing = False
                
                # Calculate bounding rectangle
                min_x = max(min(number_xcord) - BOUNDARY_INC, 0)
                max_x = min(max(number_xcord) + BOUNDARY_INC, WINDOW_SIZE_X)
                min_y = max(min(number_ycord) - BOUNDARY_INC, 0)
                max_y = min(max(number_ycord) + BOUNDARY_INC, WINDOW_SIZE_Y)
                
                if min_x < max_x and min_y < max_y:
                    # Get pixel data from drawing
                    img_arr = np.array(pygame.PixelArray(DISPLAY_SURF))[min_x:max_x, min_y:max_y].T.astype(np.float32)
                    
                    # Save image if enabled
                    if save_image:
                        cv2.imwrite(f"image_{image_cnt}.png", img_arr)
                        image_cnt += 1
                    
                    # Predict digit
                    if predict_mode and img_arr.size > 0:
                        label = predict_digit(img_arr)
                        
                        # Display prediction
                        text_surface = FONT.render(label, True, RED, WHITE)
                        text_rect = text_surface.get_rect()
                        text_rect.left, text_rect.bottom = min_x, max_y
                        DISPLAY_SURF.blit(text_surface, text_rect)
                
                # Reset coordinates
                number_xcord = []
                number_ycord = []
                
            # Handle keyboard commands
            if event.type == KEYDOWN:
                if event.unicode == "n":
                    clear_screen()
                elif event.unicode == "s":
                    save_image = not save_image
                elif event.unicode == "p":
                    predict_mode = not predict_mode
                    
        pygame.display.update()

if __name__ == "__main__":
    main()