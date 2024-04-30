
# %%
# import system libs
import os
import time
import random
import pathlib
import itertools
from glob import glob
from tqdm import tqdm_notebook, tnrange

# import data handling tools
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import label
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow, concatenate_images

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate

# %%
# function to create dataframe
def create_df(data_dir):
    images_paths = []
    masks_paths = glob(f'{data_dir}/*/*_mask*')

    for i in masks_paths:
        images_paths.append(i.replace('_mask', ''))

    df = pd.DataFrame(data= {'images_paths': images_paths, 'masks_paths': masks_paths})

    return df

#%%
data_dir = 'Data/lgg-mri-segmentation/kaggle_3m'
df = create_df(data_dir)
#%%


train_df, dummy_df = train_test_split(df, train_size= 0.8)


valid_df, test_df = train_test_split(dummy_df, train_size= 0.5)



#%%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(dataframe, augmentation_params):
    # Image and mask sizes
    target_size = (256, 256)
    # Batch size for generator
    batch_size = 40

    # Creating a common ImageDataGenerator for both images and masks
    generator = ImageDataGenerator(**augmentation_params)

    # Setting up image and mask generators
    image_generator = generator.flow_from_dataframe(
        dataframe=dataframe,
        x_col='images_paths',
        class_mode=None,
        color_mode='rgb',
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=None,
        save_prefix='image',
        seed=1
    )

    mask_generator = generator.flow_from_dataframe(
        dataframe=dataframe,
        x_col='masks_paths',
        class_mode=None,
        color_mode='grayscale',
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=None,
        save_prefix='mask',
        seed=1
    )

    # Zip the image and mask generators
    combined_generator = zip(image_generator, mask_generator)

    # Generate pairs and apply post-processing
    for (images, masks) in combined_generator:
        # Normalize images
        images = images / 255.0
        # Binarize masks
        masks = masks / 255.0
        masks[masks > 0.5] = 1
        masks[masks <= 0.5] = 0

        yield (images, masks)

#%%
# Define augmentation parameters for the training dataset
train_augmentation_params = {
    'rotation_range': 0.2,
    'width_shift_range': 0.05,
    'height_shift_range': 0.05,
    'shear_range': 0.05,
    'zoom_range': 0.05,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# Create data generators for training, validation, and testing
train_generator = create_generators(train_df, augmentation_params=train_augmentation_params)
valid_generator = create_generators(valid_df, augmentation_params={})
test_generator = create_generators(test_df, augmentation_params={})

# %%
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def show_images(image_paths, mask_paths, num_images=5):
    """
    Displays images with their corresponding masks side by side.
    
    Parameters:
        image_paths (list of str): The paths to the images.
        mask_paths (list of str): The paths to the masks.
        num_images (int): Number of images to display.
    """
    # Set the number of images to display to the minimum of num_images or the length of image_paths
    num_images = min(num_images, len(image_paths))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 3))  # Adjust size as needed
    
    # If only one image, adjust axes array to be indexable
    if num_images == 1:
        axes = [axes]
    
    for i in range(num_images):
        # Load image
        image = Image.open(image_paths[i])
        mask = Image.open(mask_paths[i])
        
        # Display image
        axes[i][0].imshow(image)
        axes[i][0].set_title('Image ' + str(i+1))
        axes[i][0].axis('off')  # Turn off axis labels
        
        # Display mask
        axes[i][1].imshow(mask, cmap='gray')  # Use gray scale for the mask
        axes[i][1].set_title('Mask ' + str(i+1))
        axes[i][1].axis('off')  # Turn off axis labels
    
    plt.tight_layout()
    plt.show()
#%%
show_images(list(train_df['images_paths']), list(train_df['masks_paths']), num_images=10)

#%%
# Count the number of images and masks and verify matching pairs
def count_and_verify_images_masks(data_df, dataset_type):
    num_images = len(data_df['images_paths'])
    num_masks = len(data_df['masks_paths'])
    
    if num_images != num_masks:
        print(f"Warning: Number of images and masks do not match in {dataset_type} dataset!")
    
    mismatch_count = 0
    # Iterate over the DataFrame to check each image-mask pair
    for img_path, mask_path in zip(data_df['images_paths'], data_df['masks_paths']):
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"Missing file: {img_path} or {mask_path}")
            mismatch_count += 1
        else:
            # Optionally add more sophisticated checks here if necessary
            pass
    
    print(f"Number of {dataset_type} images: {num_images}")
    print(f"Number of {dataset_type} masks: {num_masks}")
    print(f"Number of mismatches in {dataset_type}: {mismatch_count}")

# Example usage for training, validation, and test datasets
count_and_verify_images_masks(train_df, "training")
count_and_verify_images_masks(valid_df, "validation")
count_and_verify_images_masks(test_df, "test")


# %%
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

def unetmain(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    # Return the model
    return Model(inputs=[inputs], outputs=[outputs])


#%%\
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

def unet(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    u4 = UpSampling2D((2, 2))(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(u4)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)

    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(c5)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

    return Model(inputs=[inputs], outputs=[outputs])

# Create the model
model = unet()
model.summary()

#%%
# function to create dice coefficient
def dice_coef(y_true, y_pred, smooth=100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)

    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)

# function to create dice loss
def dice_loss(y_true, y_pred, smooth=100):
    return -dice_coef(y_true, y_pred, smooth)

# function to create iou coefficient
def iou_coef(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true * y_pred)
    sum = K.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum - intersection + smooth)
    return iou
#%%
with tf.device('/cpu:0'):
    model = unet()
    model.compile(Adamax(learning_rate= 0.001), loss= dice_loss, metrics= ['accuracy', iou_coef, dice_coef])
model.summary()
# %%
#Model training

epochs = 5
batch_size = 16
callbacks = [ModelCheckpoint('unet.keras', verbose=1, save_best_only=True)]
with tf.device("/cpu:0"):
    history = model.fit(
        train_generator,
        steps_per_epoch=int(len(train_df) / batch_size),  # Convert to integer
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=valid_generator,
        validation_steps=int(len(valid_df) / batch_size)  # Convert to integer
    )
# %%
#Chart
import matplotlib.pyplot as plt

# Create a figure and a single subplot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plotting the accuracy on the left y-axis
color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(history.history['accuracy'], color=color, label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], color=color, linestyle='dashed', label='Validation Accuracy')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

# Create a second y-axis for the loss with shared x-axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Loss', color=color)
ax2.plot(history.history['loss'], color=color, label='Train Loss')
ax2.plot(history.history['val_loss'], color=color, linestyle='dashed', label='Validation Loss')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

# Title and layout adjustments
plt.title('Training & Validation Accuracy and Loss')
fig.tight_layout()

# Show plot
plt.show()

# %%
# Model Evaluation
ts_length = len(test_df)
test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length / n <= 80]))
test_steps = ts_length // test_batch_size

# Evaluate the model on test set
test_score = model.evaluate(test_generator, steps=test_steps, verbose=1)

# Output the results  testing

print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])
print("Test IoU: ", test_score[2])
print("Test Dice: ", test_score[3])
#%%
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Iterate over random samples from the test set
for _ in range(20):
    index = np.random.randint(1, len(test_df.index))
    img = cv2.imread(test_df['images_paths'].iloc[index])
    img = cv2.resize(img, (256, 256))
    img = img/255.0
    img = img[np.newaxis, :, :, : ]

    predicted_mask = model.predict(img)

    plt.figure(figsize=(12, 12))
    
    plt.subplot(1, 3, 1)
    plt.imshow(np.squeeze(img))
    plt.axis('off')
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.imread(test_df['masks_paths'].iloc[index], cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.axis('off')
    plt.title('Original Mask')

    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(predicted_mask) > 0.5, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')
    
    plt.show()
