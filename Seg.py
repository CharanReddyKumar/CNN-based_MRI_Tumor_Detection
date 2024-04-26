
# %%
# To check folder structure
import os

def list_directory_contents(directory, indent=0):
    # List all files and directories in the provided directory
    entries = os.listdir(directory)
    prefix = " " * indent  # Creates a prefix with spaces based on the indentation level

    for entry in entries:
        # Full path of the current entry
        path = os.path.join(directory, entry)
        
        if os.path.isdir(path):
            print(f"{prefix}{entry}/ (directory)")
            # Recursively list sub-directories with increased indentation
            list_directory_contents(path, indent + 4)
        else:
            print(f"{prefix}{entry} (file)")

# Get the current directory
current_directory = os.getcwd()

# List all files and directories in the current directory with initial indentation
list_directory_contents(current_directory)

# %%
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.filters import gaussian

def preprocess_image(image_array, sigma=0.5, normalize=True):
    """Apply Gaussian blur and normalize the image."""
    if sigma > 0:
        image_array = gaussian(image_array, sigma=sigma, )
    if normalize:
        image_array = image_array / 255.0
    return image_array

def load_preprocess_data(root_dir, image_size=(256, 256)):
    images, masks, ids = [], [], []
    for subdir, dirs, files in os.walk(root_dir):
        for file in sorted(files):
            if file.endswith('.tif') and not file.endswith('_mask.tif'):
                # Correct extraction of the ID
                id = '_'.join(file.split('_')[:3])
                image_path = os.path.join(subdir, file)
                mask_path = image_path.replace('.tif', '_mask.tif')

                # Load images and masks
                image = load_img(image_path, color_mode="grayscale", target_size=image_size)
                mask = load_img(mask_path, color_mode="grayscale", target_size=image_size)
                image_array = img_to_array(image)
                mask_array = img_to_array(mask)

                # Preprocess images and masks
                image_preprocessed = preprocess_image(image_array, sigma=0.5, normalize=True)
                mask_preprocessed = preprocess_image(mask_array, sigma=0, normalize=True)  # No blurring for masks

                images.append(image_preprocessed)
                masks.append(mask_preprocessed)
                ids.append(id)

    return np.array(images), np.array(masks), ids

# Example of loading and preprocessing data
root_dir = 'Data/lgg-mri-segmentation/kaggle_3m/'
images, masks, ids = load_preprocess_data(root_dir)

#%%
from sklearn.model_selection import train_test_split

# Split data into train and test sets
train_images, test_images, train_masks, test_masks = train_test_split(
    images, masks, test_size=0.2, random_state=42)

# Optionally split train set further to create a validation set
train_images, val_images, train_masks, val_masks = train_test_split(
    train_images, train_masks, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2



# %%
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization, Activation

def unet_model(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    # Down-sampling path
    c1 = Conv2D(16, (3, 3), padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(0.1)(p1)
    
    c2 = Conv2D(32, (3, 3), padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(0.1)(p2)

    # Bottleneck
    bn = Conv2D(64, (3, 3), padding='same')(p2)
    bn = BatchNormalization()(bn)
    bn = Activation('relu')(bn)
    bn = Dropout(0.2)(bn)

    # Up-sampling path
    u1 = UpSampling2D((2, 2))(bn)
    u1 = concatenate([u1, c2])
    c3 = Conv2D(32, (3, 3), padding='same')(u1)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)

    u2 = UpSampling2D((2, 2))(c3)
    u2 = concatenate([u2, c1])
    c4 = Conv2D(16, (3, 3), padding='same')(u2)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c4)
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#%%
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Define callbacks
checkpoint = ModelCheckpoint('best_model.h5', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(patience=2, verbose=1)

# Train the model
history = model.fit(
    train_images, train_masks,
    validation_data=(val_images, val_masks),
    batch_size=32,
    epochs=10,
    callbacks=[checkpoint, early_stopping]
)

#%%

# After training, predict masks for the test images
predicted_masks = model.predict(test_images)

# Function to plot sample images, their true masks, and predictions
def plot_samples(images, true_masks, predicted_masks, n=3):
    plt.figure(figsize=(12, 8))
    for i in range(n):
        plt.subplot(3, n, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title('Image')
        plt.axis('off')
        
        plt.subplot(3, n, i+1+n)
        plt.imshow(true_masks[i].squeeze(), cmap='gray')
        plt.title('True Mask')
        plt.axis('off')
        
        plt.subplot(3, n, i+1+2*n)
        plt.imshow(predicted_masks[i].squeeze(), cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
    plt.show()

# Use the function to display the results
plot_samples(test_images, test_masks, predicted_masks)

#%%
import matplotlib.pyplot as plt

# Extract values from the history object
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']  
val_acc_values = history_dict['val_accuracy']  
epochs = range(1, len(loss_values) + 1)

# Plotting both loss and accuracy on the same graph
plt.figure(figsize=(10, 5))

# Loss Plot
plt.plot(epochs, loss_values, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss_values, 'ro-', label='Validation Loss')

# Accuracy Plot
plt.plot(epochs, acc_values, 'go-', label='Training Accuracy')
plt.plot(epochs, val_acc_values, 'yo-', label='Validation Accuracy')

plt.title('Training and Validation Loss/Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='center right')

plt.show()


#%%





# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_images, test_masks, verbose=1)

# Print the model's accuracy
print(f"Test accuracy: {test_accuracy}")


#%%
#%%
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization, Activation


# Generate the plot
plot_model(model, to_file='model_diagram.png', show_shapes=True, show_layer_names=True)

# This will save the diagram as 'model_diagram.png' and you can view this file to see the model architecture.

# %%


# %%
