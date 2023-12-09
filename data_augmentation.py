import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# Load your data and labels
#file_data = r'C:\Users\Ahmed\OneDrive - University of Florida\Desktop\FML\data_train.npy'
#file_labels = r'C:\Users\Ahmed\OneDrive - University of Florida\Desktop\FML\labels_train.npy'

data = np.load('data/data_train.npy')
data1 = data.T  # Transpose data if needed

labels = np.load('data/labels_train.npy')
labels = np.array(labels)

# Create an ImageDataGenerator with specified augmentation parameters
datagen = ImageDataGenerator(
    #vertical_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    #horizontal_flip=True,
    shear_range=0.2,
    #brightness_range=(0.5, 1.5)
    
)

# Example: Augment entire dataset
augmented_data = []
augmented_labels = []

# Define target size for resizing
target_size = (300, 300)
data = data1.reshape(8443,300,300,3)

for i in range(len(data)):
    current_image = data[i]
    current_label = labels[i]
    
    #data = np.delete(data, 0, axis=0)
    #print(current_image.shape)
    
    augmented_data.append(current_image)
    augmented_labels.append(current_label)

    current_image = np.expand_dims(current_image, axis=0)

    augmented_images = datagen.flow(current_image, batch_size=1)
    #augmented_images = np.array(augmented_images)
   # print(i)

    for _ in range(3):  
        augmented_image = augmented_images.next()[0]
        augmented_image = augmented_image.astype('uint8')
        
        augmented_image=augmented_image.reshape(300,300,3)

        
        
        
        augmented_data.append(augmented_image)
        augmented_labels.append(current_label)

# Convert augmented_data and augmented_labels to numpy arrays
augmented_data = np.array(augmented_data)
augmented_labels = np.array(augmented_labels)
data=[]
data1=[]
print(augmented_data.shape)
