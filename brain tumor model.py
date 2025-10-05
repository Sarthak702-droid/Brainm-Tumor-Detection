import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

IMAGE_SIZE = (150, 150) # VGG16 prefers 224x224, but we'll use 150x150 for compatibility with your data setup
BATCH_SIZE = 32
NUM_CLASSES = 4 # 'glioma', 'meningioma', 'notumor', 'pituitary'

# --- 2. Data Generators ---

# Training Generator with Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Testing Generator (Rescaling only)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/brain_tumor_detection/Training',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/brain_tumor_detection/Training',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    '/content/drive/MyDrive/brain_tumor_detection/Testing',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# --- 3. Model Definition: Transfer Learning with VGG16 ---

# 1. Define the input tensor shape
input_shape = IMAGE_SIZE + (3,)
input_tensor = Input(shape=input_shape)

# 2. Load the VGG16 Base Model
# weights='imagenet': loads the pre-trained weights
# include_top=False: excludes the ImageNet 1000-class classifier at the end
vgg_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_tensor=input_tensor
)

# 3. Freeze the VGG16 convolutional layers
# This prevents their pre-trained weights from being updated during training.
vgg_base.trainable = False

# 4. Build the new Classification Head
x = Flatten(name='flatten')(vgg_base.output)

# Classification Block (Your custom layers)
x = Dense(512, activation='relu', name='fc1')(x)
x = Dropout(0.5, name='dropout1')(x)
output_tensor = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x) # Output Layer for 4 classes

# Create the final model
model = Model(inputs=vgg_base.input, outputs=output_tensor)

# Display the model structure
print("\n--- VGG16 Transfer Learning Model Summary ---")
model.summary()

# --- 4. Compile and Train ---
# Note: Adam optimizer works well, but consider a smaller learning rate
# if you later decide to unfreeze and fine-tune some VGG16 layers.

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model (25 epochs is a starting point, but you might need fewer)
print("\n--- Starting Training (VGG16 Feature Extractor) ---")
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator
)

# --- 5. Evaluate ---
print("\n--- Model Evaluation on Test Data ---")
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# --- 6. Save the Model ---
MODEL_SAVE_PATH = 'brain_tumor_vgg16_transfer_model.h5'
model.save(MODEL_SAVE_PATH)
print(f"\nModel successfully saved as: {MODEL_SAVE_PATH}")
