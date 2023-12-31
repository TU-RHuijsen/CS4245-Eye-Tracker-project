from keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D, concatenate
from tensorflow.keras import mixed_precision
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf
import numpy as np
import cv2
import os
import time
from matplotlib import pyplot as plt

# Define input and output directories.
project_dir = './'
input_dir = project_dir + 'visual_test_train/'
test_dir = project_dir + 'visual_test_test/'
model_dir = project_dir + 'model/'
# Retrieve pixel width and height of the monitor.
width = 1920
height = 1080
#...
screen_width_cm = 38.3 #cm
screen_height_cm = 21.5 #cm

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
mixed_precision.set_global_policy('float32')

# activation functions
activation = 'relu'
last_activation = 'linear'

def euclidean_distance_truncated(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def euclidean_distance_pixels(y_true_scaled, y_pred_scaled):
    pixel_scaler = tf.constant([width, height], dtype=tf.float32)
    y_true = y_true_scaled*pixel_scaler
    y_pred = y_pred_scaled*pixel_scaler
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def euclidean_distance_cm(y_true_scaled, y_pred_scaled):
    cm_scaler = tf.constant([screen_width_cm, screen_height_cm], dtype=tf.float32)
    y_true = y_true_scaled*cm_scaler
    y_pred = y_pred_scaled*cm_scaler
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def get_eye_model(img_ch, img_cols, img_rows):

    eye_img_input = Input(shape=(img_cols, img_rows, img_ch))

    h = Conv2D(96, (11, 11), activation=activation)(eye_img_input)
    h = MaxPool2D(pool_size=(2, 2))(h)
    h = Conv2D(256, (5, 5), activation=activation)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    h = Conv2D(384, (3, 3), activation=activation)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    out = Conv2D(64, (1, 1), activation=activation)(h)

    model = Model(inputs=eye_img_input, outputs=out)

    return model

def get_face_model(img_ch, img_cols, img_rows):

    face_img_input = Input(shape=(img_cols, img_rows, img_ch))

    h = Conv2D(96, (11, 11), activation=activation)(face_img_input)
    h = MaxPool2D(pool_size=(2, 2))(h)
    h = Conv2D(256, (5, 5), activation=activation)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    h = Conv2D(384, (3, 3), activation=activation)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    out = Conv2D(64, (1, 1), activation=activation)(h)

    model = Model(inputs=face_img_input, outputs=out)

    return model

def get_eye_tracker_model(img_ch, img_cols, img_rows):

    # get partial models
    eye_net = get_eye_model(img_ch, img_cols, img_rows)
    face_net_part = get_face_model(img_ch, img_cols, img_rows)

    # right eye model
    right_eye_input = Input(shape=(img_cols, img_rows, img_ch))
    right_eye_net = eye_net(right_eye_input)

    # left eye model
    left_eye_input = Input(shape=(img_cols, img_rows, img_ch))
    left_eye_net = eye_net(left_eye_input)
    
    # face model
    face_input = Input(shape=(img_cols, img_rows, img_ch))
    face_net = face_net_part(face_input)

    # face grid
    face_grid = Input(shape=(25, 25, 1))

    # dense layers for eyes
    e = concatenate([left_eye_net, right_eye_net])
    e = Flatten()(e)
    fc_e1 = Dense(128, activation=activation)(e)

    # dense layers for face
    f = Flatten()(face_net)
    fc_f1 = Dense(128, activation=activation)(f)
    fc_f2 = Dense(64, activation=activation)(fc_f1)

    # dense layers for face grid
    fg = Flatten()(face_grid)
    fc_fg1 = Dense(256, activation=activation)(fg)
    fc_fg2 = Dense(128, activation=activation)(fc_fg1)

    # final dense layers
    h = concatenate([fc_e1, fc_f2, fc_fg2])
    fc1 = Dense(128, activation=activation)(h)
    fc2 = Dense(2, activation=last_activation)(fc1)

    # final model
    final_model = Model(
        inputs=[right_eye_input, left_eye_input, face_input, face_grid],
        outputs=[fc2])

    return final_model
    
# get the dataset
def get_dataset(input_dir):
    # Retrieve all the folders in the input directory
    train_ids = [name for name in os.listdir(input_dir) if os.path.isdir(input_dir + name)]
    print(str(len(train_ids)) + " images in folder!")
    
    X_face = []
    X_grid = []
    X_left_eye = []
    X_right_eye = []
    y = []
    
    for folder in train_ids:
        # If the sampled folder has only four files:
        if (len([name for name in os.listdir(input_dir + folder)]) == 4):
            
            # Retrieve the face, face grid and eye images.
            input_face = cv2.imread(input_dir + folder + "/face.png")
            input_grid = cv2.imread(input_dir + folder + "/grid.png", 0)
            input_left_eye = cv2.imread(input_dir + folder + "/left_eye.png")
            input_right_eye = cv2.imread(input_dir + folder + "/right_eye.png")
            
            # If these are the first respective images, turn them into an array, and normalize them.
            if (len(X_face) == 0):
                X_face = np.array([tf.keras.utils.img_to_array(input_face)])/255
                X_grid = np.array([tf.keras.utils.img_to_array(input_grid)])/255
                X_left_eye = np.array([tf.keras.utils.img_to_array(input_left_eye)])/255
                X_right_eye = np.array([tf.keras.utils.img_to_array(input_right_eye)])/255
            
            # For all the other ones: stack previous array on top of the new normalized entry.
            else:
                X_face = np.vstack((X_face, np.array([tf.keras.utils.img_to_array(input_face)])/255))
                X_grid = np.vstack((X_grid, np.array([tf.keras.utils.img_to_array(input_grid)])/255))
                X_left_eye = np.vstack((X_left_eye, np.array([tf.keras.utils.img_to_array(input_left_eye)])/255))
                X_right_eye = np.vstack((X_right_eye, np.array([tf.keras.utils.img_to_array(input_right_eye)])/255))
            
            # From the folder, retrieve the screen position.
            y_data = list(map(int, (folder.split('_')[0:2])))
            
            # Add the screen position as a scaled position between 0 and 1 for both x & y.
            if (len(y) == 0):
                y = [y_data[0]/width, y_data[1]/height]
            else:
                y = np.vstack((y, [y_data[0]/width, y_data[1]/height]))
    
    # Return all the input and output arrays.
    return X_face, X_grid, X_left_eye, X_right_eye, y

# Retrieve data arrays from input directory.
X_face, X_grid, X_left_eye, X_right_eye, y = get_dataset(input_dir)
# Get the eye tracker model for the given image size.
model = get_eye_tracker_model(3, 64, 64)
# Print a summary of the model.
model.summary()
# optimizer
sgd = SGD(learning_rate=1e-3, decay=5e-4, momentum=9e-1, nesterov=True)
# compile model
model.compile(optimizer=sgd, loss=euclidean_distance_truncated, metrics=[euclidean_distance_cm])

# Only save the model with the best loss on the validation set.
model_name = ".model_" + time.strftime("%Y%m%d-%H%M%S") + ".hdf5"
mcp_save = ModelCheckpoint(model_name, save_best_only=True, monitor='loss', mode='min')

# Fit the model to the data
history = model.fit([X_right_eye, X_left_eye, X_face, X_grid], y, validation_split = 0, batch_size=64, epochs=200, callbacks=[mcp_save])

# Create a plot for the loss and validation loss.
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('Euclidean distance loss - 1 person(s)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# min_loss = min(history.history['val_loss'])
# print("\nMinimum validation loss: ", min_loss)
# epoch_idx = history.history['val_loss'].index(min_loss)
# epoch = epoch_idx + 1
# print("Minimum validation loss @ epoch: ", epoch)
# print("Corresponding validation cm difference: ", history.history['val_euclidean_distance_cm'][epoch_idx])

# print("\nCorresponding training loss: ", history.history['loss'][epoch_idx])
# print("Final training loss: ", history.history['loss'][-1])

min_loss = min(history.history['loss'])
print("\nMinimum validation loss: ", min_loss)
epoch_idx = history.history['loss'].index(min_loss)
epoch = epoch_idx + 1
print("Minimum validation loss @ epoch: ", epoch)
# print("Corresponding validation cm difference: ", history.history['val_euclidean_distance_cm'][epoch_idx])

# print("\nCorresponding training loss: ", history.history['loss'][epoch_idx])
print("Final training loss: ", history.history['loss'][-1])

X_face_test_1, X_grid_test_1, X_left_eye_test_1, X_right_eye_test_1, y_test_1 = get_dataset(input_dir)
predictions_1 = model.predict([X_right_eye_test_1, X_left_eye_test_1, X_face_test_1, X_grid_test_1])

y_test_cm_1 = y_test_1.copy()
y_test_cm_1[:, 0] = y_test_cm_1[:, 0]*screen_width_cm
y_test_cm_1[:, 1] = y_test_cm_1[:, 1]*screen_height_cm
cm_diffs_1 = euclidean_distance_cm(y_test_1, predictions_1)
average_cm_diff_1 = K.mean(cm_diffs_1)
print("Average cm loss on the input images: ", average_cm_diff_1)

X_face_test_2, X_grid_test_2, X_left_eye_test_2, X_right_eye_test_2, y_test_2 = get_dataset(test_dir)
predictions_2 = model.predict([X_right_eye_test_2, X_left_eye_test_2, X_face_test_2, X_grid_test_2])

y_test_cm_2 = y_test_2.copy()
y_test_cm_2[:, 0] = y_test_cm_2[:, 0]*screen_width_cm
y_test_cm_2[:, 1] = y_test_cm_2[:, 1]*screen_height_cm
cm_diffs_2 = euclidean_distance_cm(y_test_2, predictions_2)
average_cm_diff_2 = K.mean(cm_diffs_2)
print("Average cm loss on the test images: ", average_cm_diff_2)