from keras.layers import Dense, GlobalAveragePooling2D, Dropout
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.applications import MobileNetV2
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import itertools

IMG_W = 128
IMG_H = 128
BATCH_SIZE = 3
EPOCHS = 200

base_model = MobileNetV2(weights='imagenet', input_shape=(IMG_W,IMG_H,3), include_top=False)

np.set_printoptions(formatter={'float_kind':'{:f}'.format})

def prepare_image(file):
    img = image.load_img(file, target_size=(IMG_W, IMG_H))
    img_array = image.img_to_array(img)
    img_final = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_final)

def test_model(model, dataset, path="moles/HAM10000"):
    results = np.zeros((2,2))
    for i in range(len(dataset)):
        img = image.load_img(path+'/'+dataset['name'][i]+'.jpg', target_size=(IMG_W, IMG_H))
        img_array = image.img_to_array(img)
        img_final = np.expand_dims(img_array, axis=0)
        image_full = preprocess_input(img_final)
        prediction = np.round(model.predict(image_full)[0][0])
        y = 1
        if dataset['meta.clinical.benign_malignant'][i] == 'benign':
            y = 0
        if prediction == 0:
            if y == prediction:
                results[0][0]+=1
            else:
                results[1][0]+=1
        else:
            if y == prediction:
                results[1][1]+=1
            else:
                results[0][1]+=1
    return results
        
out_ten = base_model.output
out_ten = GlobalAveragePooling2D()(out_ten)
out_ten = Dense(256, activation='relu')(out_ten)
out_ten = Dropout(0.5)(out_ten)
out_ten = Dense(128, activation='relu')(out_ten)
out_ten = Dropout(0.25)(out_ten)
out_ten = Dense(64, activation='relu')(out_ten)
out_ten = Dense(1, activation='sigmoid')(out_ten)

model = Model(inputs=base_model.input, outputs=out_ten)

for layer in base_model.layers:
    layer.trainable = False

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


training_set = train_datagen.flow_from_directory(directory='moles/trainf',
                                                 target_size=(IMG_W, IMG_H),
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=True,
                                                 class_mode='binary',
                                                 color_mode='rgb')

test_set = train_datagen.flow_from_directory(directory='moles/testf',
                                                 target_size=(IMG_W, IMG_H),
                                                 batch_size=1,
                                                 shuffle=False,
                                                 class_mode='binary',
                                                 color_mode='rgb')

optimizer = optimizers.Adam(lr=0.000015)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_acc:.2f}.hdf5', 
                             verbose=1,
                             monitor='val_acc',
                             save_best_only=True, mode='max')
callbacks = [checkpoint]

model.fit_generator(generator=training_set, 
                    steps_per_epoch=training_set.n//training_set.batch_size,
                    epochs=EPOCHS,
                    callbacks=callbacks,
                    validation_data=test_set,
                    validation_steps=test_set.n//test_set.batch_size)

model.load_weights('80_train.h5')
predict_set = model.predict_generator(test_set, 800)
fpr, tpr, thresholds = roc_curve(test_set.classes, predict_set)
cm = confusion_matrix(test_set.classes, np.round(predict_set))

def plot_roc(fpr, tpr):
    auc_value = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC (area = {:.3f})'.format(auc_value))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def test_image(model, path):
    test_image = prepare_image(path)
    predictions = model.predict(test_image)
    results = ['{:f}'.format(item) for item in predictions[0]]
    return results

model.save("model_full.h5")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save("model.h5")

