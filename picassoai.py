# Created by Deji

from PIL import Image
import numpy as np
import os as os
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import time


def loadDatasetOne():
    file_dir = './img'
    files = os.listdir(file_dir)
    dataset = []
    cnt = 0
    for f in files:
        im = Image.open(file_dir + '/' + f)
        im = im.resize((512, 512))
        if np.array(im).shape == (512, 512, 3):
            dataset.append(np.array(im).flatten())
            cnt = cnt + 1

    dataset = np.vstack(dataset)
    train_data = dataset[:71]
    test_data = dataset[71:]
    print(dataset.shape)
    print(test_data.shape)
    print(train_data.shape)
    print(dataset)
    train_x = train_data[:, : int(.75 * 786432)]
    train_y = train_data[:, int(.75 * 786432):]
    test_x = test_data[:, : int(.75 * 786432)]
    return train_x, train_y, test_x


def loadDatasetThree():
    file_dir = './img'
    files = os.listdir(file_dir)
    dataset = []
    cnt = 0
    for f in files:
        im = Image.open(file_dir + '/' + f)
        im = im.resize((512, 512))
        if np.array(im).shape == (512, 512, 3):
            print(np.array(im))
            dataset.append(np.array(im).flatten())
            cnt = cnt + 1

    dataset = np.vstack(dataset)
    np.random.shuffle(dataset)
    train_data = np.transpose(dataset[:48])
    test_data = np.transpose(dataset[48:])
    train_x = train_data[:, :-1]
    train_y = train_data[:, -1]
    test_x = test_data
    return train_x, train_y, test_x


def train_model():
    global train_x, train_y, test_x
    gbr = GradientBoostingRegressor()
    cv_score = cross_val_score(gbr, train_x, train_y).mean()
    print(cv_score)
    nn = MLPRegressor()
    cv_score = cross_val_score(nn, train_x, train_y).mean()
    print(cv_score)
    rft = RandomForestRegressor()
    cv_score = cross_val_score(rft, train_x, train_y).mean()
    print(cv_score)


def prediction():
    global train_x, train_y, test_x
    gbr = GradientBoostingRegressor()
    gbr.fit(train_x, train_y)
    print('Accuracy of gbr, on the training set: ' + str(gbr.score(train_x, train_y)))
    start_time = time.time()
    predictions = gbr.predict(test_x)
    predict_time = time.time() - start_time
    print('Prediction time for gbr is ' + str(predict_time) + '\n')
    predictions = predictions.astype('uint8')
    print(predictions)
    return predictions


def generate_img(predictions):
    img_array = predictions.reshape((512, 512, 3))
    print(img_array)
    img = Image.fromarray(img_array)
    img.save("./generated_imgs/picassoai_" + str(np.random.randint(100)) + ".jpg")


def main():
    global train_x, train_y, test_x
    train_x, train_y, test_x = loadDatasetThree()
    # train_model()
    img_data = prediction()
    generate_img(img_data)


if __name__ == '__main__':
    main()
