# Created by Deji
from PIL import Image
import numpy as np
import os as os
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import time
import pandas as pd
import requests
import json
import asyncio

class PicassoAI:
    dataset = ''
    train_x = ''
    train_y = ''
    test_x = ''

    def __init__(self):
        self.train_x, self.train_y, self.test_x, self.dataset = self.loadDataset()

    def loadDataset(self):
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
        np.random.shuffle(dataset)
        df = pd.DataFrame(np.transpose(dataset))
        df.to_csv('./dataset.csv')
        train_data = np.transpose(dataset[:48])
        test_data = np.transpose(dataset[48:])
        train_x = train_data[:, :-1]
        train_y = train_data[:, -1]
        test_x = test_data
        return train_x, train_y, test_x, np.transpose(dataset)

    def train_model(self):
        #  Tried other model such MLP neural network regressor and random forest trees, but GBR performed best
        global train_x, train_y, test_x
        cvscore = []
        range = [4, 5, 6, 7, 8]
        for i in range:
            print(i)
            gbr = GradientBoostingRegressor(max_leaf_nodes=i)
            cv_score = cross_val_score(gbr, train_x, train_y, scoring='neg_mean_squared_error').mean()
            cvscore.append(cv_score)
        print(cvscore)

    def predict_using_local_model(self):
        gbr = GradientBoostingRegressor()
        gbr.fit(self.train_x, self.train_y)
        print('Accuracy of gbr, on the training set: ' + str(gbr.score(train_x, train_y)))
        start_time = time.time()
        predictions = gbr.predict(self.test_x)
        predict_time = time.time() - start_time
        print('Prediction time for gbr is ' + str(predict_time) + '\n')
        predictions = predictions.astype('uint8')
        return predictions

    async def predict_using_AutoML_model(self):
        # output = []
        # for i in range(1, 786432):
        #     test_samples = json.dumps({"data": self.dataset[:i, :94].tolist()})
        #     headers = {'Content-Type':'application/json'}
        #     scoring_uri = 'http://94a794ab-09cc-4689-a89f-1367649d19cb.westus2.azurecontainer.io/score'
        #     resp = requests.post(scoring_uri, test_samples, headers = headers)
        #     y = json.loads(json.loads(resp.text))["result"][0]
        #     output.append(y)
        # return output

        test_samples = json.dumps({"data": self.dataset[:, :94].tolist()})
        headers = {'Content-Type':'application/json'}
        scoring_uri = 'http://94a794ab-09cc-4689-a89f-1367649d19cb.westus2.azurecontainer.io/score'
        resp = await requests.post(scoring_uri, test_samples, headers = headers)
        return resp

    def generate_img(self, predictions):
        img_array = predictions.reshape((512, 512, 3))
        img = Image.fromarray(img_array)
        img.save("./generated_imgs/picassoai_" + str(np.random.randint(100)) + ".jpg")


if __name__ == '__main__':
    ai = PicassoAI()
    loop = asyncio.get_event_loop()
    y = loop.run_until_complete(ai.predict_using_AutoML_model())
    ai.generate_img(y)
