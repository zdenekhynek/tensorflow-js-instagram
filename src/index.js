// Based on https://towardsdatascience.com/diabetes-prediction-using-logistic-regression-with-tensorflow-js-35371e47c49d

import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import * as Papa from "papaparse";
import _ from "lodash";

const features = ["profile pic", "nums/length username", "fullname words", "nums/length fullname",
  "name==username", "description length", "external URL", "private", "#posts", "#followers", "#follows"];

Papa.parsePromise = function (file) {
  return new Promise(function (complete, error) {
    Papa.parse(file, {
      header: true,
      download: true,
      dynamicTyping: true,
      complete,
      error
    });
  });
};

const oneHot = outcome => Array.from(tf.oneHot(outcome, 2).dataSync());


export const trainLogisticRegression = async (featureCount, trainDs, validDs) => {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 2,
      activation: "softmax",
      inputShape: [featureCount]
    })
  );
  const optimizer = tf.train.adam(0.005);
  model.compile({
    optimizer: optimizer,
    loss: "binaryCrossentropy",
    metrics: ["accuracy"]
  });

  const trainLogs = [];
  const lossContainer = document.getElementById("loss-cont");
  const accContainer = document.getElementById("acc-cont");
  console.log("Training...");

  await model.fitDataset(trainDs, {
    epochs: 100,
    validationData: validDs,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        trainLogs.push(logs);
        tfvis.show.history(lossContainer, trainLogs, ["loss", "val_loss"]);
        tfvis.show.history(accContainer, trainLogs, ["acc", "val_acc"]);
      }
    }
  });

  return model;
};

export async function trainComplexModel(featureCount, trainDs, validDs) {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 12,
      activation: "relu",
      inputShape: [featureCount]
    })
  );
  model.add(
    tf.layers.dense({
      units: 2,
      activation: "softmax"
    })
  );
  const optimizer = tf.train.adam(0.0001);
  model.compile({
    optimizer: optimizer,
    loss: "binaryCrossentropy",
    metrics: ["accuracy"]
  });
  const trainLogs = [];
  const lossContainer = document.getElementById("loss-cont");
  const accContainer = document.getElementById("acc-cont");
  
  console.log("Training...");
  await model.fitDataset(trainDs, {
    epochs: 25,
    validationData: validDs,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        trainLogs.push(logs);
        tfvis.show.history(lossContainer, trainLogs, ["loss", "val_loss"]);
        tfvis.show.history(accContainer, trainLogs, ["acc", "val_acc"]);
      }
    }
  });

  return model;
};

export async function prepareData(url) {
  const csv = await Papa.parsePromise(url);
  return csv.data;
};

export async function loadPretrainedModel() {
  console.log(tf);
  const model = await tf.loadLayersModel('./pretrained-model.json');

  const optimizer = tf.train.adam(0.001);
  model.compile({
    optimizer: optimizer,
    loss: "binaryCrossentropy",
    metrics: ["accuracy"]
  });

  return model;
}

const createFeatures = (data, features) => {
  return data.map(row =>
    features.map(feature => {
      const val = row[feature];
      return val === undefined ? 0 : val;
    })
  );
};

const createDataSets = (data, features, testSize, batchSize) => {
  const X = data.map(r =>
    features.map(f => {
      const val = r[f];
      return val === undefined ? 0 : val;
    })
  );
  const y = data.map(r => {
    const fake = r.fake === undefined ? 0 : r.fake;
    return oneHot(fake);
  });

  const splitIdx = parseInt((1 - testSize) * data.length, 10);
  console.log('X', X, 'Y', y);

  const ds = tf.data
    .zip({ xs: tf.data.array(X), ys: tf.data.array(y) })
    .shuffle(data.length, 42);

  return [
    ds.take(splitIdx).batch(batchSize),
    ds.skip(splitIdx + 1).batch(batchSize),
    tf.tensor(X.slice(splitIdx)),
    tf.tensor(y.slice(splitIdx))
  ];
};


const run = async () => {
  //  1. LOAD DATA
  const data = await prepareData('./instagram_train.csv');
  console.log('data', data);

  //  2. PREPARE DATA

  //  3. CREATE MODEL

  //  4. TRAIN MODEL

  //  5. VISUALISE RESULTS
  
  // const features = ["profile pic", "nums/length username", "fullname words", "nums/length fullname",
  //   "name==username", "description length", "external URL", "private", "#posts", "#followers", "#follows"];

  const [trainDs, validDs, xTest, yTest] = createDataSets(
    data,
    features,
    0.25,
    16
  );

  const model = await trainLogisticRegression(
    features.length,
    trainDs,
    validDs
  );

  const preds = model.predict(xTest).argMax(-1);
  const labels = yTest.argMax(-1);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);

  const container = document.getElementById("confusion-matrix");

  tfvis.render.confusionMatrix(container, {
    values: confusionMatrix,
    tickLabels: ["Fake", "Not-Fake"]
  });

  await model.save('downloads://pretrained-model');
};

if (document.readyState !== "loading") {
  run();
} else {
  document.addEventListener("DOMContentLoaded", run);
}
