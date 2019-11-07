import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import { loadCsv, oneHot } from "./utils.js";

// container for visualisations
const lossContainerEl = document.getElementById("loss-cont");
const accContainerEl = document.getElementById("acc-cont");
const matrixContainerEl = document.getElementById("confusion-matrix");

/**
 * 1. Load data from the train CSV file
 * 
 * Returns:
 *  an array of objects with the following structure
 *  e.g. [{#followers: 1000, #follows: 955, #posts: 32, ...}, ...]
 */
export async function loadData() {
  console.log("1. Loading data ...");
  const data = await loadCsv('./instagram_train.csv');
  return data;
} 

/**
 * 2. Prepare data 
 * Takes:
 * 
 * Returns:
 *  X - an array of arrays [[],[], ...]
 *  y - an array of labels
 */
export async function prepareData(data, features) {
  console.log("2. Preparing data ...");
  const X = data.map((row) => {
    return features.map((feature) => {
      const value = row[feature];
      return value !== undefined ? value : 0;
    });
  });

  const y = data.map((row) => {
    const fake = row.fake !== undefined ? row.fake : 0;
    return oneHot(fake);
  });

  return [X, y];
}

/**
 * 3. Validation split
 * 
 */
export function splitData(X, y, validationSplit, batchSize) {
  console.log("3. Splitting data ...");
  //  convert to tensor
  const ds = tf.data
    .zip({ xs: tf.data.array(X), ys: tf.data.array(y) })
    .shuffle(X.length, 42);

  //  split into validation set 
  const splitIdx = X.length * validationSplit;
  console.log('X', X, 'Y', y);

  const trainDs = ds.take(splitIdx).batch(batchSize);
  const validDs = ds.skip(splitIdx + 1).batch(batchSize);

  return [trainDs, validDs];
}

/**
 * 4. Define tensorflow sequential model with:
 *  - one output layer with 2 neurons (we're predicting probabilities of two outcomes)
 *  - with input shape of the number of features
 *  - softmax activation layer (because it's a classification problem)
 * 
 * Returns:
 *    tensorflow model
 */
export function getTfModel(features) {
  console.log("4. Getting tf model ...");
  const model = tf.sequential();
  
  model.add(
    tf.layers.dense({
      units: 2,
      activation: "softmax",
      inputShape: [features.length]
    })
  );

  return model;
}

/**
 * 5. Compile model
 *  - using adam optimizer, binaryCrossentropy and accuracy
 */
export function compileTfModel(model) {
  console.log("5. Compiling tf model ...");
  const optimizer = tf.train.adam(0.005);
  
  model.compile({
    optimizer: optimizer,
    loss: "binaryCrossentropy",
    metrics: ["accuracy"]
  });
}

/**
 * 6. Train tf model by:
 *  - converting training data into tf tensors
 *  - train the model itself using model.fitDataset
 */
export async function trainTfModel(model, trainDs, validationDs) {
  console.log("6. Training tf model ...");
  
  const trainLogs = [];
  await model.fitDataset(trainDs, {
    epochs: 100,
    validationData: validationDs,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        trainLogs.push(logs);
        tfvis.show.history(lossContainerEl, trainLogs, ["loss", "val_loss"]);
        tfvis.show.history(accContainerEl, trainLogs, ["acc", "val_acc"]);
      }
    }
  });

  return model;
}

/**
 * 7. Test results
 *  - get testing data as a tensor
 *  - predict lables using model.predict
 *  - visualise results with tfvis confusionMatrix
 */
export async function testResults(model, X, y, split) {
  const splitIdx = X.length * split;
  const xTest = tf.tensor(X.slice(splitIdx));
  const yTest = tf.tensor(y.slice(splitIdx));

  const preds = model.predict(xTest).argMax(-1);
  const labels = yTest.argMax(-1);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);

  tfvis.render.confusionMatrix(matrixContainerEl, {
    values: confusionMatrix,
    tickLabels: ["Fake", "Not-Fake"]
  });
}


(async () => {
  //  1. LOAD DATA
  const data = await loadData();
  console.log('Loaded data:', data);

  //  2. PREPARE DATA
  const features = ["profile pic", "nums/length username", "fullname words", "nums/length fullname",
    "name==username", "description length", "external URL", "private", "#posts", "#followers", "#follows"];
  const [X, y] = await prepareData(data, features);

  //  3. SPLIT DATA
  const [trainDs, validationDs] = splitData(X, y, .8, 16);

  //  4. GET TF MODEL
  const model = getTfModel(features);

  //  5. COMPILE TF MODEL
  compileTfModel(model);

  //  6. TRAIN TF MODEL
  await trainTfModel(model, trainDs, validationDs);

  //  7. TEST RESULTS
  testResults(model, X, y, .8);

  //  8. SAVE MODEL
  await model.save('downloads://pretrained-model');
})();
