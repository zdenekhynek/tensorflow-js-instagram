import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import * as Papa from "papaparse";


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

const prepareData = async (url) => {
  // const csv = await Papa.parsePromise(
  //   "https://raw.githubusercontent.com/curiousily/Logistic-Regression-with-TensorFlow-js/master/src/data/diabetes.csv"
  // );
  const csv = await Papa.parsePromise(
    url
  );

  return csv.data;
};

const createFeatures = (data, features) => {
  return data.map(row =>
    features.map(feature => {
      const val = row[feature];
      return val === undefined ? 0 : val;
    })
  );
};

async function loadPretrainedModel() {
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

async function predictData(model) {
  const testData = await prepareData('./tiz.alocci-followers.csv');
  //  const testData = await prepareData('./signal_noise-followers.csv');
  const testX = tf.tensor(createFeatures(testData, features));
  console.log('testData', testData, 'textX', testX);
  const testPredictions = model.predict(testX).argMax(-1);
    
  // console.log('testPredictions', testPredictions.dataSync(), 'labels', labels.dataSync());
  console.log('copare lengths',  testPredictions.dataSync().length, testData.length);
  
  testPredictions.dataSync().forEach((d, i) => {
    console.log(d, testData[i]);
  });
}


(async() => {
  const model = await loadPretrainedModel();
  predictData(model);
})();
  