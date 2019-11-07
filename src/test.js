console.log('test.js');

import $ from "jquery";
import dt from 'datatables.net';

dt();

// import * as tf from "@tensorflow/tfjs";
// import * as tfvis from "@tensorflow/tfjs-vis";
// import * as Papa from "papaparse";


// const features = ["profile pic", "nums/length username", "fullname words", "nums/length fullname",
// "name==username", "description length", "external URL", "private", "#posts", "#followers", "#follows"];


// Papa.parsePromise = function (file) {
//   return new Promise(function (complete, error) {
//     Papa.parse(file, {
//       header: true,
//       download: true,
//       dynamicTyping: true,
//       complete,
//       error
//     });
//   });
// };

// const oneHot = outcome => Array.from(tf.oneHot(outcome, 2).dataSync());

// const prepareData = async (url) => {
//   // const csv = await Papa.parsePromise(
//   //   "https://raw.githubusercontent.com/curiousily/Logistic-Regression-with-TensorFlow-js/master/src/data/diabetes.csv"
//   // );
//   const csv = await Papa.parsePromise(
//     url
//   );

//   return csv.data;
// };

// const createFeatures = (data, features) => {
//   return data.map(row =>
//     features.map(feature => {
//       const val = row[feature];
//       return val === undefined ? 0 : val;
//     })
//   );
// };

// async function loadPretrainedModel() {
//   console.log(tf);
//   const model = await tf.loadLayersModel('./pretrained-model.json');

//   const optimizer = tf.train.adam(0.001);
//   model.compile({
//     optimizer: optimizer,
//     loss: "binaryCrossentropy",
//     metrics: ["accuracy"]
//   });

//   return model;
// }

// async function predictData(model) {
//   const testData = await prepareData('./tiz.alocci-followers.csv');
//   //  const testData = await prepareData('./signal_noise-followers.csv');
//   const testX = tf.tensor(createFeatures(testData, features));
//   console.log('testData', testData, 'textX', testX);
//   const testPredictions = model.predict(testX).argMax(-1);
    
//   // console.log('testPredictions', testPredictions.dataSync(), 'labels', labels.dataSync());
//   console.log('copare lengths',  testPredictions.dataSync().length, testData.length);
  
//   testPredictions.dataSync().forEach((d, i) => {
//     console.log(d, testData[i]);
//   });
// }

import * as tf from "@tensorflow/tfjs";
import { loadCsv } from "./utils.js";

export async function loadData(url) {
  const data = await loadCsv(url);
  return data;
} 

/**
 * Load definition and weights and compile
 */
export async function loadPretrainedModel() {
  const model = await tf.loadLayersModel('./pretrained-model.json');

  const optimizer = tf.train.adam(0.005);
  model.compile({
    optimizer: optimizer,
    loss: "binaryCrossentropy",
    metrics: ["accuracy"]
  });
  return model;
}

export async function prepareData(data, features) {
  console.log("2. Preparing data ...");
  const X = data.map((row) => {
    return features.map((feature) => {
      const value = row[feature];
      return value !== undefined ? value : 0;
    });
  });
  return X;
}

export async function predictData(data, model, features) {
  const X = await prepareData(data, features);
  const tensorX = tf.tensor(X);
  
  console.log('testData', X, 'textX', tensorX);
  const testPredictions = model.predict(tensorX).argMax(-1);
    
  return [X, testPredictions.dataSync()];
}

export function displayResults(data, features) {
  const columns = features.map(f => ({ title: f }));
  console.log(features, columns);
  console.log(data);

  columns.unshift({ title: 'username' });
  columns.unshift({ title: 'fake' });

  // //  amend data
  // const data = X.map((x, i) => {
  //   return [y[i], 'username', ...x];
  // })

  $('#results').DataTable({
    'pageLength': 35,
    data,
    columns,
  });
}

(async() => {
  const dataPath = './instagram_test.csv';

  //  make sure this is the same with 
  const features = ["profile pic", "nums/length username", "fullname words", "nums/length fullname",
    "name==username", "description length", "external URL", "private", "#posts", "#followers", "#follows"];
  
  const model = await loadPretrainedModel();
  const data = await loadData(dataPath);
  const [X, y] = await predictData(data, model, features);
  
  //  amend data 
  const fullData = X.map((x, i) => {
    return [y[i], data[i].username, ...x];
  })

  displayResults(fullData, features);
})();
