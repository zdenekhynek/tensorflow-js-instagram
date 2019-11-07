import $ from "jquery";
import * as tf from "@tensorflow/tfjs";

// load generic datagrid component
import dt from 'datatables.net';
dt();

import { loadCsv } from "./utils.js";

//  which data you want predict
const DATA_PATH = './instagram_test.csv';


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


/**
 * Returns [[], [], []]
 */
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

  $('#results').DataTable({
    'pageLength': 35,
    data,
    columns,
  });
}

(async() => {
  
  //  make sure this is the same with index.js
  const features = ["profile pic", "nums/length username", "fullname words", "nums/length fullname",
    "name==username", "description length", "external URL", "private", "#posts", "#followers", "#follows"];
  
  const model = await loadPretrainedModel();
  const data = await loadData(DATA_PATH);
  const [X, y] = await predictData(data, model, features);
  
  //  amend data 
  const fullData = X.map((x, i) => {
    return [y[i], data[i].username, ...x];
  })

  displayResults(fullData, features);
})();
