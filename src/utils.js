import * as Papa from "papaparse";
import * as tf from "@tensorflow/tfjs";

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

export async function loadCsv(url) {
  const csv = await Papa.parsePromise(url);
  return csv.data;
};

export function oneHot(outcome, numClasses = 2) {
  return Array.from(tf.oneHot(outcome, numClasses).dataSync());
};
