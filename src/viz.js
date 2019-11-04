const Plotly = window.Plotly;

const renderOutcomes = data => {
  const outcomes = data.map(r => r.fake);

  const [fake, real] = _.partition(outcomes, o => o === 1);

  const chartData = [
    {
      labels: ["Fake", "real"],
      values: [fake.length, real.length],
      type: "pie",
      opacity: 0.6,
      marker: {
        colors: ["gold", "forestgreen"]
      }
    }
  ];

  Plotly.newPlot("outcome-cont", chartData, {
    title: "Fake vs real"
  });
};

const renderHistogram = (container, data, column, config, xAxisRange = null) => {
  const fake = data.filter(r => r.fake === 1).map(r => r[column]);
  const real = data.filter(r => r.fake === 0).map(r => r[column]);

  const dTrace = {
    name: "fake",
    x: fake,
    type: "histogram",
    opacity: 0.6,
    marker: {
      color: "gold"
    }
  };

  const hTrace = {
    name: "real",
    x: real,
    type: "histogram",
    opacity: 0.4,
    marker: {
      color: "forestgreen"
    }
  };

  Plotly.newPlot(container, [dTrace, hTrace], {
    barmode: "overlay",
    xaxis: {
      title: config.xLabel,
      range: xAxisRange
    },
    yaxis: { title: "Count" },
    title: config.title
  });
};

const renderScatter = (container, data, columns, config) => {
  const fake = data.filter(r => r.fake === 1);
  const real = data.filter(r => r.fake === 0);

  var dTrace = {
    x: fake.map(r => r[columns[0]]),
    y: fake.map(r => r[columns[1]]),
    mode: "markers",
    type: "scatter",
    name: "Fake",
    opacity: 0.4,
    marker: {
      color: "gold"
    }
  };

  var hTrace = {
    x: real.map(r => r[columns[0]]),
    y: real.map(r => r[columns[1]]),
    mode: "markers",
    type: "scatter",
    name: "Real",
    opacity: 0.4,
    marker: {
      color: "forestgreen"
    }
  };

  var chartData = [dTrace, hTrace];

  Plotly.newPlot(container, chartData, {
    title: config.title,
    xaxis: {
      title: config.xLabel
    },
    yaxis: { title: config.yLabel }
  });
};


renderOutcomes(data);

renderHistogram("followers-cont", data, "#followers", {
  title: "#followers",
}, [0, 20000]);

renderHistogram("follows-cont", data, "#follows", {
  title: "#follows",
});

// renderScatter("glucose-age-cont", data, ["Glucose", "Age"], {
//   title: "Glucose vs Age",
//   xLabel: "Glucose",
//   yLabel: "Age"
// });
