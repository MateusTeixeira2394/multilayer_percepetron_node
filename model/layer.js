function Layer(numberNeuron,numberInputs){
  this.neurons = getNeurons(numberNeuron,numberInputs);
  this.arrI = []
  this.arrY = []
}

function getNeurons(numberNeuron,numberInputs){
  var arr = []
  for (var i = 0; i < numberNeuron; i++) {
    var neuron = require('./neuron')(numberInputs);
    arr.push(neuron);
  }
  return arr;
}

module.exports = function(numberNeuron,numberInputs){
  var layer = new Layer(numberNeuron,numberInputs);
  return layer;
}
