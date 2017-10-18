function Neuron(numberInputs){
  this.weights = getWeights(numberInputs);
}

function getWeights(numberInputs){
  var arr = []
  for (var i = 0; i < numberInputs; i++) {
    arr.push(Math.random());
  }
  return arr;
}

module.exports = function(numberInputs){
  var neuron = new Neuron(numberInputs);
  return neuron;
}
