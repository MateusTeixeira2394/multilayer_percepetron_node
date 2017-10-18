function Mlp(inputs,outputs,layers){
  this.inputs = inputs
  this.outputs = outputs
  this.layers = getLayers(layers,inputs[0].length);
}

function getLayers(layers,numberInputs){
  var arr = []
  for (var i = 0; i < layers.length; i++) {
    // if it is the first layer, each neuron will have the number of entries
    // equal to the number of 'x'
    if (i==0) {
      var layer = require('./layer')(layers[i],numberInputs);
    }
    // otherwise, each neuron will have the number of entries equal to
    // number of neurons of the previous layer plus 1 (because of the '-1' input)
    else{
      var layer = require('./layer')(layers[i], layers[i-1] + 1);
    }
    arr.push(layer);
  }
  return arr;
}


module.exports = function(inputs,outputs,layers){
  var mlp = new Mlp(inputs,outputs,layers);
  return mlp;
}
