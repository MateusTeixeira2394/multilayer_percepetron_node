function Mlp(inputs,outputs,layers,n,e){
  this.inputs = inputs
  this.outputs = outputs
  this.layers = getLayers(layers,inputs[0].length);
  this.n = n
  this.e = e
  this.epoch = 0
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

Mlp.prototype.feedForward = function() {
  //for each row of the data sample
  for (var i = 0; i < this.inputs.length; i++) {

    //for each layer of the mlp
    for (var j = 0; j < this.layers.length; j++) {

      //if is the first layer, it will pass the mlp inputs as entry
      if (j == 0) {
        this.layers[j].arrI = calculateI(this.inputs[i],this.layers[j]);

      //if not, it will pass the array Y of the last layer as entry
      }else{
        
      }

    }
  }
};

function hiperbolicFunction(value) {
  var e = Math.E
  var negValue = value*-1

  return ((e**value)-(e**negValue))/((e**value)+(e**negValue))
};

function logisticFunction(value) {
  var e = Math.E
  var negValue = value*-1

  return 1/(1+(e**negValue))
};

function calculateY(){

}

//calculate the array I of the layer
function calculateI(arrInputs,layer){
  var arrI = []
  for (var i = 0; i < layer.neurons.length; i++) {
    arrI.push( calculateU( arrInputs,layer.neurons[i] ) );
  }
  return arrI;
}

//calculate the U of the each neuron of the layer
function calculateU(arrInputs,neuron){
  var u = 0.0
  for (var i = 0; i < neuron.weights.length; i++) {
    u = u + (arrInputs[i]*neuron.weights[i])
  }
  return u;
}

module.exports = function(inputs,outputs,layers,n,e){
  var mlp = new Mlp(inputs,outputs,layers,n,e);
  return mlp;
}
