function Mlp(inputs,outputs,layers,n,e){
  this.inputs = inputs
  this.outputs = outputs
  this.layers = getLayers(layers,inputs[0].length);
  this.n = n
  this.e = e
  this.epoch = 0
  this.arrY = []
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


Mlp.prototype.calculateEqm = function() {
  var eqm = 0.0
  var p = this.outputs.length
  for (var i = 0; i < p; i++) {
    eqm = eqm + calculateError(this.arrY[i],this.outputs[i])
  }
  return eqm/p
};


calculateError = function(rowArrY,rowOutputs){
  var error = 0.0

  for (var i = 0; i < rowOutputs.length; i++) {
    error = error + ((rowOutputs[i]-rowArrY[i])**2)
  }
  return error/2
}

Mlp.prototype.calculateArrY = function () {
  for (var i = 0; i < this.inputs.length; i++) {
    this.arrY.push( feedForward(this.inputs[i],this.layers) );
  }
};

Mlp.prototype.training = function() {

  // clean the array Y
  this.arrY = []

  //For each line of the inputs do:
  for (var i = 0; i < this.inputs.length; i++) {

    //Make a feed for ward and add the response to array Y
    this.arrY.push( feedForward(this.inputs[i],this.layers) );

    //Adjust the weigths with back propagation
    this.layers = backPropagation(this.layers,this.outputs[i]);
  }
};

backPropagation = function (layers,outputs) {

  //for each mlp layer do:
  for (var i = layers.length-1; i >= 0 ; i--) {

    // get the sigma array of the layer
    if (i == layers.length-1) {
      layers[i].sigmas = getDeltasLastLayer(outputs,layers[i].arrY,layers[i].arrI)
    }else {
      layers[i].sigmas = getDeltasHiddenLayers(layers[i],layers[i+1]);
    }

  }

  return layers;
};


getDeltasLastLayer = function(arrD,arrY,arrI){
  var sigmas = []
  for (var i = 0; i < arrD.length; i++) {
    sigmas[i] = (arrD[i]-arrY[i])*derivateHiperbolic(arrI[i]);
  }
  return sigmas;
}

getDeltasHiddenLayers = function(currentLayer,previousLayer){
  var sigmas = []

  for (var i = 0; i < currentLayer.neurons.length; i++) {
    sigmas[i] = getGradientAux(previousLayer,i)*derivateHiperbolic(currentLayer.arrI[i]);
  }

  return sigmas;
}

getGradientAux = function(layer,j){
  var value = 0

  for (var k = 0; k < layer.sigmas.length; k++) {
    value = value + layer.neurons[k].weigths[j]*layer.sigmas[k]
  }

  return value
}


derivateHiperbolic = function(value){
  var e = Math.E
  var negValue = value*(-1)
  return 4/(((e**negValue)+(e**value))**2);
}

feedForward = function(inputs,layers) {

    var arr = []

    //for each layer of the mlp
    for (var j = 0; j < layers.length; j++) {

      //if is the first layer, it will pass the mlp inputs as entry
      if (j == 0) {
        layers[j].arrI = calculateI(inputs,layers[j]);

      //if not, it will pass the array Y of the last layer as entry
      //Obs: The array Y must be added with -1 in begin of it
      }else{
        layers[j].arrI = calculateI(
          addLessOneToBegin(layers[j-1].arrY),
          layers[j] );

        //remove the element '-1' of the layer array Y
        layers[j-1].arrY.shift();
      }

      layers[j].arrY = calculateY(layers[j].arrI);

    }

    return arr = layers[layers.length-1].arrY
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

//method to calculate the array Y of the layer
//To this case were used by hiperbolic function,
//but could be used by logistic function
function calculateY(arrI){
  var arrY = []
  for (var i = 0; i < arrI.length; i++) {
    arrY[i] = hiperbolicFunction(arrI[i]);
  }
  return arrY;
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

function addLessOneToBegin(arr){
  arr.unshift(-1);
  return arr
}

module.exports = function(inputs,outputs,layers,n,e){
  var mlp = new Mlp(inputs,outputs,layers,n,e);
  return mlp;
}
