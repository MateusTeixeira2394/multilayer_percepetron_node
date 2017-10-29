var importCSV = require('./controller/importCSV');

var data = importCSV('./csv_treino.txt');

//the number of outputs
var numberOutputs = 1

//module to separate the csv object generated in inputs and outputs
var input_output = require('./controller/get_inputs_outputs')(data,numberOutputs);

var inputs = input_output[0]

var outputs = input_output[1]

//learning rate
var n = 0.5

//precision required
var e = 0.0001

var progress = true

//array to represent the layers of the mlp
//each  array elements say the number of neurons
var layers = [2,1]

var mlp = require('./model/mlp')(inputs,outputs,layers,n,e);

mlp.calculateArrY();

while (progress) {

  var eqm_before = mlp.calculateEqm();

  mlp.training();

  var eqm_after = mlp.calculateEqm();

  mlp.incrementEpoch();

  console.log(Math.abs(eqm_after-eqm_before));
  console.log(mlp.epoch);

  if (Math.abs(eqm_after-eqm_before) <= e) {
    progress = false
  }
}

var data_teste = importCSV('./csv_teste.txt');

var input_output_test = require('./controller/get_inputs_outputs')(data_teste,numberOutputs);

var inputs_test = input_output_test[0]

var outputs_test = input_output_test[1]

mlp.test(inputs_test)
