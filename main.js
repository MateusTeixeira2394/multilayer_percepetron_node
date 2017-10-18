var importCSV = require('./controller/importCSV');

var data = importCSV('./csv_treino.txt');

//the number of outputs
var numberOutputs = 1

//module to separate the csv object generated in inputs and outputs
var input_output = require('./controller/get_inputs_outputs')(data,numberOutputs);

var inputs = input_output[0]

var outputs = input_output[1]

//array to represent the layers of the mlp
//each  array elements say the number of neurons
var layers = [3,2,1]

var mlp = require('./model/mlp')(inputs,outputs,layers);
