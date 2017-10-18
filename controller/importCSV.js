fs = require('fs');

function csvToData(csv){

  var aux = csv.split('\n');

  var arr = []

  for (var i = 0; i < aux.length-1; i++) {
    arr[i] = aux[i].split(',').map(parseFloat);
  }

  return arr;
}


module.exports = function(path){
  var csv = fs.readFileSync(path,'utf8');
  return csvToData(csv);
}
