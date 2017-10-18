
function addLessOneToInputs(data){
  for (var i = 0; i < data.length; i++) {
    data[i].unshift(-1);
  }
  return data;
}

module.exports = function(data,numberOutputs){
  var saidas = []

  for (var i = 0; i < data.length; i++) {
    var arr = []

    for (var j = 0; j < numberOutputs; j++) {
      arr.unshift(data[i].pop())
    }

    saidas[i] = arr
  }

  return [addLessOneToInputs(data),saidas];
}
