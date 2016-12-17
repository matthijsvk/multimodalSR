function createIdmaker() {
  function pickFromArrayAtRandom(array) {
    return array[~~(Math.random() * array.length)];
  }

  var idChars = 
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'.split('');

  // Creates a string of random characters of the length specified.
  function randomId(len) {
    var id = '';
    for (var i = 0; i < len; ++i) {
      id += pickFromArrayAtRandom(idChars);
    }
    return id;
  }

  return {
    randomId: randomId
  };
}

if (typeof module === 'object' && typeof module.exports === 'object') {
  module.exports = createIdmaker();
}

