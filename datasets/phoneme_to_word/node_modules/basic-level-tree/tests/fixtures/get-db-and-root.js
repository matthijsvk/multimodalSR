var createLevelTree = require('../../index');
var level = require('level');

function getDbAndRoot(done) {
  var db = level(
    __dirname + '/../test.db',
    {
      valueEncoding: 'json'
    }
  );

  createLevelTree(
    {
      db: db,
      treeName: 'subcon'
    },
    passBackDbAndRoot
  );

  function passBackDbAndRoot(error, root) {
    if (error) {
      done(error);
    }
    else {
      done(error, db, root);
    }
  }
}

module.exports = getDbAndRoot;
