var level = require('level');
var sublevel = require('level-sublevel');
var _ = require('lodash');

function setUpSubleveledDB(opts) {
  // opts:
  //  db: LevelDB instance
  //  sublevels: Dictionary of immediate sublevel names and their 
  //    delimiters. e.g. {words: 'w'}
  var leveldb = opts.db;
  var subleveldb = sublevel(leveldb);

  for (var levelname in opts.sublevels) {
    var sl = subleveldb.sublevel(opts.sublevels[levelname]);
    sl.readAllValues = _.curry(readAllValuesFromSublevel)(sl);
    leveldb[levelname] = sl;
  }

  return leveldb;
}

function readAllValuesFromSublevel(sublevel, done) {
  var values = [];
  var valueStream = sublevel.createValueStream();
  valueStream.on('data', function addValue(value) {
    values.push(value);
  });

  valueStream.on('close', passBackValues);

  function passBackValues(error) {
    if (error) {
      done(error);
    }
    else {
      done(error, values);
    }
  }
}

// TODO: readAllKeys.

module.exports = {
  setUpSubleveledDB: setUpSubleveledDB,
  readAllValuesFromSublevel: readAllValuesFromSublevel
};
