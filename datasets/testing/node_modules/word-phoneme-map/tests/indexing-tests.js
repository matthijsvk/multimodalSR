var test = require('tape');
var indexWordsAndPhonemes = require('../index-words-and-phonemes');
var createWordPhonemeMap = require('../word-phoneme-map');
var fs = require('fs');
var rimraf = require('rimraf');
var callNextTick = require('call-next-tick');

var indexOpts = {
  dbLocation: __dirname + '/test.db',
  numberOfLinesToIndex: 6000
};

rimraf.sync(indexOpts.dbLocation);

function setUpMiniIndex(done) {
  if (fs.existsSync(indexOpts.dbLocation)) {
    callNextTick(done);
  }
  else {
    indexWordsAndPhonemes(indexOpts, done);
  }
}

test('Try map without db', function noDb(t) {
  t.plan(1);

  t.throws(createMapWithNoDb);

  function createMapWithNoDb() {
    createWordPhonemeMap({
      dbLocation: null
    });
  }
});

test('Index', function indexTest(t) {
  t.plan(2);
  setUpMiniIndex(checkDb);

  function checkDb(error) {
    t.ok(!error, 'No error occurred while indexing.');
    t.ok(fs.existsSync(indexOpts.dbLocation), 'Database file was created.');
  }
});

