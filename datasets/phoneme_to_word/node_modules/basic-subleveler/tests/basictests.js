var test = require('tape');
var basicSubleveler = require('../index');
var queue = require('queue-async');
var level = require('level');

var leveldb;

((function prepare() {
  leveldb = level(
    __dirname + '/test.db',
    {
      valueEncoding: 'json'
    }
  );
})());

test('Basic test', function basicTest(t) {
  t.plan(11);

  var db = basicSubleveler.setUpSubleveledDB({
    db: leveldb,
    sublevels: {
      meats: 'm',
      vegetables: 'v'
    }
  });

  t.ok(db, 'Creates a db object');
  t.equal(typeof db.meats, 'object', 'db has a meats level.');
  t.equal(typeof db.vegetables, 'object', 'db has a vegetables level.');

  var q = queue();

  q.defer(db.meats.put, 'beef', 'delicious');
  q.defer(db.meats.put, 'fish', 'so good');
  q.defer(db.vegetables.put, 'broccoli', 'tasty');
  q.defer(db.vegetables.put, 'tomato', 'juicy');
  q.defer(db.vegetables.put, 'arugula', 'elitist');

  q.awaitAll(checkPuts);

  function checkPuts(error) {
    t.ok(!error, 'Saves without error');
    var readQueue = queue();
    readQueue.defer(db.meats.readAllValues);
    readQueue.defer(db.vegetables.readAllValues);
    readQueue.await(checkReads);
  }

  function checkReads(error, meatValues, vegValues) {
    t.ok(!error, 'Reads all values without error.');
    t.ok(meatValues.indexOf('delicious') !== -1, 'Can read value.');
    t.ok(meatValues.indexOf('so good') !== -1, 'Can read value.');
    t.ok(vegValues.indexOf('tasty') !== -1, 'Can read value.');
    t.ok(vegValues.indexOf('juicy') !== -1, 'Can read value.');
    t.ok(vegValues.indexOf('elitist') !== -1, 'Can read value.');

    db.close(checkClose);
  }

  function checkClose(error) {
    t.ok(!error, 'There is no error while closing the LevelDB instance.');
  }
});
