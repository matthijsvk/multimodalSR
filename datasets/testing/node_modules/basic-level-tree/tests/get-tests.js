var test = require('tape');
var createLevelTree = require('../index');
var level = require('level');
var callNextTick = require('call-next-tick');
var _ = require('lodash');
var testData = require('./fixtures/get-test-data');
var populateFreshDb = require('./fixtures/populate-fresh-db');

var session = {};

test('Prepare', function prepare(t) {
  t.plan(1);

  populateFreshDb(checkPopulate);

  function checkPopulate(error) {
    if (error) {
      console.log('Populate error:', error);
    }

    session.db = level(
      __dirname + '/test.db',
      {
        valueEncoding: 'json'
      }
    );

    t.ok(!error, 'No error while preparing db for tests.');
  }
});

test('Get tree', function treeTest(t) {
  t.plan(3);

  createLevelTree(
    {
      db: session.db,
      treeName: 'subcon'
    },
    checkTree
  );

  function checkTree(error, root) {
    t.ok(!error, 'No error when getting tree.');
    t.equal(typeof root, 'object');
    t.deepEqual(root.value, testData.get('Wart').value, 'Root value got.');
    session.root = root;

    callNextTick(runGetChildTest, root);
  }
});

function runGetChildTest(node) {
  var testDatum = testData.get(node.value.name);

  test('Get ' + node.value.name + ' children', function getTest(t) {
    t.plan(5);

    t.equal(typeof node.getChildren, 'function', 'Has a getChildren method.');
    t.deepEqual(node.value, testDatum.value, 'Node value is correct.');

    node.getChildren(checkGet);

    function checkGet(error, children) {
      t.ok(!error, 'No error while getting.');
      t.equal(typeof children, 'object');

      var childNames = _.pluck(_.pluck(children, 'value'), 'name');
      t.deepEqual(
        childNames,
        testDatum.sessionKeysOfExpectedChildren,
        'Children`s names are correct.'
      );

      children.forEach(runGetChildTest);
    }
  });
}

