var test = require('tape');
var testData = require('./fixtures/get-test-data');
var populateFreshDb = require('./fixtures/populate-fresh-db');
var getDbAndRoot = require('./fixtures/get-db-and-root');
var async = require('async');

var session = {};

test('Prepare', function prepare(t) {
  t.plan(1);

  async.waterfall([populateFreshDb, getDbAndRoot], saveDbAndRoot);

  function saveDbAndRoot(error, db, root) {
    if (error) {
      console.log('Setup error:', error);
    }

    session.db = db;
    session.root = root;
    t.ok(!error, 'No error while setting up for tests.');
  }
});

var pathTestCases = [
  {
    path: [],
    expectedTestDataKey: 'Wart'
  },
  {
    path: ['Fryguy'],
    expectedTestDataKey: 'Fryguy'
  },
  {
    path: ['Tryclyde', 'Panser'],
    expectedTestDataKey: 'Panser'
  }
];

pathTestCases.forEach(runPathTest);

function runPathTest(testCase) {
  test('Get child at path ' + testCase.path.join('/'), function getPathTest(t) {
    t.plan(2);

    session.root.getChildAtPath(testCase.path, checkResult);

    function checkResult(error, node) {
      t.ok(!error, 'No error while getting from path.');
      t.deepEqual(
        node.value,
        testData.get(testCase.expectedTestDataKey).value,
        'Correct child is retrieved.'
      );
    }
  });
}

var badPaths = [
  ['Bowser'],
  ['Tryclyde', 'Trouter']
];

badPaths.forEach(runBadPathTest);

function runBadPathTest(badPath) {
  test('Handle bad path ' + badPath.join('/'), function getBadPathTest(t) {
    t.plan(2);

    session.root.getChildAtPath(badPath, checkResult);

    function checkResult(error, node) {
      t.ok(!error, 'No error while getting from path.');
      t.equal(node, undefined, 'undefined is retrieved.');
    }
  });
}
