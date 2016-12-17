var test = require('tape');
var testData = require('./fixtures/get-test-data');
var populateFreshDb = require('./fixtures/populate-fresh-db');
var getDbAndRoot = require('./fixtures/get-db-and-root');
var async = require('async');

var fullTree = {
  value: testData.get('Wart').value,
  children: [
    {
      value: testData.get('Tryclyde').value,
      children: [
        {
          value: testData.get('Cobrat').value,
          children: []
        },
        {
          value: testData.get('Pokey').value,
          children: []
        },
        {
          value: testData.get('Panser').value,
          children: []
        },
      ]
    },
    {
      value: testData.get('Fryguy').value,
      children: [
        {
          value: testData.get('Flurry').value,
          children: []
        },
        {
          value: testData.get('Autobomb').value,
          children: []
        }
      ]
    }
  ]
};

var subtreeTestCases = [
  {
    path: [],
    expectedTree: fullTree
  },
  {
    path: ['Fryguy'],
    expectedTree: fullTree.children[1]
  },
  {
    path: ['Tryclyde', 'Panser'],
    expectedTree: fullTree.children[0].children[2]
  }
];


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

subtreeTestCases.forEach(runSubtreeTest);
subtreeTestCases.forEach(runSubtreeAtPathTest);

function runSubtreeTest(testCase) {
  test('Get subtree ' + testCase.path.join('/'),
    function subtreeTest(t) {
      t.plan(3);

      session.root.getChildAtPath(testCase.path, getChildSubtree);

      function getChildSubtree(error, child) {
        t.ok(!error, 'No error while getting child from path.');
        child.getSubtree(checkResult);
      }

      function checkResult(error, subtree) {
        // console.log('subtree:');
        // console.log(JSON.stringify(subtree, null, '  '));
        t.ok(!error, 'No error while getting subtree from node.');
        t.deepEqual(
          subtree,
          testCase.expectedTree,
          'Correct subtree is retrieved.'
        );
      }
    }
  );
}

function runSubtreeAtPathTest(testCase) {
  test('Get subtree at path ' + testCase.path.join('/'),
    function subtreeAtPathTest(t) {
      t.plan(2);

      session.root.getSubtreeAtPath(testCase.path, checkResult);

      function checkResult(error, subtree) {
        t.ok(!error, 'No error while getting subtree from node at path.');
        t.deepEqual(
          subtree,
          testCase.expectedTree,
          'Correct subtree is retrieved from path.'
        );
      }
    }
  );
}

var badPaths = [
  ['Bowser'],
  ['Tryclyde', 'Trouter']
];

badPaths.forEach(runBadPathTest);

function runBadPathTest(badPath) {
  test('Handle bad path ' + badPath.join('/'), function getBadPathTest(t) {
    t.plan(2);

    session.root.getSubtreeAtPath(badPath, checkResult);

    function checkResult(error, subtree) {
      t.ok(!error, 'No error while getting from path.');
      t.equal(subtree, undefined, 'undefined is retrieved.');
    }
  });
}
