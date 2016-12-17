var test = require('tape');
var createLevelTree = require('../index');
var rimraf = require('rimraf');
var level = require('level');
var testCases = require('./fixtures/add-test-cases').toJS();

var session = {};

((function prepare() {
  var dbPath = __dirname + '/test.db';
  rimraf.sync(dbPath);
  session.db = level(
    dbPath,
    {
      valueEncoding: 'json'
    }
  );
})());

test('Create tree', function treeTest(t) {
  t.plan(2);

  createLevelTree(
    {
      db: session.db,
      treeName: 'subcon',
      root: {
        name: 'Wart',
        weakness: 'vegetables'
      }
    },
    checkTree
  );

  function checkTree(error, root) {
    t.ok(!error, 'No error while creating tree.');
    t.equal(typeof root, 'object');
    session.root = root;
  }
});

testCases.forEach(runAddChildTest);

function runAddChildTest(testCase) {
  test('Add ' + testCase.name, function addTest(t) {
    t.plan(3);
    session[testCase.parentKey].addChild(testCase.value, checkAdd);

    function checkAdd(error, added) {
      t.ok(!error, 'No error while adding.');
      t.equal(typeof added, 'object');
      t.deepEqual(
        added.value, testCase.value, 'Value is stored correctly for node.'
      );
      session[testCase.value.name] = added;
    }
  });
}

test('Do not add again if already there.', function doNotAddIfAlreadyThere(t) {
  t.plan(3);

  var numberOfChildren = session['Fryguy'].children.length;

  session['Fryguy'].addChildIfNotThere(
    {
      value: {
        name: 'Flurry',
        weakness: 'fire'
      },
      equalityFn: function hasSameName(a, b) {
        return a.name === b.name;
      }
    },
    checkAdd
  );

  function checkAdd(error, result) {
    t.ok(!error, 'No error while adding.');
    t.deepEqual(
      result.value, testCases[5].value, 'Existing value is returned.'
    );
    t.equal(
      session['Fryguy'].children.length,
      numberOfChildren,
      'Number of children remains the same.'
    );
  }
});


test('Add if not already there.', function addChildIfNotThere(t) {
  t.plan(3);

  var numberOfChildren = session['Fryguy'].children.length;
  var newValue = {
    name: 'Shyguy',
    weakness: 'cliffs'
  };

  session['Fryguy'].addChildIfNotThere(
    {
      value: newValue,
      equalityFn: function hasSameName(a, b) {
        return a.name === b.name;
      }
    },
    checkAdd
  );

  function checkAdd(error, result) {
    t.ok(!error, 'No error while adding.');
    t.deepEqual(result.value, newValue, 'New value is returned.');
    t.equal(
      session['Fryguy'].children.length,
      numberOfChildren + 1,
      'Number of children increases by 1.'
    );
  }
});

test('Default to deepEqual in addChildIfNotThere.', function defaultEq(t) {
  t.plan(3);

  var numberOfChildren = session['Fryguy'].children.length;
  var newValue = {
    name: 'Flurry',
    weakness: 'fire'
  };

  session['Fryguy'].addChildIfNotThere(
    {
      value: newValue
      // No equalityFn specified.
    },
    checkAdd
  );

  function checkAdd(error, result) {
    t.ok(!error, 'No error while adding.');
    t.deepEqual(result.value, newValue, 'New value is returned.');
    t.equal(
      session['Fryguy'].children.length,
      numberOfChildren + 1,
      'Number of children increases by 1.'
    );
  }
});

test('Close db', function close(t) {
  t.plan(1);

  session.db.close(checkClose);

  function checkClose(error) {
    t.ok(!error, 'No error while closing database.');
  }
});
