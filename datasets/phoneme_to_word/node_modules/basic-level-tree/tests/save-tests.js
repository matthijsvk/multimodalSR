var test = require('tape');
var createLevelTree = require('../index');
var populateFreshDb = require('./fixtures/populate-fresh-db');
var getDbAndRoot = require('./fixtures/get-db-and-root');
var async = require('async');
var _ = require('lodash');

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

test('Update child', function updateChildTest(t) {
  t.plan(9);

  session.root.getChildAtPath(['Tryclyde'], updateChild);

  var child;

  function updateChild(error, node) {
    t.ok(!error, 'No error while getting from path.');

    child = node;
    child.value.weakness = 'pits';
    child.save(closeDb);
  }

  function closeDb(error) {
    t.ok(!error, 'No error while saving.');
    session.db.close(reopen);
  }

  function reopen(error) {
    t.ok(!error, 'No error while closing.');
    getDbAndRoot(useNewDbAndRootInstances);
  }

  function useNewDbAndRootInstances(error, db, root) {
    t.ok(!error, 'No error while reopening.');
    session.db = db;
    session.root = root;

    session.root.getChildAtPath(['Tryclyde'], checkSave);
  }

  function checkSave(error, savedNode) {
    t.ok(!error, 'No error while getting from path after reopening.');
    t.ok(savedNode, 'savedNode is returned.');
    t.equal(typeof savedNode, 'object', 'savedNode is an object.');
    t.equal(savedNode.value.weakness, 'pits', 'savedNode has updated value');
    t.deepEqual(
      _.pick(child, 'id', 'value', 'children'),
      _.pick(savedNode, 'id', 'value', 'children'),
      'The saved node is equal to the updated child.'
    );
  }
});
