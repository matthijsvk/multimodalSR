var createLevelTree = require('../../index');
var rimraf = require('rimraf');
var level = require('level');
var testCases = require('./add-test-cases').toJS();
var queue = require('queue-async');

var session = {};

function populateFreshDb(populateDone) {
  var dbPath = __dirname + '/../test.db';

  rimraf.sync(dbPath);
  session.db = level(
    dbPath,
    {
      valueEncoding: 'json'
    }
  );

  createLevelTree(
    {
      db: session.db,
      treeName: 'subcon',
      root: {
        name: 'Wart',
        weakness: 'vegetables'
      }
    },
    addNodes
  );

  function addNodes(error, root) {
    if (error) {
      populateDone(error);      
    }
    else {
      session.root = root;

      var q = queue(1);
      testCases.forEach(queueAddNode);

      function queueAddNode(testCase) {
        q.defer(addNode, testCase);
      }

      q.awaitAll(cleanUp);
    }
  }

  function cleanUp(error) {
    if (error) {
      populateDone(error);
    }
    else {
      session.db.close(populateDone);
    }
  }
}

function addNode(testCase, done) {
  session[testCase.parentKey].addChild(testCase.value, saveAdded);

  function saveAdded(error, added) {
    if (error) {
      done(error);
    }
    else {
      session[testCase.value.name] = added;
      done();
    }
  }
}

module.exports = populateFreshDb;
