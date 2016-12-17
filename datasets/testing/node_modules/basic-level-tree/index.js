var exportMethods = require('export-methods');
var _ = require('lodash');
var Sublevel = require('level-sublevel');
var idmaker = require('idmaker');
var queue = require('queue-async');
var getChildAtPath = require('./get-child-at-path');
var getSubtree = require('./get-subtree');

function createLevelTree(opts, done) {
  var treeDb;
  var treeName;

  if (opts) {
    treeName = opts.treeName;
  }

  if (!treeName) {
    done(new Error('Missing treeName. Cannot create tree.'));
    return;
  }

  var treeDb = Sublevel(opts.db).sublevel(treeName);

  if (!treeDb) {
    done(new Error('Missing db. Cannot create tree.'));
    return;
  }

  var rootId = 'root';
  var rootNode;

  treeDb.get(rootId, checkForRoot);

  function checkForRoot(error, root) {
    if (error && error.type === 'NotFoundError') {
      rootNode = {
        id: rootId,
        value: opts.root,
        children: []
      };
      treeDb.put(rootId, rootNode, packageRoot);
    }
    else {
      rootNode = root;
      packageRoot(error);
    }
  }

  function packageRoot(error) {
    attachMethodsToNode(rootNode);

    if (error) {
      done(error);
    }
    else {
      done(error, rootNode);
    }
  }

  function addChildToNode(parent, childValue, addDone) {
    var childId = idmaker.randomId(8);

    var childNode = {
      id: childId,
      value: childValue,
      children: []
    };
    attachMethodsToNode(childNode);

    parent.children.push(childId);

    var q = queue();
    q.defer(treeDb.put, parent.id, parent);
    q.defer(treeDb.put, childId, childNode);
    q.await(passBackChild);

    function passBackChild(error) {
      if (error) {
        addDone(error);
      }
      else {
        addDone(error, childNode);
      }
    }
  }

  function getChildrenOfParent(parent, getDone) {
    var q = queue(10);
    parent.children.forEach(queueGet);
    function queueGet(childKey) {
      q.defer(treeDb.get, childKey);
    }
    q.awaitAll(attachMethodsToChildren);

    function attachMethodsToChildren(error, children) {
      if (error) {
        getDone(error);
      }
      else {
        children.forEach(attachMethodsToNode);
        getDone(error, children);
      }
    }
  }

  function getSubtreeAtPath(node, path, getSubtreeDone) {
    node.getChildAtPath(path, getChildSubtree);

    function getChildSubtree(error, child) {
      if (error) {
        getSubtreeDone(error);
      }
      else if (!child) {
        getSubtreeDone(error, child);
      }
      else {
        child.getSubtree(getSubtreeDone);
      }
    }
  }

  function addChildIfNotThere(parent, opts, addDone) {
    var childValue;
    var equalityFn;

    if (opts) {
      childValue = opts.value;
      equalityFn = opts.equalityFn;
    }

    if (!equalityFn) {
      equalityFn = _.isEqual;
    }

    parent.getChildren(checkChildrenForMatch);

    function checkChildrenForMatch(error, children) {
      if (error) {
        addDone(error);
      }
      else {
        var existingChild = _.find(children, hasValueEqualToValueToAdd);
        if (existingChild) {
          addDone(null, existingChild);
        }
        else {
          parent.addChild(childValue, addDone);
        }
      }
    }

    function hasValueEqualToValueToAdd(child) {
      return equalityFn(child.value, childValue);
    }
  }

  function saveNode(node, saveDone) {
    treeDb.put(node.id, node, saveDone);
  }

  function attachMethodsToNode(node) {
    node.addChild = _.curry(addChildToNode)(node);
    node.addChildIfNotThere = _.curry(addChildIfNotThere)(node);
    node.getChildren = _.curry(getChildrenOfParent)(node);
    node.getChildAtPath = _.curry(getChildAtPath)(treeDb)(node);
    node.getSubtree = _.curry(getSubtree)(node);
    node.getSubtreeAtPath = _.curry(getSubtreeAtPath)(node);
    node.save = _.curry(saveNode)(node);
  }
}

module.exports = createLevelTree;
