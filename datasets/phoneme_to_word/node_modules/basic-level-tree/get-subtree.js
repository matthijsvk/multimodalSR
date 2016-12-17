var queue = require('queue-async');
var _ = require('lodash');

function getSubtree(node, done) {
  var summary = {
    // If there's performance issues, reconsider cloneDeep.
    value: _.cloneDeep(node.value)
  };

  node.getChildren(addChildrenToSummary);

  function addChildrenToSummary(error, children) {
    var q;

    if (error) {
      done(error);
    }
    else {
      q = queue(4);
      children.forEach(queueRecursion);
      q.awaitAll(passBackChildTrees);
    }

    function queueRecursion(child) {
      q.defer(getSubtree, child);
    }

    function passBackChildTrees(error, childTrees) {
      if (error) {
        done(error);
      }
      else {
        summary.children = childTrees;
        done(error, summary);
      }
    }
  }
}

module.exports = getSubtree;
