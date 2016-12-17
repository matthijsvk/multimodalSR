var callNextTick = require('call-next-tick');
var _ = require('lodash');

function getChildAtPath(db, parent, path, done) {
  if (path.length < 1) {
    callNextTick(done, null, parent);
    return;
  }

  var childName = path[0];
  parent.getChildren(pickChild);

  function pickChild(error, children) {
    if (error) {
      done(error);
    }
    else {
      var child = _.find(children, childHasNameFromPath);
      if (!child) {
        callNextTick(done, error);
      }
      else {
        callNextTick(getChildAtPath, db, child, path.slice(1), done);
      }
    }
  }

  function childHasNameFromPath(child) {
    return child && child.value && child.value.name === childName;
  }
}

module.exports = getChildAtPath;
