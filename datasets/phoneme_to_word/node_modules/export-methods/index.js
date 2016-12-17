function exportMethods(fns) {
  var fnArray;

  if (Array.isArray(fns)) {
    fnArray = fns;
  }
  else {
    fnArray = Array.prototype.slice.call(arguments);
  }

  return fnArray.reduce(addFnToExports, {});
}

function addFnToExports(exports, fn) {
  exports[fn.name] = fn;
  return exports;
}

module.exports = exportMethods;
