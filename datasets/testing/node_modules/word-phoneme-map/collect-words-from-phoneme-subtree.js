function collectWordsFromPhonemeSubtree(subtree) {
  var collected = [];
  var currentNodes = [subtree];
  var nextNodes = [];

  while (currentNodes.length > 0) {
    currentNodes.forEach(visitCurrentNode);
    currentNodes = nextNodes.slice();
    nextNodes.length = 0;
  }

  function visitCurrentNode(node) {
    if (node) {
      if (node.value && node.value.words) {
        collected = collected.concat(node.value.words);
      }
      if (node.children) {
        nextNodes = nextNodes.concat(node.children);
      }
    }
  }
  
  return collected;
}

module.exports = collectWordsFromPhonemeSubtree;
