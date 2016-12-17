var createLevelTree = require('basic-level-tree');
var collectWordsFromPhonemeSubtree = require('./collect-words-from-phoneme-subtree');

function createForwardPhonemeMap(opts, createDone) {
  var db;
  var root;

  if (opts) {
    db = opts.db;
  }

  if (!db) {
    createDone(new Error('Cannot create forward phonemes map without db.'));
    return;
  }

  var levelTree = createLevelTree(
    {
      db: db,
      treeName: 'forward-phonemes'
    },
    passBackMethod
  );

  function passBackMethod(error, levelTreeRoot) {
    if (error) {
      createDone(error);
    }
    else {
      root = levelTreeRoot;
      createDone(error, wordsForPhonemeStartSequence);
    }
  }

  function wordsForPhonemeStartSequence(phonemesInOrder, done) {
    root.getSubtreeAtPath(phonemesInOrder, gatherWords);

    function gatherWords(error, subtree) {
      if (error) {
        done(error);
      }
      else {
        var words = collectWordsFromPhonemeSubtree(subtree);
        if (words) {
          done(error, words);
        }
        else {
          done(error);
        }
      }
    }
  } 
}

module.exports = createForwardPhonemeMap;
