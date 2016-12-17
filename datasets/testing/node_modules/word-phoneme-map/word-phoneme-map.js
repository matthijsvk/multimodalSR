var basicSubleveler = require('basic-subleveler');
var level = require('level');
var createReversePhonemeMap = require('./reverse-phoneme-map');
var createForwardPhonemeMap = require('./forward-phoneme-map');
var queue = require('queue-async');

function createWordPhonemeMap(opts, createDone) {
  if (!opts || !opts.dbLocation) {
    createDone(new Error('Cannot create wordPhonemeMap without dbLocation.'));
    return;
  }

  var db = level(
    opts.dbLocation,
    {
      valueEncoding: 'json'
    }
  );

  var db = basicSubleveler.setUpSubleveledDB({
    db: db,
    sublevels: {
      words: 'w',
      phonemes: 'p'
    }
  });

  var mapOpts = {
    db: db
  };

  var q = queue();
  q.defer(createReversePhonemeMap, mapOpts);
  q.defer(createForwardPhonemeMap, mapOpts);
  q.await(passBackMethods);

  function passBackMethods(
    error, wordsForPhonemeEndSequence, wordsForPhonemeStartSequence) {

    if (error) {
      createDone(error);
    }
    else {
      createDone(
        error,
        {
          wordsForPhonemeSequence: wordsForPhonemeSequence,
          phonemeSequencesForWord: phonemeSequencesForWord,
          wordsForPhonemeEndSequence: wordsForPhonemeEndSequence,
          wordsForPhonemeStartSequence: wordsForPhonemeStartSequence,
          close: db.close.bind(db)
        }
      );
    }
  }

  function wordsForPhonemeSequence(sequence, done) {
    var sequenceString = sequence.join('_');
    var seqLevel = db.phonemes.sublevel(sequenceString);
    basicSubleveler.readAllValuesFromSublevel(seqLevel, done);
  }

  function phonemeSequencesForWord(word, done) {
    var wordLevel = db.words.sublevel(word);
    basicSubleveler.readAllValuesFromSublevel(wordLevel, done);
  }
}

module.exports = createWordPhonemeMap;
