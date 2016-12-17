var queue = require('queue-async');
var level = require('level');
var basicSubleveler = require('basic-subleveler');
var phonemeTypes = require('phoneme-types');
var callNextTick = require('call-next-tick');
var createReversePhonemeIndexer = require('./reverse-phoneme-indexer');
var createForwardPhonemeIndexer = require('./forward-phoneme-indexer')
var queue = require('queue-async');

function createPhonemeIndexer(opts, done) {
  var indexWordByReversePhonemes;
  var indexWordByForwardPhonemes;

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

  var specialIndexerOpts = {
    db: db
  };

  var specialIndexerQueue = queue();
  specialIndexerQueue.defer(createReversePhonemeIndexer, specialIndexerOpts);
  specialIndexerQueue.defer(createForwardPhonemeIndexer, specialIndexerOpts);
  specialIndexerQueue.await(passBackMethods);

  function passBackMethods(error, reverseIndexMethod, forwardIndexMethod) {
    if (error) {
      done(error);
    }
    else {
      indexWordByReversePhonemes = reverseIndexMethod;
      indexWordByForwardPhonemes = forwardIndexMethod;

      var indexerMethods = {
        index: index,
        closeDb: db.close.bind(db)
      };
      done(error, indexerMethods);
    }
  }

  function index(word, cmuDictPhonemeString, done) {
    var phonemeString = phonemeTypes.stripStressor(cmuDictPhonemeString);
    var phonemes = phonemeString.split(' ');
    phonemeString = phonemes.join('_');

    if (stringIsEmpty(word)) {
      callNextTick(done, new Error('Missing word.'));
      return;
    }
    if (stringIsEmpty(phonemeString)) {
      callNextTick(done, new Error('Missing phonemeString.'));
      return;
    }
    
    var q = queue();

    // Index by word.
    var cleanedWord = stripOrdinal(word);
    var wordLevel = db.words.sublevel(cleanedWord);
    q.defer(wordLevel.put, phonemeString, phonemes);

    // Index by phoneme string.
    var phonemeLevel = db.phonemes.sublevel(phonemeString);

    q.defer(phonemeLevel.put, cleanedWord, cleanedWord);

    // Reverse index.
    q.defer(indexWordByReversePhonemes, cleanedWord, phonemes);
    // Forward index.
    q.defer(indexWordByForwardPhonemes, cleanedWord, phonemes);

    q.awaitAll(done);
  }
}

function stringIsEmpty(s) {
  return (typeof s !== 'string' || s.length < 1);
}

var ordinalRegex = /\(\d\)/;

function stripOrdinal(word) {
  return word.replace(ordinalRegex, '');
}


module.exports = createPhonemeIndexer;
