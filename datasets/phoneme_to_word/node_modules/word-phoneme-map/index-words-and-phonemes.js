var callNextTick = require('call-next-tick');
var fs = require('fs');
var split = require('split');
var queue = require('queue-async');
var createPhonemeIndexer = require('./phoneme-indexer');
var Writable = require('stream').Writable;

function indexWordsAndPhonemes(opts, done) {
  createPhonemeIndexer(
    {
      dbLocation: opts.dbLocation
    },
    startIndexing
  );

  function startIndexing(error, indexer) {
    if (error) {
      done(error);
      return;
    }

    var q = queue(4);
    var linesIndexed = 0;
    var readStream = fs.createReadStream(__dirname + '/data/cmudict.0.7a');
    var lineStream = split();
    var indexStream = Writable({
      objectMode: true
    });
    indexStream._write = writeChunkToIndex;

    readStream.pipe(lineStream);
    lineStream.pipe(indexStream);

    lineStream.on('end', cleanUp);

    function writeChunkToIndex(chunk, enc, callback) {
      if (opts.numberOfLinesToIndex === undefined ||
        linesIndexed < opts.numberOfLinesToIndex) {

        linesIndexed += 1;
        indexLine(chunk, callback)
      }
      else {
        callback();
      }
    }

    function indexLine(line, indexDone) {
      if (!line || line.indexOf(';;;') === 0) {
        indexDone();
        return;
      }

      var pieces = line.split('  ');
      if (pieces.length < 2) {
        indexDone();
        return;
      }

      var word = pieces[0];
      var phonemeString = pieces[1];

      if (stringIsValid(word) && stringIsValid(phonemeString)) {
        indexer.index(word, phonemeString, indexDone);
      }
      else {
        // It is not an error if the line is not parseable.
        callNextTick(indexDone);
      }
    }

    function cleanUp(error) {
      indexer.closeDb(passError);

      function passError() {
        done(error);
      }
    }
  }
}

function stringIsValid(s) {
  return (typeof s === 'string' && s.length > 0);
}

module.exports = indexWordsAndPhonemes;
