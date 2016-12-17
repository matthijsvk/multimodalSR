// Depends on indexing-tests having been run first, unfortunately.

var test = require('tape');
var createWordPhonemeMap = require('../word-phoneme-map');
var callNextTick = require('call-next-tick');

var dbLocation = __dirname + '/test.db';

test('Create and use map', function fullPhonemeSequenceMatch(t) {
  var expectedWordsForSequences = [
    {
      sequences: [
        ['AA', 'R', 'K']
      ],
      endWords: ['ARC', 'ARK']
    },
    {
      sequences: [
        ['AE', 'B', 'N', 'AO', 'R', 'M', 'AH', 'L', 'IY']
      ],
      endWords: ['ABNORMALLY']
    },
    {
      sequences: [
        ['EY', 'B', 'AH', 'L', 'ER'],
        ['EY', 'B', 'L', 'ER']
      ],
      endWords: ['ABLER']
    }
  ];


  t.plan(expectedWordsForSequences.length * 4 + 2);

  createWordPhonemeMap(
    {
      dbLocation: dbLocation
    },
    useMap
  );

  function useMap(error, wordPhonemeMap) {
    t.ok(!error, 'No error while creating map.');

    expectedWordsForSequences.forEach(runWordsForSequenceTest);

    function runWordsForSequenceTest(pair) {
      wordPhonemeMap.wordsForPhonemeSequence(pair.sequences[0], checkWords);

      function checkWords(error, words) {
        t.ok(!error, 'No error occured while looking for words.');
        t.deepEqual(words, pair.endWords, 'Expected words are returned.');
      }
    }

    expectedWordsForSequences.forEach(runSequencesForWordsTest);

    function runSequencesForWordsTest(pair) {
      wordPhonemeMap.phonemeSequencesForWord(pair.endWords[0], checkSequences);

      function checkSequences(error, sequences) {
        t.ok(!error, 'No error occured while looking for sequence.');
        t.deepEqual(
          sequences, pair.sequences, 'Expected sequence is returned.'
        );
      }
    }

    wordPhonemeMap.close(checkClose);
  }

  function checkClose(error) {
    t.ok(!error, 'Database closes successfully.');
  }
});

var expectedWordsForSequences = [
  {
    sequence: ['AA', 'R', 'K'],
    endWords: ['ARC', 'ARK', 'AARDVARK', '?QUESTION-MARK'],
    startWords:  ['ARC', 'ARK', 'ARCHEY', 'ARCO', 'ARKO', 'ARCS', 'ARX', 'ARCADE', 'ARCANE', 'ARCHERD', 'ARKIN', 'ARCHIVE', 'ARCO\'S', 'ARCOS', 'ARKOSE', 'ARKLA', 'ARQUETTE', 'ARCADES', 'ARKADY', 'ARCHAIC', 'ARCANA', 'ARCAND', 'ARKADI', 'ARKAROW', 'ARCARO', 'ARCATA', 'ARCTIC', 'ARKIN\'S', 'ARCHIVES', 'ARCOLA', 'ARCURI', 'ARKLA\'S', 'ARKWRIGHT', 'ARCADIA', 'ARCANUM', 'ARKANSAS', 'ARCTURUS', 'ARCHETYPE', 'ARCHIVAL', 'ARQUILLA', 'ARCADIAN', 'ARCHANGEL', 'ARKANSAN', 'ARCANUM\'S', 'ARKANSAS\'', 'ARKANSAS\'S', 'ARCHITECT', 'ARCHITRAVE', 'ARCHIVIST', 'ARCHETYPAL', 'ARCTURUS', 'ARCHETYPES', 'ARCHIVIST', 'ARCHULETA', 'ARCHULETTA', 'ARCADIANS', 'ARCHANGELS', 'ARKANSANS', 'ARCHIMEDES', 'ARCHITECT\'S', 'ARCHITECTS', 'ARCHITECTURE', 'ARCHITRAVES', 'ARCHIVISTS', 'ARCOSANTI', 'ARCHAEOLOGY', 'ARCHEOLOGY', 'ARCHIVISTS', 'ARCHIPELAGO', 'ARCHITECTURE\'S', 'ARCHITECTURES', 'ARKADELPHIA', 'ARCHITECTURAL', 'ARCHAEOLOGIST', 'ARCHAEOLOGICAL', 'ARCHEOLOGICAL', 'ARCHAEOLOGISTS', 'ARCHITECTURALLY', 'ARCHITECTURALLY']
  },
  {
    sequence: ['AH', 'L', 'IY'],
    endWords: ['ALLEE', 'AMALIE', 'ACTUALLY', 'ANOMALY', 'ACTUALLY', 'ANNUALLY', 'ANGRILY', 'ARTFULLY', 'ABYSMALLY', 'ADDITIONALLY', 'ABNORMALLY', 'ADDITIONALLY', 'ANECDOTALLY', 'ANECDOTALLY', 'ACCIDENTALLY', 'ARTIFICIALLY', 'ANENCEPHALY', 'ACCIDENTALLY', 'ARBITRARILY', 'AGRICULTURALLY', 'ARCHITECTURALLY', 'AGRICULTURALLY', 'ARCHITECTURALLY'],
    startWords: ['ALLEE', 'ALEEN', 'ALENE', 'ALEVE', 'ALIYAH', 'ALEDO', 'ALINA', 'ALETHA', 'ALISA', 'ALITO', 'ALENIA', 'ALERIA', 'ALLEVIATE', 'ALLEGIANCE', 'ALLEVIATES', 'ALLEVIATED', 'ALLEVIATED', 'ALLEVIATING', 'ALLEVIATION', 'ALLEGIANCES']
  },
  {
    sequence: ['L', 'ER'],
    endWords: ['AILOR', 'ALLER', 'ALLOR', 'ABLER', 'ADLER', 'AGLER', 'ABLER', 'AMBLER', 'ANDLER', 'ANGLER', 'AKSLER', 'AMSLER', 'ANTLER', 'ANNULAR', 'ALACHLOR', 'ALTSCHILLER', 'ALTSCHULER', 'ALTSHULER', 'ALVEOLAR', 'ANGULAR', 'ALTSCHULER', 'ALTSHULER', 'APPENZELLER'],
    startWords: []
  },
  {
    sequence: ['AA', 'AA'],
    endWords: [],
    startWords: []
  }
];

expectedWordsForSequences.forEach(runReverseMatchTest);

function runReverseMatchTest(pair) {
  test('Partial matching from end', function matchingFromEnd(t) {
    t.plan(6);

    createWordPhonemeMap(
      {
        dbLocation: dbLocation
      },
      useMap
    );

    function useMap(error, wordPhonemeMap) {
      t.ok(!error, 'No error while creating map.');

      var backwardWordsChecked = false;
      var forwardWordsChecked = false;

      wordPhonemeMap.wordsForPhonemeEndSequence(
        pair.sequence, checkWords
      );
      wordPhonemeMap.wordsForPhonemeStartSequence(
        pair.sequence, checkStartWords
      );

      function checkWords(error, words) {
        // console.log('words!', words);
        t.ok(!error, 'No error occured while looking for words.');
        t.deepEqual(words, pair.endWords, 'Expected words are returned.');

        backwardWordsChecked = true;
        closeIfChecksAreDone();
      }

      function checkStartWords(error, words) {
        // console.log('words!', words);
        t.ok(!error, 'No error occured while looking for words.');
        t.deepEqual(words, pair.startWords, 'Expected words are returned.');

        forwardWordsChecked = true;
        closeIfChecksAreDone();
      }

      function closeIfChecksAreDone() {
        if (backwardWordsChecked && forwardWordsChecked) {
          wordPhonemeMap.close(checkClose);
        }
      }
    }

    function checkClose(error) {
      t.ok(!error, 'Database closes successfully.');
    }
  });
}
