var assert = require('assert');
var phonemeTypes = require('../phoneme-types');

suite('Phoneme navigator', function navigatorSuite() {
  suite('Phoneme classifcation', function classificationSuite() {
    function checkClassifyPhoneme(phoneme, expectedClassification) {
      assert.equal(phonemeTypes.classifyPhoneme(phoneme), 
        expectedClassification
      );
    }

    test('Verify that that phonemes are identified as vowels', 
      function vowelTest() {
        checkClassifyPhoneme('AA', 'vowel');
        checkClassifyPhoneme('AE', 'vowel');
        checkClassifyPhoneme('AH', 'vowel');
        checkClassifyPhoneme('AO', 'vowel');
        checkClassifyPhoneme('AW', 'vowel');
        checkClassifyPhoneme('EY', 'vowel');
        checkClassifyPhoneme('IY', 'vowel');
        checkClassifyPhoneme('UH', 'vowel');
        checkClassifyPhoneme('UW', 'vowel');
      }
    );

    test('Verify that that phonemes are identified as affricates', 
      function affricateTest() {
        checkClassifyPhoneme('CH', 'affricate');
        checkClassifyPhoneme('JH', 'affricate');
      }
    );

    test('Verify that that phonemes are identified as aspirates', 
      function aspirateTest() {
        checkClassifyPhoneme('HH', 'aspirate');
      }
    );

    test('Verify that that phonemes are identified as fricatives', 
      function fricativeTest() {
        checkClassifyPhoneme('DH', 'fricative');
        checkClassifyPhoneme('F', 'fricative');
        checkClassifyPhoneme('S', 'fricative');
        checkClassifyPhoneme('SH', 'fricative');
        checkClassifyPhoneme('TH', 'fricative');
        checkClassifyPhoneme('V', 'fricative');
        checkClassifyPhoneme('Z', 'fricative');
        checkClassifyPhoneme('ZH', 'fricative');
      }
    );

    test('Verify that that phonemes are identified as liquids', 
      function liquidTest() {
        checkClassifyPhoneme('L', 'liquid');
        checkClassifyPhoneme('R', 'liquid');
      }
    );

    test('Verify that that phonemes are identified as nasals', 
      function nasalTest() {
        checkClassifyPhoneme('M', 'nasal');
        checkClassifyPhoneme('N', 'nasal');
        checkClassifyPhoneme('NG', 'nasal');
      }
    );

    test('Verify that that phonemes are identified as semivowels', 
      function semivowelTest() {
        checkClassifyPhoneme('W', 'semivowel');
        checkClassifyPhoneme('Y', 'semivowel');
      }
    );

    test('Verify that that phonemes are identified as stops', 
      function stopTest() {
        checkClassifyPhoneme('B', 'stop');
        checkClassifyPhoneme('D', 'stop');
        checkClassifyPhoneme('G', 'stop');
        checkClassifyPhoneme('K', 'stop');
        checkClassifyPhoneme('P', 'stop');
        checkClassifyPhoneme('T', 'stop');
      }
    );
  });

  suite('Related phonemes', function relatedSuite() {
    test('Verify that related phonemes are found for a vowel', 
      function testRelatedVowels() {
        assert.deepEqual(
          phonemeTypes.getPhonemesInSameClass('AY'),
          [
            'AA',
            'AE',
            'AH',
            'AO',
            'AW',
            'EH',
            'ER',
            'EY',
            'IH',
            'IY',
            'OW',
            'OY',
            'UH',
            'UW'
          ]
        );
      }
    );

    test('Verify that related phonemes are found for an affricate', 
      function testRelatedAffricates() {
        assert.deepEqual(
          phonemeTypes.getPhonemesInSameClass('JH'),
          ['CH']
        );
      }
    );

    test('Verify that related phonemes are found for an aspirate', 
      function testRelated() {
        assert.deepEqual(
          phonemeTypes.getPhonemesInSameClass('HH'),
          []
        );
      }
    );

    test('Verify that related phonemes are found for a fricative', 
      function testRelatedFricatives() {
        assert.deepEqual(
          phonemeTypes.getPhonemesInSameClass('V'),
          [
            'DH',
            'F',
            'S',
            'SH',
            'TH',
            'Z',
            'ZH'
          ]
        );
      }
    );

    test('Verify that related phonemes are found for a liquid', 
      function testRelatedLiquid() {
        assert.deepEqual(
          phonemeTypes.getPhonemesInSameClass('L'),
          ['R']
        );
      }
    );

    test('Verify that related phonemes are found for a nasal', 
      function testRelatedNasals() {
        assert.deepEqual(
          phonemeTypes.getPhonemesInSameClass('N'),
          ['M', 'NG']
        );
      }
    );

    test('Verify that related phonemes are found for a semivowel', 
      function testRelatedSemivowels() {
        assert.deepEqual(
          phonemeTypes.getPhonemesInSameClass('Y'),
          ['W']
        );
      }
    );

    test('Verify that related phonemes are found for a stop', 
      function testRelatedStops() {
        assert.deepEqual(
          phonemeTypes.getPhonemesInSameClass('T'),
          [
            'B',
            'D',
            'G',
            'K',
            'P'
          ]
        );
      }
    );
  });

});
