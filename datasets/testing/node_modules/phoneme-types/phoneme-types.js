// This is a singleton-style module.
var _ = require('lodash');

var phonemeListsByClassification = {
  vowel: [
    'AA',
    'AE',
    'AH',
    'AO',
    'AW',
    'AY',
    'EH',
    'ER',
    'EY',
    'IH',
    'IY',
    'OW',
    'OY',
    'UH',
    'UW'
  ],
  affricate: [
    'CH',
    'JH'
  ],
  aspirate: [
    'HH'
  ],
  fricative: [
    'DH',
    'F',
    'S',
    'SH',
    'TH',
    'V',
    'Z',
    'ZH'
  ],
  liquid: [
    'L',
    'R'
  ],
  nasal: [
    'M',
    'N',
    'NG'
  ],
  semivowel: [
    'W',
    'Y'
  ],
  stop: [
    'B',
    'D',
    'G',
    'K',
    'P',
    'T'
  ]
};

var classificationsByPhoneme = {
  AA: 'vowel',
  AE: 'vowel',
  AH: 'vowel',
  AO: 'vowel',
  AW: 'vowel',
  AY: 'vowel',
  EH: 'vowel',
  ER: 'vowel',
  EY: 'vowel',
  IH: 'vowel',
  IY: 'vowel',
  OW: 'vowel',
  OY: 'vowel',
  UH: 'vowel',
  UW: 'vowel',
  CH: 'affricate',
  JH:'affricate',
  HH: 'aspirate',
  DH: 'fricative',
  F: 'fricative',
  S: 'fricative',
  SH: 'fricative',
  TH: 'fricative',
  V: 'fricative',
  Z: 'fricative',
  ZH: 'fricative',
  L: 'liquid',
  R: 'liquid',
  M: 'nasal',
  N: 'nasal',
  NG: 'nasal',
  W: 'semivowel',
  Y: 'semivowel',
  B: 'stop',
  D: 'stop',
  G: 'stop',
  K: 'stop',
  P: 'stop',
  T: 'stop'
};

// Phonemes that cannot be followed within a syllable.
var syllableEndingPhonemes = [
  'CH',
  'JH',
  'DH',
  'SH',
  'TH',
  'V',
  'Z',
  'ZH',
  'M',
  'NG',
  'B',
  'D',
  'G',
  'K',
  'T'
];

function classifyPhoneme(phoneme) {
  return (phoneme in classificationsByPhoneme) ? 
    classificationsByPhoneme[phoneme] : null;
}

function getPhonemesInSameClass(phoneme) {
  var fellows = [];
  var phonemeClass = classifyPhoneme(phoneme);
  if (phonemeClass) {
    fellows = _.without(phonemeListsByClassification[phonemeClass], phoneme);
  }
  return fellows;
}

function stripStressor(phonemeWithStressor) {
  return phonemeWithStressor.replace(/\d/g, '');
}

function isConsonantish(phoneme) {
  return classificationsByPhoneme[phoneme] !== 'vowel';
}

function isVowelish(phoneme) {
  return classificationsByPhoneme[phoneme] === 'vowel';
}

function isSyllableEnder(phoneme) {
  return syllableEndingPhonemes.indexOf(phoneme) !== -1;
}

module.exports = {
  classifyPhoneme: classifyPhoneme,
  stripStressor: stripStressor,
  getPhonemesInSameClass: getPhonemesInSameClass,
  isConsonantish: isConsonantish,
  isVowelish: isVowelish,
  isSyllableEnder: isSyllableEnder
};
