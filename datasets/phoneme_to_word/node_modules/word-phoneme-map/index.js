var indexWordsAndPhonemes = require('./index-words-and-phonemes');
var createWordPhonemeMap = require('./word-phoneme-map');

module.exports = {
  setUpDatabase: indexWordsAndPhonemes,
  createWordPhonemeMap: createWordPhonemeMap
};
