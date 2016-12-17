phoneme-types
==================

Classifies [CMU/ARPAbet-style phonemes](http://www.speech.cs.cmu.edu/cgi-bin/cmudict#phones).

Installation
------------

    npm install phoneme-types

Usage
-----

    var phonemeTypes = require('phoneme-types');
    phonemeTypes.isConsonantish('K');
    // true

Other methods:

- classifyPhoneme: Tells you if a phoneme is one of the following:
  - vowel
  - affricate
  - aspirate
  - fricative
  - liquid
  - nasal
  - semivowel
  - stop
- getPhonemesInSameClass: Given a phoneme, returns all of the other phonemes of the same class.

Tests
-----

Run tests with `make test`.

License
-------

MIT.
