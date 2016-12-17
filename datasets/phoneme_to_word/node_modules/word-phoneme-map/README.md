word-phoneme-map
==================

Provides a two-way map between the words and phonemes listed in the [CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict). You can look up words that correspond to phoneme sequences and look up phoneme sequences that correspond to words.

Installation
------------

    npm install word-phoneme-map

Usage
-----

Sometime before you use the map, you need to call `setUpDatabase` to set up a database, like so:

    var setUpDatabase = require('word-phoneme-map').setUpDatabase;
    setUpDatabase(
      {
        dbLocation: __dirname + '/where-you-want-it/a-word-phoneme.db'
      },
      done
    );

    function done(error) {
      if (error) {
        console.log(error)
      }
      else {
        console.log('Successfully set up database.');
      }
    }

You only need to do this once.

From then on, you can use the map like this:

    var createWordPhonemeMap = require('word-phoneme-map').createWordPhonemeMap;
    var wordPhonemeMap = createWordPhonemeMap({
      dbLocation: __dirname + '/where-you-want-it-/a-word-phoneme.db'
    });

    wordPhonemeMap.wordsForPhonemeSequence(['AA', 'R', 'K'], showWords);

    function showWords(error, words) {
      if (error) {
        console.log(error);
      }
      else {
        console.log(words);
      }
    }

Output:

    [
      'ARC',
      'ARK'
    ]

To do lookups in the opposite direction, do this:

    wordPhonemeMap.phonemeSequencesForWord('ARK', showPhonemes);

    function showPhonemes(error, phonemeSequences) {
      if (error) {
        console.log(error);
      }
      else {
        console.log(phonemeSequences);
      }
    }

Output:

    [
      ['AA', 'R', 'K']
    ]

Tests
-----

Run tests with `make test`.

About CMUdict (the Carnegie Mellon Pronouncing Dictionary)
----------------------------------------------------------

"[It is a free pronouncing dictionary of English](http://www.speech.cs.cmu.edu/cgi-bin/cmudict), suitable for uses in speech
technology and is maintained by the Speech Group in the School of
Computer Science at Carnegie Mellon University."

"The Carnegie Mellon Pronouncing Dictionary, in its current and
previous versions is Copyright (C) 1993-2014 by Carnegie Mellon
University.  Use of this dictionary for any research or commercial
purpose is completely unrestricted.  If you make use of or
redistribute this material we request that you acknowledge its
origin in your descriptions."

License
-------

The MIT License (MIT)

Copyright (c) 2015 Jim Kang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
