export-methods
==============

Creates an export object for an array of functions, using the functions' names as keys.

Why
---

Sometimes, you have a bunch of functions you want to export, either via `module.exports` or as a return value for a constructor function. This is what you end up writing:

    function getDoc() {
      ...
    }

    function getTree() {
      ...
    }

    function getNode() {
      ...
    }

    module.exports = {
      getDoc: getDoc,
      getTree: getTree,
      getNode: getNode
    };

Now, writing `getNode: getNode` won't kill you, but it does feel kinda stupid.

You can use this module to avoid that unsightly situation like so:

    var exportMethods = require('export-methods');
    ...
    module.exports = exportMethods(getDoc, getTree, getNode);

Or, if the situation favors it:

    module.exports = exportMethods([getDoc, getTree, getNode]);

Installation
------------

    npm install export-methods

Tests
-----

Run tests with `make test`.

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
