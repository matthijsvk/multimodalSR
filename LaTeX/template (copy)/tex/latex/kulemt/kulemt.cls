%%
%% This is file `kulemt.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% kulemt-src.dtx  (with options: `class')
%% 
%% This is a generated file.
%% 
%% Copyright (C) 2012 by Luc Van Eycken <Luc.VanEycken@esat.kuleuven.be>
%% 
%% This file may be distributed and/or modified under the conditions of
%% the LaTeX Project Public License, either version 1.2 of this license
%% or (at your option) any later version.  The latest version of this
%% license is in:
%% 
%%    http://www.latex-project.org/lppl.txt
%% 
%% and version 1.2 or later is part of all distributions of LaTeX version
%% 1999/12/01 or later.
%% 
\NeedsTeXFormat{LaTeX2e}[2001/06/01]
\ProvidesClass{kulemt}
    [2015/05/25 v1.8
     KU Leuven engineering master thesis document class]
\def\kulemt@cls{kulemt}
\def\kulemt@error#1{%
  \ClassError\kulemt@cls{#1}{Exit, correct this error and rerun.}}
\def\kulemt@opt@missingpkg#1#2{\kulemt@error{%
    The option `#1' is ignored because\MessageBreak
    it requires the installation of the package `#2'}}
\def\kulemt@ifdutch{\protect\iflanguage{dutch}}
\@ifundefined{l@dutch}{\kulemt@error{%
    The language `dutch' is not available.\MessageBreak
    You must install Dutch hyphenation patterns}}{}
\def\kulemt@ifand#1{\expandafter\kulemt@ifand@ #1\and\@nil}
\def\kulemt@ifand@#1\and #2\@nil{%
  \if=\noexpand#2=\expandafter\@secondoftwo\else
    \expandafter\@firstoftwo\fi}
\RequirePackage{keyval}
\newcommand*\setup{%
  \kulemt@catcode@setup
  \kulemt@setup}
\@onlypreamble\setup
\def\kulemt@setup#1{%
  \setkeys{kulemt}{#1}%
  \kulemt@uncatcode@setup
  \kulemt@do@once@opts}
\def\kulemt@catcode@setup{%
  \csname extras\languagename\endcsname
  \kulemt@ifdutch{\catcode`\"\active}{}}
\def\kulemt@uncatcode@setup{%
  \csname noextras\languagename\endcsname
  \@makeother\"}
\def\kulemt@invalidate@key#1#2{%
  \define@key{kulemt}{#1}{\ClassWarning\kulemt@cls{%
      The key `#1' can only be used #2.\MessageBreak
      It is ignored}}}
\def\kulemt@keynovalue#1#2{%
  \define@key{kulemt}{#1}[]{%
    \def\@tempa{##1}\ifx\@tempa\@empty\else
      \PackageWarningNoLine\kulemt@cls{Value of option `#1' ignored}\fi
    #2}}
\def\kulemt@clskeys{}
\def\kulemt@clskey#1{%
  \edef\kulemt@clskeys{\kulemt@clskeys,#1}%
  \define@key{kulemt}{#1}}
\def\kulemt@clsopt#1{%
  \edef\kulemt@clskeys{\kulemt@clskeys,#1}%
  \kulemt@keynovalue{#1}}
\def\kulemt@process@ptions{%
  \@ifundefined{opt@\@currname.\@currext}{}%
   {\begingroup
    \toks@\expandafter\expandafter\expandafter{%
      \csname opt@\@currname.\@currext\endcsname}%
    \edef\CurrentOption{\the\toks@}%
    \toks@{}%
    \@for\CurrentOption:=\CurrentOption\do{%
      \@ifundefined{%
        KV@kulemt@\expandafter\kulemt@getkey\CurrentOption=\@nil}%
       {\ifx\@unusedoptionlist\@empty
          \global\let\@unusedoptionlist\CurrentOption
        \else
          \expandafter\expandafter\expandafter\gdef
          \expandafter\expandafter\expandafter\@unusedoptionlist
          \expandafter\expandafter\expandafter{%
            \expandafter\@unusedoptionlist
            \expandafter,\CurrentOption}%
        \fi}%
       {\toks@\expandafter{%
          \the\expandafter\toks@\expandafter,\CurrentOption}%
        \expandafter\kulemt@update@classoptions\CurrentOption=aa=\@nil
       }}%
    \edef\@tempa{\endgroup
      \noexpand\setkeys{kulemt}{\the\toks@}}%
    \@tempa
    \let\CurrentOption\@empty}%
  \@for\@tempa:=\kulemt@clskeys\do{%
    \expandafter\kulemt@invalidate@key\expandafter{\@tempa}{%
      as a class option}}}
\def\kulemt@getkey#1=#2\@nil{#1}
\def\kulemt@update@classoptions#1=#2#3=#4\@nil{%
  \def\@tempa{#3}\ifx\@tempa\@empty
    \def\@tempa##1,#1=##2,##3\@nil{##1,##3\@nil}%
    \def\@tempb,##1,\@nil{##1}%
    \xdef\@classoptionslist{\expandafter\expandafter\expandafter\@tempb
      \expandafter\@tempa\expandafter,\@classoptionslist,\@nil}%
  \fi}
\kulemt@clskey{master}{\lowercase{\edef\kulemt@opt@master{#1}}}
\def\kulemt@opt@master{}
\kulemt@clsopt{10pt}{\def\kulemt@ptsize{0}}
\kulemt@clsopt{11pt}{\def\kulemt@ptsize{1}}
\def\kulemt@ptsize{1}
\kulemt@clsopt{openright}{\PassOptionsToClass{open}{memoir}}
\kulemt@clsopt{openany}{\PassOptionsToClass{openany}{memoir}}
\kulemt@clsopt{openleft}{\PassOptionsToClass{openleft}{memoir}}
\kulemt@clsopt{oneside}{\PassOptionsToClass{oneside}{memoir}}
\kulemt@clsopt{twoside}{\PassOptionsToClass{twoside}{memoir}}
\kulemt@clskey{bind}{\setlength\kulemt@bind{#1}}
\newdimen\kulemt@bind
\kulemt@bind\z@
\kulemt@clsopt{draft}{\PassOptionsToClass{draft}{memoir}}
\def\kulemt@babel@opt{english,dutch}
\kulemt@clsopt{dutch}{\def\kulemt@language{dutch}}
\kulemt@clsopt{english}{\def\kulemt@language{english}}
\def\kulemt@language{\kulemt@master@language}
\kulemt@clskey{extralanguage}{%
  \edef\@tempa{{,#1,}{,\kulemt@babel@opt,}}\expandafter\in@\@tempa
  \ifin@\else \edef\kulemt@babel@opt{#1,\kulemt@babel@opt}\fi}
\kulemt@clsopt{nomicrotype}{\kulemt@microtypefalse}
\newif\ifkulemt@microtype \kulemt@microtypetrue
\kulemt@clsopt{fleqn}{\PassOptionsToClass{fleqn}{memoir}}
\gdef\kulemt@do@once@opts{}
\def\kulemt@add@once@opts#1#2#3{\g@addto@macro\kulemt@do@once@opts{%
    \ifx #2\@empty\else
      \kulemt@invalidate@key{#1}{once}%
      \def\@tempa{#3\let#2\@empty}%
      \expandafter\@tempa
    \fi}}
\define@key{kulemt}{masteroption}{\def\kulemt@opt@masteroption{#1}}
\def\kulemt@opt@masteroption{}
\def\kulemt@master@option{}
\kulemt@add@once@opts{masteroption}\kulemt@opt@masteroption{%
  \global\let\kulemt@master@option\@empty
  \ifx\kulemt@master@options@vl\relax
    \ifx\kulemt@opt@masteroption\@empty\else
      \kulemt@error{%
        The option `masteroption' is ignored because\MessageBreak
        your program disallows a master option on front pages}%
    \fi
  \else
    \@for\@tempa:=\kulemt@opt@masteroption\do{%
      \let\@tempc\@tempa
      \expandafter\kulemt@handle@mo\expandafter\kulemt@master@options
      \expandafter{%
        \expandafter\kulemt@add@mo@tempc\expandafter{\@tempa}}%
      \expandafter\g@addto@macro\expandafter\kulemt@master@option
        \expandafter{\@tempc,}}%
  \fi}
\def\kulemt@add@mo@tempc#1#2#3{%
  \lowercase{\def\@tempa{#1}}\def\@tempb{#2}\ifx\@tempa\@tempb
    \def\@tempc{#3}\fi}
\def\kulemt@handle@mo#1#2{%
  \expandafter\@tfor\expandafter\@tempd\expandafter:\expandafter=#1\do{%
    \expandafter\kulemt@handle@mo@\@tempd\@nil{#2}}}
\def\kulemt@handle@mo@ #1:#2\@nil #3{#3{#1}{#2}}
\RequirePackage[ascii]{inputenc}
\define@key{kulemt}{inputenc}{\def\kulemt@opt@inputenc{#1}}
\def\kulemt@opt@inputenc{}
\kulemt@add@once@opts{inputenc}\kulemt@opt@inputenc{%
  \inputencoding\kulemt@opt@inputenc}
\define@key{kulemt}{font}{%
  \edef\@tempa{\zap@space#1 \@empty}%
  \expandafter\kulemt@set@font\@tempa::\@empty}
\def\kulemt@set@font#1:#2:#3\@empty{%
  \expandafter\let\expandafter\@tempa
    \csname kulemt@set@font@#1\endcsname
  \ifx\@tempa\relax
    \kulemt@error{Font `#1' is ignored because it is an unknown font}%
  \else \@tempa{#2}\fi}
\let\kulemt@loadfont@\@empty
\kulemt@add@once@opts{font}\kulemt@loadfont@{%
  \kulemt@loadfont@ \kulemt@loadmt}
\def\kulemt@set@font@cm#1{\let\kulemt@loadfont@\relax}
\def\kulemt@set@font@lm#1{%
  \IfFileExists{lmodern.sty}%
   {\def\kulemt@loadfont@{%
      \RequirePackage[T1]{fontenc}%
      \RequirePackage{lmodern}}}%
   {\kulemt@opt@missingpkg{font=lm}{lm}}}
\def\kulemt@set@font@palatino#1{%
  \def\kulemt@loadfont@{%
    \RequirePackage[T1]{fontenc}%
    \RequirePackage[#1]{mathpazo}%
    \RequirePackage[scaled=.9]{helvet}}%
  \IfFileExists{lmodern.sty}{%
    \g@addto@macro\kulemt@loadfont@{%
      \renewcommand{\ttdefault}{lmtt}}}{}}
\def\kulemt@set@font@times#1{%
  \def\kulemt@loadfont@{%
    \RequirePackage[T1]{fontenc}%
    \RequirePackage[#1]{mathptmx}%
    \RequirePackage[scaled=.9]{helvet}}%
  \IfFileExists{lmodern.sty}{%
    \g@addto@macro\kulemt@loadfont@{%
      \renewcommand{\ttdefault}{lmtt}}}{}}
\def\kulemt@set@font@utopia#1{%
  \IfFileExists{fourier.sty}%
    {\def\kulemt@loadfont@{%
       \RequirePackage[T1]{fontenc}%
       \RequirePackage[#1]{fourier}%
       \RequirePackage[scaled=.9]{helvet}}%
     \IfFileExists{lmodern.sty}{%
       \g@addto@macro\kulemt@loadfont@{%
         \renewcommand{\ttdefault}{lmtt}}}{}}%
    {\kulemt@opt@missingpkg{font=utopia}{fourier}}}
\define@key{kulemt}{title}{\title{#1}}
\def\@title{\kulemt@error{No title given}}
\define@key{kulemt}{subtitle}{\def\kulemt@subtitle{#1}}
\def\kulemt@subtitle{}
\define@key{kulemt}{author}{\author{#1}}
\def\@author{\kulemt@error{No author given}}
\define@key{kulemt}{promotor}{%
  \def\@tempa{#1}\ifx\@tempa\@empty\else \def\kulemt@promotor{#1}\fi}
\let\KV@kulemt@promoter\KV@kulemt@promotor
\def\kulemt@promotor{\kulemt@error{No promoter given}}
\define@key{kulemt}{assessor}{\def\kulemt@assessor{#1}}
\def\kulemt@assessor{\kulemt@error{No assessor given}}
\define@key{kulemt}{assistant}{\def\kulemt@assistant{#1}}
\def\kulemt@assistant{\kulemt@error{No assistant given}}
\define@key{kulemt}{acyear}{\def\kulemt@acyear{#1}}
\def\kulemt@acyear{}
\def\kulemt@acyear@t{%
  \ifx\kulemt@acyear\@empty
    \@tempcnta\year \ifnum\month<10 \advance\@tempcnta\m@ne \fi
    \@tempcntb\@tempcnta \advance\@tempcntb\@ne
    \xdef\kulemt@acyear{\the\@tempcnta\space -- \the\@tempcntb}%
  \fi
  \kulemt@acyear}
\define@key{kulemt}{translatedtitle}{\def\kulemt@translatedtitle{#1}}
\def\kulemt@translatedtitle{%
  \kulemt@error{No translated title available}}
\define@key{kulemt}{shortabstract}{\def\kulemt@shortabstract{#1}}
\def\kulemt@shortabstract{%
  \kulemt@error{No short abstract available}}
\define@key{kulemt}{udc}{\def\kulemt@udc{#1}}
\def\kulemt@udc{%
  \kulemt@error{UDC number missing}}
\define@key{kulemt}{keywords}{\def\kulemt@keywords{#1}}
\def\kulemt@keywords{}
\define@key{kulemt}{articletitle}{\def\kulemt@article@title{#1}}
\def\kulemt@article@title{}
\kulemt@keynovalue{coverpageonly}{\kulemt@coverpagetrue}
\newif\ifkulemt@coverpage
\kulemt@keynovalue{frontpagesonly}{\kulemt@frontpagestrue}
\newif\ifkulemt@frontpages
\kulemt@keynovalue{filingcard}{\kulemt@filingcardtrue}
\newif\ifkulemt@filingcard
\define@key{kulemt}{bindcover}{\ClassWarningNoLine\kulemt@cls{%
    Option `bindcover' is no longer supported}}
\def\kulemt@def@master#1{\@namedef{kulemt@m@#1}}
\newcommand*\kulemt@obsolete@master[2][]{%
  \@namedef{kulemt@m@#2\if &#1&\else .#1\fi}}
\def\kulemt@set@master#1#2#3#4#5#6#7#8#9\@nil{%
  \edef\kulemt@master@language{%
    \if N\@car#1\@nil dutch\else english\fi}%
  \kulemt@facnum=#2\relax
  \if F\@car#3\@nil \kulemt@filingcardtrue \fi
  \def\kulemt@master@colors{#4}%
  \def\kulemt@master@title{#5}%
  \def\kulemt@copyright@contact{#6}%
  \ifx\kulemt@copyright@contact\@empty
    \protected@edef\kulemt@copyright@contact{\kulemt@fac@copyright}%
  \fi
  \@ifnextchar[\kulemt@set@mo{\kulemt@set@mo[]}#7\@nil{#8}}
\def\kulemt@set@mo[#1]#2\@nil#3{%
  \def\kulemt@master@options@vl{#1}%
  \def\kulemt@master@options@v{-}%
  \ifx\kulemt@master@options@vl\kulemt@master@options@v
    \let\kulemt@master@options@vl\relax
    \let\kulemt@master@options@v\@empty
    \let\kulemt@master@options@o\@empty
    \let\kulemt@master@options\@empty
  \else
    \def\kulemt@master@options@v{#2}%
    \def\kulemt@master@options@o{#3}%
    \def\kulemt@master@options{#2#3}%
  \fi}
\def\kulemt@getcolors#1:#2:#3\@nil{%
  \edef\kulemt@color@bg{\if !#1!{white}\else [cmyk]{#1}\fi}%
  \edef\kulemt@color@fg{\if !#2!{black}\else [cmyk]{#2}\fi}}
\let\kulemt@div@master\@gobble
\let\kulemt@end@master@def\relax
\newcount\kulemt@facnum
\kulemt@facnum\@ne
\def\kulemt@fac@name{%
  \ifcase\kulemt@facnum \or
    Facult\kulemt@ifdutch{eit Ingenieurswetenschappen}%
                         {y of Engineering}%
  \fi}
\def\kulemt@kulfac@logo#1{%
  \edef\@tempa{[#1]{logokul\ifnum\kulemt@facnum=\@ne eng\fi}}%
  \expandafter\includegraphics\@tempa}
\def\kulemt@fac@logo#1{}
\def\kulemt@fac@copyright{%
  \ifnum\kulemt@facnum >\@ne
    \kulemt@ifdutch d{th}e \MakeLowercase{\noexpand\kulemt@paa@0}::%
  \else
    Faculteit Ingenieurswetenschappen, Kasteelpark Arenberg 1 bus 2200,
    B-3001 Heverlee:+32-16-321350:%
  \fi}
\def\kulemt@paa@#1{%
  \ifcase #1%
    Promot\kulemt@ifand\kulemt@promotor{\kulemt@ifdutch{oren}{ers}}{}%
  \or
    Assessor\kulemt@ifand\kulemt@assessor{\kulemt@ifdutch{en}s}{}%
  \or
    \kulemt@ifdutch{Begeleider}{Assistant}%
    \kulemt@ifand\kulemt@assistant{s}{}%
  \fi}
\providecommand*\kulemt@cfgfile{kulemt.cfg}
\ifx\kulemt@cfgfile\@empty\else
  \InputIfFileExists\kulemt@cfgfile{}{\kulemt@error{%
      Configuration file `\kulemt@cfgfile' is not installed}}
\fi
\kulemt@process@ptions
\ifx\kulemt@opt@master\@empty
  \kulemt@error{Required option `master' missing}\fi
\expandafter\let\expandafter\@tempa
  \csname kulemt@m@\kulemt@opt@master \endcsname
\ifx\@tempa\relax
  \kulemt@error{%
    Unsupported value `\kulemt@opt@master' for option `master'}\fi
\expandafter\kulemt@set@master\@tempa{}{}{}\@nil
\@ifundefined{kulemt@cfgfile@m}{%
  \edef\kulemt@cfgfile@m{kulemt-\kulemt@opt@master.cfg}}{}
\ifx\kulemt@cfgfile@m\@empty\else
  \InputIfFileExists\kulemt@cfgfile@m{}{}
\fi
\ifx\kulemt@master@options@vl\@empty\else
  \ifx\kulemt@master@options@vl\relax\else
    \def\kulemt@check@masteroption{%
      \ifx\kulemt@master@option\@empty
        \kulemt@error{%
          When using option `master=\kulemt@opt@master',\MessageBreak
          you should specify at least one master option.\MessageBreak
          Allowed master options are: \kulemt@master@options@vl}%
      \fi}
    \AtBeginDocument{\kulemt@check@masteroption}
  \fi
\fi
\LoadClass[a4paper,1\kulemt@ptsize pt]{memoir}[2004/04/05]
\def\and{\\}
\let\andnext\and
\RequirePackage[\kulemt@babel@opt]{babel}
\addto\captionsenglish{%
  \def\appendixtocname{Appendices}%
  \def\appendixpagename{Appendices}%
  \def\figurerefname{Figure}%
  \def\tablerefname{Table}%
  \def\pagerefname{page}%
  \def\partrefname{Part~}%
  \def\chapterrefname{Chapter~}%
  \def\listfiguresandtablesname{List of Figures and Tables}}
\begingroup
  \catcode`\"\active
  \@firstofone{\endgroup
    \addto\captionsdutch{%
      \def\appendixtocname{B"ylagen}%
      \def\appendixpagename{B"ylagen}%
      \def\figurerefname{figuur}%
      \def\tablerefname{tabel}%
      \def\pagerefname{pagina}%
      \def\partrefname{Deel~}%
      \def\chapterrefname{hoofdstuk~}%
      \def\listfiguresandtablesname{L"yst van figuren en tabellen}}}
\AtBeginDocument{\gdef\latinencoding{T1}%
  \ifx\cf@encoding\latinencoding\else \xdef\latinencoding{OT1}\fi}
\edef\kulemt@language{\kulemt@language}
\expandafter\main@language\expandafter{\kulemt@language}
\def\kulemt@selectmasterlanguage{%
  \expandafter\selectlanguage\expandafter{\kulemt@master@language}}
\def\kulemt@selecttextlanguage{%
  \expandafter\selectlanguage\expandafter{\kulemt@language}}
\RequirePackage{graphicx}
\RequirePackage{color}
\def\kulemt@loadmt{%
  \@ifpackageloaded{microtype}{}{\RequirePackage{microtype}}}
\AtBeginDocument{\kulemt@loadmt}
\ifkulemt@microtype
  \ifpdf
    \IfFileExists{microtype.sty}{}{%
      \def\@tempa{the package is not installed}%
      \kulemt@microtypefalse}
  \else
    \def\@tempa{you're not using pdflatex in pdf mode}
    \kulemt@microtypefalse
  \fi
\else
  \def\@tempa{option `nomicrotype' was used}
\fi
\ifkulemt@microtype\else
  \let\kulemt@loadmt\relax
  \ClassWarningNoLine\kulemt@cls{%
    Package `microtype' not used because\MessageBreak
    \@tempa}%
\fi
\def\kulemt@check@hyperref{%
  \@ifpackageloaded{hyperref}{%
    \@ifundefined{HyPsd@babel@dutch}{}{%
      \addto\HyPsd@babel@dutch{%
        \declare@shorthand{dutch}{"y}{ij}%
        \declare@shorthand{dutch}{"Y}{IJ}}}%
    \@ifpackageloaded{memhfixc}{}{%
      \RequirePackage{memhfixc}}%
    \providecommand*\theHsubfigure{\theHfigure.\arabic{subfigure}}%
    \providecommand*\theHsubtable{\theHtable.\arabic{subtable}}%
  }{}}
\AtBeginDocument{\kulemt@check@hyperref}
\RequirePackage[T1,OT1]{fontenc}
\kulemt@do@once@opts
\ifcase\kulemt@ptsize\relax
  \textwidth=13cm
  \textheight=20cm
\or
  \textwidth=14cm
  \textheight=215mm
\fi
\spinemargin\paperwidth
\advance\spinemargin -\textwidth
\foremargin\spinemargin
\advance\foremargin -\kulemt@bind\relax
\foremargin .\if@twoside 6\else 5\fi\foremargin
\advance\spinemargin -\foremargin
\marginparwidth=56pt
\marginparsep=1.2\onelineskip
\marginparpush=\onelineskip
\setulmargins{*}{*}{1.2}
\setheaderspaces{*}{\headsep}{*}
\checkthelayout\fixthelayout
\AtBeginDocument{\checkandfixthelayout}
\renewcommand*\clearforchapter{%
  \if@mainmatter
    \if@openleft
      \cleartoverso
    \else
      \if@openright
        \cleartorecto
      \else
        \clearpage
      \fi
    \fi
  \else
    \clearpage
  \fi}
\makeoddhead{ruled}{}{}{%
  \if@mainmatter \rightmark \else \scshape\leftmark \fi}
\pagestyle{ruled}
\makepagestyle{nohead}
\makeevenfoot{nohead}{\thepage}{}{}
\makeoddfoot{nohead}{}{}{\thepage}
\aliaspagestyle{chapter}{nohead}
\maxsecnumdepth{subsection}
\maxtocdepth{section}
\def\tocheadstart{\clearforchapter\chapterheadstart}
\def\lofheadstart{\clearforchapter\chapterheadstart}
\def\lotheadstart{\clearforchapter\chapterheadstart}
\newcommand*\listoffiguresandtables{%
  \chapter\listfiguresandtablesname
  \def\@lofmaketitle{\section*\listfigurename}%
  \listoffigures*%
  \let\listoffigures\relax
  \def\@lotmaketitle{\section*\listtablename}%
  \listoftables*%
  \let\listoftables\relax}
\newcommand*\listfiguresandtablesname{List of Figures and Tables}
\setrmarg{2.55em plus1fil}
\setlength{\cftbeforechapterskip}{1ex \@plus\p@}
\captionnamefont{\scshape}
\captiontitlefont{\upshape}
\captionstyle[\centering]{\centerlastline}
\DeclareFontFamily{T1}{kulemtfpf}{}
\DeclareFontShape{T1}{kulemtfpf}{m}{n}{<->phvr8t}{}
\DeclareFontShape{T1}{kulemtfpf}{m}{sc}{<->phvrc8t}{}
\DeclareFontShape{T1}{kulemtfpf}{m}{sl}{<->phvro8t}{}
\DeclareFontShape{T1}{kulemtfpf}{m}{it}{<->ssub*kulemtfpf/m/sl}{}
\DeclareFontShape{T1}{kulemtfpf}{bx}{n}{<->phvb8t}{}
\DeclareFontShape{T1}{kulemtfpf}{bx}{sc}{<->phvbc8t}{}
\DeclareFontShape{T1}{kulemtfpf}{bx}{sl}{<->phvbo8t}{}
\DeclareFontShape{T1}{kulemtfpf}{bx}{it}{<->ssub*kulemtfpf/bx/it}{}
\DeclareFontShape{T1}{kulemtfpf}{b}{n}{<->ssub*kulemtfpf/bx/n}{}
\DeclareFontShape{T1}{kulemtfpf}{b}{sc}{<->ssub*kulemtfpf/bx/sc}{}
\DeclareFontShape{T1}{kulemtfpf}{b}{sl}{<->ssub*kulemtfpf/bx/sl}{}
\DeclareFontShape{T1}{kulemtfpf}{b}{it}{<->ssub*kulemtfpf/bx/sl}{}
\def\kulemt@fpf@title{\fontsize\@xxvpt{30}\selectfont}
\def\kulemt@fpf@subtitle{\fontsize\@xviipt{22}\selectfont}
\def\kulemt@fpf@author{\fontsize\@xivpt{18}\selectfont}
\def\kulemt@fpf@txthead{\fontsize\@xiipt{14.5}%
  \fontseries\bfdefault\selectfont}
\def\kulemt@fpf@text{\fontsize\@xiipt{14}\selectfont}
\def\kulemt@fpf@banner{\fontsize\@xivpt{18}%
  \fontseries\bfdefault\selectfont}
\def\kulemt@master@text{Thesis
  \kulemt@ifdutch
    {voorgedragen tot het behalen van de graad van}%
    {submitted for the degree of}
  \kulemt@master@title
  \ifx\kulemt@master@option\@empty\else
    \def\@tempb{, }%
    \@for\@tempa:=\kulemt@master@option\do{%
      \ifx\@tempa\@empty\else
        \@tempb \def\@tempb{ \kulemt@ifdutch{en}{and} }%
        \@tempa
      \fi}%
  \fi}
\def\kulemt@paa@fp#1{%
  \begingroup
    \ifcase #1\relax
      \let\@tempa\kulemt@promotor
    \or
      \let\@tempa\kulemt@assessor
    \or
      \let\@tempa\kulemt@assistant
    \else
       \let\@tempa\@empty
    \fi
    \ifx\@tempa\@empty\else
      \medskip \begingroup
        \kulemt@fpf@txthead \kulemt@paa@{#1}:\vskip 2\p@
      \endgroup \@tempa \par
    \fi
  \endgroup}
\def\kulemt@frontpage{\clearpage
  \setcounter{page}\m@ne
  \thispagestyle{empty}%
  \@tempdima\uppermargin \advance\@tempdima\topskip
  \advance\@tempdima\baselineskip \advance\@tempdima -1cm%
  \null \vskip -\@tempdima
%% The margins are 16\,mm, resulting in a text width of 178\,mm on A4 paper.
  \hbox to\hsize{%
    \@tempdima 2cm\advance\@tempdima -\spinemargin \hskip\@tempdima
    \vbox to\z@{\hsize 17cm\relax
      \lineskip\z@skip \parskip\z@skip
      \@ifundefined{microtypesetup}{}{\microtypesetup{activate=false}}%
      \fontencoding{T1}\fontfamily{kulemtfpf}\kulemt@fpf@text
      \raggedright \kulemt@selectmasterlanguage
      \noindent \hskip-1cm%
      \vbox to3cm{\hbox{\kulemt@kulfac@logo{width=56mm}}\vss}\hfill
      \vbox to3cm{\hbox{\kulemt@fac@logo{height=2cm}}\vss}%
      \hskip-1cm\hskip\z@skip
      \vskip 40\p@ \@plus 2fill\relax
      \begingroup \kulemt@selecttextlanguage
        \kulemt@fpf@title \@title \par
        \ifx\kulemt@subtitle\@empty\else
          \vskip 1em\relax
          \kulemt@fpf@subtitle \kulemt@subtitle \par
        \fi
      \endgroup
      \vskip 40\p@ \@plus .3fill%
      \begingroup \kulemt@fpf@author \@author
        \vskip 30\p@ \@plus 1fill\endgroup
      \noindent \hfill \vbox{\hsize .5\textwidth \raggedleft
        \kulemt@master@text \par
        \kulemt@paa@fp0%
        \ifkulemt@coverpage\else
          \kulemt@paa@fp1%
          \kulemt@paa@fp2%
        \fi}%
      \vskip 20\p@ \@plus 2fill\relax
      \centering \kulemt@fpf@banner
      \textmd{Academi\kulemt@ifdutch{ejaar}{c year} \kulemt@acyear@t}%
      \vskip 15\p@
      \ifkulemt@coverpage
        \ifx\kulemt@master@colors\@empty\else
          \vskip 15\p@
          \centerline{\fboxsep\z@
            \expandafter\kulemt@getcolors\kulemt@master@colors::\@nil
            \expandafter\colorbox\kulemt@color@bg{%
              \vbox to 15mm{\hsize 19cm\vss
                \expandafter\textcolor\kulemt@color@fg{%
                  \kulemt@master@title}\vss}}}%
        \fi
      \fi
      \vskip -277mm}%
    \hss}%
  \clearpage}
\let\maketitle\relax
\def\kulemt@contact@print{%
  \expandafter\kulemt@contact@scan\kulemt@copyright@contact:::\@nil}
\def\kulemt@contact@scan#1:#2:#3:#4\@nil{#1%
  \def\@tempa{#2}\def\@tempb{#3}%
  \ifx\@tempa\@empty
    \ifx\@tempb\@empty\else , \texttt{#3}\fi
  \else
    , #2%
    \ifx\@tempb\@empty\else
      \space o\kulemt@ifdutch{f via e-}{r by e}mail \texttt{#3}%
    \fi
  \fi}
\def\kulemt@copyright@english{\selectlanguage{english}%
  Without written permission of the \MakeLowercase{\kulemt@paa@0} and
  the author\kulemt@ifand\@author{s}{} it is forbidden to reproduce
  or adapt in any form or by any means any part of this publication.
  Requests for obtaining the right to reproduce or utilize parts of
  this publication should be addressed to \kulemt@contact@print.\par
  A written permission of the \MakeLowercase{\kulemt@paa@0} is also
  required to use the methods, products, schematics and programs
  described in this work for industrial or commercial use, and for
  submitting this publication in scientific contests.\par}
\def\kulemt@copyright@dutch{\selectlanguage{dutch}%
  Zonder voorafgaande schriftelijke toestemming van zowel de
  \MakeLowercase{\kulemt@paa@0} als de auteur\kulemt@ifand\@author{s}{}
  is overnemen, kopi\"eren, gebruiken of realiseren van deze uitgave
  of gedeelten ervan verboden. Voor aanvragen tot of informatie
  i.v.m.\ het overnemen en/of gebruik en/of realisatie van gedeelten
  uit deze publicatie, wend u tot \kulemt@contact@print.\par
  Voorafgaande schriftelijke toestemming van de
  \MakeLowercase{\kulemt@paa@0} is eveneens vereist voor het
  aanwenden van de in deze masterproef beschreven (originele)
  methoden, producten, schakelingen en programma's voor industrieel
  of commercieel nut en voor de inzending van deze publicatie ter
  deelname aan wetenschappelijke prijzen of wedstrijden.\par}
\def\kulemt@copyrightpage{\clearpage
  \thispagestyle{empty}%
  \null \vfill
  \begingroup
    \parindent\z@ \parskip .5\baselineskip \sloppypar
    \copyright\space Copyright KU~Leuven\vskip\baselineskip
    \expandafter\let\expandafter\@tempa
      \csname kulemt@copyright@\kulemt@master@language\endcsname
    \expandafter\let\expandafter\@tempb
      \csname kulemt@copyright@\kulemt@language\endcsname
    \ifx\@tempa\@tempb \@tempa \else
      \ifx\@tempb\kulemt@copyright@english
        \let\@tempb\@tempa \let\@tempa\kulemt@copyright@english \fi
      \def\@tempc{\@tempa \vskip\baselineskip}%
      \expandafter\@tempc\@tempb
    \fi
  \endgroup
  \clearpage}
\makepagestyle{filingcard}
\makeheadrule{filingcard}{\textwidth}{\normalrulethickness}
\makeevenhead{filingcard}{\kulemt@selectmasterlanguage
  KU~Leuven \kulemt@fac@name}{}{\kulemt@acyear@t}
\makeoddhead{filingcard}{\kulemt@selectmasterlanguage
  KU~Leuven \kulemt@fac@name}{}{\kulemt@acyear@t}
\def\kulemt@filingcard{\clearforchapter
  \thispagestyle{filingcard}%
  \begingroup
    \kulemt@selectmasterlanguage
    \begingroup
      \centering \Large
      \kulemt@ifdutch{Fiche masterproef}{Master thesis filing card}%
      \vskip 1em
    \endgroup
    \begingroup
      \parskip\medskipamount
      \@hangfrom{\textit{%
          Student\kulemt@ifand\@author{\kulemt@ifdutch{en}s}}{}: }%
        \@author\par
      \@hangfrom{\textit{Tit\kulemt@ifdutch{el}{le}}: }%
        {\kulemt@selecttextlanguage \@title}\par
      \ifx\kulemt@translatedtitle\@empty\else
        \@hangfrom{\textit{%
            \kulemt@ifdutch
             {\kulemt@selecttextlanguage
              \kulemt@ifdutch{Engel}{Nederland}se titel}%
             {\kulemt@selecttextlanguage
              \kulemt@ifdutch{English}{Dutch} title}}: }%
        \kulemt@translatedtitle\par
      \fi
      \noindent \textit{UDC}: \kulemt@udc\par
      \ifx\kulemt@keywords\@empty\else
        \@hangfrom{\textit{Keywords}: }\kulemt@keywords\par
      \fi
      \ifx\kulemt@article@title\@empty\else
        \@hangfrom{\textit{%
            \kulemt@ifdutch{Titel van het artikel}{Article title}}: }%
          \kulemt@article@title\par
      \fi
      \vskip\medskipamount
    \endgroup
    \noindent \textit{\kulemt@ifdutch{Korte inhoud}{Abstract}}:%
    \vskip 2\p@
    \begingroup \kulemt@selecttextlanguage
      \noindent\ignorespaces \kulemt@shortabstract
    \endgroup
    \vfill \parskip\smallskipamount
    \begingroup \raggedright
      \noindent \kulemt@master@text \par
    \endgroup
    \@hangfrom{\textit{\kulemt@paa@0}: }\kulemt@promotor\par
    \@hangfrom{\textit{\kulemt@paa@1}: }\kulemt@assessor\par
    \@hangfrom{\textit{\kulemt@paa@2}: }\kulemt@assistant\par
  \endgroup
  \clearpage}
\g@addto@macro\document{\kulemt@frontpage
  \ifkulemt@coverpage
    \def\@tempa{\end{document}}%
  \else
    \kulemt@copyrightpage
    \ifkulemt@frontpages
      \def\@tempa{\end{document}}%
    \else
      \let\@tempa\frontmatter
    \fi
  \fi
  \@tempa}
\def\kulemt@error@mm{\kulemt@error{%
    You forgot to use \string\mainmatter}}
\expandafter\def\expandafter\mainmatter\expandafter{%
  \expandafter\let\expandafter\kulemt@error@mm\expandafter\relax
  \mainmatter}
\AtEndDocument{%
  \ifkulemt@coverpage\else
    \ifkulemt@frontpages\else \kulemt@error@mm \fi
    \ifkulemt@filingcard
      \if@mainmatter \backmatter \fi
      \kulemt@filingcard
    \fi
  \fi}
\newenvironment{preface}[1][\@author]%
 {\chapter\prefacename
  \def\kulemt@preface@{#1}}%
 {\par
  \ifx\kulemt@preface@\@empty\else
    \bigskip \raggedleft \itshape \kulemt@preface@
  \fi
  \vfill \clearpage}
\renewenvironment{abstract}%
 {\chapter\abstractname}%
 {\clearpage}
\newenvironment{abstract*}[1][\kulemt@master@language]%
 {\expandafter\selectlanguage\expandafter{#1}%
  \abstract}%
 {\endabstract}
\endinput
%%
%% End of file `kulemt.cls'.
