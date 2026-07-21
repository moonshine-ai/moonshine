import * as onig from '@shikijs/engine-oniguruma';
import { createOnigurumaEngine as createOnigurumaEngine$1, getDefaultWasmLoader } from '@shikijs/engine-oniguruma';
import { readFile } from 'fs/promises';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ShikiError as ShikiError$1 } from '@shikijs/types';
import { INITIAL, EncodedTokenMetadata, FontStyle, Registry as Registry$1, Theme } from '@shikijs/vscode-textmate';

const bundledLanguagesInfo = [
  {
    "id": "abap",
    "name": "ABAP",
    "import": () => import('./abap-BOmX0VAN.js')
  },
  {
    "id": "actionscript-3",
    "name": "ActionScript",
    "import": () => import('./actionscript-3-ClrteL-g.js')
  },
  {
    "id": "ada",
    "name": "Ada",
    "import": () => import('./ada-C5bIzhJC.js')
  },
  {
    "id": "angular-html",
    "name": "Angular HTML",
    "import": () => import('./angular-html-gJWxiwCK.js').then(function (n) { return n.f; })
  },
  {
    "id": "angular-ts",
    "name": "Angular TypeScript",
    "import": () => import('./angular-ts-BlNqvL5c.js')
  },
  {
    "id": "apache",
    "name": "Apache Conf",
    "import": () => import('./apache-Y3WsrcoV.js')
  },
  {
    "id": "apex",
    "name": "Apex",
    "import": () => import('./apex-e_nhEYJq.js')
  },
  {
    "id": "apl",
    "name": "APL",
    "import": () => import('./apl-CeJZjKFK.js')
  },
  {
    "id": "applescript",
    "name": "AppleScript",
    "import": () => import('./applescript-DsGdbAlU.js')
  },
  {
    "id": "ara",
    "name": "Ara",
    "import": () => import('./ara-aYrDhzdm.js')
  },
  {
    "id": "asciidoc",
    "name": "AsciiDoc",
    "aliases": [
      "adoc"
    ],
    "import": () => import('./asciidoc-7Be60ISU.js')
  },
  {
    "id": "asm",
    "name": "Assembly",
    "import": () => import('./asm-BDYFrKjk.js')
  },
  {
    "id": "astro",
    "name": "Astro",
    "import": () => import('./astro-BRXI1FYL.js')
  },
  {
    "id": "awk",
    "name": "AWK",
    "import": () => import('./awk-DySjvGlj.js')
  },
  {
    "id": "ballerina",
    "name": "Ballerina",
    "import": () => import('./ballerina-D9wtHKUG.js')
  },
  {
    "id": "bat",
    "name": "Batch File",
    "aliases": [
      "batch"
    ],
    "import": () => import('./bat-B3qwsOoT.js')
  },
  {
    "id": "beancount",
    "name": "Beancount",
    "import": () => import('./beancount-CnprpJTT.js')
  },
  {
    "id": "berry",
    "name": "Berry",
    "aliases": [
      "be"
    ],
    "import": () => import('./berry-DTdRNf--.js')
  },
  {
    "id": "bibtex",
    "name": "BibTeX",
    "import": () => import('./bibtex-ChLwU6ot.js')
  },
  {
    "id": "bicep",
    "name": "Bicep",
    "import": () => import('./bicep-c7hN9ddp.js')
  },
  {
    "id": "blade",
    "name": "Blade",
    "import": () => import('./blade-inuj8OAI.js')
  },
  {
    "id": "bsl",
    "name": "1C (Enterprise)",
    "aliases": [
      "1c"
    ],
    "import": () => import('./bsl-Bkf8d1lh.js')
  },
  {
    "id": "c",
    "name": "C",
    "import": () => import('./c-DWyntzpU.js')
  },
  {
    "id": "cadence",
    "name": "Cadence",
    "aliases": [
      "cdc"
    ],
    "import": () => import('./cadence-Yd66v9R5.js')
  },
  {
    "id": "cairo",
    "name": "Cairo",
    "import": () => import('./cairo-1CeC2fMe.js')
  },
  {
    "id": "clarity",
    "name": "Clarity",
    "import": () => import('./clarity-kcNLu_ft.js')
  },
  {
    "id": "clojure",
    "name": "Clojure",
    "aliases": [
      "clj"
    ],
    "import": () => import('./clojure-DOZ92VXR.js')
  },
  {
    "id": "cmake",
    "name": "CMake",
    "import": () => import('./cmake-CSVRClCE.js')
  },
  {
    "id": "cobol",
    "name": "COBOL",
    "import": () => import('./cobol-CPFOxTGc.js')
  },
  {
    "id": "codeowners",
    "name": "CODEOWNERS",
    "import": () => import('./codeowners-DujmmhGy.js')
  },
  {
    "id": "codeql",
    "name": "CodeQL",
    "aliases": [
      "ql"
    ],
    "import": () => import('./codeql-UMT4Ns2b.js')
  },
  {
    "id": "coffee",
    "name": "CoffeeScript",
    "aliases": [
      "coffeescript"
    ],
    "import": () => import('./coffee-CSQ8Q4yP.js')
  },
  {
    "id": "common-lisp",
    "name": "Common Lisp",
    "aliases": [
      "lisp"
    ],
    "import": () => import('./common-lisp-Cci_oeEP.js')
  },
  {
    "id": "coq",
    "name": "Coq",
    "import": () => import('./coq-DS075fv2.js')
  },
  {
    "id": "cpp",
    "name": "C++",
    "aliases": [
      "c++"
    ],
    "import": () => import('./cpp-Do3vLwZR.js')
  },
  {
    "id": "crystal",
    "name": "Crystal",
    "import": () => import('./crystal-D5gueAH-.js')
  },
  {
    "id": "csharp",
    "name": "C#",
    "aliases": [
      "c#",
      "cs"
    ],
    "import": () => import('./csharp-BUDpKPju.js')
  },
  {
    "id": "css",
    "name": "CSS",
    "import": () => import('./css-CcnpzXBr.js')
  },
  {
    "id": "csv",
    "name": "CSV",
    "import": () => import('./csv-DU-9qBnR.js')
  },
  {
    "id": "cue",
    "name": "CUE",
    "import": () => import('./cue-BpuFp0U4.js')
  },
  {
    "id": "cypher",
    "name": "Cypher",
    "aliases": [
      "cql"
    ],
    "import": () => import('./cypher-CJs5KXff.js')
  },
  {
    "id": "d",
    "name": "D",
    "import": () => import('./d-CfNeOiGR.js')
  },
  {
    "id": "dart",
    "name": "Dart",
    "import": () => import('./dart-Do_QFP4L.js')
  },
  {
    "id": "dax",
    "name": "DAX",
    "import": () => import('./dax-B-JVuGHQ.js')
  },
  {
    "id": "desktop",
    "name": "Desktop",
    "import": () => import('./desktop-BlY81DH0.js')
  },
  {
    "id": "diff",
    "name": "Diff",
    "import": () => import('./diff-CckWcOjT.js')
  },
  {
    "id": "docker",
    "name": "Dockerfile",
    "aliases": [
      "dockerfile"
    ],
    "import": () => import('./docker-sHh5R1An.js')
  },
  {
    "id": "dotenv",
    "name": "dotEnv",
    "import": () => import('./dotenv-DLX2bz0J.js')
  },
  {
    "id": "dream-maker",
    "name": "Dream Maker",
    "import": () => import('./dream-maker-DizMCl3g.js')
  },
  {
    "id": "edge",
    "name": "Edge",
    "import": () => import('./edge-BsmsFX9V.js')
  },
  {
    "id": "elixir",
    "name": "Elixir",
    "import": () => import('./elixir-CGb5JUiL.js')
  },
  {
    "id": "elm",
    "name": "Elm",
    "import": () => import('./elm-GRE9VLcJ.js')
  },
  {
    "id": "emacs-lisp",
    "name": "Emacs Lisp",
    "aliases": [
      "elisp"
    ],
    "import": () => import('./emacs-lisp-CzGD3FYu.js')
  },
  {
    "id": "erb",
    "name": "ERB",
    "import": () => import('./erb-BXu5GRwd.js')
  },
  {
    "id": "erlang",
    "name": "Erlang",
    "aliases": [
      "erl"
    ],
    "import": () => import('./erlang-IzIbKXju.js')
  },
  {
    "id": "fennel",
    "name": "Fennel",
    "import": () => import('./fennel-CVix6oar.js')
  },
  {
    "id": "fish",
    "name": "Fish",
    "import": () => import('./fish-C5yh10iG.js')
  },
  {
    "id": "fluent",
    "name": "Fluent",
    "aliases": [
      "ftl"
    ],
    "import": () => import('./fluent-Hyl3PwKq.js')
  },
  {
    "id": "fortran-fixed-form",
    "name": "Fortran (Fixed Form)",
    "aliases": [
      "f",
      "for",
      "f77"
    ],
    "import": () => import('./fortran-fixed-form-DkN5qxmg.js')
  },
  {
    "id": "fortran-free-form",
    "name": "Fortran (Free Form)",
    "aliases": [
      "f90",
      "f95",
      "f03",
      "f08",
      "f18"
    ],
    "import": () => import('./fortran-free-form-Bj70omIK.js')
  },
  {
    "id": "fsharp",
    "name": "F#",
    "aliases": [
      "f#",
      "fs"
    ],
    "import": () => import('./fsharp-CPj5b6hR.js')
  },
  {
    "id": "gdresource",
    "name": "GDResource",
    "import": () => import('./gdresource-cuXUn82g.js')
  },
  {
    "id": "gdscript",
    "name": "GDScript",
    "import": () => import('./gdscript-27H8_EuK.js')
  },
  {
    "id": "gdshader",
    "name": "GDShader",
    "import": () => import('./gdshader-2QFg8sFN.js')
  },
  {
    "id": "genie",
    "name": "Genie",
    "import": () => import('./genie-CyQe-g4_.js')
  },
  {
    "id": "gherkin",
    "name": "Gherkin",
    "import": () => import('./gherkin-CaMDiJB1.js')
  },
  {
    "id": "git-commit",
    "name": "Git Commit Message",
    "import": () => import('./git-commit-DpXt6lSL.js')
  },
  {
    "id": "git-rebase",
    "name": "Git Rebase Message",
    "import": () => import('./git-rebase-DqXwfZYu.js')
  },
  {
    "id": "gleam",
    "name": "Gleam",
    "import": () => import('./gleam-tUVnEaC5.js')
  },
  {
    "id": "glimmer-js",
    "name": "Glimmer JS",
    "aliases": [
      "gjs"
    ],
    "import": () => import('./glimmer-js-CXXgqzw8.js')
  },
  {
    "id": "glimmer-ts",
    "name": "Glimmer TS",
    "aliases": [
      "gts"
    ],
    "import": () => import('./glimmer-ts-I2Ri0vHp.js')
  },
  {
    "id": "glsl",
    "name": "GLSL",
    "import": () => import('./glsl-DNxBKKNq.js')
  },
  {
    "id": "gnuplot",
    "name": "Gnuplot",
    "import": () => import('./gnuplot-DMfwcbYR.js')
  },
  {
    "id": "go",
    "name": "Go",
    "import": () => import('./go-CG6jXbKP.js')
  },
  {
    "id": "graphql",
    "name": "GraphQL",
    "aliases": [
      "gql"
    ],
    "import": () => import('./graphql-DqfhDByg.js')
  },
  {
    "id": "groovy",
    "name": "Groovy",
    "import": () => import('./groovy-qVXuHEdY.js')
  },
  {
    "id": "hack",
    "name": "Hack",
    "import": () => import('./hack-csj6_okc.js')
  },
  {
    "id": "haml",
    "name": "Ruby Haml",
    "import": () => import('./haml-BD7qFoCJ.js')
  },
  {
    "id": "handlebars",
    "name": "Handlebars",
    "aliases": [
      "hbs"
    ],
    "import": () => import('./handlebars-DjJaV7wU.js')
  },
  {
    "id": "haskell",
    "name": "Haskell",
    "aliases": [
      "hs"
    ],
    "import": () => import('./haskell-BvmqBUUl.js')
  },
  {
    "id": "haxe",
    "name": "Haxe",
    "import": () => import('./haxe-D4R93kjU.js')
  },
  {
    "id": "hcl",
    "name": "HashiCorp HCL",
    "import": () => import('./hcl-DXXGusUm.js')
  },
  {
    "id": "hjson",
    "name": "Hjson",
    "import": () => import('./hjson-Thw2PQkJ.js')
  },
  {
    "id": "hlsl",
    "name": "HLSL",
    "import": () => import('./hlsl-BcCxlkQo.js')
  },
  {
    "id": "html",
    "name": "HTML",
    "import": () => import('./html-DJjKhWOc.js')
  },
  {
    "id": "html-derivative",
    "name": "HTML (Derivative)",
    "import": () => import('./html-derivative-BEAtTNr1.js')
  },
  {
    "id": "http",
    "name": "HTTP",
    "import": () => import('./http-DfD1Gi2b.js')
  },
  {
    "id": "hxml",
    "name": "HXML",
    "import": () => import('./hxml-DHVR79aP.js')
  },
  {
    "id": "hy",
    "name": "Hy",
    "import": () => import('./hy-GEk5y3Nl.js')
  },
  {
    "id": "imba",
    "name": "Imba",
    "import": () => import('./imba-BYIKuhRM.js')
  },
  {
    "id": "ini",
    "name": "INI",
    "aliases": [
      "properties"
    ],
    "import": () => import('./ini-Clz0070y.js')
  },
  {
    "id": "java",
    "name": "Java",
    "import": () => import('./java-J9ifFwHK.js')
  },
  {
    "id": "javascript",
    "name": "JavaScript",
    "aliases": [
      "js"
    ],
    "import": () => import('./javascript-CjVMUwBu.js')
  },
  {
    "id": "jinja",
    "name": "Jinja",
    "import": () => import('./jinja-c6LDET89.js')
  },
  {
    "id": "jison",
    "name": "Jison",
    "import": () => import('./jison-BpP1S9rG.js')
  },
  {
    "id": "json",
    "name": "JSON",
    "import": () => import('./json-Cfy1KxwD.js')
  },
  {
    "id": "json5",
    "name": "JSON5",
    "import": () => import('./json5-C0u1cfcn.js')
  },
  {
    "id": "jsonc",
    "name": "JSON with Comments",
    "import": () => import('./jsonc-DIWaKIEg.js')
  },
  {
    "id": "jsonl",
    "name": "JSON Lines",
    "import": () => import('./jsonl-COVMViRA.js')
  },
  {
    "id": "jsonnet",
    "name": "Jsonnet",
    "import": () => import('./jsonnet-DsAYm9aF.js')
  },
  {
    "id": "jssm",
    "name": "JSSM",
    "aliases": [
      "fsl"
    ],
    "import": () => import('./jssm-BXzBgqRl.js')
  },
  {
    "id": "jsx",
    "name": "JSX",
    "import": () => import('./jsx-C6hsJK1G.js')
  },
  {
    "id": "julia",
    "name": "Julia",
    "aliases": [
      "jl"
    ],
    "import": () => import('./julia-CuzX8Z7x.js')
  },
  {
    "id": "kotlin",
    "name": "Kotlin",
    "aliases": [
      "kt",
      "kts"
    ],
    "import": () => import('./kotlin-D6h4w0pC.js')
  },
  {
    "id": "kusto",
    "name": "Kusto",
    "aliases": [
      "kql"
    ],
    "import": () => import('./kusto-BLWh8Zuz.js')
  },
  {
    "id": "latex",
    "name": "LaTeX",
    "import": () => import('./latex-pqcdd-Dp.js')
  },
  {
    "id": "lean",
    "name": "Lean 4",
    "aliases": [
      "lean4"
    ],
    "import": () => import('./lean-BOcIAQWM.js')
  },
  {
    "id": "less",
    "name": "Less",
    "import": () => import('./less-BrwQf5_Z.js')
  },
  {
    "id": "liquid",
    "name": "Liquid",
    "import": () => import('./liquid-B9PD4Dq-.js')
  },
  {
    "id": "log",
    "name": "Log file",
    "import": () => import('./log-BtD96ao3.js')
  },
  {
    "id": "logo",
    "name": "Logo",
    "import": () => import('./logo-DItzB5Eg.js')
  },
  {
    "id": "lua",
    "name": "Lua",
    "import": () => import('./lua-CBfHEbQR.js')
  },
  {
    "id": "luau",
    "name": "Luau",
    "import": () => import('./luau-Dkil-FGj.js')
  },
  {
    "id": "make",
    "name": "Makefile",
    "aliases": [
      "makefile"
    ],
    "import": () => import('./make-b2-MkGps.js')
  },
  {
    "id": "markdown",
    "name": "Markdown",
    "aliases": [
      "md"
    ],
    "import": () => import('./markdown-bqe6s7EU.js')
  },
  {
    "id": "marko",
    "name": "Marko",
    "import": () => import('./marko-CUzJbaFl.js')
  },
  {
    "id": "matlab",
    "name": "MATLAB",
    "import": () => import('./matlab-Bz7pVSlE.js')
  },
  {
    "id": "mdc",
    "name": "MDC",
    "import": () => import('./mdc-CPImvEGa.js')
  },
  {
    "id": "mdx",
    "name": "MDX",
    "import": () => import('./mdx-C7R6ENO1.js')
  },
  {
    "id": "mermaid",
    "name": "Mermaid",
    "aliases": [
      "mmd"
    ],
    "import": () => import('./mermaid-Dmt1ILLy.js')
  },
  {
    "id": "mipsasm",
    "name": "MIPS Assembly",
    "aliases": [
      "mips"
    ],
    "import": () => import('./mipsasm-CJKbxpwt.js')
  },
  {
    "id": "mojo",
    "name": "Mojo",
    "import": () => import('./mojo-CR3BHjjf.js')
  },
  {
    "id": "move",
    "name": "Move",
    "import": () => import('./move-Bxd2lAfi.js')
  },
  {
    "id": "narrat",
    "name": "Narrat Language",
    "aliases": [
      "nar"
    ],
    "import": () => import('./narrat-D6eaxwjN.js')
  },
  {
    "id": "nextflow",
    "name": "Nextflow",
    "aliases": [
      "nf"
    ],
    "import": () => import('./nextflow-BVwNoED2.js')
  },
  {
    "id": "nginx",
    "name": "Nginx",
    "import": () => import('./nginx-BUdJQSF0.js')
  },
  {
    "id": "nim",
    "name": "Nim",
    "import": () => import('./nim-BGhgTKnC.js')
  },
  {
    "id": "nix",
    "name": "Nix",
    "import": () => import('./nix-GExN55d0.js')
  },
  {
    "id": "nushell",
    "name": "nushell",
    "aliases": [
      "nu"
    ],
    "import": () => import('./nushell-CItaKqhj.js')
  },
  {
    "id": "objective-c",
    "name": "Objective-C",
    "aliases": [
      "objc"
    ],
    "import": () => import('./objective-c-DAoXoumH.js')
  },
  {
    "id": "objective-cpp",
    "name": "Objective-C++",
    "import": () => import('./objective-cpp-CRbSLkwX.js')
  },
  {
    "id": "ocaml",
    "name": "OCaml",
    "import": () => import('./ocaml-D9p0L4sG.js')
  },
  {
    "id": "pascal",
    "name": "Pascal",
    "import": () => import('./pascal-C7Z-jAV5.js')
  },
  {
    "id": "perl",
    "name": "Perl",
    "import": () => import('./perl-DWIXe5mg.js')
  },
  {
    "id": "php",
    "name": "PHP",
    "import": () => import('./php-DBeJUqIG.js')
  },
  {
    "id": "plsql",
    "name": "PL/SQL",
    "import": () => import('./plsql-B683bAA-.js')
  },
  {
    "id": "po",
    "name": "Gettext PO",
    "aliases": [
      "pot",
      "potx"
    ],
    "import": () => import('./po-c7rwMACo.js')
  },
  {
    "id": "polar",
    "name": "Polar",
    "import": () => import('./polar-BaWoaHC_.js')
  },
  {
    "id": "postcss",
    "name": "PostCSS",
    "import": () => import('./postcss-B2omATDc.js')
  },
  {
    "id": "powerquery",
    "name": "PowerQuery",
    "import": () => import('./powerquery-0X2HetX3.js')
  },
  {
    "id": "powershell",
    "name": "PowerShell",
    "aliases": [
      "ps",
      "ps1"
    ],
    "import": () => import('./powershell-D9TBvCW4.js')
  },
  {
    "id": "prisma",
    "name": "Prisma",
    "import": () => import('./prisma-CnApJfVM.js')
  },
  {
    "id": "prolog",
    "name": "Prolog",
    "import": () => import('./prolog-C_qzsQ-D.js')
  },
  {
    "id": "proto",
    "name": "Protocol Buffer 3",
    "aliases": [
      "protobuf"
    ],
    "import": () => import('./proto-DPqkgrwu.js')
  },
  {
    "id": "pug",
    "name": "Pug",
    "aliases": [
      "jade"
    ],
    "import": () => import('./pug-DO9VkxOi.js')
  },
  {
    "id": "puppet",
    "name": "Puppet",
    "import": () => import('./puppet-CFjOU4zd.js')
  },
  {
    "id": "purescript",
    "name": "PureScript",
    "import": () => import('./purescript-DO5Czfo2.js')
  },
  {
    "id": "python",
    "name": "Python",
    "aliases": [
      "py"
    ],
    "import": () => import('./python-BWZACR-G.js')
  },
  {
    "id": "qml",
    "name": "QML",
    "import": () => import('./qml-Sow7dYUt.js')
  },
  {
    "id": "qmldir",
    "name": "QML Directory",
    "import": () => import('./qmldir-qwv8bvdg.js')
  },
  {
    "id": "qss",
    "name": "Qt Style Sheets",
    "import": () => import('./qss-BBbc5pbj.js')
  },
  {
    "id": "r",
    "name": "R",
    "import": () => import('./r-2ACUfFDs.js')
  },
  {
    "id": "racket",
    "name": "Racket",
    "import": () => import('./racket-pndzfVaQ.js')
  },
  {
    "id": "raku",
    "name": "Raku",
    "aliases": [
      "perl6"
    ],
    "import": () => import('./raku-BB00rLRm.js')
  },
  {
    "id": "razor",
    "name": "ASP.NET Razor",
    "import": () => import('./razor-BVzx9cIz.js')
  },
  {
    "id": "reg",
    "name": "Windows Registry Script",
    "import": () => import('./reg-BDHTWrV-.js')
  },
  {
    "id": "regexp",
    "name": "RegExp",
    "aliases": [
      "regex"
    ],
    "import": () => import('./regexp-D6QN1sWt.js')
  },
  {
    "id": "rel",
    "name": "Rel",
    "import": () => import('./rel-BKK0Z6ir.js')
  },
  {
    "id": "riscv",
    "name": "RISC-V",
    "import": () => import('./riscv-fn5hNgLY.js')
  },
  {
    "id": "rst",
    "name": "reStructuredText",
    "import": () => import('./rst-C61EwAAT.js')
  },
  {
    "id": "ruby",
    "name": "Ruby",
    "aliases": [
      "rb"
    ],
    "import": () => import('./ruby-B35gCjOI.js')
  },
  {
    "id": "rust",
    "name": "Rust",
    "aliases": [
      "rs"
    ],
    "import": () => import('./rust-D3MDq5o6.js')
  },
  {
    "id": "sas",
    "name": "SAS",
    "import": () => import('./sas-BuZOAse9.js')
  },
  {
    "id": "sass",
    "name": "Sass",
    "import": () => import('./sass-BmNeXdl1.js')
  },
  {
    "id": "scala",
    "name": "Scala",
    "import": () => import('./scala-CzXUO94T.js')
  },
  {
    "id": "scheme",
    "name": "Scheme",
    "import": () => import('./scheme-C8V8eTvR.js')
  },
  {
    "id": "scss",
    "name": "SCSS",
    "import": () => import('./scss-BxhDGjtM.js')
  },
  {
    "id": "sdbl",
    "name": "1C (Query)",
    "aliases": [
      "1c-query"
    ],
    "import": () => import('./sdbl-Ds6BtD9c.js')
  },
  {
    "id": "shaderlab",
    "name": "ShaderLab",
    "aliases": [
      "shader"
    ],
    "import": () => import('./shaderlab-BUyDy3Yv.js')
  },
  {
    "id": "shellscript",
    "name": "Shell",
    "aliases": [
      "bash",
      "sh",
      "shell",
      "zsh"
    ],
    "import": () => import('./shellscript-DvfPQ4T_.js')
  },
  {
    "id": "shellsession",
    "name": "Shell Session",
    "aliases": [
      "console"
    ],
    "import": () => import('./shellsession-DW3IyT3o.js')
  },
  {
    "id": "smalltalk",
    "name": "Smalltalk",
    "import": () => import('./smalltalk-DLJFTASC.js')
  },
  {
    "id": "solidity",
    "name": "Solidity",
    "import": () => import('./solidity-Cnnm7z2F.js')
  },
  {
    "id": "soy",
    "name": "Closure Templates",
    "aliases": [
      "closure-templates"
    ],
    "import": () => import('./soy-DVN-3N05.js')
  },
  {
    "id": "sparql",
    "name": "SPARQL",
    "import": () => import('./sparql-BC9RdhT7.js')
  },
  {
    "id": "splunk",
    "name": "Splunk Query Language",
    "aliases": [
      "spl"
    ],
    "import": () => import('./splunk-ARl4th5X.js')
  },
  {
    "id": "sql",
    "name": "SQL",
    "import": () => import('./sql-CwwRRajN.js')
  },
  {
    "id": "ssh-config",
    "name": "SSH Config",
    "import": () => import('./ssh-config-BeNpcEMY.js')
  },
  {
    "id": "stata",
    "name": "Stata",
    "import": () => import('./stata-_t3JKC5S.js')
  },
  {
    "id": "stylus",
    "name": "Stylus",
    "aliases": [
      "styl"
    ],
    "import": () => import('./stylus-Cxc8GeSb.js')
  },
  {
    "id": "svelte",
    "name": "Svelte",
    "import": () => import('./svelte-CT2HhgzH.js')
  },
  {
    "id": "swift",
    "name": "Swift",
    "import": () => import('./swift-CgYJHmDN.js')
  },
  {
    "id": "system-verilog",
    "name": "SystemVerilog",
    "import": () => import('./system-verilog-BdJNtzP9.js')
  },
  {
    "id": "systemd",
    "name": "Systemd Units",
    "import": () => import('./systemd-DMOdNjaZ.js')
  },
  {
    "id": "talonscript",
    "name": "TalonScript",
    "aliases": [
      "talon"
    ],
    "import": () => import('./talonscript-3oHtx0QF.js')
  },
  {
    "id": "tasl",
    "name": "Tasl",
    "import": () => import('./tasl-CT-l6Y9v.js')
  },
  {
    "id": "tcl",
    "name": "Tcl",
    "import": () => import('./tcl-DtshbUef.js')
  },
  {
    "id": "templ",
    "name": "Templ",
    "import": () => import('./templ-BwtyTtcV.js')
  },
  {
    "id": "terraform",
    "name": "Terraform",
    "aliases": [
      "tf",
      "tfvars"
    ],
    "import": () => import('./terraform-C0Rxenyp.js')
  },
  {
    "id": "tex",
    "name": "TeX",
    "import": () => import('./tex-D1HUc06E.js')
  },
  {
    "id": "toml",
    "name": "TOML",
    "import": () => import('./toml-GFeQ6yHI.js')
  },
  {
    "id": "ts-tags",
    "name": "TypeScript with Tags",
    "aliases": [
      "lit"
    ],
    "import": () => import('./ts-tags-CrKaQRpw.js')
  },
  {
    "id": "tsv",
    "name": "TSV",
    "import": () => import('./tsv-Ko2eBya-.js')
  },
  {
    "id": "tsx",
    "name": "TSX",
    "import": () => import('./tsx-zCkZ74Uo.js')
  },
  {
    "id": "turtle",
    "name": "Turtle",
    "import": () => import('./turtle-ChKZJk3-.js')
  },
  {
    "id": "twig",
    "name": "Twig",
    "import": () => import('./twig-C98t3vWG.js')
  },
  {
    "id": "typescript",
    "name": "TypeScript",
    "aliases": [
      "ts"
    ],
    "import": () => import('./typescript-DFjkS2ah.js')
  },
  {
    "id": "typespec",
    "name": "TypeSpec",
    "aliases": [
      "tsp"
    ],
    "import": () => import('./typespec-B2WY4yAd.js')
  },
  {
    "id": "typst",
    "name": "Typst",
    "aliases": [
      "typ"
    ],
    "import": () => import('./typst-DIxzjN7Z.js')
  },
  {
    "id": "v",
    "name": "V",
    "import": () => import('./v-DH2oYZQ5.js')
  },
  {
    "id": "vala",
    "name": "Vala",
    "import": () => import('./vala-CxmQQ9y4.js')
  },
  {
    "id": "vb",
    "name": "Visual Basic",
    "aliases": [
      "cmd"
    ],
    "import": () => import('./vb-BxZnMNQR.js')
  },
  {
    "id": "verilog",
    "name": "Verilog",
    "import": () => import('./verilog-DeRObUln.js')
  },
  {
    "id": "vhdl",
    "name": "VHDL",
    "import": () => import('./vhdl-BMRzAllk.js')
  },
  {
    "id": "viml",
    "name": "Vim Script",
    "aliases": [
      "vim",
      "vimscript"
    ],
    "import": () => import('./viml-I8CLENc8.js')
  },
  {
    "id": "vue",
    "name": "Vue",
    "import": () => import('./vue-DeRZa5ej.js')
  },
  {
    "id": "vue-html",
    "name": "Vue HTML",
    "import": () => import('./vue-html-rUkGiilo.js')
  },
  {
    "id": "vyper",
    "name": "Vyper",
    "aliases": [
      "vy"
    ],
    "import": () => import('./vyper-CcS3VNO0.js')
  },
  {
    "id": "wasm",
    "name": "WebAssembly",
    "import": () => import('./wasm-DgHa91K3.js')
  },
  {
    "id": "wenyan",
    "name": "Wenyan",
    "aliases": [
      "\u6587\u8A00"
    ],
    "import": () => import('./wenyan-BESVSALc.js')
  },
  {
    "id": "wgsl",
    "name": "WGSL",
    "import": () => import('./wgsl-BDjL_6V5.js')
  },
  {
    "id": "wikitext",
    "name": "Wikitext",
    "aliases": [
      "mediawiki",
      "wiki"
    ],
    "import": () => import('./wikitext-BsTpwCuu.js')
  },
  {
    "id": "wolfram",
    "name": "Wolfram",
    "aliases": [
      "wl"
    ],
    "import": () => import('./wolfram-CsP5ezCu.js')
  },
  {
    "id": "xml",
    "name": "XML",
    "import": () => import('./xml-DNQ2AKLY.js')
  },
  {
    "id": "xsl",
    "name": "XSL",
    "import": () => import('./xsl-BKoRQ8ik.js')
  },
  {
    "id": "yaml",
    "name": "YAML",
    "aliases": [
      "yml"
    ],
    "import": () => import('./yaml-DfQjDO74.js')
  },
  {
    "id": "zenscript",
    "name": "ZenScript",
    "import": () => import('./zenscript-CdvyPjGb.js')
  },
  {
    "id": "zig",
    "name": "Zig",
    "import": () => import('./zig-B8F_EXWu.js')
  }
];
const bundledLanguagesBase = Object.fromEntries(bundledLanguagesInfo.map((i) => [i.id, i.import]));
const bundledLanguagesAlias = Object.fromEntries(bundledLanguagesInfo.flatMap((i) => i.aliases?.map((a) => [a, i.import]) || []));
const bundledLanguages = {
  ...bundledLanguagesBase,
  ...bundledLanguagesAlias
};

const bundledThemesInfo = [
  {
    "id": "andromeeda",
    "displayName": "Andromeeda",
    "type": "dark",
    "import": () => import('./andromeeda-Cm29CUIt.js')
  },
  {
    "id": "aurora-x",
    "displayName": "Aurora X",
    "type": "dark",
    "import": () => import('./aurora-x-C61gCcFC.js')
  },
  {
    "id": "ayu-dark",
    "displayName": "Ayu Dark",
    "type": "dark",
    "import": () => import('./ayu-dark-4nAHfw6e.js')
  },
  {
    "id": "catppuccin-frappe",
    "displayName": "Catppuccin Frapp\xE9",
    "type": "dark",
    "import": () => import('./catppuccin-frappe-DRWFYqVL.js')
  },
  {
    "id": "catppuccin-latte",
    "displayName": "Catppuccin Latte",
    "type": "light",
    "import": () => import('./catppuccin-latte-57WLnpvu.js')
  },
  {
    "id": "catppuccin-macchiato",
    "displayName": "Catppuccin Macchiato",
    "type": "dark",
    "import": () => import('./catppuccin-macchiato-DElSEHTs.js')
  },
  {
    "id": "catppuccin-mocha",
    "displayName": "Catppuccin Mocha",
    "type": "dark",
    "import": () => import('./catppuccin-mocha-CmCEFnHh.js')
  },
  {
    "id": "dark-plus",
    "displayName": "Dark Plus",
    "type": "dark",
    "import": () => import('./dark-plus-CfGTxS_N.js')
  },
  {
    "id": "dracula",
    "displayName": "Dracula Theme",
    "type": "dark",
    "import": () => import('./dracula-C9Ywc-To.js')
  },
  {
    "id": "dracula-soft",
    "displayName": "Dracula Theme Soft",
    "type": "dark",
    "import": () => import('./dracula-soft-jXDHlsrl.js')
  },
  {
    "id": "everforest-dark",
    "displayName": "Everforest Dark",
    "type": "dark",
    "import": () => import('./everforest-dark-Cv8Sz0RF.js')
  },
  {
    "id": "everforest-light",
    "displayName": "Everforest Light",
    "type": "light",
    "import": () => import('./everforest-light-C-pKK-ml.js')
  },
  {
    "id": "github-dark",
    "displayName": "GitHub Dark",
    "type": "dark",
    "import": () => import('./github-dark-CFENc4o3.js')
  },
  {
    "id": "github-dark-default",
    "displayName": "GitHub Dark Default",
    "type": "dark",
    "import": () => import('./github-dark-default-Det7zS2Q.js')
  },
  {
    "id": "github-dark-dimmed",
    "displayName": "GitHub Dark Dimmed",
    "type": "dark",
    "import": () => import('./github-dark-dimmed-DTliyBIC.js')
  },
  {
    "id": "github-dark-high-contrast",
    "displayName": "GitHub Dark High Contrast",
    "type": "dark",
    "import": () => import('./github-dark-high-contrast-CS7JCXAt.js')
  },
  {
    "id": "github-light",
    "displayName": "GitHub Light",
    "type": "light",
    "import": () => import('./github-light-IT3VqcP-.js')
  },
  {
    "id": "github-light-default",
    "displayName": "GitHub Light Default",
    "type": "light",
    "import": () => import('./github-light-default-DbiuWJKc.js')
  },
  {
    "id": "github-light-high-contrast",
    "displayName": "GitHub Light High Contrast",
    "type": "light",
    "import": () => import('./github-light-high-contrast-B8162YsG.js')
  },
  {
    "id": "houston",
    "displayName": "Houston",
    "type": "dark",
    "import": () => import('./houston-B0u6hL7J.js')
  },
  {
    "id": "kanagawa-dragon",
    "displayName": "Kanagawa Dragon",
    "type": "dark",
    "import": () => import('./kanagawa-dragon-B3shUtba.js')
  },
  {
    "id": "kanagawa-lotus",
    "displayName": "Kanagawa Lotus",
    "type": "light",
    "import": () => import('./kanagawa-lotus-DBk1m4eC.js')
  },
  {
    "id": "kanagawa-wave",
    "displayName": "Kanagawa Wave",
    "type": "dark",
    "import": () => import('./kanagawa-wave-dXN0Ocj_.js')
  },
  {
    "id": "laserwave",
    "displayName": "LaserWave",
    "type": "dark",
    "import": () => import('./laserwave-BLuI0m3M.js')
  },
  {
    "id": "light-plus",
    "displayName": "Light Plus",
    "type": "light",
    "import": () => import('./light-plus-Cpqqyhhd.js')
  },
  {
    "id": "material-theme",
    "displayName": "Material Theme",
    "type": "dark",
    "import": () => import('./material-theme-xCpzc19L.js')
  },
  {
    "id": "material-theme-darker",
    "displayName": "Material Theme Darker",
    "type": "dark",
    "import": () => import('./material-theme-darker-B57xbZTk.js')
  },
  {
    "id": "material-theme-lighter",
    "displayName": "Material Theme Lighter",
    "type": "light",
    "import": () => import('./material-theme-lighter-Be4O9DpB.js')
  },
  {
    "id": "material-theme-ocean",
    "displayName": "Material Theme Ocean",
    "type": "dark",
    "import": () => import('./material-theme-ocean-CjeVYc3W.js')
  },
  {
    "id": "material-theme-palenight",
    "displayName": "Material Theme Palenight",
    "type": "dark",
    "import": () => import('./material-theme-palenight-Ckn1LvMS.js')
  },
  {
    "id": "min-dark",
    "displayName": "Min Dark",
    "type": "dark",
    "import": () => import('./min-dark-DEZ95ueN.js')
  },
  {
    "id": "min-light",
    "displayName": "Min Light",
    "type": "light",
    "import": () => import('./min-light-DXhfqFL2.js')
  },
  {
    "id": "monokai",
    "displayName": "Monokai",
    "type": "dark",
    "import": () => import('./monokai-B8tfXJnN.js')
  },
  {
    "id": "night-owl",
    "displayName": "Night Owl",
    "type": "dark",
    "import": () => import('./night-owl-2LEfVnVV.js')
  },
  {
    "id": "nord",
    "displayName": "Nord",
    "type": "dark",
    "import": () => import('./nord-CkWB1cCu.js')
  },
  {
    "id": "one-dark-pro",
    "displayName": "One Dark Pro",
    "type": "dark",
    "import": () => import('./one-dark-pro-BiM_MmqQ.js')
  },
  {
    "id": "one-light",
    "displayName": "One Light",
    "type": "light",
    "import": () => import('./one-light-BGtdMtJw.js')
  },
  {
    "id": "plastic",
    "displayName": "Plastic",
    "type": "dark",
    "import": () => import('./plastic-DGoJ7BO5.js')
  },
  {
    "id": "poimandres",
    "displayName": "Poimandres",
    "type": "dark",
    "import": () => import('./poimandres-8X9GgTNF.js')
  },
  {
    "id": "red",
    "displayName": "Red",
    "type": "dark",
    "import": () => import('./red-DWB3RHH9.js')
  },
  {
    "id": "rose-pine",
    "displayName": "Ros\xE9 Pine",
    "type": "dark",
    "import": () => import('./rose-pine-D49OWO-f.js')
  },
  {
    "id": "rose-pine-dawn",
    "displayName": "Ros\xE9 Pine Dawn",
    "type": "light",
    "import": () => import('./rose-pine-dawn-CLgQrv_4.js')
  },
  {
    "id": "rose-pine-moon",
    "displayName": "Ros\xE9 Pine Moon",
    "type": "dark",
    "import": () => import('./rose-pine-moon-DinphuwM.js')
  },
  {
    "id": "slack-dark",
    "displayName": "Slack Dark",
    "type": "dark",
    "import": () => import('./slack-dark-CDockJGe.js')
  },
  {
    "id": "slack-ochin",
    "displayName": "Slack Ochin",
    "type": "light",
    "import": () => import('./slack-ochin-DJf7NwAf.js')
  },
  {
    "id": "snazzy-light",
    "displayName": "Snazzy Light",
    "type": "light",
    "import": () => import('./snazzy-light-KLRpMvz7.js')
  },
  {
    "id": "solarized-dark",
    "displayName": "Solarized Dark",
    "type": "dark",
    "import": () => import('./solarized-dark-DjvRd57_.js')
  },
  {
    "id": "solarized-light",
    "displayName": "Solarized Light",
    "type": "light",
    "import": () => import('./solarized-light-CGdpSOBq.js')
  },
  {
    "id": "synthwave-84",
    "displayName": "Synthwave '84",
    "type": "dark",
    "import": () => import('./synthwave-84-DsOgpoep.js')
  },
  {
    "id": "tokyo-night",
    "displayName": "Tokyo Night",
    "type": "dark",
    "import": () => import('./tokyo-night-QjbeeVFG.js')
  },
  {
    "id": "vesper",
    "displayName": "Vesper",
    "type": "dark",
    "import": () => import('./vesper-BY47GlXd.js')
  },
  {
    "id": "vitesse-black",
    "displayName": "Vitesse Black",
    "type": "dark",
    "import": () => import('./vitesse-black-C1qWXfk2.js')
  },
  {
    "id": "vitesse-dark",
    "displayName": "Vitesse Dark",
    "type": "dark",
    "import": () => import('./vitesse-dark-FE9UVeyX.js')
  },
  {
    "id": "vitesse-light",
    "displayName": "Vitesse Light",
    "type": "light",
    "import": () => import('./vitesse-light-D27naoFF.js')
  }
];
const bundledThemes = Object.fromEntries(bundledThemesInfo.map((i) => [i.id, i.import]));

function toArray(x) {
  return Array.isArray(x) ? x : [x];
}
function splitLines(code, preserveEnding = false) {
  const parts = code.split(/(\r?\n)/g);
  let index = 0;
  const lines = [];
  for (let i = 0; i < parts.length; i += 2) {
    const line = preserveEnding ? parts[i] + (parts[i + 1] || "") : parts[i];
    lines.push([line, index]);
    index += parts[i].length;
    index += parts[i + 1]?.length || 0;
  }
  return lines;
}
function isPlainLang(lang) {
  return !lang || ["plaintext", "txt", "text", "plain"].includes(lang);
}
function isSpecialLang(lang) {
  return lang === "ansi" || isPlainLang(lang);
}
function isNoneTheme(theme) {
  return theme === "none";
}
function isSpecialTheme(theme) {
  return isNoneTheme(theme);
}
async function normalizeGetter(p) {
  return Promise.resolve(typeof p === "function" ? p() : p).then((r) => r.default || r);
}
function resolveColorReplacements(theme, options) {
  const replacements = typeof theme === "string" ? {} : { ...theme.colorReplacements };
  const themeName = typeof theme === "string" ? theme : theme.name;
  for (const [key, value] of Object.entries(options?.colorReplacements || {})) {
    if (typeof value === "string")
      replacements[key] = value;
    else if (key === themeName)
      Object.assign(replacements, value);
  }
  return replacements;
}
function applyColorReplacements(color, replacements) {
  if (!color)
    return color;
  return replacements?.[color?.toLowerCase()] || color;
}

class ShikiError extends Error {
  constructor(message) {
    super(message);
    this.name = "ShikiError";
  }
}

const _grammarStateMap = /* @__PURE__ */ new WeakMap();
function setLastGrammarStateToMap(keys, state) {
  _grammarStateMap.set(keys, state);
}
function getLastGrammarStateFromMap(keys) {
  return _grammarStateMap.get(keys);
}
class GrammarState {
  /**
   * Theme to Stack mapping
   */
  _stacks = {};
  lang;
  get themes() {
    return Object.keys(this._stacks);
  }
  get theme() {
    return this.themes[0];
  }
  get _stack() {
    return this._stacks[this.theme];
  }
  /**
   * Static method to create a initial grammar state.
   */
  static initial(lang, themes) {
    return new GrammarState(
      Object.fromEntries(toArray(themes).map((theme) => [theme, INITIAL])),
      lang
    );
  }
  constructor(...args) {
    if (args.length === 2) {
      const [stacksMap, lang] = args;
      this.lang = lang;
      this._stacks = stacksMap;
    } else {
      const [stack, lang, theme] = args;
      this.lang = lang;
      this._stacks = { [theme]: stack };
    }
  }
  /**
   * Get the internal stack object.
   * @internal
   */
  getInternalStack(theme = this.theme) {
    return this._stacks[theme];
  }
  /**
   * @deprecated use `getScopes` instead
   */
  get scopes() {
    return getScopes(this._stacks[this.theme]);
  }
  getScopes(theme = this.theme) {
    return getScopes(this._stacks[theme]);
  }
  toJSON() {
    return {
      lang: this.lang,
      theme: this.theme,
      themes: this.themes,
      scopes: this.scopes
    };
  }
}
function getScopes(stack) {
  const scopes = [];
  const visited = /* @__PURE__ */ new Set();
  function pushScope(stack2) {
    if (visited.has(stack2))
      return;
    visited.add(stack2);
    const name = stack2?.nameScopesList?.scopeName;
    if (name)
      scopes.push(name);
    if (stack2.parent)
      pushScope(stack2.parent);
  }
  pushScope(stack);
  return scopes;
}
function getGrammarStack(state, theme) {
  if (!(state instanceof GrammarState))
    throw new ShikiError("Invalid grammar state");
  return state.getInternalStack(theme);
}

// src/colors.ts
var namedColors = [
  "black",
  "red",
  "green",
  "yellow",
  "blue",
  "magenta",
  "cyan",
  "white",
  "brightBlack",
  "brightRed",
  "brightGreen",
  "brightYellow",
  "brightBlue",
  "brightMagenta",
  "brightCyan",
  "brightWhite"
];

// src/decorations.ts
var decorations = {
  1: "bold",
  2: "dim",
  3: "italic",
  4: "underline",
  7: "reverse",
  9: "strikethrough"
};

// src/parser.ts
function findSequence(value, position) {
  const nextEscape = value.indexOf("\x1B[", position);
  if (nextEscape !== -1) {
    const nextClose = value.indexOf("m", nextEscape);
    return {
      sequence: value.substring(nextEscape + 2, nextClose).split(";"),
      startPosition: nextEscape,
      position: nextClose + 1
    };
  }
  return {
    position: value.length
  };
}
function parseColor(sequence, index) {
  let offset = 1;
  const colorMode = sequence[index + offset++];
  let color;
  if (colorMode === "2") {
    const rgb = [
      sequence[index + offset++],
      sequence[index + offset++],
      sequence[index + offset]
    ].map((x) => Number.parseInt(x));
    if (rgb.length === 3 && !rgb.some((x) => Number.isNaN(x))) {
      color = {
        type: "rgb",
        rgb
      };
    }
  } else if (colorMode === "5") {
    const colorIndex = Number.parseInt(sequence[index + offset]);
    if (!Number.isNaN(colorIndex)) {
      color = { type: "table", index: Number(colorIndex) };
    }
  }
  return [offset, color];
}
function parseSequence(sequence) {
  const commands = [];
  for (let i = 0; i < sequence.length; i++) {
    const code = sequence[i];
    const codeInt = Number.parseInt(code);
    if (Number.isNaN(codeInt))
      continue;
    if (codeInt === 0) {
      commands.push({ type: "resetAll" });
    } else if (codeInt <= 9) {
      const decoration = decorations[codeInt];
      if (decoration) {
        commands.push({
          type: "setDecoration",
          value: decorations[codeInt]
        });
      }
    } else if (codeInt <= 29) {
      const decoration = decorations[codeInt - 20];
      if (decoration) {
        commands.push({
          type: "resetDecoration",
          value: decoration
        });
      }
    } else if (codeInt <= 37) {
      commands.push({
        type: "setForegroundColor",
        value: { type: "named", name: namedColors[codeInt - 30] }
      });
    } else if (codeInt === 38) {
      const [offset, color] = parseColor(sequence, i);
      if (color) {
        commands.push({
          type: "setForegroundColor",
          value: color
        });
      }
      i += offset;
    } else if (codeInt === 39) {
      commands.push({
        type: "resetForegroundColor"
      });
    } else if (codeInt <= 47) {
      commands.push({
        type: "setBackgroundColor",
        value: { type: "named", name: namedColors[codeInt - 40] }
      });
    } else if (codeInt === 48) {
      const [offset, color] = parseColor(sequence, i);
      if (color) {
        commands.push({
          type: "setBackgroundColor",
          value: color
        });
      }
      i += offset;
    } else if (codeInt === 49) {
      commands.push({
        type: "resetBackgroundColor"
      });
    } else if (codeInt >= 90 && codeInt <= 97) {
      commands.push({
        type: "setForegroundColor",
        value: { type: "named", name: namedColors[codeInt - 90 + 8] }
      });
    } else if (codeInt >= 100 && codeInt <= 107) {
      commands.push({
        type: "setBackgroundColor",
        value: { type: "named", name: namedColors[codeInt - 100 + 8] }
      });
    }
  }
  return commands;
}
function createAnsiSequenceParser() {
  let foreground = null;
  let background = null;
  let decorations2 = /* @__PURE__ */ new Set();
  return {
    parse(value) {
      const tokens = [];
      let position = 0;
      do {
        const findResult = findSequence(value, position);
        const text = findResult.sequence ? value.substring(position, findResult.startPosition) : value.substring(position);
        if (text.length > 0) {
          tokens.push({
            value: text,
            foreground,
            background,
            decorations: new Set(decorations2)
          });
        }
        if (findResult.sequence) {
          const commands = parseSequence(findResult.sequence);
          for (const styleToken of commands) {
            if (styleToken.type === "resetAll") {
              foreground = null;
              background = null;
              decorations2.clear();
            } else if (styleToken.type === "resetForegroundColor") {
              foreground = null;
            } else if (styleToken.type === "resetBackgroundColor") {
              background = null;
            } else if (styleToken.type === "resetDecoration") {
              decorations2.delete(styleToken.value);
            }
          }
          for (const styleToken of commands) {
            if (styleToken.type === "setForegroundColor") {
              foreground = styleToken.value;
            } else if (styleToken.type === "setBackgroundColor") {
              background = styleToken.value;
            } else if (styleToken.type === "setDecoration") {
              decorations2.add(styleToken.value);
            }
          }
        }
        position = findResult.position;
      } while (position < value.length);
      return tokens;
    }
  };
}

// src/palette.ts
var defaultNamedColorsMap = {
  black: "#000000",
  red: "#bb0000",
  green: "#00bb00",
  yellow: "#bbbb00",
  blue: "#0000bb",
  magenta: "#ff00ff",
  cyan: "#00bbbb",
  white: "#eeeeee",
  brightBlack: "#555555",
  brightRed: "#ff5555",
  brightGreen: "#00ff00",
  brightYellow: "#ffff55",
  brightBlue: "#5555ff",
  brightMagenta: "#ff55ff",
  brightCyan: "#55ffff",
  brightWhite: "#ffffff"
};
function createColorPalette(namedColorsMap = defaultNamedColorsMap) {
  function namedColor(name) {
    return namedColorsMap[name];
  }
  function rgbColor(rgb) {
    return `#${rgb.map((x) => Math.max(0, Math.min(x, 255)).toString(16).padStart(2, "0")).join("")}`;
  }
  let colorTable;
  function getColorTable() {
    if (colorTable) {
      return colorTable;
    }
    colorTable = [];
    for (let i = 0; i < namedColors.length; i++) {
      colorTable.push(namedColor(namedColors[i]));
    }
    let levels = [0, 95, 135, 175, 215, 255];
    for (let r = 0; r < 6; r++) {
      for (let g = 0; g < 6; g++) {
        for (let b = 0; b < 6; b++) {
          colorTable.push(rgbColor([levels[r], levels[g], levels[b]]));
        }
      }
    }
    let level = 8;
    for (let i = 0; i < 24; i++, level += 10) {
      colorTable.push(rgbColor([level, level, level]));
    }
    return colorTable;
  }
  function tableColor(index) {
    return getColorTable()[index];
  }
  function value(color) {
    switch (color.type) {
      case "named":
        return namedColor(color.name);
      case "rgb":
        return rgbColor(color.rgb);
      case "table":
        return tableColor(color.index);
    }
  }
  return {
    value
  };
}

function tokenizeAnsiWithTheme(theme, fileContents, options) {
  const colorReplacements = resolveColorReplacements(theme, options);
  const lines = splitLines(fileContents);
  const colorPalette = createColorPalette(
    Object.fromEntries(
      namedColors.map((name) => [
        name,
        theme.colors?.[`terminal.ansi${name[0].toUpperCase()}${name.substring(1)}`]
      ])
    )
  );
  const parser = createAnsiSequenceParser();
  return lines.map(
    (line) => parser.parse(line[0]).map((token) => {
      let color;
      let bgColor;
      if (token.decorations.has("reverse")) {
        color = token.background ? colorPalette.value(token.background) : theme.bg;
        bgColor = token.foreground ? colorPalette.value(token.foreground) : theme.fg;
      } else {
        color = token.foreground ? colorPalette.value(token.foreground) : theme.fg;
        bgColor = token.background ? colorPalette.value(token.background) : undefined;
      }
      color = applyColorReplacements(color, colorReplacements);
      bgColor = applyColorReplacements(bgColor, colorReplacements);
      if (token.decorations.has("dim"))
        color = dimColor(color);
      let fontStyle = FontStyle.None;
      if (token.decorations.has("bold"))
        fontStyle |= FontStyle.Bold;
      if (token.decorations.has("italic"))
        fontStyle |= FontStyle.Italic;
      if (token.decorations.has("underline"))
        fontStyle |= FontStyle.Underline;
      return {
        content: token.value,
        offset: line[1],
        // TODO: more accurate offset? might need to fork ansi-sequence-parser
        color,
        bgColor,
        fontStyle
      };
    })
  );
}
function dimColor(color) {
  const hexMatch = color.match(/#([0-9a-f]{3})([0-9a-f]{3})?([0-9a-f]{2})?/);
  if (hexMatch) {
    if (hexMatch[3]) {
      const alpha = Math.round(Number.parseInt(hexMatch[3], 16) / 2).toString(16).padStart(2, "0");
      return `#${hexMatch[1]}${hexMatch[2]}${alpha}`;
    } else if (hexMatch[2]) {
      return `#${hexMatch[1]}${hexMatch[2]}80`;
    } else {
      return `#${Array.from(hexMatch[1]).map((x) => `${x}${x}`).join("")}80`;
    }
  }
  const cssVarMatch = color.match(/var\((--[\w-]+-ansi-[\w-]+)\)/);
  if (cssVarMatch)
    return `var(${cssVarMatch[1]}-dim)`;
  return color;
}

function codeToTokensBase(internal, code, options = {}) {
  const {
    lang = "text",
    theme: themeName = internal.getLoadedThemes()[0]
  } = options;
  if (isPlainLang(lang) || isNoneTheme(themeName))
    return splitLines(code).map((line) => [{ content: line[0], offset: line[1] }]);
  const { theme, colorMap } = internal.setTheme(themeName);
  if (lang === "ansi")
    return tokenizeAnsiWithTheme(theme, code, options);
  const _grammar = internal.getLanguage(lang);
  if (options.grammarState) {
    if (options.grammarState.lang !== _grammar.name) {
      throw new ShikiError$1(`Grammar state language "${options.grammarState.lang}" does not match highlight language "${_grammar.name}"`);
    }
    if (!options.grammarState.themes.includes(theme.name)) {
      throw new ShikiError$1(`Grammar state themes "${options.grammarState.themes}" do not contain highlight theme "${theme.name}"`);
    }
  }
  return tokenizeWithTheme(code, _grammar, theme, colorMap, options);
}
function tokenizeWithTheme(code, grammar, theme, colorMap, options) {
  const result = _tokenizeWithTheme(code, grammar, theme, colorMap, options);
  const grammarState = new GrammarState(
    _tokenizeWithTheme(code, grammar, theme, colorMap, options).stateStack,
    grammar.name,
    theme.name
  );
  setLastGrammarStateToMap(result.tokens, grammarState);
  return result.tokens;
}
function _tokenizeWithTheme(code, grammar, theme, colorMap, options) {
  const colorReplacements = resolveColorReplacements(theme, options);
  const {
    tokenizeMaxLineLength = 0,
    tokenizeTimeLimit = 500
  } = options;
  const lines = splitLines(code);
  let stateStack = options.grammarState ? getGrammarStack(options.grammarState, theme.name) ?? INITIAL : options.grammarContextCode != null ? _tokenizeWithTheme(
    options.grammarContextCode,
    grammar,
    theme,
    colorMap,
    {
      ...options,
      grammarState: undefined,
      grammarContextCode: undefined
    }
  ).stateStack : INITIAL;
  let actual = [];
  const final = [];
  for (let i = 0, len = lines.length; i < len; i++) {
    const [line, lineOffset] = lines[i];
    if (line === "") {
      actual = [];
      final.push([]);
      continue;
    }
    if (tokenizeMaxLineLength > 0 && line.length >= tokenizeMaxLineLength) {
      actual = [];
      final.push([{
        content: line,
        offset: lineOffset,
        color: "",
        fontStyle: 0
      }]);
      continue;
    }
    let resultWithScopes;
    let tokensWithScopes;
    let tokensWithScopesIndex;
    if (options.includeExplanation) {
      resultWithScopes = grammar.tokenizeLine(line, stateStack);
      tokensWithScopes = resultWithScopes.tokens;
      tokensWithScopesIndex = 0;
    }
    const result = grammar.tokenizeLine2(line, stateStack, tokenizeTimeLimit);
    const tokensLength = result.tokens.length / 2;
    for (let j = 0; j < tokensLength; j++) {
      const startIndex = result.tokens[2 * j];
      const nextStartIndex = j + 1 < tokensLength ? result.tokens[2 * j + 2] : line.length;
      if (startIndex === nextStartIndex)
        continue;
      const metadata = result.tokens[2 * j + 1];
      const color = applyColorReplacements(
        colorMap[EncodedTokenMetadata.getForeground(metadata)],
        colorReplacements
      );
      const fontStyle = EncodedTokenMetadata.getFontStyle(metadata);
      const token = {
        content: line.substring(startIndex, nextStartIndex),
        offset: lineOffset + startIndex,
        color,
        fontStyle
      };
      if (options.includeExplanation) {
        const themeSettingsSelectors = [];
        if (options.includeExplanation !== "scopeName") {
          for (const setting of theme.settings) {
            let selectors;
            switch (typeof setting.scope) {
              case "string":
                selectors = setting.scope.split(/,/).map((scope) => scope.trim());
                break;
              case "object":
                selectors = setting.scope;
                break;
              default:
                continue;
            }
            themeSettingsSelectors.push({
              settings: setting,
              selectors: selectors.map((selector) => selector.split(/ /))
            });
          }
        }
        token.explanation = [];
        let offset = 0;
        while (startIndex + offset < nextStartIndex) {
          const tokenWithScopes = tokensWithScopes[tokensWithScopesIndex];
          const tokenWithScopesText = line.substring(
            tokenWithScopes.startIndex,
            tokenWithScopes.endIndex
          );
          offset += tokenWithScopesText.length;
          token.explanation.push({
            content: tokenWithScopesText,
            scopes: options.includeExplanation === "scopeName" ? explainThemeScopesNameOnly(
              tokenWithScopes.scopes
            ) : explainThemeScopesFull(
              themeSettingsSelectors,
              tokenWithScopes.scopes
            )
          });
          tokensWithScopesIndex += 1;
        }
      }
      actual.push(token);
    }
    final.push(actual);
    actual = [];
    stateStack = result.ruleStack;
  }
  return {
    tokens: final,
    stateStack
  };
}
function explainThemeScopesNameOnly(scopes) {
  return scopes.map((scope) => ({ scopeName: scope }));
}
function explainThemeScopesFull(themeSelectors, scopes) {
  const result = [];
  for (let i = 0, len = scopes.length; i < len; i++) {
    const scope = scopes[i];
    result[i] = {
      scopeName: scope,
      themeMatches: explainThemeScope(themeSelectors, scope, scopes.slice(0, i))
    };
  }
  return result;
}
function matchesOne(selector, scope) {
  return selector === scope || scope.substring(0, selector.length) === selector && scope[selector.length] === ".";
}
function matches(selectors, scope, parentScopes) {
  if (!matchesOne(selectors[selectors.length - 1], scope))
    return false;
  let selectorParentIndex = selectors.length - 2;
  let parentIndex = parentScopes.length - 1;
  while (selectorParentIndex >= 0 && parentIndex >= 0) {
    if (matchesOne(selectors[selectorParentIndex], parentScopes[parentIndex]))
      selectorParentIndex -= 1;
    parentIndex -= 1;
  }
  if (selectorParentIndex === -1)
    return true;
  return false;
}
function explainThemeScope(themeSettingsSelectors, scope, parentScopes) {
  const result = [];
  for (const { selectors, settings } of themeSettingsSelectors) {
    for (const selectorPieces of selectors) {
      if (matches(selectorPieces, scope, parentScopes)) {
        result.push(settings);
        break;
      }
    }
  }
  return result;
}

function codeToTokensWithThemes(internal, code, options) {
  const themes = Object.entries(options.themes).filter((i) => i[1]).map((i) => ({ color: i[0], theme: i[1] }));
  const themedTokens = themes.map((t) => {
    const tokens2 = codeToTokensBase(internal, code, {
      ...options,
      theme: t.theme
    });
    const state = getLastGrammarStateFromMap(tokens2);
    const theme = typeof t.theme === "string" ? t.theme : t.theme.name;
    return {
      tokens: tokens2,
      state,
      theme
    };
  });
  const tokens = syncThemesTokenization(
    ...themedTokens.map((i) => i.tokens)
  );
  const mergedTokens = tokens[0].map(
    (line, lineIdx) => line.map((_token, tokenIdx) => {
      const mergedToken = {
        content: _token.content,
        variants: {},
        offset: _token.offset
      };
      if ("includeExplanation" in options && options.includeExplanation) {
        mergedToken.explanation = _token.explanation;
      }
      tokens.forEach((t, themeIdx) => {
        const {
          content: _,
          explanation: __,
          offset: ___,
          ...styles
        } = t[lineIdx][tokenIdx];
        mergedToken.variants[themes[themeIdx].color] = styles;
      });
      return mergedToken;
    })
  );
  const mergedGrammarState = themedTokens[0].state ? new GrammarState(
    Object.fromEntries(themedTokens.map((s) => [s.theme, s.state?.getInternalStack(s.theme)])),
    themedTokens[0].state.lang
  ) : undefined;
  if (mergedGrammarState)
    setLastGrammarStateToMap(mergedTokens, mergedGrammarState);
  return mergedTokens;
}
function syncThemesTokenization(...themes) {
  const outThemes = themes.map(() => []);
  const count = themes.length;
  for (let i = 0; i < themes[0].length; i++) {
    const lines = themes.map((t) => t[i]);
    const outLines = outThemes.map(() => []);
    outThemes.forEach((t, i2) => t.push(outLines[i2]));
    const indexes = lines.map(() => 0);
    const current = lines.map((l) => l[0]);
    while (current.every((t) => t)) {
      const minLength = Math.min(...current.map((t) => t.content.length));
      for (let n = 0; n < count; n++) {
        const token = current[n];
        if (token.content.length === minLength) {
          outLines[n].push(token);
          indexes[n] += 1;
          current[n] = lines[n][indexes[n]];
        } else {
          outLines[n].push({
            ...token,
            content: token.content.slice(0, minLength)
          });
          current[n] = {
            ...token,
            content: token.content.slice(minLength),
            offset: token.offset + minLength
          };
        }
      }
    }
  }
  return outThemes;
}

const VSCODE_FALLBACK_EDITOR_FG = { light: "#333333", dark: "#bbbbbb" };
const VSCODE_FALLBACK_EDITOR_BG = { light: "#fffffe", dark: "#1e1e1e" };
const RESOLVED_KEY = "__shiki_resolved";
function normalizeTheme(rawTheme) {
  if (rawTheme?.[RESOLVED_KEY])
    return rawTheme;
  const theme = {
    ...rawTheme
  };
  if (theme.tokenColors && !theme.settings) {
    theme.settings = theme.tokenColors;
    delete theme.tokenColors;
  }
  theme.type ||= "dark";
  theme.colorReplacements = { ...theme.colorReplacements };
  theme.settings ||= [];
  let { bg, fg } = theme;
  if (!bg || !fg) {
    const globalSetting = theme.settings ? theme.settings.find((s) => !s.name && !s.scope) : undefined;
    if (globalSetting?.settings?.foreground)
      fg = globalSetting.settings.foreground;
    if (globalSetting?.settings?.background)
      bg = globalSetting.settings.background;
    if (!fg && theme?.colors?.["editor.foreground"])
      fg = theme.colors["editor.foreground"];
    if (!bg && theme?.colors?.["editor.background"])
      bg = theme.colors["editor.background"];
    if (!fg)
      fg = theme.type === "light" ? VSCODE_FALLBACK_EDITOR_FG.light : VSCODE_FALLBACK_EDITOR_FG.dark;
    if (!bg)
      bg = theme.type === "light" ? VSCODE_FALLBACK_EDITOR_BG.light : VSCODE_FALLBACK_EDITOR_BG.dark;
    theme.fg = fg;
    theme.bg = bg;
  }
  if (!(theme.settings[0] && theme.settings[0].settings && !theme.settings[0].scope)) {
    theme.settings.unshift({
      settings: {
        foreground: theme.fg,
        background: theme.bg
      }
    });
  }
  let replacementCount = 0;
  const replacementMap = /* @__PURE__ */ new Map();
  function getReplacementColor(value) {
    if (replacementMap.has(value))
      return replacementMap.get(value);
    replacementCount += 1;
    const hex = `#${replacementCount.toString(16).padStart(8, "0").toLowerCase()}`;
    if (theme.colorReplacements?.[`#${hex}`])
      return getReplacementColor(value);
    replacementMap.set(value, hex);
    return hex;
  }
  theme.settings = theme.settings.map((setting) => {
    const replaceFg = setting.settings?.foreground && !setting.settings.foreground.startsWith("#");
    const replaceBg = setting.settings?.background && !setting.settings.background.startsWith("#");
    if (!replaceFg && !replaceBg)
      return setting;
    const clone = {
      ...setting,
      settings: {
        ...setting.settings
      }
    };
    if (replaceFg) {
      const replacement = getReplacementColor(setting.settings.foreground);
      theme.colorReplacements[replacement] = setting.settings.foreground;
      clone.settings.foreground = replacement;
    }
    if (replaceBg) {
      const replacement = getReplacementColor(setting.settings.background);
      theme.colorReplacements[replacement] = setting.settings.background;
      clone.settings.background = replacement;
    }
    return clone;
  });
  for (const key of Object.keys(theme.colors || {})) {
    if (key === "editor.foreground" || key === "editor.background" || key.startsWith("terminal.ansi")) {
      if (!theme.colors[key]?.startsWith("#")) {
        const replacement = getReplacementColor(theme.colors[key]);
        theme.colorReplacements[replacement] = theme.colors[key];
        theme.colors[key] = replacement;
      }
    }
  }
  Object.defineProperty(theme, RESOLVED_KEY, {
    enumerable: false,
    writable: false,
    value: true
  });
  return theme;
}

async function resolveLangs(langs) {
  return Array.from(new Set((await Promise.all(
    langs.filter((l) => !isSpecialLang(l)).map(async (lang) => await normalizeGetter(lang).then((r) => Array.isArray(r) ? r : [r]))
  )).flat()));
}
async function resolveThemes(themes) {
  const resolved = await Promise.all(
    themes.map(
      async (theme) => isSpecialTheme(theme) ? null : normalizeTheme(await normalizeGetter(theme))
    )
  );
  return resolved.filter((i) => !!i);
}

class Registry extends Registry$1 {
  constructor(_resolver, _themes, _langs, _alias = {}) {
    super(_resolver);
    this._resolver = _resolver;
    this._themes = _themes;
    this._langs = _langs;
    this._alias = _alias;
    this._themes.map((t) => this.loadTheme(t));
    this.loadLanguages(this._langs);
  }
  _resolvedThemes = /* @__PURE__ */ new Map();
  _resolvedGrammars = /* @__PURE__ */ new Map();
  _langMap = /* @__PURE__ */ new Map();
  _langGraph = /* @__PURE__ */ new Map();
  _textmateThemeCache = /* @__PURE__ */ new WeakMap();
  _loadedThemesCache = null;
  _loadedLanguagesCache = null;
  getTheme(theme) {
    if (typeof theme === "string")
      return this._resolvedThemes.get(theme);
    else
      return this.loadTheme(theme);
  }
  loadTheme(theme) {
    const _theme = normalizeTheme(theme);
    if (_theme.name) {
      this._resolvedThemes.set(_theme.name, _theme);
      this._loadedThemesCache = null;
    }
    return _theme;
  }
  getLoadedThemes() {
    if (!this._loadedThemesCache)
      this._loadedThemesCache = [...this._resolvedThemes.keys()];
    return this._loadedThemesCache;
  }
  // Override and re-implement this method to cache the textmate themes as `TextMateTheme.createFromRawTheme`
  // is expensive. Themes can switch often especially for dual-theme support.
  //
  // The parent class also accepts `colorMap` as the second parameter, but since we don't use that,
  // we omit here so it's easier to cache the themes.
  setTheme(theme) {
    let textmateTheme = this._textmateThemeCache.get(theme);
    if (!textmateTheme) {
      textmateTheme = Theme.createFromRawTheme(theme);
      this._textmateThemeCache.set(theme, textmateTheme);
    }
    this._syncRegistry.setTheme(textmateTheme);
  }
  getGrammar(name) {
    if (this._alias[name]) {
      const resolved = /* @__PURE__ */ new Set([name]);
      while (this._alias[name]) {
        name = this._alias[name];
        if (resolved.has(name))
          throw new ShikiError(`Circular alias \`${Array.from(resolved).join(" -> ")} -> ${name}\``);
        resolved.add(name);
      }
    }
    return this._resolvedGrammars.get(name);
  }
  loadLanguage(lang) {
    if (this.getGrammar(lang.name))
      return;
    const embeddedLazilyBy = new Set(
      [...this._langMap.values()].filter((i) => i.embeddedLangsLazy?.includes(lang.name))
    );
    this._resolver.addLanguage(lang);
    const grammarConfig = {
      balancedBracketSelectors: lang.balancedBracketSelectors || ["*"],
      unbalancedBracketSelectors: lang.unbalancedBracketSelectors || []
    };
    this._syncRegistry._rawGrammars.set(lang.scopeName, lang);
    const g = this.loadGrammarWithConfiguration(lang.scopeName, 1, grammarConfig);
    g.name = lang.name;
    this._resolvedGrammars.set(lang.name, g);
    if (lang.aliases) {
      lang.aliases.forEach((alias) => {
        this._alias[alias] = lang.name;
      });
    }
    this._loadedLanguagesCache = null;
    if (embeddedLazilyBy.size) {
      for (const e of embeddedLazilyBy) {
        this._resolvedGrammars.delete(e.name);
        this._loadedLanguagesCache = null;
        this._syncRegistry?._injectionGrammars?.delete(e.scopeName);
        this._syncRegistry?._grammars?.delete(e.scopeName);
        this.loadLanguage(this._langMap.get(e.name));
      }
    }
  }
  dispose() {
    super.dispose();
    this._resolvedThemes.clear();
    this._resolvedGrammars.clear();
    this._langMap.clear();
    this._langGraph.clear();
    this._loadedThemesCache = null;
  }
  loadLanguages(langs) {
    for (const lang of langs)
      this.resolveEmbeddedLanguages(lang);
    const langsGraphArray = Array.from(this._langGraph.entries());
    const missingLangs = langsGraphArray.filter(([_, lang]) => !lang);
    if (missingLangs.length) {
      const dependents = langsGraphArray.filter(([_, lang]) => lang && lang.embeddedLangs?.some((l) => missingLangs.map(([name]) => name).includes(l))).filter((lang) => !missingLangs.includes(lang));
      throw new ShikiError(`Missing languages ${missingLangs.map(([name]) => `\`${name}\``).join(", ")}, required by ${dependents.map(([name]) => `\`${name}\``).join(", ")}`);
    }
    for (const [_, lang] of langsGraphArray)
      this._resolver.addLanguage(lang);
    for (const [_, lang] of langsGraphArray)
      this.loadLanguage(lang);
  }
  getLoadedLanguages() {
    if (!this._loadedLanguagesCache) {
      this._loadedLanguagesCache = [
        .../* @__PURE__ */ new Set([...this._resolvedGrammars.keys(), ...Object.keys(this._alias)])
      ];
    }
    return this._loadedLanguagesCache;
  }
  resolveEmbeddedLanguages(lang) {
    this._langMap.set(lang.name, lang);
    this._langGraph.set(lang.name, lang);
    if (lang.embeddedLangs) {
      for (const embeddedLang of lang.embeddedLangs)
        this._langGraph.set(embeddedLang, this._langMap.get(embeddedLang));
    }
  }
}

class Resolver {
  _langs = /* @__PURE__ */ new Map();
  _scopeToLang = /* @__PURE__ */ new Map();
  _injections = /* @__PURE__ */ new Map();
  _onigLib;
  constructor(engine, langs) {
    this._onigLib = {
      createOnigScanner: (patterns) => engine.createScanner(patterns),
      createOnigString: (s) => engine.createString(s)
    };
    langs.forEach((i) => this.addLanguage(i));
  }
  get onigLib() {
    return this._onigLib;
  }
  getLangRegistration(langIdOrAlias) {
    return this._langs.get(langIdOrAlias);
  }
  loadGrammar(scopeName) {
    return this._scopeToLang.get(scopeName);
  }
  addLanguage(l) {
    this._langs.set(l.name, l);
    if (l.aliases) {
      l.aliases.forEach((a) => {
        this._langs.set(a, l);
      });
    }
    this._scopeToLang.set(l.scopeName, l);
    if (l.injectTo) {
      l.injectTo.forEach((i) => {
        if (!this._injections.get(i))
          this._injections.set(i, []);
        this._injections.get(i).push(l.scopeName);
      });
    }
  }
  getInjections(scopeName) {
    const scopeParts = scopeName.split(".");
    let injections = [];
    for (let i = 1; i <= scopeParts.length; i++) {
      const subScopeName = scopeParts.slice(0, i).join(".");
      injections = [...injections, ...this._injections.get(subScopeName) || []];
    }
    return injections;
  }
}

let instancesCount = 0;
function createShikiInternalSync(options) {
  instancesCount += 1;
  if (options.warnings !== false && instancesCount >= 10 && instancesCount % 10 === 0)
    console.warn(`[Shiki] ${instancesCount} instances have been created. Shiki is supposed to be used as a singleton, consider refactoring your code to cache your highlighter instance; Or call \`highlighter.dispose()\` to release unused instances.`);
  let isDisposed = false;
  if (!options.engine)
    throw new ShikiError("`engine` option is required for synchronous mode");
  const langs = (options.langs || []).flat(1);
  const themes = (options.themes || []).flat(1).map(normalizeTheme);
  const resolver = new Resolver(options.engine, langs);
  const _registry = new Registry(resolver, themes, langs, options.langAlias);
  let _lastTheme;
  function getLanguage(name) {
    ensureNotDisposed();
    const _lang = _registry.getGrammar(typeof name === "string" ? name : name.name);
    if (!_lang)
      throw new ShikiError(`Language \`${name}\` not found, you may need to load it first`);
    return _lang;
  }
  function getTheme(name) {
    if (name === "none")
      return { bg: "", fg: "", name: "none", settings: [], type: "dark" };
    ensureNotDisposed();
    const _theme = _registry.getTheme(name);
    if (!_theme)
      throw new ShikiError(`Theme \`${name}\` not found, you may need to load it first`);
    return _theme;
  }
  function setTheme(name) {
    ensureNotDisposed();
    const theme = getTheme(name);
    if (_lastTheme !== name) {
      _registry.setTheme(theme);
      _lastTheme = name;
    }
    const colorMap = _registry.getColorMap();
    return {
      theme,
      colorMap
    };
  }
  function getLoadedThemes() {
    ensureNotDisposed();
    return _registry.getLoadedThemes();
  }
  function getLoadedLanguages() {
    ensureNotDisposed();
    return _registry.getLoadedLanguages();
  }
  function loadLanguageSync(...langs2) {
    ensureNotDisposed();
    _registry.loadLanguages(langs2.flat(1));
  }
  async function loadLanguage(...langs2) {
    return loadLanguageSync(await resolveLangs(langs2));
  }
  function loadThemeSync(...themes2) {
    ensureNotDisposed();
    for (const theme of themes2.flat(1)) {
      _registry.loadTheme(theme);
    }
  }
  async function loadTheme(...themes2) {
    ensureNotDisposed();
    return loadThemeSync(await resolveThemes(themes2));
  }
  function ensureNotDisposed() {
    if (isDisposed)
      throw new ShikiError("Shiki instance has been disposed");
  }
  function dispose() {
    if (isDisposed)
      return;
    isDisposed = true;
    _registry.dispose();
    instancesCount -= 1;
  }
  return {
    setTheme,
    getTheme,
    getLanguage,
    getLoadedThemes,
    getLoadedLanguages,
    loadLanguage,
    loadLanguageSync,
    loadTheme,
    loadThemeSync,
    dispose,
    [Symbol.dispose]: dispose
  };
}

async function createShikiInternal(options = {}) {
  if (options.loadWasm) ;
  const [
    themes,
    langs,
    engine
  ] = await Promise.all([
    resolveThemes(options.themes || []),
    resolveLangs(options.langs || []),
    options.engine || createOnigurumaEngine$1(options.loadWasm || getDefaultWasmLoader())
  ]);
  return createShikiInternalSync({
    ...options,
    loadWasm: undefined,
    themes,
    langs,
    engine
  });
}

async function loadBuiltinWasm() {
    const __dirname = dirname(fileURLToPath(import.meta.url));
    await loadWasm(readFile(__dirname + "/onig.wasm"));
}
function loadWasm(options) {
    return onig.loadWasm(options);
}
function createOnigurumaEngine(options) {
    return onig.createOnigurumaEngine(options);
}

export { bundledLanguages, bundledLanguagesInfo, bundledThemes, bundledThemesInfo, codeToTokensWithThemes, createOnigurumaEngine, createShikiInternal, loadBuiltinWasm, loadWasm };
