exports.id = 794
exports.ids = [794]
exports.modules = {
  /***/ 2508: /***/ () => {
    Prism.languages.clike = {
      comment: [
        { pattern: /(^|[^\\])\/\*[\s\S]*?(?:\*\/|$)/, lookbehind: !0, greedy: !0 },
        { pattern: /(^|[^\\:])\/\/.*/, lookbehind: !0, greedy: !0 },
      ],
      string: { pattern: /(["'])(?:\\(?:\r\n|[\s\S])|(?!\1)[^\\\r\n])*\1/, greedy: !0 },
      'class-name': {
        pattern:
          /(\b(?:class|extends|implements|instanceof|interface|new|trait)\s+|\bcatch\s+\()[\w.\\]+/i,
        lookbehind: !0,
        inside: { punctuation: /[.\\]/ },
      },
      keyword:
        /\b(?:break|catch|continue|do|else|finally|for|function|if|in|instanceof|new|null|return|throw|try|while)\b/,
      boolean: /\b(?:false|true)\b/,
      function: /\b\w+(?=\()/,
      number: /\b0x[\da-f]+\b|(?:\b\d+(?:\.\d*)?|\B\.\d+)(?:e[+-]?\d+)?/i,
      operator: /[<>]=?|[!=]=?=?|--?|\+\+?|&&?|\|\|?|[?*/~^%]/,
      punctuation: /[{}[\];(),.:]/,
    }

    /***/
  },

  /***/ 1151: /***/ () => {
    !(function (e) {
      var a,
        n = /("|')(?:\\(?:\r\n|[\s\S])|(?!\1)[^\\\r\n])*\1/
      ;(e.languages.css.selector = {
        pattern: e.languages.css.selector.pattern,
        lookbehind: !0,
        inside: (a = {
          'pseudo-element': /:(?:after|before|first-letter|first-line|selection)|::[-\w]+/,
          'pseudo-class': /:[-\w]+/,
          class: /\.[-\w]+/,
          id: /#[-\w]+/,
          attribute: {
            pattern: RegExp('\\[(?:[^[\\]"\']|' + n.source + ')*\\]'),
            greedy: !0,
            inside: {
              punctuation: /^\[|\]$/,
              'case-sensitivity': { pattern: /(\s)[si]$/i, lookbehind: !0, alias: 'keyword' },
              namespace: {
                pattern: /^(\s*)(?:(?!\s)[-*\w\xA0-\uFFFF])*\|(?!=)/,
                lookbehind: !0,
                inside: { punctuation: /\|$/ },
              },
              'attr-name': { pattern: /^(\s*)(?:(?!\s)[-\w\xA0-\uFFFF])+/, lookbehind: !0 },
              'attr-value': [
                n,
                { pattern: /(=\s*)(?:(?!\s)[-\w\xA0-\uFFFF])+(?=\s*$)/, lookbehind: !0 },
              ],
              operator: /[|~*^$]?=/,
            },
          },
          'n-th': [
            {
              pattern: /(\(\s*)[+-]?\d*[\dn](?:\s*[+-]\s*\d+)?(?=\s*\))/,
              lookbehind: !0,
              inside: { number: /[\dn]+/, operator: /[+-]/ },
            },
            { pattern: /(\(\s*)(?:even|odd)(?=\s*\))/i, lookbehind: !0 },
          ],
          combinator: />|\+|~|\|\|/,
          punctuation: /[(),]/,
        }),
      }),
        (e.languages.css.atrule.inside['selector-function-argument'].inside = a),
        e.languages.insertBefore('css', 'property', {
          variable: {
            pattern: /(^|[^-\w\xA0-\uFFFF])--(?!\s)[-_a-z\xA0-\uFFFF](?:(?!\s)[-\w\xA0-\uFFFF])*/i,
            lookbehind: !0,
          },
        })
      var r = { pattern: /(\b\d+)(?:%|[a-z]+(?![\w-]))/, lookbehind: !0 },
        i = { pattern: /(^|[^\w.-])-?(?:\d+(?:\.\d+)?|\.\d+)/, lookbehind: !0 }
      e.languages.insertBefore('css', 'function', {
        operator: { pattern: /(\s)[+\-*\/](?=\s)/, lookbehind: !0 },
        hexcode: { pattern: /\B#[\da-f]{3,8}\b/i, alias: 'color' },
        color: [
          {
            pattern:
              /(^|[^\w-])(?:AliceBlue|AntiqueWhite|Aqua|Aquamarine|Azure|Beige|Bisque|Black|BlanchedAlmond|Blue|BlueViolet|Brown|BurlyWood|CadetBlue|Chartreuse|Chocolate|Coral|CornflowerBlue|Cornsilk|Crimson|Cyan|DarkBlue|DarkCyan|DarkGoldenRod|DarkGr[ae]y|DarkGreen|DarkKhaki|DarkMagenta|DarkOliveGreen|DarkOrange|DarkOrchid|DarkRed|DarkSalmon|DarkSeaGreen|DarkSlateBlue|DarkSlateGr[ae]y|DarkTurquoise|DarkViolet|DeepPink|DeepSkyBlue|DimGr[ae]y|DodgerBlue|FireBrick|FloralWhite|ForestGreen|Fuchsia|Gainsboro|GhostWhite|Gold|GoldenRod|Gr[ae]y|Green|GreenYellow|HoneyDew|HotPink|IndianRed|Indigo|Ivory|Khaki|Lavender|LavenderBlush|LawnGreen|LemonChiffon|LightBlue|LightCoral|LightCyan|LightGoldenRodYellow|LightGr[ae]y|LightGreen|LightPink|LightSalmon|LightSeaGreen|LightSkyBlue|LightSlateGr[ae]y|LightSteelBlue|LightYellow|Lime|LimeGreen|Linen|Magenta|Maroon|MediumAquaMarine|MediumBlue|MediumOrchid|MediumPurple|MediumSeaGreen|MediumSlateBlue|MediumSpringGreen|MediumTurquoise|MediumVioletRed|MidnightBlue|MintCream|MistyRose|Moccasin|NavajoWhite|Navy|OldLace|Olive|OliveDrab|Orange|OrangeRed|Orchid|PaleGoldenRod|PaleGreen|PaleTurquoise|PaleVioletRed|PapayaWhip|PeachPuff|Peru|Pink|Plum|PowderBlue|Purple|RebeccaPurple|Red|RosyBrown|RoyalBlue|SaddleBrown|Salmon|SandyBrown|SeaGreen|SeaShell|Sienna|Silver|SkyBlue|SlateBlue|SlateGr[ae]y|Snow|SpringGreen|SteelBlue|Tan|Teal|Thistle|Tomato|Transparent|Turquoise|Violet|Wheat|White|WhiteSmoke|Yellow|YellowGreen)(?![\w-])/i,
            lookbehind: !0,
          },
          {
            pattern:
              /\b(?:hsl|rgb)\(\s*\d{1,3}\s*,\s*\d{1,3}%?\s*,\s*\d{1,3}%?\s*\)\B|\b(?:hsl|rgb)a\(\s*\d{1,3}\s*,\s*\d{1,3}%?\s*,\s*\d{1,3}%?\s*,\s*(?:0|0?\.\d+|1)\s*\)\B/i,
            inside: { unit: r, number: i, function: /[\w-]+(?=\()/, punctuation: /[(),]/ },
          },
        ],
        entity: /\\[\da-f]{1,8}/i,
        unit: r,
        number: i,
      })
    })(Prism)

    /***/
  },

  /***/ 1139: /***/ () => {
    !(function (s) {
      var e = /(?:"(?:\\(?:\r\n|[\s\S])|[^"\\\r\n])*"|'(?:\\(?:\r\n|[\s\S])|[^'\\\r\n])*')/
      ;(s.languages.css = {
        comment: /\/\*[\s\S]*?\*\//,
        atrule: {
          pattern: RegExp('@[\\w-](?:[^;{\\s"\']|\\s+(?!\\s)|' + e.source + ')*?(?:;|(?=\\s*\\{))'),
          inside: {
            rule: /^@[\w-]+/,
            'selector-function-argument': {
              pattern:
                /(\bselector\s*\(\s*(?![\s)]))(?:[^()\s]|\s+(?![\s)])|\((?:[^()]|\([^()]*\))*\))+(?=\s*\))/,
              lookbehind: !0,
              alias: 'selector',
            },
            keyword: { pattern: /(^|[^\w-])(?:and|not|only|or)(?![\w-])/, lookbehind: !0 },
          },
        },
        url: {
          pattern: RegExp('\\burl\\((?:' + e.source + '|(?:[^\\\\\r\n()"\']|\\\\[^])*)\\)', 'i'),
          greedy: !0,
          inside: {
            function: /^url/i,
            punctuation: /^\(|\)$/,
            string: { pattern: RegExp('^' + e.source + '$'), alias: 'url' },
          },
        },
        selector: {
          pattern: RegExp(
            '(^|[{}\\s])[^{}\\s](?:[^{};"\'\\s]|\\s+(?![\\s{])|' + e.source + ')*(?=\\s*\\{)'
          ),
          lookbehind: !0,
        },
        string: { pattern: e, greedy: !0 },
        property: {
          pattern:
            /(^|[^-\w\xA0-\uFFFF])(?!\s)[-_a-z\xA0-\uFFFF](?:(?!\s)[-\w\xA0-\uFFFF])*(?=\s*:)/i,
          lookbehind: !0,
        },
        important: /!important\b/i,
        function: { pattern: /(^|[^-a-z0-9])[-a-z0-9]+(?=\()/i, lookbehind: !0 },
        punctuation: /[(){};:,]/,
      }),
        (s.languages.css.atrule.inside.rest = s.languages.css)
      var t = s.languages.markup
      t && (t.tag.addInlined('style', 'css'), t.tag.addAttribute('style', 'css'))
    })(Prism)

    /***/
  },

  /***/ 1855: /***/ () => {
    ;(Prism.languages.javascript = Prism.languages.extend('clike', {
      'class-name': [
        Prism.languages.clike['class-name'],
        {
          pattern:
            /(^|[^$\w\xA0-\uFFFF])(?!\s)[_$A-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\.(?:constructor|prototype))/,
          lookbehind: !0,
        },
      ],
      keyword: [
        { pattern: /((?:^|\})\s*)catch\b/, lookbehind: !0 },
        {
          pattern:
            /(^|[^.]|\.\.\.\s*)\b(?:as|assert(?=\s*\{)|async(?=\s*(?:function\b|\(|[$\w\xA0-\uFFFF]|$))|await|break|case|class|const|continue|debugger|default|delete|do|else|enum|export|extends|finally(?=\s*(?:\{|$))|for|from(?=\s*(?:['"]|$))|function|(?:get|set)(?=\s*(?:[#\[$\w\xA0-\uFFFF]|$))|if|implements|import|in|instanceof|interface|let|new|null|of|package|private|protected|public|return|static|super|switch|this|throw|try|typeof|undefined|var|void|while|with|yield)\b/,
          lookbehind: !0,
        },
      ],
      function:
        /#?(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\s*(?:\.\s*(?:apply|bind|call)\s*)?\()/,
      number: {
        pattern: RegExp(
          '(^|[^\\w$])(?:NaN|Infinity|0[bB][01]+(?:_[01]+)*n?|0[oO][0-7]+(?:_[0-7]+)*n?|0[xX][\\dA-Fa-f]+(?:_[\\dA-Fa-f]+)*n?|\\d+(?:_\\d+)*n|(?:\\d+(?:_\\d+)*(?:\\.(?:\\d+(?:_\\d+)*)?)?|\\.\\d+(?:_\\d+)*)(?:[Ee][+-]?\\d+(?:_\\d+)*)?)(?![\\w$])'
        ),
        lookbehind: !0,
      },
      operator:
        /--|\+\+|\*\*=?|=>|&&=?|\|\|=?|[!=]==|<<=?|>>>?=?|[-+*/%&|^!=<>]=?|\.{3}|\?\?=?|\?\.?|[~:]/,
    })),
      (Prism.languages.javascript['class-name'][0].pattern =
        /(\b(?:class|extends|implements|instanceof|interface|new)\s+)[\w.\\]+/),
      Prism.languages.insertBefore('javascript', 'keyword', {
        regex: {
          pattern: RegExp(
            '((?:^|[^$\\w\\xA0-\\uFFFF."\'\\])\\s]|\\b(?:return|yield))\\s*)/(?:(?:\\[(?:[^\\]\\\\\r\n]|\\\\.)*\\]|\\\\.|[^/\\\\\\[\r\n])+/[dgimyus]{0,7}|(?:\\[(?:[^[\\]\\\\\r\n]|\\\\.|\\[(?:[^[\\]\\\\\r\n]|\\\\.|\\[(?:[^[\\]\\\\\r\n]|\\\\.)*\\])*\\])*\\]|\\\\.|[^/\\\\\\[\r\n])+/[dgimyus]{0,7}v[dgimyus]{0,7})(?=(?:\\s|/\\*(?:[^*]|\\*(?!/))*\\*/)*(?:$|[\r\n,.;:})\\]]|//))'
          ),
          lookbehind: !0,
          greedy: !0,
          inside: {
            'regex-source': {
              pattern: /^(\/)[\s\S]+(?=\/[a-z]*$)/,
              lookbehind: !0,
              alias: 'language-regex',
              inside: Prism.languages.regex,
            },
            'regex-delimiter': /^\/|\/$/,
            'regex-flags': /^[a-z]+$/,
          },
        },
        'function-variable': {
          pattern:
            /#?(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\s*[=:]\s*(?:async\s*)?(?:\bfunction\b|(?:\((?:[^()]|\([^()]*\))*\)|(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*)\s*=>))/,
          alias: 'function',
        },
        parameter: [
          {
            pattern:
              /(function(?:\s+(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*)?\s*\(\s*)(?!\s)(?:[^()\s]|\s+(?![\s)])|\([^()]*\))+(?=\s*\))/,
            lookbehind: !0,
            inside: Prism.languages.javascript,
          },
          {
            pattern:
              /(^|[^$\w\xA0-\uFFFF])(?!\s)[_$a-z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\s*=>)/i,
            lookbehind: !0,
            inside: Prism.languages.javascript,
          },
          {
            pattern: /(\(\s*)(?!\s)(?:[^()\s]|\s+(?![\s)])|\([^()]*\))+(?=\s*\)\s*=>)/,
            lookbehind: !0,
            inside: Prism.languages.javascript,
          },
          {
            pattern:
              /((?:\b|\s|^)(?!(?:as|async|await|break|case|catch|class|const|continue|debugger|default|delete|do|else|enum|export|extends|finally|for|from|function|get|if|implements|import|in|instanceof|interface|let|new|null|of|package|private|protected|public|return|set|static|super|switch|this|throw|try|typeof|undefined|var|void|while|with|yield)(?![$\w\xA0-\uFFFF]))(?:(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*\s*)\(\s*|\]\s*\(\s*)(?!\s)(?:[^()\s]|\s+(?![\s)])|\([^()]*\))+(?=\s*\)\s*\{)/,
            lookbehind: !0,
            inside: Prism.languages.javascript,
          },
        ],
        constant: /\b[A-Z](?:[A-Z_]|\dx?)*\b/,
      }),
      Prism.languages.insertBefore('javascript', 'string', {
        hashbang: { pattern: /^#!.*/, greedy: !0, alias: 'comment' },
        'template-string': {
          pattern: /`(?:\\[\s\S]|\$\{(?:[^{}]|\{(?:[^{}]|\{[^}]*\})*\})+\}|(?!\$\{)[^\\`])*`/,
          greedy: !0,
          inside: {
            'template-punctuation': { pattern: /^`|`$/, alias: 'string' },
            interpolation: {
              pattern: /((?:^|[^\\])(?:\\{2})*)\$\{(?:[^{}]|\{(?:[^{}]|\{[^}]*\})*\})+\}/,
              lookbehind: !0,
              inside: {
                'interpolation-punctuation': { pattern: /^\$\{|\}$/, alias: 'punctuation' },
                rest: Prism.languages.javascript,
              },
            },
            string: /[\s\S]+/,
          },
        },
        'string-property': {
          pattern: /((?:^|[,{])[ \t]*)(["'])(?:\\(?:\r\n|[\s\S])|(?!\2)[^\\\r\n])*\2(?=\s*:)/m,
          lookbehind: !0,
          greedy: !0,
          alias: 'property',
        },
      }),
      Prism.languages.insertBefore('javascript', 'operator', {
        'literal-property': {
          pattern:
            /((?:^|[,{])[ \t]*)(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\s*:)/m,
          lookbehind: !0,
          alias: 'property',
        },
      }),
      Prism.languages.markup &&
        (Prism.languages.markup.tag.addInlined('script', 'javascript'),
        Prism.languages.markup.tag.addAttribute(
          'on(?:abort|blur|change|click|composition(?:end|start|update)|dblclick|error|focus(?:in|out)?|key(?:down|up)|load|mouse(?:down|enter|leave|move|out|over|up)|reset|resize|scroll|select|slotchange|submit|unload|wheel)',
          'javascript'
        )),
      (Prism.languages.js = Prism.languages.javascript)

    /***/
  },

  /***/ 3784: /***/ () => {
    !(function (a) {
      function e(a, e) {
        return RegExp(
          a.replace(/<ID>/g, function () {
            return '(?!\\s)[_$a-zA-Z\\xA0-\\uFFFF](?:(?!\\s)[$\\w\\xA0-\\uFFFF])*'
          }),
          e
        )
      }
      a.languages.insertBefore('javascript', 'function-variable', {
        'method-variable': {
          pattern: RegExp('(\\.\\s*)' + a.languages.javascript['function-variable'].pattern.source),
          lookbehind: !0,
          alias: ['function-variable', 'method', 'function', 'property-access'],
        },
      }),
        a.languages.insertBefore('javascript', 'function', {
          method: {
            pattern: RegExp('(\\.\\s*)' + a.languages.javascript.function.source),
            lookbehind: !0,
            alias: ['function', 'property-access'],
          },
        }),
        a.languages.insertBefore('javascript', 'constant', {
          'known-class-name': [
            {
              pattern:
                /\b(?:(?:Float(?:32|64)|(?:Int|Uint)(?:8|16|32)|Uint8Clamped)?Array|ArrayBuffer|BigInt|Boolean|DataView|Date|Error|Function|Intl|JSON|(?:Weak)?(?:Map|Set)|Math|Number|Object|Promise|Proxy|Reflect|RegExp|String|Symbol|WebAssembly)\b/,
              alias: 'class-name',
            },
            { pattern: /\b(?:[A-Z]\w*)Error\b/, alias: 'class-name' },
          ],
        }),
        a.languages.insertBefore('javascript', 'keyword', {
          imports: {
            pattern: e(
              '(\\bimport\\b\\s*)(?:<ID>(?:\\s*,\\s*(?:\\*\\s*as\\s+<ID>|\\{[^{}]*\\}))?|\\*\\s*as\\s+<ID>|\\{[^{}]*\\})(?=\\s*\\bfrom\\b)'
            ),
            lookbehind: !0,
            inside: a.languages.javascript,
          },
          exports: {
            pattern: e(
              '(\\bexport\\b\\s*)(?:\\*(?:\\s*as\\s+<ID>)?(?=\\s*\\bfrom\\b)|\\{[^{}]*\\})'
            ),
            lookbehind: !0,
            inside: a.languages.javascript,
          },
        }),
        a.languages.javascript.keyword.unshift(
          { pattern: /\b(?:as|default|export|from|import)\b/, alias: 'module' },
          {
            pattern:
              /\b(?:await|break|catch|continue|do|else|finally|for|if|return|switch|throw|try|while|yield)\b/,
            alias: 'control-flow',
          },
          { pattern: /\bnull\b/, alias: ['null', 'nil'] },
          { pattern: /\bundefined\b/, alias: 'nil' }
        ),
        a.languages.insertBefore('javascript', 'operator', {
          spread: { pattern: /\.{3}/, alias: 'operator' },
          arrow: { pattern: /=>/, alias: 'operator' },
        }),
        a.languages.insertBefore('javascript', 'punctuation', {
          'property-access': { pattern: e('(\\.\\s*)#?<ID>'), lookbehind: !0 },
          'maybe-class-name': {
            pattern: /(^|[^$\w\xA0-\uFFFF])[A-Z][$\w\xA0-\uFFFF]+/,
            lookbehind: !0,
          },
          dom: {
            pattern:
              /\b(?:document|(?:local|session)Storage|location|navigator|performance|window)\b/,
            alias: 'variable',
          },
          console: { pattern: /\bconsole(?=\s*\.)/, alias: 'class-name' },
        })
      for (
        var t = ['function', 'function-variable', 'method', 'method-variable', 'property-access'],
          r = 0;
        r < t.length;
        r++
      ) {
        var n = t[r],
          s = a.languages.javascript[n]
        'RegExp' === a.util.type(s) && (s = a.languages.javascript[n] = { pattern: s })
        var o = s.inside || {}
        ;(s.inside = o), (o['maybe-class-name'] = /^[A-Z][\s\S]*/)
      }
    })(Prism)

    /***/
  },

  /***/ 5139: /***/ () => {
    ;(Prism.languages.json = {
      property: { pattern: /(^|[^\\])"(?:\\.|[^\\"\r\n])*"(?=\s*:)/, lookbehind: !0, greedy: !0 },
      string: { pattern: /(^|[^\\])"(?:\\.|[^\\"\r\n])*"(?!\s*:)/, lookbehind: !0, greedy: !0 },
      comment: { pattern: /\/\/.*|\/\*[\s\S]*?(?:\*\/|$)/, greedy: !0 },
      number: /-?\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b/i,
      punctuation: /[{}[\],]/,
      operator: /:/,
      boolean: /\b(?:false|true)\b/,
      null: { pattern: /\bnull\b/, alias: 'keyword' },
    }),
      (Prism.languages.webmanifest = Prism.languages.json)

    /***/
  },

  /***/ 9146: /***/ () => {
    !(function (t) {
      var n = t.util.clone(t.languages.javascript),
        e = '(?:\\{<S>*\\.{3}(?:[^{}]|<BRACES>)*\\})'
      function a(t, n) {
        return (
          (t = t
            .replace(/<S>/g, function () {
              return '(?:\\s|//.*(?!.)|/\\*(?:[^*]|\\*(?!/))\\*/)'
            })
            .replace(/<BRACES>/g, function () {
              return '(?:\\{(?:\\{(?:\\{[^{}]*\\}|[^{}])*\\}|[^{}])*\\})'
            })
            .replace(/<SPREAD>/g, function () {
              return e
            })),
          RegExp(t, n)
        )
      }
      ;(e = a(e).source),
        (t.languages.jsx = t.languages.extend('markup', n)),
        (t.languages.jsx.tag.pattern = a(
          '</?(?:[\\w.:-]+(?:<S>+(?:[\\w.:$-]+(?:=(?:"(?:\\\\[^]|[^\\\\"])*"|\'(?:\\\\[^]|[^\\\\\'])*\'|[^\\s{\'"/>=]+|<BRACES>))?|<SPREAD>))*<S>*/?)?>'
        )),
        (t.languages.jsx.tag.inside.tag.pattern = /^<\/?[^\s>\/]*/),
        (t.languages.jsx.tag.inside['attr-value'].pattern =
          /=(?!\{)(?:"(?:\\[\s\S]|[^\\"])*"|'(?:\\[\s\S]|[^\\'])*'|[^\s'">]+)/),
        (t.languages.jsx.tag.inside.tag.inside['class-name'] = /^[A-Z]\w*(?:\.[A-Z]\w*)*$/),
        (t.languages.jsx.tag.inside.comment = n.comment),
        t.languages.insertBefore(
          'inside',
          'attr-name',
          { spread: { pattern: a('<SPREAD>'), inside: t.languages.jsx } },
          t.languages.jsx.tag
        ),
        t.languages.insertBefore(
          'inside',
          'special-attr',
          {
            script: {
              pattern: a('=<BRACES>'),
              alias: 'language-javascript',
              inside: {
                'script-punctuation': { pattern: /^=(?=\{)/, alias: 'punctuation' },
                rest: t.languages.jsx,
              },
            },
          },
          t.languages.jsx.tag
        )
      var s = function (t) {
          return t
            ? 'string' == typeof t
              ? t
              : 'string' == typeof t.content
              ? t.content
              : t.content.map(s).join('')
            : ''
        },
        g = function (n) {
          for (var e = [], a = 0; a < n.length; a++) {
            var o = n[a],
              i = !1
            if (
              ('string' != typeof o &&
                ('tag' === o.type && o.content[0] && 'tag' === o.content[0].type
                  ? '</' === o.content[0].content[0].content
                    ? e.length > 0 &&
                      e[e.length - 1].tagName === s(o.content[0].content[1]) &&
                      e.pop()
                    : '/>' === o.content[o.content.length - 1].content ||
                      e.push({ tagName: s(o.content[0].content[1]), openedBraces: 0 })
                  : e.length > 0 && 'punctuation' === o.type && '{' === o.content
                  ? e[e.length - 1].openedBraces++
                  : e.length > 0 &&
                    e[e.length - 1].openedBraces > 0 &&
                    'punctuation' === o.type &&
                    '}' === o.content
                  ? e[e.length - 1].openedBraces--
                  : (i = !0)),
              (i || 'string' == typeof o) && e.length > 0 && 0 === e[e.length - 1].openedBraces)
            ) {
              var r = s(o)
              a < n.length - 1 &&
                ('string' == typeof n[a + 1] || 'plain-text' === n[a + 1].type) &&
                ((r += s(n[a + 1])), n.splice(a + 1, 1)),
                a > 0 &&
                  ('string' == typeof n[a - 1] || 'plain-text' === n[a - 1].type) &&
                  ((r = s(n[a - 1]) + r), n.splice(a - 1, 1), a--),
                (n[a] = new t.Token('plain-text', r, null, r))
            }
            o.content && 'string' != typeof o.content && g(o.content)
          }
        }
      t.hooks.add('after-tokenize', function (t) {
        ;('jsx' !== t.language && 'tsx' !== t.language) || g(t.tokens)
      })
    })(Prism)

    /***/
  },

  /***/ 4520: /***/ () => {
    !(function (e) {
      var a = e.util.clone(e.languages.typescript)
      ;(e.languages.tsx = e.languages.extend('jsx', a)),
        delete e.languages.tsx.parameter,
        delete e.languages.tsx['literal-property']
      var t = e.languages.tsx.tag
      ;(t.pattern = RegExp('(^|[^\\w$]|(?=</))(?:' + t.pattern.source + ')', t.pattern.flags)),
        (t.lookbehind = !0)
    })(Prism)

    /***/
  },

  /***/ 3416: /***/ () => {
    !(function (e) {
      ;(e.languages.typescript = e.languages.extend('javascript', {
        'class-name': {
          pattern:
            /(\b(?:class|extends|implements|instanceof|interface|new|type)\s+)(?!keyof\b)(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?:\s*<(?:[^<>]|<(?:[^<>]|<[^<>]*>)*>)*>)?/,
          lookbehind: !0,
          greedy: !0,
          inside: null,
        },
        builtin:
          /\b(?:Array|Function|Promise|any|boolean|console|never|number|string|symbol|unknown)\b/,
      })),
        e.languages.typescript.keyword.push(
          /\b(?:abstract|declare|is|keyof|readonly|require)\b/,
          /\b(?:asserts|infer|interface|module|namespace|type)\b(?=\s*(?:[{_$a-zA-Z\xA0-\uFFFF]|$))/,
          /\btype\b(?=\s*(?:[\{*]|$))/
        ),
        delete e.languages.typescript.parameter,
        delete e.languages.typescript['literal-property']
      var s = e.languages.extend('typescript', {})
      delete s['class-name'],
        (e.languages.typescript['class-name'].inside = s),
        e.languages.insertBefore('typescript', 'function', {
          decorator: {
            pattern: /@[$\w\xA0-\uFFFF]+/,
            inside: { at: { pattern: /^@/, alias: 'operator' }, function: /^[\s\S]+/ },
          },
          'generic-function': {
            pattern:
              /#?(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*\s*<(?:[^<>]|<(?:[^<>]|<[^<>]*>)*>)*>(?=\s*\()/,
            greedy: !0,
            inside: {
              function: /^#?(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*/,
              generic: { pattern: /<[\s\S]+/, alias: 'class-name', inside: s },
            },
          },
        }),
        (e.languages.ts = e.languages.typescript)
    })(Prism)

    /***/
  },

  /***/ 9499: /***/ (module) => {
    /* **********************************************
     Begin prism-core.js
********************************************** */

    /// <reference lib="WebWorker"/>

    var _self =
      typeof window !== 'undefined'
        ? window // if in browser
        : typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope
        ? self // if in worker
        : {} // if in node js

    /**
     * Prism: Lightweight, robust, elegant syntax highlighting
     *
     * @license MIT <https://opensource.org/licenses/MIT>
     * @author Lea Verou <https://lea.verou.me>
     * @namespace
     * @public
     */
    var Prism = (function (_self) {
      // Private helper vars
      var lang = /(?:^|\s)lang(?:uage)?-([\w-]+)(?=\s|$)/i
      var uniqueId = 0

      // The grammar object for plaintext
      var plainTextGrammar = {}

      var _ = {
        /**
         * By default, Prism will attempt to highlight all code elements (by calling {@link Prism.highlightAll}) on the
         * current page after the page finished loading. This might be a problem if e.g. you wanted to asynchronously load
         * additional languages or plugins yourself.
         *
         * By setting this value to `true`, Prism will not automatically highlight all code elements on the page.
         *
         * You obviously have to change this value before the automatic highlighting started. To do this, you can add an
         * empty Prism object into the global scope before loading the Prism script like this:
         *
         * ```js
         * window.Prism = window.Prism || {};
         * Prism.manual = true;
         * // add a new <script> to load Prism's script
         * ```
         *
         * @default false
         * @type {boolean}
         * @memberof Prism
         * @public
         */
        manual: _self.Prism && _self.Prism.manual,
        /**
         * By default, if Prism is in a web worker, it assumes that it is in a worker it created itself, so it uses
         * `addEventListener` to communicate with its parent instance. However, if you're using Prism manually in your
         * own worker, you don't want it to do this.
         *
         * By setting this value to `true`, Prism will not add its own listeners to the worker.
         *
         * You obviously have to change this value before Prism executes. To do this, you can add an
         * empty Prism object into the global scope before loading the Prism script like this:
         *
         * ```js
         * window.Prism = window.Prism || {};
         * Prism.disableWorkerMessageHandler = true;
         * // Load Prism's script
         * ```
         *
         * @default false
         * @type {boolean}
         * @memberof Prism
         * @public
         */
        disableWorkerMessageHandler: _self.Prism && _self.Prism.disableWorkerMessageHandler,

        /**
         * A namespace for utility methods.
         *
         * All function in this namespace that are not explicitly marked as _public_ are for __internal use only__ and may
         * change or disappear at any time.
         *
         * @namespace
         * @memberof Prism
         */
        util: {
          encode: function encode(tokens) {
            if (tokens instanceof Token) {
              return new Token(tokens.type, encode(tokens.content), tokens.alias)
            } else if (Array.isArray(tokens)) {
              return tokens.map(encode)
            } else {
              return tokens
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/\u00a0/g, ' ')
            }
          },

          /**
           * Returns the name of the type of the given value.
           *
           * @param {any} o
           * @returns {string}
           * @example
           * type(null)      === 'Null'
           * type(undefined) === 'Undefined'
           * type(123)       === 'Number'
           * type('foo')     === 'String'
           * type(true)      === 'Boolean'
           * type([1, 2])    === 'Array'
           * type({})        === 'Object'
           * type(String)    === 'Function'
           * type(/abc+/)    === 'RegExp'
           */
          type: function (o) {
            return Object.prototype.toString.call(o).slice(8, -1)
          },

          /**
           * Returns a unique number for the given object. Later calls will still return the same number.
           *
           * @param {Object} obj
           * @returns {number}
           */
          objId: function (obj) {
            if (!obj['__id']) {
              Object.defineProperty(obj, '__id', { value: ++uniqueId })
            }
            return obj['__id']
          },

          /**
           * Creates a deep clone of the given object.
           *
           * The main intended use of this function is to clone language definitions.
           *
           * @param {T} o
           * @param {Record<number, any>} [visited]
           * @returns {T}
           * @template T
           */
          clone: function deepClone(o, visited) {
            visited = visited || {}

            var clone
            var id
            switch (_.util.type(o)) {
              case 'Object':
                id = _.util.objId(o)
                if (visited[id]) {
                  return visited[id]
                }
                clone = /** @type {Record<string, any>} */ ({})
                visited[id] = clone

                for (var key in o) {
                  if (o.hasOwnProperty(key)) {
                    clone[key] = deepClone(o[key], visited)
                  }
                }

                return /** @type {any} */ (clone)

              case 'Array':
                id = _.util.objId(o)
                if (visited[id]) {
                  return visited[id]
                }
                clone = []
                visited[id] = clone

                ;/** @type {Array} */ (/** @type {any} */ (o)).forEach(function (v, i) {
                  clone[i] = deepClone(v, visited)
                })

                return /** @type {any} */ (clone)

              default:
                return o
            }
          },

          /**
           * Returns the Prism language of the given element set by a `language-xxxx` or `lang-xxxx` class.
           *
           * If no language is set for the element or the element is `null` or `undefined`, `none` will be returned.
           *
           * @param {Element} element
           * @returns {string}
           */
          getLanguage: function (element) {
            while (element) {
              var m = lang.exec(element.className)
              if (m) {
                return m[1].toLowerCase()
              }
              element = element.parentElement
            }
            return 'none'
          },

          /**
           * Sets the Prism `language-xxxx` class of the given element.
           *
           * @param {Element} element
           * @param {string} language
           * @returns {void}
           */
          setLanguage: function (element, language) {
            // remove all `language-xxxx` classes
            // (this might leave behind a leading space)
            element.className = element.className.replace(RegExp(lang, 'gi'), '')

            // add the new `language-xxxx` class
            // (using `classList` will automatically clean up spaces for us)
            element.classList.add('language-' + language)
          },

          /**
           * Returns the script element that is currently executing.
           *
           * This does __not__ work for line script element.
           *
           * @returns {HTMLScriptElement | null}
           */
          currentScript: function () {
            if (typeof document === 'undefined') {
              return null
            }
            if ('currentScript' in document && 1 < 2 /* hack to trip TS' flow analysis */) {
              return /** @type {any} */ (document.currentScript)
            }

            // IE11 workaround
            // we'll get the src of the current script by parsing IE11's error stack trace
            // this will not work for inline scripts

            try {
              throw new Error()
            } catch (err) {
              // Get file src url from stack. Specifically works with the format of stack traces in IE.
              // A stack will look like this:
              //
              // Error
              //    at _.util.currentScript (http://localhost/components/prism-core.js:119:5)
              //    at Global code (http://localhost/components/prism-core.js:606:1)

              var src = (/at [^(\r\n]*\((.*):[^:]+:[^:]+\)$/i.exec(err.stack) || [])[1]
              if (src) {
                var scripts = document.getElementsByTagName('script')
                for (var i in scripts) {
                  if (scripts[i].src == src) {
                    return scripts[i]
                  }
                }
              }
              return null
            }
          },

          /**
           * Returns whether a given class is active for `element`.
           *
           * The class can be activated if `element` or one of its ancestors has the given class and it can be deactivated
           * if `element` or one of its ancestors has the negated version of the given class. The _negated version_ of the
           * given class is just the given class with a `no-` prefix.
           *
           * Whether the class is active is determined by the closest ancestor of `element` (where `element` itself is
           * closest ancestor) that has the given class or the negated version of it. If neither `element` nor any of its
           * ancestors have the given class or the negated version of it, then the default activation will be returned.
           *
           * In the paradoxical situation where the closest ancestor contains __both__ the given class and the negated
           * version of it, the class is considered active.
           *
           * @param {Element} element
           * @param {string} className
           * @param {boolean} [defaultActivation=false]
           * @returns {boolean}
           */
          isActive: function (element, className, defaultActivation) {
            var no = 'no-' + className

            while (element) {
              var classList = element.classList
              if (classList.contains(className)) {
                return true
              }
              if (classList.contains(no)) {
                return false
              }
              element = element.parentElement
            }
            return !!defaultActivation
          },
        },

        /**
         * This namespace contains all currently loaded languages and the some helper functions to create and modify languages.
         *
         * @namespace
         * @memberof Prism
         * @public
         */
        languages: {
          /**
           * The grammar for plain, unformatted text.
           */
          plain: plainTextGrammar,
          plaintext: plainTextGrammar,
          text: plainTextGrammar,
          txt: plainTextGrammar,

          /**
           * Creates a deep copy of the language with the given id and appends the given tokens.
           *
           * If a token in `redef` also appears in the copied language, then the existing token in the copied language
           * will be overwritten at its original position.
           *
           * ## Best practices
           *
           * Since the position of overwriting tokens (token in `redef` that overwrite tokens in the copied language)
           * doesn't matter, they can technically be in any order. However, this can be confusing to others that trying to
           * understand the language definition because, normally, the order of tokens matters in Prism grammars.
           *
           * Therefore, it is encouraged to order overwriting tokens according to the positions of the overwritten tokens.
           * Furthermore, all non-overwriting tokens should be placed after the overwriting ones.
           *
           * @param {string} id The id of the language to extend. This has to be a key in `Prism.languages`.
           * @param {Grammar} redef The new tokens to append.
           * @returns {Grammar} The new language created.
           * @public
           * @example
           * Prism.languages['css-with-colors'] = Prism.languages.extend('css', {
           *     // Prism.languages.css already has a 'comment' token, so this token will overwrite CSS' 'comment' token
           *     // at its original position
           *     'comment': { ... },
           *     // CSS doesn't have a 'color' token, so this token will be appended
           *     'color': /\b(?:red|green|blue)\b/
           * });
           */
          extend: function (id, redef) {
            var lang = _.util.clone(_.languages[id])

            for (var key in redef) {
              lang[key] = redef[key]
            }

            return lang
          },

          /**
           * Inserts tokens _before_ another token in a language definition or any other grammar.
           *
           * ## Usage
           *
           * This helper method makes it easy to modify existing languages. For example, the CSS language definition
           * not only defines CSS highlighting for CSS documents, but also needs to define highlighting for CSS embedded
           * in HTML through `<style>` elements. To do this, it needs to modify `Prism.languages.markup` and add the
           * appropriate tokens. However, `Prism.languages.markup` is a regular JavaScript object literal, so if you do
           * this:
           *
           * ```js
           * Prism.languages.markup.style = {
           *     // token
           * };
           * ```
           *
           * then the `style` token will be added (and processed) at the end. `insertBefore` allows you to insert tokens
           * before existing tokens. For the CSS example above, you would use it like this:
           *
           * ```js
           * Prism.languages.insertBefore('markup', 'cdata', {
           *     'style': {
           *         // token
           *     }
           * });
           * ```
           *
           * ## Special cases
           *
           * If the grammars of `inside` and `insert` have tokens with the same name, the tokens in `inside`'s grammar
           * will be ignored.
           *
           * This behavior can be used to insert tokens after `before`:
           *
           * ```js
           * Prism.languages.insertBefore('markup', 'comment', {
           *     'comment': Prism.languages.markup.comment,
           *     // tokens after 'comment'
           * });
           * ```
           *
           * ## Limitations
           *
           * The main problem `insertBefore` has to solve is iteration order. Since ES2015, the iteration order for object
           * properties is guaranteed to be the insertion order (except for integer keys) but some browsers behave
           * differently when keys are deleted and re-inserted. So `insertBefore` can't be implemented by temporarily
           * deleting properties which is necessary to insert at arbitrary positions.
           *
           * To solve this problem, `insertBefore` doesn't actually insert the given tokens into the target object.
           * Instead, it will create a new object and replace all references to the target object with the new one. This
           * can be done without temporarily deleting properties, so the iteration order is well-defined.
           *
           * However, only references that can be reached from `Prism.languages` or `insert` will be replaced. I.e. if
           * you hold the target object in a variable, then the value of the variable will not change.
           *
           * ```js
           * var oldMarkup = Prism.languages.markup;
           * var newMarkup = Prism.languages.insertBefore('markup', 'comment', { ... });
           *
           * assert(oldMarkup !== Prism.languages.markup);
           * assert(newMarkup === Prism.languages.markup);
           * ```
           *
           * @param {string} inside The property of `root` (e.g. a language id in `Prism.languages`) that contains the
           * object to be modified.
           * @param {string} before The key to insert before.
           * @param {Grammar} insert An object containing the key-value pairs to be inserted.
           * @param {Object<string, any>} [root] The object containing `inside`, i.e. the object that contains the
           * object to be modified.
           *
           * Defaults to `Prism.languages`.
           * @returns {Grammar} The new grammar object.
           * @public
           */
          insertBefore: function (inside, before, insert, root) {
            root = root || /** @type {any} */ (_.languages)
            var grammar = root[inside]
            /** @type {Grammar} */
            var ret = {}

            for (var token in grammar) {
              if (grammar.hasOwnProperty(token)) {
                if (token == before) {
                  for (var newToken in insert) {
                    if (insert.hasOwnProperty(newToken)) {
                      ret[newToken] = insert[newToken]
                    }
                  }
                }

                // Do not insert token which also occur in insert. See #1525
                if (!insert.hasOwnProperty(token)) {
                  ret[token] = grammar[token]
                }
              }
            }

            var old = root[inside]
            root[inside] = ret

            // Update references in other language definitions
            _.languages.DFS(_.languages, function (key, value) {
              if (value === old && key != inside) {
                this[key] = ret
              }
            })

            return ret
          },

          // Traverse a language definition with Depth First Search
          DFS: function DFS(o, callback, type, visited) {
            visited = visited || {}

            var objId = _.util.objId

            for (var i in o) {
              if (o.hasOwnProperty(i)) {
                callback.call(o, i, o[i], type || i)

                var property = o[i]
                var propertyType = _.util.type(property)

                if (propertyType === 'Object' && !visited[objId(property)]) {
                  visited[objId(property)] = true
                  DFS(property, callback, null, visited)
                } else if (propertyType === 'Array' && !visited[objId(property)]) {
                  visited[objId(property)] = true
                  DFS(property, callback, i, visited)
                }
              }
            }
          },
        },

        plugins: {},

        /**
         * This is the most high-level function in Prism’s API.
         * It fetches all the elements that have a `.language-xxxx` class and then calls {@link Prism.highlightElement} on
         * each one of them.
         *
         * This is equivalent to `Prism.highlightAllUnder(document, async, callback)`.
         *
         * @param {boolean} [async=false] Same as in {@link Prism.highlightAllUnder}.
         * @param {HighlightCallback} [callback] Same as in {@link Prism.highlightAllUnder}.
         * @memberof Prism
         * @public
         */
        highlightAll: function (async, callback) {
          _.highlightAllUnder(document, async, callback)
        },

        /**
         * Fetches all the descendants of `container` that have a `.language-xxxx` class and then calls
         * {@link Prism.highlightElement} on each one of them.
         *
         * The following hooks will be run:
         * 1. `before-highlightall`
         * 2. `before-all-elements-highlight`
         * 3. All hooks of {@link Prism.highlightElement} for each element.
         *
         * @param {ParentNode} container The root element, whose descendants that have a `.language-xxxx` class will be highlighted.
         * @param {boolean} [async=false] Whether each element is to be highlighted asynchronously using Web Workers.
         * @param {HighlightCallback} [callback] An optional callback to be invoked on each element after its highlighting is done.
         * @memberof Prism
         * @public
         */
        highlightAllUnder: function (container, async, callback) {
          var env = {
            callback: callback,
            container: container,
            selector:
              'code[class*="language-"], [class*="language-"] code, code[class*="lang-"], [class*="lang-"] code',
          }

          _.hooks.run('before-highlightall', env)

          env.elements = Array.prototype.slice.apply(env.container.querySelectorAll(env.selector))

          _.hooks.run('before-all-elements-highlight', env)

          for (var i = 0, element; (element = env.elements[i++]); ) {
            _.highlightElement(element, async === true, env.callback)
          }
        },

        /**
         * Highlights the code inside a single element.
         *
         * The following hooks will be run:
         * 1. `before-sanity-check`
         * 2. `before-highlight`
         * 3. All hooks of {@link Prism.highlight}. These hooks will be run by an asynchronous worker if `async` is `true`.
         * 4. `before-insert`
         * 5. `after-highlight`
         * 6. `complete`
         *
         * Some the above hooks will be skipped if the element doesn't contain any text or there is no grammar loaded for
         * the element's language.
         *
         * @param {Element} element The element containing the code.
         * It must have a class of `language-xxxx` to be processed, where `xxxx` is a valid language identifier.
         * @param {boolean} [async=false] Whether the element is to be highlighted asynchronously using Web Workers
         * to improve performance and avoid blocking the UI when highlighting very large chunks of code. This option is
         * [disabled by default](https://prismjs.com/faq.html#why-is-asynchronous-highlighting-disabled-by-default).
         *
         * Note: All language definitions required to highlight the code must be included in the main `prism.js` file for
         * asynchronous highlighting to work. You can build your own bundle on the
         * [Download page](https://prismjs.com/download.html).
         * @param {HighlightCallback} [callback] An optional callback to be invoked after the highlighting is done.
         * Mostly useful when `async` is `true`, since in that case, the highlighting is done asynchronously.
         * @memberof Prism
         * @public
         */
        highlightElement: function (element, async, callback) {
          // Find language
          var language = _.util.getLanguage(element)
          var grammar = _.languages[language]

          // Set language on the element, if not present
          _.util.setLanguage(element, language)

          // Set language on the parent, for styling
          var parent = element.parentElement
          if (parent && parent.nodeName.toLowerCase() === 'pre') {
            _.util.setLanguage(parent, language)
          }

          var code = element.textContent

          var env = {
            element: element,
            language: language,
            grammar: grammar,
            code: code,
          }

          function insertHighlightedCode(highlightedCode) {
            env.highlightedCode = highlightedCode

            _.hooks.run('before-insert', env)

            env.element.innerHTML = env.highlightedCode

            _.hooks.run('after-highlight', env)
            _.hooks.run('complete', env)
            callback && callback.call(env.element)
          }

          _.hooks.run('before-sanity-check', env)

          // plugins may change/add the parent/element
          parent = env.element.parentElement
          if (
            parent &&
            parent.nodeName.toLowerCase() === 'pre' &&
            !parent.hasAttribute('tabindex')
          ) {
            parent.setAttribute('tabindex', '0')
          }

          if (!env.code) {
            _.hooks.run('complete', env)
            callback && callback.call(env.element)
            return
          }

          _.hooks.run('before-highlight', env)

          if (!env.grammar) {
            insertHighlightedCode(_.util.encode(env.code))
            return
          }

          if (async && _self.Worker) {
            var worker = new Worker(_.filename)

            worker.onmessage = function (evt) {
              insertHighlightedCode(evt.data)
            }

            worker.postMessage(
              JSON.stringify({
                language: env.language,
                code: env.code,
                immediateClose: true,
              })
            )
          } else {
            insertHighlightedCode(_.highlight(env.code, env.grammar, env.language))
          }
        },

        /**
         * Low-level function, only use if you know what you’re doing. It accepts a string of text as input
         * and the language definitions to use, and returns a string with the HTML produced.
         *
         * The following hooks will be run:
         * 1. `before-tokenize`
         * 2. `after-tokenize`
         * 3. `wrap`: On each {@link Token}.
         *
         * @param {string} text A string with the code to be highlighted.
         * @param {Grammar} grammar An object containing the tokens to use.
         *
         * Usually a language definition like `Prism.languages.markup`.
         * @param {string} language The name of the language definition passed to `grammar`.
         * @returns {string} The highlighted HTML.
         * @memberof Prism
         * @public
         * @example
         * Prism.highlight('var foo = true;', Prism.languages.javascript, 'javascript');
         */
        highlight: function (text, grammar, language) {
          var env = {
            code: text,
            grammar: grammar,
            language: language,
          }
          _.hooks.run('before-tokenize', env)
          if (!env.grammar) {
            throw new Error('The language "' + env.language + '" has no grammar.')
          }
          env.tokens = _.tokenize(env.code, env.grammar)
          _.hooks.run('after-tokenize', env)
          return Token.stringify(_.util.encode(env.tokens), env.language)
        },

        /**
         * This is the heart of Prism, and the most low-level function you can use. It accepts a string of text as input
         * and the language definitions to use, and returns an array with the tokenized code.
         *
         * When the language definition includes nested tokens, the function is called recursively on each of these tokens.
         *
         * This method could be useful in other contexts as well, as a very crude parser.
         *
         * @param {string} text A string with the code to be highlighted.
         * @param {Grammar} grammar An object containing the tokens to use.
         *
         * Usually a language definition like `Prism.languages.markup`.
         * @returns {TokenStream} An array of strings and tokens, a token stream.
         * @memberof Prism
         * @public
         * @example
         * let code = `var foo = 0;`;
         * let tokens = Prism.tokenize(code, Prism.languages.javascript);
         * tokens.forEach(token => {
         *     if (token instanceof Prism.Token && token.type === 'number') {
         *         console.log(`Found numeric literal: ${token.content}`);
         *     }
         * });
         */
        tokenize: function (text, grammar) {
          var rest = grammar.rest
          if (rest) {
            for (var token in rest) {
              grammar[token] = rest[token]
            }

            delete grammar.rest
          }

          var tokenList = new LinkedList()
          addAfter(tokenList, tokenList.head, text)

          matchGrammar(text, tokenList, grammar, tokenList.head, 0)

          return toArray(tokenList)
        },

        /**
         * @namespace
         * @memberof Prism
         * @public
         */
        hooks: {
          all: {},

          /**
           * Adds the given callback to the list of callbacks for the given hook.
           *
           * The callback will be invoked when the hook it is registered for is run.
           * Hooks are usually directly run by a highlight function but you can also run hooks yourself.
           *
           * One callback function can be registered to multiple hooks and the same hook multiple times.
           *
           * @param {string} name The name of the hook.
           * @param {HookCallback} callback The callback function which is given environment variables.
           * @public
           */
          add: function (name, callback) {
            var hooks = _.hooks.all

            hooks[name] = hooks[name] || []

            hooks[name].push(callback)
          },

          /**
           * Runs a hook invoking all registered callbacks with the given environment variables.
           *
           * Callbacks will be invoked synchronously and in the order in which they were registered.
           *
           * @param {string} name The name of the hook.
           * @param {Object<string, any>} env The environment variables of the hook passed to all callbacks registered.
           * @public
           */
          run: function (name, env) {
            var callbacks = _.hooks.all[name]

            if (!callbacks || !callbacks.length) {
              return
            }

            for (var i = 0, callback; (callback = callbacks[i++]); ) {
              callback(env)
            }
          },
        },

        Token: Token,
      }
      _self.Prism = _

      // Typescript note:
      // The following can be used to import the Token type in JSDoc:
      //
      //   @typedef {InstanceType<import("./prism-core")["Token"]>} Token

      /**
       * Creates a new token.
       *
       * @param {string} type See {@link Token#type type}
       * @param {string | TokenStream} content See {@link Token#content content}
       * @param {string|string[]} [alias] The alias(es) of the token.
       * @param {string} [matchedStr=""] A copy of the full string this token was created from.
       * @class
       * @global
       * @public
       */
      function Token(type, content, alias, matchedStr) {
        /**
         * The type of the token.
         *
         * This is usually the key of a pattern in a {@link Grammar}.
         *
         * @type {string}
         * @see GrammarToken
         * @public
         */
        this.type = type
        /**
         * The strings or tokens contained by this token.
         *
         * This will be a token stream if the pattern matched also defined an `inside` grammar.
         *
         * @type {string | TokenStream}
         * @public
         */
        this.content = content
        /**
         * The alias(es) of the token.
         *
         * @type {string|string[]}
         * @see GrammarToken
         * @public
         */
        this.alias = alias
        // Copy of the full string this token was created from
        this.length = (matchedStr || '').length | 0
      }

      /**
       * A token stream is an array of strings and {@link Token Token} objects.
       *
       * Token streams have to fulfill a few properties that are assumed by most functions (mostly internal ones) that process
       * them.
       *
       * 1. No adjacent strings.
       * 2. No empty strings.
       *
       *    The only exception here is the token stream that only contains the empty string and nothing else.
       *
       * @typedef {Array<string | Token>} TokenStream
       * @global
       * @public
       */

      /**
       * Converts the given token or token stream to an HTML representation.
       *
       * The following hooks will be run:
       * 1. `wrap`: On each {@link Token}.
       *
       * @param {string | Token | TokenStream} o The token or token stream to be converted.
       * @param {string} language The name of current language.
       * @returns {string} The HTML representation of the token or token stream.
       * @memberof Token
       * @static
       */
      Token.stringify = function stringify(o, language) {
        if (typeof o == 'string') {
          return o
        }
        if (Array.isArray(o)) {
          var s = ''
          o.forEach(function (e) {
            s += stringify(e, language)
          })
          return s
        }

        var env = {
          type: o.type,
          content: stringify(o.content, language),
          tag: 'span',
          classes: ['token', o.type],
          attributes: {},
          language: language,
        }

        var aliases = o.alias
        if (aliases) {
          if (Array.isArray(aliases)) {
            Array.prototype.push.apply(env.classes, aliases)
          } else {
            env.classes.push(aliases)
          }
        }

        _.hooks.run('wrap', env)

        var attributes = ''
        for (var name in env.attributes) {
          attributes +=
            ' ' + name + '="' + (env.attributes[name] || '').replace(/"/g, '&quot;') + '"'
        }

        return (
          '<' +
          env.tag +
          ' class="' +
          env.classes.join(' ') +
          '"' +
          attributes +
          '>' +
          env.content +
          '</' +
          env.tag +
          '>'
        )
      }

      /**
       * @param {RegExp} pattern
       * @param {number} pos
       * @param {string} text
       * @param {boolean} lookbehind
       * @returns {RegExpExecArray | null}
       */
      function matchPattern(pattern, pos, text, lookbehind) {
        pattern.lastIndex = pos
        var match = pattern.exec(text)
        if (match && lookbehind && match[1]) {
          // change the match to remove the text matched by the Prism lookbehind group
          var lookbehindLength = match[1].length
          match.index += lookbehindLength
          match[0] = match[0].slice(lookbehindLength)
        }
        return match
      }

      /**
       * @param {string} text
       * @param {LinkedList<string | Token>} tokenList
       * @param {any} grammar
       * @param {LinkedListNode<string | Token>} startNode
       * @param {number} startPos
       * @param {RematchOptions} [rematch]
       * @returns {void}
       * @private
       *
       * @typedef RematchOptions
       * @property {string} cause
       * @property {number} reach
       */
      function matchGrammar(text, tokenList, grammar, startNode, startPos, rematch) {
        for (var token in grammar) {
          if (!grammar.hasOwnProperty(token) || !grammar[token]) {
            continue
          }

          var patterns = grammar[token]
          patterns = Array.isArray(patterns) ? patterns : [patterns]

          for (var j = 0; j < patterns.length; ++j) {
            if (rematch && rematch.cause == token + ',' + j) {
              return
            }

            var patternObj = patterns[j]
            var inside = patternObj.inside
            var lookbehind = !!patternObj.lookbehind
            var greedy = !!patternObj.greedy
            var alias = patternObj.alias

            if (greedy && !patternObj.pattern.global) {
              // Without the global flag, lastIndex won't work
              var flags = patternObj.pattern.toString().match(/[imsuy]*$/)[0]
              patternObj.pattern = RegExp(patternObj.pattern.source, flags + 'g')
            }

            /** @type {RegExp} */
            var pattern = patternObj.pattern || patternObj

            for (
              // iterate the token list and keep track of the current token/string position
              var currentNode = startNode.next, pos = startPos;
              currentNode !== tokenList.tail;
              pos += currentNode.value.length, currentNode = currentNode.next
            ) {
              if (rematch && pos >= rematch.reach) {
                break
              }

              var str = currentNode.value

              if (tokenList.length > text.length) {
                // Something went terribly wrong, ABORT, ABORT!
                return
              }

              if (str instanceof Token) {
                continue
              }

              var removeCount = 1 // this is the to parameter of removeBetween
              var match

              if (greedy) {
                match = matchPattern(pattern, pos, text, lookbehind)
                if (!match || match.index >= text.length) {
                  break
                }

                var from = match.index
                var to = match.index + match[0].length
                var p = pos

                // find the node that contains the match
                p += currentNode.value.length
                while (from >= p) {
                  currentNode = currentNode.next
                  p += currentNode.value.length
                }
                // adjust pos (and p)
                p -= currentNode.value.length
                pos = p

                // the current node is a Token, then the match starts inside another Token, which is invalid
                if (currentNode.value instanceof Token) {
                  continue
                }

                // find the last node which is affected by this match
                for (
                  var k = currentNode;
                  k !== tokenList.tail && (p < to || typeof k.value === 'string');
                  k = k.next
                ) {
                  removeCount++
                  p += k.value.length
                }
                removeCount--

                // replace with the new match
                str = text.slice(pos, p)
                match.index -= pos
              } else {
                match = matchPattern(pattern, 0, str, lookbehind)
                if (!match) {
                  continue
                }
              }

              // eslint-disable-next-line no-redeclare
              var from = match.index
              var matchStr = match[0]
              var before = str.slice(0, from)
              var after = str.slice(from + matchStr.length)

              var reach = pos + str.length
              if (rematch && reach > rematch.reach) {
                rematch.reach = reach
              }

              var removeFrom = currentNode.prev

              if (before) {
                removeFrom = addAfter(tokenList, removeFrom, before)
                pos += before.length
              }

              removeRange(tokenList, removeFrom, removeCount)

              var wrapped = new Token(
                token,
                inside ? _.tokenize(matchStr, inside) : matchStr,
                alias,
                matchStr
              )
              currentNode = addAfter(tokenList, removeFrom, wrapped)

              if (after) {
                addAfter(tokenList, currentNode, after)
              }

              if (removeCount > 1) {
                // at least one Token object was removed, so we have to do some rematching
                // this can only happen if the current pattern is greedy

                /** @type {RematchOptions} */
                var nestedRematch = {
                  cause: token + ',' + j,
                  reach: reach,
                }
                matchGrammar(text, tokenList, grammar, currentNode.prev, pos, nestedRematch)

                // the reach might have been extended because of the rematching
                if (rematch && nestedRematch.reach > rematch.reach) {
                  rematch.reach = nestedRematch.reach
                }
              }
            }
          }
        }
      }

      /**
       * @typedef LinkedListNode
       * @property {T} value
       * @property {LinkedListNode<T> | null} prev The previous node.
       * @property {LinkedListNode<T> | null} next The next node.
       * @template T
       * @private
       */

      /**
       * @template T
       * @private
       */
      function LinkedList() {
        /** @type {LinkedListNode<T>} */
        var head = { value: null, prev: null, next: null }
        /** @type {LinkedListNode<T>} */
        var tail = { value: null, prev: head, next: null }
        head.next = tail

        /** @type {LinkedListNode<T>} */
        this.head = head
        /** @type {LinkedListNode<T>} */
        this.tail = tail
        this.length = 0
      }

      /**
       * Adds a new node with the given value to the list.
       *
       * @param {LinkedList<T>} list
       * @param {LinkedListNode<T>} node
       * @param {T} value
       * @returns {LinkedListNode<T>} The added node.
       * @template T
       */
      function addAfter(list, node, value) {
        // assumes that node != list.tail && values.length >= 0
        var next = node.next

        var newNode = { value: value, prev: node, next: next }
        node.next = newNode
        next.prev = newNode
        list.length++

        return newNode
      }
      /**
       * Removes `count` nodes after the given node. The given node will not be removed.
       *
       * @param {LinkedList<T>} list
       * @param {LinkedListNode<T>} node
       * @param {number} count
       * @template T
       */
      function removeRange(list, node, count) {
        var next = node.next
        for (var i = 0; i < count && next !== list.tail; i++) {
          next = next.next
        }
        node.next = next
        next.prev = node
        list.length -= i
      }
      /**
       * @param {LinkedList<T>} list
       * @returns {T[]}
       * @template T
       */
      function toArray(list) {
        var array = []
        var node = list.head.next
        while (node !== list.tail) {
          array.push(node.value)
          node = node.next
        }
        return array
      }

      if (!_self.document) {
        if (!_self.addEventListener) {
          // in Node.js
          return _
        }

        if (!_.disableWorkerMessageHandler) {
          // In worker
          _self.addEventListener(
            'message',
            function (evt) {
              var message = JSON.parse(evt.data)
              var lang = message.language
              var code = message.code
              var immediateClose = message.immediateClose

              _self.postMessage(_.highlight(code, _.languages[lang], lang))
              if (immediateClose) {
                _self.close()
              }
            },
            false
          )
        }

        return _
      }

      // Get current script and highlight
      var script = _.util.currentScript()

      if (script) {
        _.filename = script.src

        if (script.hasAttribute('data-manual')) {
          _.manual = true
        }
      }

      function highlightAutomaticallyCallback() {
        if (!_.manual) {
          _.highlightAll()
        }
      }

      if (!_.manual) {
        // If the document state is "loading", then we'll use DOMContentLoaded.
        // If the document state is "interactive" and the prism.js script is deferred, then we'll also use the
        // DOMContentLoaded event because there might be some plugins or languages which have also been deferred and they
        // might take longer one animation frame to execute which can create a race condition where only some plugins have
        // been loaded when Prism.highlightAll() is executed, depending on how fast resources are loaded.
        // See https://github.com/PrismJS/prism/issues/2102
        var readyState = document.readyState
        if (readyState === 'loading' || (readyState === 'interactive' && script && script.defer)) {
          document.addEventListener('DOMContentLoaded', highlightAutomaticallyCallback)
        } else {
          if (window.requestAnimationFrame) {
            window.requestAnimationFrame(highlightAutomaticallyCallback)
          } else {
            window.setTimeout(highlightAutomaticallyCallback, 16)
          }
        }
      }

      return _
    })(_self)

    if (true && module.exports) {
      module.exports = Prism
    }

    // hack for components to work correctly in node.js
    if (typeof global !== 'undefined') {
      global.Prism = Prism
    }

    // some additional documentation/types

    /**
     * The expansion of a simple `RegExp` literal to support additional properties.
     *
     * @typedef GrammarToken
     * @property {RegExp} pattern The regular expression of the token.
     * @property {boolean} [lookbehind=false] If `true`, then the first capturing group of `pattern` will (effectively)
     * behave as a lookbehind group meaning that the captured text will not be part of the matched text of the new token.
     * @property {boolean} [greedy=false] Whether the token is greedy.
     * @property {string|string[]} [alias] An optional alias or list of aliases.
     * @property {Grammar} [inside] The nested grammar of this token.
     *
     * The `inside` grammar will be used to tokenize the text value of each token of this kind.
     *
     * This can be used to make nested and even recursive language definitions.
     *
     * Note: This can cause infinite recursion. Be careful when you embed different languages or even the same language into
     * each another.
     * @global
     * @public
     */

    /**
     * @typedef Grammar
     * @type {Object<string, RegExp | GrammarToken | Array<RegExp | GrammarToken>>}
     * @property {Grammar} [rest] An optional grammar object that will be appended to this grammar.
     * @global
     * @public
     */

    /**
     * A function which will invoked after an element was successfully highlighted.
     *
     * @callback HighlightCallback
     * @param {Element} element The element successfully highlighted.
     * @returns {void}
     * @global
     * @public
     */

    /**
     * @callback HookCallback
     * @param {Object<string, any>} env The environment variables of the hook.
     * @returns {void}
     * @global
     * @public
     */

    /* **********************************************
     Begin prism-markup.js
********************************************** */

    Prism.languages.markup = {
      comment: {
        pattern: /<!--(?:(?!<!--)[\s\S])*?-->/,
        greedy: true,
      },
      prolog: {
        pattern: /<\?[\s\S]+?\?>/,
        greedy: true,
      },
      doctype: {
        // https://www.w3.org/TR/xml/#NT-doctypedecl
        pattern:
          /<!DOCTYPE(?:[^>"'[\]]|"[^"]*"|'[^']*')+(?:\[(?:[^<"'\]]|"[^"]*"|'[^']*'|<(?!!--)|<!--(?:[^-]|-(?!->))*-->)*\]\s*)?>/i,
        greedy: true,
        inside: {
          'internal-subset': {
            pattern: /(^[^\[]*\[)[\s\S]+(?=\]>$)/,
            lookbehind: true,
            greedy: true,
            inside: null, // see below
          },
          string: {
            pattern: /"[^"]*"|'[^']*'/,
            greedy: true,
          },
          punctuation: /^<!|>$|[[\]]/,
          'doctype-tag': /^DOCTYPE/i,
          name: /[^\s<>'"]+/,
        },
      },
      cdata: {
        pattern: /<!\[CDATA\[[\s\S]*?\]\]>/i,
        greedy: true,
      },
      tag: {
        pattern:
          /<\/?(?!\d)[^\s>\/=$<%]+(?:\s(?:\s*[^\s>\/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s'">=]+(?=[\s>]))|(?=[\s/>])))+)?\s*\/?>/,
        greedy: true,
        inside: {
          tag: {
            pattern: /^<\/?[^\s>\/]+/,
            inside: {
              punctuation: /^<\/?/,
              namespace: /^[^\s>\/:]+:/,
            },
          },
          'special-attr': [],
          'attr-value': {
            pattern: /=\s*(?:"[^"]*"|'[^']*'|[^\s'">=]+)/,
            inside: {
              punctuation: [
                {
                  pattern: /^=/,
                  alias: 'attr-equals',
                },
                {
                  pattern: /^(\s*)["']|["']$/,
                  lookbehind: true,
                },
              ],
            },
          },
          punctuation: /\/?>/,
          'attr-name': {
            pattern: /[^\s>\/]+/,
            inside: {
              namespace: /^[^\s>\/:]+:/,
            },
          },
        },
      },
      entity: [
        {
          pattern: /&[\da-z]{1,8};/i,
          alias: 'named-entity',
        },
        /&#x?[\da-f]{1,8};/i,
      ],
    }

    Prism.languages.markup['tag'].inside['attr-value'].inside['entity'] =
      Prism.languages.markup['entity']
    Prism.languages.markup['doctype'].inside['internal-subset'].inside = Prism.languages.markup

    // Plugin to make entity title show the real entity, idea by Roman Komarov
    Prism.hooks.add('wrap', function (env) {
      if (env.type === 'entity') {
        env.attributes['title'] = env.content.replace(/&amp;/, '&')
      }
    })

    Object.defineProperty(Prism.languages.markup.tag, 'addInlined', {
      /**
       * Adds an inlined language to markup.
       *
       * An example of an inlined language is CSS with `<style>` tags.
       *
       * @param {string} tagName The name of the tag that contains the inlined language. This name will be treated as
       * case insensitive.
       * @param {string} lang The language key.
       * @example
       * addInlined('style', 'css');
       */
      value: function addInlined(tagName, lang) {
        var includedCdataInside = {}
        includedCdataInside['language-' + lang] = {
          pattern: /(^<!\[CDATA\[)[\s\S]+?(?=\]\]>$)/i,
          lookbehind: true,
          inside: Prism.languages[lang],
        }
        includedCdataInside['cdata'] = /^<!\[CDATA\[|\]\]>$/i

        var inside = {
          'included-cdata': {
            pattern: /<!\[CDATA\[[\s\S]*?\]\]>/i,
            inside: includedCdataInside,
          },
        }
        inside['language-' + lang] = {
          pattern: /[\s\S]+/,
          inside: Prism.languages[lang],
        }

        var def = {}
        def[tagName] = {
          pattern: RegExp(
            /(<__[^>]*>)(?:<!\[CDATA\[(?:[^\]]|\](?!\]>))*\]\]>|(?!<!\[CDATA\[)[\s\S])*?(?=<\/__>)/.source.replace(
              /__/g,
              function () {
                return tagName
              }
            ),
            'i'
          ),
          lookbehind: true,
          greedy: true,
          inside: inside,
        }

        Prism.languages.insertBefore('markup', 'cdata', def)
      },
    })
    Object.defineProperty(Prism.languages.markup.tag, 'addAttribute', {
      /**
       * Adds an pattern to highlight languages embedded in HTML attributes.
       *
       * An example of an inlined language is CSS with `style` attributes.
       *
       * @param {string} attrName The name of the tag that contains the inlined language. This name will be treated as
       * case insensitive.
       * @param {string} lang The language key.
       * @example
       * addAttribute('style', 'css');
       */
      value: function (attrName, lang) {
        Prism.languages.markup.tag.inside['special-attr'].push({
          pattern: RegExp(
            /(^|["'\s])/.source +
              '(?:' +
              attrName +
              ')' +
              /\s*=\s*(?:"[^"]*"|'[^']*'|[^\s'">=]+(?=[\s>]))/.source,
            'i'
          ),
          lookbehind: true,
          inside: {
            'attr-name': /^[^\s=]+/,
            'attr-value': {
              pattern: /=[\s\S]+/,
              inside: {
                value: {
                  pattern: /(^=\s*(["']|(?!["'])))\S[\s\S]*(?=\2$)/,
                  lookbehind: true,
                  alias: [lang, 'language-' + lang],
                  inside: Prism.languages[lang],
                },
                punctuation: [
                  {
                    pattern: /^=/,
                    alias: 'attr-equals',
                  },
                  /"|'/,
                ],
              },
            },
          },
        })
      },
    })

    Prism.languages.html = Prism.languages.markup
    Prism.languages.mathml = Prism.languages.markup
    Prism.languages.svg = Prism.languages.markup

    Prism.languages.xml = Prism.languages.extend('markup', {})
    Prism.languages.ssml = Prism.languages.xml
    Prism.languages.atom = Prism.languages.xml
    Prism.languages.rss = Prism.languages.xml

    /* **********************************************
     Begin prism-css.js
********************************************** */

    ;(function (Prism) {
      var string = /(?:"(?:\\(?:\r\n|[\s\S])|[^"\\\r\n])*"|'(?:\\(?:\r\n|[\s\S])|[^'\\\r\n])*')/

      Prism.languages.css = {
        comment: /\/\*[\s\S]*?\*\//,
        atrule: {
          pattern: RegExp(
            '@[\\w-](?:' +
              /[^;{\s"']|\s+(?!\s)/.source +
              '|' +
              string.source +
              ')*?' +
              /(?:;|(?=\s*\{))/.source
          ),
          inside: {
            rule: /^@[\w-]+/,
            'selector-function-argument': {
              pattern:
                /(\bselector\s*\(\s*(?![\s)]))(?:[^()\s]|\s+(?![\s)])|\((?:[^()]|\([^()]*\))*\))+(?=\s*\))/,
              lookbehind: true,
              alias: 'selector',
            },
            keyword: {
              pattern: /(^|[^\w-])(?:and|not|only|or)(?![\w-])/,
              lookbehind: true,
            },
            // See rest below
          },
        },
        url: {
          // https://drafts.csswg.org/css-values-3/#urls
          pattern: RegExp(
            '\\burl\\((?:' + string.source + '|' + /(?:[^\\\r\n()"']|\\[\s\S])*/.source + ')\\)',
            'i'
          ),
          greedy: true,
          inside: {
            function: /^url/i,
            punctuation: /^\(|\)$/,
            string: {
              pattern: RegExp('^' + string.source + '$'),
              alias: 'url',
            },
          },
        },
        selector: {
          pattern: RegExp(
            '(^|[{}\\s])[^{}\\s](?:[^{};"\'\\s]|\\s+(?![\\s{])|' + string.source + ')*(?=\\s*\\{)'
          ),
          lookbehind: true,
        },
        string: {
          pattern: string,
          greedy: true,
        },
        property: {
          pattern:
            /(^|[^-\w\xA0-\uFFFF])(?!\s)[-_a-z\xA0-\uFFFF](?:(?!\s)[-\w\xA0-\uFFFF])*(?=\s*:)/i,
          lookbehind: true,
        },
        important: /!important\b/i,
        function: {
          pattern: /(^|[^-a-z0-9])[-a-z0-9]+(?=\()/i,
          lookbehind: true,
        },
        punctuation: /[(){};:,]/,
      }

      Prism.languages.css['atrule'].inside.rest = Prism.languages.css

      var markup = Prism.languages.markup
      if (markup) {
        markup.tag.addInlined('style', 'css')
        markup.tag.addAttribute('style', 'css')
      }
    })(Prism)

    /* **********************************************
     Begin prism-clike.js
********************************************** */

    Prism.languages.clike = {
      comment: [
        {
          pattern: /(^|[^\\])\/\*[\s\S]*?(?:\*\/|$)/,
          lookbehind: true,
          greedy: true,
        },
        {
          pattern: /(^|[^\\:])\/\/.*/,
          lookbehind: true,
          greedy: true,
        },
      ],
      string: {
        pattern: /(["'])(?:\\(?:\r\n|[\s\S])|(?!\1)[^\\\r\n])*\1/,
        greedy: true,
      },
      'class-name': {
        pattern:
          /(\b(?:class|extends|implements|instanceof|interface|new|trait)\s+|\bcatch\s+\()[\w.\\]+/i,
        lookbehind: true,
        inside: {
          punctuation: /[.\\]/,
        },
      },
      keyword:
        /\b(?:break|catch|continue|do|else|finally|for|function|if|in|instanceof|new|null|return|throw|try|while)\b/,
      boolean: /\b(?:false|true)\b/,
      function: /\b\w+(?=\()/,
      number: /\b0x[\da-f]+\b|(?:\b\d+(?:\.\d*)?|\B\.\d+)(?:e[+-]?\d+)?/i,
      operator: /[<>]=?|[!=]=?=?|--?|\+\+?|&&?|\|\|?|[?*/~^%]/,
      punctuation: /[{}[\];(),.:]/,
    }

    /* **********************************************
     Begin prism-javascript.js
********************************************** */

    Prism.languages.javascript = Prism.languages.extend('clike', {
      'class-name': [
        Prism.languages.clike['class-name'],
        {
          pattern:
            /(^|[^$\w\xA0-\uFFFF])(?!\s)[_$A-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\.(?:constructor|prototype))/,
          lookbehind: true,
        },
      ],
      keyword: [
        {
          pattern: /((?:^|\})\s*)catch\b/,
          lookbehind: true,
        },
        {
          pattern:
            /(^|[^.]|\.\.\.\s*)\b(?:as|assert(?=\s*\{)|async(?=\s*(?:function\b|\(|[$\w\xA0-\uFFFF]|$))|await|break|case|class|const|continue|debugger|default|delete|do|else|enum|export|extends|finally(?=\s*(?:\{|$))|for|from(?=\s*(?:['"]|$))|function|(?:get|set)(?=\s*(?:[#\[$\w\xA0-\uFFFF]|$))|if|implements|import|in|instanceof|interface|let|new|null|of|package|private|protected|public|return|static|super|switch|this|throw|try|typeof|undefined|var|void|while|with|yield)\b/,
          lookbehind: true,
        },
      ],
      // Allow for all non-ASCII characters (See http://stackoverflow.com/a/2008444)
      function:
        /#?(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\s*(?:\.\s*(?:apply|bind|call)\s*)?\()/,
      number: {
        pattern: RegExp(
          /(^|[^\w$])/.source +
            '(?:' +
            // constant
            (/NaN|Infinity/.source +
              '|' +
              // binary integer
              /0[bB][01]+(?:_[01]+)*n?/.source +
              '|' +
              // octal integer
              /0[oO][0-7]+(?:_[0-7]+)*n?/.source +
              '|' +
              // hexadecimal integer
              /0[xX][\dA-Fa-f]+(?:_[\dA-Fa-f]+)*n?/.source +
              '|' +
              // decimal bigint
              /\d+(?:_\d+)*n/.source +
              '|' +
              // decimal number (integer or float) but no bigint
              /(?:\d+(?:_\d+)*(?:\.(?:\d+(?:_\d+)*)?)?|\.\d+(?:_\d+)*)(?:[Ee][+-]?\d+(?:_\d+)*)?/
                .source) +
            ')' +
            /(?![\w$])/.source
        ),
        lookbehind: true,
      },
      operator:
        /--|\+\+|\*\*=?|=>|&&=?|\|\|=?|[!=]==|<<=?|>>>?=?|[-+*/%&|^!=<>]=?|\.{3}|\?\?=?|\?\.?|[~:]/,
    })

    Prism.languages.javascript['class-name'][0].pattern =
      /(\b(?:class|extends|implements|instanceof|interface|new)\s+)[\w.\\]+/

    Prism.languages.insertBefore('javascript', 'keyword', {
      regex: {
        pattern: RegExp(
          // lookbehind
          // eslint-disable-next-line regexp/no-dupe-characters-character-class
          /((?:^|[^$\w\xA0-\uFFFF."'\])\s]|\b(?:return|yield))\s*)/.source +
            // Regex pattern:
            // There are 2 regex patterns here. The RegExp set notation proposal added support for nested character
            // classes if the `v` flag is present. Unfortunately, nested CCs are both context-free and incompatible
            // with the only syntax, so we have to define 2 different regex patterns.
            /\//.source +
            '(?:' +
            /(?:\[(?:[^\]\\\r\n]|\\.)*\]|\\.|[^/\\\[\r\n])+\/[dgimyus]{0,7}/.source +
            '|' +
            // `v` flag syntax. This supports 3 levels of nested character classes.
            /(?:\[(?:[^[\]\\\r\n]|\\.|\[(?:[^[\]\\\r\n]|\\.|\[(?:[^[\]\\\r\n]|\\.)*\])*\])*\]|\\.|[^/\\\[\r\n])+\/[dgimyus]{0,7}v[dgimyus]{0,7}/
              .source +
            ')' +
            // lookahead
            /(?=(?:\s|\/\*(?:[^*]|\*(?!\/))*\*\/)*(?:$|[\r\n,.;:})\]]|\/\/))/.source
        ),
        lookbehind: true,
        greedy: true,
        inside: {
          'regex-source': {
            pattern: /^(\/)[\s\S]+(?=\/[a-z]*$)/,
            lookbehind: true,
            alias: 'language-regex',
            inside: Prism.languages.regex,
          },
          'regex-delimiter': /^\/|\/$/,
          'regex-flags': /^[a-z]+$/,
        },
      },
      // This must be declared before keyword because we use "function" inside the look-forward
      'function-variable': {
        pattern:
          /#?(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\s*[=:]\s*(?:async\s*)?(?:\bfunction\b|(?:\((?:[^()]|\([^()]*\))*\)|(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*)\s*=>))/,
        alias: 'function',
      },
      parameter: [
        {
          pattern:
            /(function(?:\s+(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*)?\s*\(\s*)(?!\s)(?:[^()\s]|\s+(?![\s)])|\([^()]*\))+(?=\s*\))/,
          lookbehind: true,
          inside: Prism.languages.javascript,
        },
        {
          pattern:
            /(^|[^$\w\xA0-\uFFFF])(?!\s)[_$a-z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\s*=>)/i,
          lookbehind: true,
          inside: Prism.languages.javascript,
        },
        {
          pattern: /(\(\s*)(?!\s)(?:[^()\s]|\s+(?![\s)])|\([^()]*\))+(?=\s*\)\s*=>)/,
          lookbehind: true,
          inside: Prism.languages.javascript,
        },
        {
          pattern:
            /((?:\b|\s|^)(?!(?:as|async|await|break|case|catch|class|const|continue|debugger|default|delete|do|else|enum|export|extends|finally|for|from|function|get|if|implements|import|in|instanceof|interface|let|new|null|of|package|private|protected|public|return|set|static|super|switch|this|throw|try|typeof|undefined|var|void|while|with|yield)(?![$\w\xA0-\uFFFF]))(?:(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*\s*)\(\s*|\]\s*\(\s*)(?!\s)(?:[^()\s]|\s+(?![\s)])|\([^()]*\))+(?=\s*\)\s*\{)/,
          lookbehind: true,
          inside: Prism.languages.javascript,
        },
      ],
      constant: /\b[A-Z](?:[A-Z_]|\dx?)*\b/,
    })

    Prism.languages.insertBefore('javascript', 'string', {
      hashbang: {
        pattern: /^#!.*/,
        greedy: true,
        alias: 'comment',
      },
      'template-string': {
        pattern: /`(?:\\[\s\S]|\$\{(?:[^{}]|\{(?:[^{}]|\{[^}]*\})*\})+\}|(?!\$\{)[^\\`])*`/,
        greedy: true,
        inside: {
          'template-punctuation': {
            pattern: /^`|`$/,
            alias: 'string',
          },
          interpolation: {
            pattern: /((?:^|[^\\])(?:\\{2})*)\$\{(?:[^{}]|\{(?:[^{}]|\{[^}]*\})*\})+\}/,
            lookbehind: true,
            inside: {
              'interpolation-punctuation': {
                pattern: /^\$\{|\}$/,
                alias: 'punctuation',
              },
              rest: Prism.languages.javascript,
            },
          },
          string: /[\s\S]+/,
        },
      },
      'string-property': {
        pattern: /((?:^|[,{])[ \t]*)(["'])(?:\\(?:\r\n|[\s\S])|(?!\2)[^\\\r\n])*\2(?=\s*:)/m,
        lookbehind: true,
        greedy: true,
        alias: 'property',
      },
    })

    Prism.languages.insertBefore('javascript', 'operator', {
      'literal-property': {
        pattern:
          /((?:^|[,{])[ \t]*)(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\s*:)/m,
        lookbehind: true,
        alias: 'property',
      },
    })

    if (Prism.languages.markup) {
      Prism.languages.markup.tag.addInlined('script', 'javascript')

      // add attribute support for all DOM events.
      // https://developer.mozilla.org/en-US/docs/Web/Events#Standard_events
      Prism.languages.markup.tag.addAttribute(
        /on(?:abort|blur|change|click|composition(?:end|start|update)|dblclick|error|focus(?:in|out)?|key(?:down|up)|load|mouse(?:down|enter|leave|move|out|over|up)|reset|resize|scroll|select|slotchange|submit|unload|wheel)/
          .source,
        'javascript'
      )
    }

    Prism.languages.js = Prism.languages.javascript

    /* **********************************************
     Begin prism-file-highlight.js
********************************************** */

    ;(function () {
      if (typeof Prism === 'undefined' || typeof document === 'undefined') {
        return
      }

      // https://developer.mozilla.org/en-US/docs/Web/API/Element/matches#Polyfill
      if (!Element.prototype.matches) {
        Element.prototype.matches =
          Element.prototype.msMatchesSelector || Element.prototype.webkitMatchesSelector
      }

      var LOADING_MESSAGE = 'Loading…'
      var FAILURE_MESSAGE = function (status, message) {
        return '✖ Error ' + status + ' while fetching file: ' + message
      }
      var FAILURE_EMPTY_MESSAGE = '✖ Error: File does not exist or is empty'

      var EXTENSIONS = {
        js: 'javascript',
        py: 'python',
        rb: 'ruby',
        ps1: 'powershell',
        psm1: 'powershell',
        sh: 'bash',
        bat: 'batch',
        h: 'c',
        tex: 'latex',
      }

      var STATUS_ATTR = 'data-src-status'
      var STATUS_LOADING = 'loading'
      var STATUS_LOADED = 'loaded'
      var STATUS_FAILED = 'failed'

      var SELECTOR =
        'pre[data-src]:not([' +
        STATUS_ATTR +
        '="' +
        STATUS_LOADED +
        '"])' +
        ':not([' +
        STATUS_ATTR +
        '="' +
        STATUS_LOADING +
        '"])'

      /**
       * Loads the given file.
       *
       * @param {string} src The URL or path of the source file to load.
       * @param {(result: string) => void} success
       * @param {(reason: string) => void} error
       */
      function loadFile(src, success, error) {
        var xhr = new XMLHttpRequest()
        xhr.open('GET', src, true)
        xhr.onreadystatechange = function () {
          if (xhr.readyState == 4) {
            if (xhr.status < 400 && xhr.responseText) {
              success(xhr.responseText)
            } else {
              if (xhr.status >= 400) {
                error(FAILURE_MESSAGE(xhr.status, xhr.statusText))
              } else {
                error(FAILURE_EMPTY_MESSAGE)
              }
            }
          }
        }
        xhr.send(null)
      }

      /**
       * Parses the given range.
       *
       * This returns a range with inclusive ends.
       *
       * @param {string | null | undefined} range
       * @returns {[number, number | undefined] | undefined}
       */
      function parseRange(range) {
        var m = /^\s*(\d+)\s*(?:(,)\s*(?:(\d+)\s*)?)?$/.exec(range || '')
        if (m) {
          var start = Number(m[1])
          var comma = m[2]
          var end = m[3]

          if (!comma) {
            return [start, start]
          }
          if (!end) {
            return [start, undefined]
          }
          return [start, Number(end)]
        }
        return undefined
      }

      Prism.hooks.add('before-highlightall', function (env) {
        env.selector += ', ' + SELECTOR
      })

      Prism.hooks.add('before-sanity-check', function (env) {
        var pre = /** @type {HTMLPreElement} */ (env.element)
        if (pre.matches(SELECTOR)) {
          env.code = '' // fast-path the whole thing and go to complete

          pre.setAttribute(STATUS_ATTR, STATUS_LOADING) // mark as loading

          // add code element with loading message
          var code = pre.appendChild(document.createElement('CODE'))
          code.textContent = LOADING_MESSAGE

          var src = pre.getAttribute('data-src')

          var language = env.language
          if (language === 'none') {
            // the language might be 'none' because there is no language set;
            // in this case, we want to use the extension as the language
            var extension = (/\.(\w+)$/.exec(src) || [, 'none'])[1]
            language = EXTENSIONS[extension] || extension
          }

          // set language classes
          Prism.util.setLanguage(code, language)
          Prism.util.setLanguage(pre, language)

          // preload the language
          var autoloader = Prism.plugins.autoloader
          if (autoloader) {
            autoloader.loadLanguages(language)
          }

          // load file
          loadFile(
            src,
            function (text) {
              // mark as loaded
              pre.setAttribute(STATUS_ATTR, STATUS_LOADED)

              // handle data-range
              var range = parseRange(pre.getAttribute('data-range'))
              if (range) {
                var lines = text.split(/\r\n?|\n/g)

                // the range is one-based and inclusive on both ends
                var start = range[0]
                var end = range[1] == null ? lines.length : range[1]

                if (start < 0) {
                  start += lines.length
                }
                start = Math.max(0, Math.min(start - 1, lines.length))
                if (end < 0) {
                  end += lines.length
                }
                end = Math.max(0, Math.min(end, lines.length))

                text = lines.slice(start, end).join('\n')

                // add data-start for line numbers
                if (!pre.hasAttribute('data-start')) {
                  pre.setAttribute('data-start', String(start + 1))
                }
              }

              // highlight code
              code.textContent = text
              Prism.highlightElement(code)
            },
            function (error) {
              // mark as failed
              pre.setAttribute(STATUS_ATTR, STATUS_FAILED)

              code.textContent = error
            }
          )
        }
      })

      Prism.plugins.fileHighlight = {
        /**
         * Executes the File Highlight plugin for all matching `pre` elements under the given container.
         *
         * Note: Elements which are already loaded or currently loading will not be touched by this method.
         *
         * @param {ParentNode} [container=document]
         */
        highlight: function highlight(container) {
          var elements = (container || document).querySelectorAll(SELECTOR)

          for (var i = 0, element; (element = elements[i++]); ) {
            Prism.highlightElement(element)
          }
        },
      }

      var logged = false
      /** @deprecated Use `Prism.plugins.fileHighlight.highlight` instead. */
      Prism.fileHighlight = function () {
        if (!logged) {
          console.warn(
            'Prism.fileHighlight is deprecated. Use `Prism.plugins.fileHighlight.highlight` instead.'
          )
          logged = true
        }
        Prism.plugins.fileHighlight.highlight.apply(this, arguments)
      }
    })()

    /***/
  },

  /***/ 1794: /***/ (__webpack_module__, __webpack_exports__, __webpack_require__) => {
    'use strict'
    __webpack_require__.a(
      __webpack_module__,
      async (__webpack_handle_async_dependencies__, __webpack_async_result__) => {
        try {
          __webpack_require__.r(__webpack_exports__)
          /* harmony export */ __webpack_require__.d(__webpack_exports__, {
            /* harmony export */ Code: () => /* binding */ Code,
            /* harmony export */
          })
          /* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(6689)
          /* harmony import */ var notion_utils__WEBPACK_IMPORTED_MODULE_1__ =
            __webpack_require__(8751)
          /* harmony import */ var prismjs__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(9499)
          /* harmony import */ var prismjs_components_prism_clike_min_js__WEBPACK_IMPORTED_MODULE_3__ =
            __webpack_require__(2508)
          /* harmony import */ var prismjs_components_prism_css_extras_min_js__WEBPACK_IMPORTED_MODULE_4__ =
            __webpack_require__(1151)
          /* harmony import */ var prismjs_components_prism_css_min_js__WEBPACK_IMPORTED_MODULE_5__ =
            __webpack_require__(1139)
          /* harmony import */ var prismjs_components_prism_javascript_min_js__WEBPACK_IMPORTED_MODULE_6__ =
            __webpack_require__(1855)
          /* harmony import */ var prismjs_components_prism_js_extras_min_js__WEBPACK_IMPORTED_MODULE_7__ =
            __webpack_require__(3784)
          /* harmony import */ var prismjs_components_prism_json_min_js__WEBPACK_IMPORTED_MODULE_8__ =
            __webpack_require__(5139)
          /* harmony import */ var prismjs_components_prism_jsx_min_js__WEBPACK_IMPORTED_MODULE_9__ =
            __webpack_require__(9146)
          /* harmony import */ var prismjs_components_prism_tsx_min_js__WEBPACK_IMPORTED_MODULE_10__ =
            __webpack_require__(4520)
          /* harmony import */ var prismjs_components_prism_typescript_min_js__WEBPACK_IMPORTED_MODULE_11__ =
            __webpack_require__(3416)
          /* harmony import */ var react_image__WEBPACK_IMPORTED_MODULE_12__ =
            __webpack_require__(9358)
          /* harmony import */ var react_lazy_images__WEBPACK_IMPORTED_MODULE_13__ =
            __webpack_require__(5830)
          /* harmony import */ var react_hotkeys_hook__WEBPACK_IMPORTED_MODULE_14__ =
            __webpack_require__(2784)
          /* harmony import */ var react_fast_compare__WEBPACK_IMPORTED_MODULE_15__ =
            __webpack_require__(258)
          var __webpack_async_dependencies__ = __webpack_handle_async_dependencies__([
            notion_utils__WEBPACK_IMPORTED_MODULE_1__,
          ])
          notion_utils__WEBPACK_IMPORTED_MODULE_1__ = (
            __webpack_async_dependencies__.then
              ? (await __webpack_async_dependencies__)()
              : __webpack_async_dependencies__
          )[0]
          var __create = Object.create
          var __defProp = Object.defineProperty
          var __defProps = Object.defineProperties
          var __getOwnPropDesc = Object.getOwnPropertyDescriptor
          var __getOwnPropDescs = Object.getOwnPropertyDescriptors
          var __getOwnPropNames = Object.getOwnPropertyNames
          var __getOwnPropSymbols = Object.getOwnPropertySymbols
          var __getProtoOf = Object.getPrototypeOf
          var __hasOwnProp = Object.prototype.hasOwnProperty
          var __propIsEnum = Object.prototype.propertyIsEnumerable
          var __defNormalProp = (obj, key, value) =>
            key in obj
              ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value })
              : (obj[key] = value)
          var __spreadValues = (a, b) => {
            for (var prop in b || (b = {}))
              if (__hasOwnProp.call(b, prop)) __defNormalProp(a, prop, b[prop])
            if (__getOwnPropSymbols)
              for (var prop of __getOwnPropSymbols(b)) {
                if (__propIsEnum.call(b, prop)) __defNormalProp(a, prop, b[prop])
              }
            return a
          }
          var __spreadProps = (a, b) => __defProps(a, __getOwnPropDescs(b))
          var __objRest = (source, exclude) => {
            var target = {}
            for (var prop in source)
              if (__hasOwnProp.call(source, prop) && exclude.indexOf(prop) < 0)
                target[prop] = source[prop]
            if (source != null && __getOwnPropSymbols)
              for (var prop of __getOwnPropSymbols(source)) {
                if (exclude.indexOf(prop) < 0 && __propIsEnum.call(source, prop))
                  target[prop] = source[prop]
              }
            return target
          }
          var __commonJS = (cb, mod) =>
            function __require() {
              return (
                mod || (0, cb[__getOwnPropNames(cb)[0]])((mod = { exports: {} }).exports, mod),
                mod.exports
              )
            }
          var __copyProps = (to, from, except, desc) => {
            if ((from && typeof from === 'object') || typeof from === 'function') {
              for (let key of __getOwnPropNames(from))
                if (!__hasOwnProp.call(to, key) && key !== except)
                  __defProp(to, key, {
                    get: () => from[key],
                    enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable,
                  })
            }
            return to
          }
          var __toESM = (mod, isNodeMode, target) => (
            (target = mod != null ? __create(__getProtoOf(mod)) : {}),
            __copyProps(
              isNodeMode || !mod || !mod.__esModule
                ? __defProp(target, 'default', { value: mod, enumerable: true })
                : target,
              mod
            )
          )
          var __async = (__this, __arguments, generator) => {
            return new Promise((resolve, reject) => {
              var fulfilled = (value) => {
                try {
                  step(generator.next(value))
                } catch (e) {
                  reject(e)
                }
              }
              var rejected = (value) => {
                try {
                  step(generator.throw(value))
                } catch (e) {
                  reject(e)
                }
              }
              var step = (x) =>
                x.done ? resolve(x.value) : Promise.resolve(x.value).then(fulfilled, rejected)
              step((generator = generator.apply(__this, __arguments)).next())
            })
          }

          // ../../node_modules/clipboard-copy/index.js
          var require_clipboard_copy = __commonJS({
            '../../node_modules/clipboard-copy/index.js'(exports, module) {
              module.exports = clipboardCopy
              function makeError() {
                return new DOMException('The request is not allowed', 'NotAllowedError')
              }
              function copyClipboardApi(text) {
                return __async(this, null, function* () {
                  if (!navigator.clipboard) {
                    throw makeError()
                  }
                  return navigator.clipboard.writeText(text)
                })
              }
              function copyExecCommand(text) {
                return __async(this, null, function* () {
                  const span = document.createElement('span')
                  span.textContent = text
                  span.style.whiteSpace = 'pre'
                  span.style.webkitUserSelect = 'auto'
                  span.style.userSelect = 'all'
                  document.body.appendChild(span)
                  const selection = window.getSelection()
                  const range = window.document.createRange()
                  selection.removeAllRanges()
                  range.selectNode(span)
                  selection.addRange(range)
                  let success = false
                  try {
                    success = window.document.execCommand('copy')
                  } finally {
                    selection.removeAllRanges()
                    window.document.body.removeChild(span)
                  }
                  if (!success) throw makeError()
                })
              }
              function clipboardCopy(text) {
                return __async(this, null, function* () {
                  try {
                    yield copyClipboardApi(text)
                  } catch (err) {
                    try {
                      yield copyExecCommand(text)
                    } catch (err2) {
                      throw err2 || err || makeError()
                    }
                  }
                })
              }
            },
          })

          // ../../node_modules/lodash.throttle/index.js
          var require_lodash = __commonJS({
            '../../node_modules/lodash.throttle/index.js'(exports, module) {
              var FUNC_ERROR_TEXT = 'Expected a function'
              var NAN = 0 / 0
              var symbolTag = '[object Symbol]'
              var reTrim = /^\s+|\s+$/g
              var reIsBadHex = /^[-+]0x[0-9a-f]+$/i
              var reIsBinary = /^0b[01]+$/i
              var reIsOctal = /^0o[0-7]+$/i
              var freeParseInt = parseInt
              var freeGlobal =
                typeof global == 'object' && global && global.Object === Object && global
              var freeSelf = typeof self == 'object' && self && self.Object === Object && self
              var root = freeGlobal || freeSelf || Function('return this')()
              var objectProto = Object.prototype
              var objectToString = objectProto.toString
              var nativeMax = Math.max
              var nativeMin = Math.min
              var now = function () {
                return root.Date.now()
              }
              function debounce(func, wait, options) {
                var lastArgs,
                  lastThis,
                  maxWait,
                  result,
                  timerId,
                  lastCallTime,
                  lastInvokeTime = 0,
                  leading = false,
                  maxing = false,
                  trailing = true
                if (typeof func != 'function') {
                  throw new TypeError(FUNC_ERROR_TEXT)
                }
                wait = toNumber(wait) || 0
                if (isObject(options)) {
                  leading = !!options.leading
                  maxing = 'maxWait' in options
                  maxWait = maxing ? nativeMax(toNumber(options.maxWait) || 0, wait) : maxWait
                  trailing = 'trailing' in options ? !!options.trailing : trailing
                }
                function invokeFunc(time) {
                  var args = lastArgs,
                    thisArg = lastThis
                  lastArgs = lastThis = void 0
                  lastInvokeTime = time
                  result = func.apply(thisArg, args)
                  return result
                }
                function leadingEdge(time) {
                  lastInvokeTime = time
                  timerId = setTimeout(timerExpired, wait)
                  return leading ? invokeFunc(time) : result
                }
                function remainingWait(time) {
                  var timeSinceLastCall = time - lastCallTime,
                    timeSinceLastInvoke = time - lastInvokeTime,
                    result2 = wait - timeSinceLastCall
                  return maxing ? nativeMin(result2, maxWait - timeSinceLastInvoke) : result2
                }
                function shouldInvoke(time) {
                  var timeSinceLastCall = time - lastCallTime,
                    timeSinceLastInvoke = time - lastInvokeTime
                  return (
                    lastCallTime === void 0 ||
                    timeSinceLastCall >= wait ||
                    timeSinceLastCall < 0 ||
                    (maxing && timeSinceLastInvoke >= maxWait)
                  )
                }
                function timerExpired() {
                  var time = now()
                  if (shouldInvoke(time)) {
                    return trailingEdge(time)
                  }
                  timerId = setTimeout(timerExpired, remainingWait(time))
                }
                function trailingEdge(time) {
                  timerId = void 0
                  if (trailing && lastArgs) {
                    return invokeFunc(time)
                  }
                  lastArgs = lastThis = void 0
                  return result
                }
                function cancel() {
                  if (timerId !== void 0) {
                    clearTimeout(timerId)
                  }
                  lastInvokeTime = 0
                  lastArgs = lastCallTime = lastThis = timerId = void 0
                }
                function flush() {
                  return timerId === void 0 ? result : trailingEdge(now())
                }
                function debounced() {
                  var time = now(),
                    isInvoking = shouldInvoke(time)
                  lastArgs = arguments
                  lastThis = this
                  lastCallTime = time
                  if (isInvoking) {
                    if (timerId === void 0) {
                      return leadingEdge(lastCallTime)
                    }
                    if (maxing) {
                      timerId = setTimeout(timerExpired, wait)
                      return invokeFunc(lastCallTime)
                    }
                  }
                  if (timerId === void 0) {
                    timerId = setTimeout(timerExpired, wait)
                  }
                  return result
                }
                debounced.cancel = cancel
                debounced.flush = flush
                return debounced
              }
              function throttle2(func, wait, options) {
                var leading = true,
                  trailing = true
                if (typeof func != 'function') {
                  throw new TypeError(FUNC_ERROR_TEXT)
                }
                if (isObject(options)) {
                  leading = 'leading' in options ? !!options.leading : leading
                  trailing = 'trailing' in options ? !!options.trailing : trailing
                }
                return debounce(func, wait, {
                  leading: leading,
                  maxWait: wait,
                  trailing: trailing,
                })
              }
              function isObject(value) {
                var type = typeof value
                return !!value && (type == 'object' || type == 'function')
              }
              function isObjectLike(value) {
                return !!value && typeof value == 'object'
              }
              function isSymbol(value) {
                return (
                  typeof value == 'symbol' ||
                  (isObjectLike(value) && objectToString.call(value) == symbolTag)
                )
              }
              function toNumber(value) {
                if (typeof value == 'number') {
                  return value
                }
                if (isSymbol(value)) {
                  return NAN
                }
                if (isObject(value)) {
                  var other = typeof value.valueOf == 'function' ? value.valueOf() : value
                  value = isObject(other) ? other + '' : other
                }
                if (typeof value != 'string') {
                  return value === 0 ? value : +value
                }
                value = value.replace(reTrim, '')
                var isBinary = reIsBinary.test(value)
                return isBinary || reIsOctal.test(value)
                  ? freeParseInt(value.slice(2), isBinary ? 2 : 8)
                  : reIsBadHex.test(value)
                  ? NAN
                  : +value
              }
              module.exports = throttle2
            },
          })

          // src/third-party/code.tsx
          var import_clipboard_copy = __toESM(require_clipboard_copy(), 1)

          // src/components/text.tsx

          // src/context.tsx

          // src/components/asset-wrapper.tsx

          // src/utils.ts

          // src/map-image-url.ts
          var defaultMapImageUrl = (url, block) => {
            if (!url) {
              return null
            }
            if (url.startsWith('data:')) {
              return url
            }
            if (url.startsWith('https://images.unsplash.com')) {
              return url
            }
            try {
              const u = new URL(url)
              if (
                u.pathname.startsWith('/secure.notion-static.com') &&
                u.hostname.endsWith('.amazonaws.com')
              ) {
                if (
                  u.searchParams.has('X-Amz-Credential') &&
                  u.searchParams.has('X-Amz-Signature') &&
                  u.searchParams.has('X-Amz-Algorithm')
                ) {
                  return url
                }
              }
            } catch (e) {}
            if (url.startsWith('/images')) {
              url = `https://www.notion.so${url}`
            }
            url = `https://www.notion.so${
              url.startsWith('/image') ? url : `/image/${encodeURIComponent(url)}`
            }`
            const notionImageUrlV2 = new URL(url)
            let table = block.parent_table === 'space' ? 'block' : block.parent_table
            if (table === 'collection' || table === 'team') {
              table = 'block'
            }
            notionImageUrlV2.searchParams.set('table', table)
            notionImageUrlV2.searchParams.set('id', block.id)
            notionImageUrlV2.searchParams.set('cache', 'v2')
            url = notionImageUrlV2.toString()
            return url
          }

          // src/map-page-url.ts
          var defaultMapPageUrl = (rootPageId) => (pageId) => {
            pageId = (pageId || '').replace(/-/g, '')
            if (rootPageId && pageId === rootPageId) {
              return '/'
            } else {
              return `/${pageId}`
            }
          }

          // src/utils.ts
          var cs = (...classes) => classes.filter((a) => !!a).join(' ')
          var getHashFragmentValue = (url) => {
            return url.includes('#') ? url.replace(/^.+(#.+)$/, '$1') : ''
          }
          var isBrowser = typeof window !== 'undefined'
          var youtubeDomains = /* @__PURE__ */ new Set([
            'youtu.be',
            'youtube.com',
            'www.youtube.com',
            'youtube-nocookie.com',
            'www.youtube-nocookie.com',
          ])
          var getYoutubeId = (url) => {
            try {
              const { hostname } = new URL(url)
              if (!youtubeDomains.has(hostname)) {
                return null
              }
              const regExp = /^.*(youtu\.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/i
              const match = url.match(regExp)
              if (match && match[2].length == 11) {
                return match[2]
              }
            } catch (e) {}
            return null
          }

          // src/components/eoi.tsx

          // src/icons/type-github.tsx

          function SvgTypeGitHub(props) {
            return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              'svg',
              __spreadValues(
                {
                  viewBox: '0 0 260 260',
                },
                props
              ),
              /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                'g',
                null,
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('path', {
                  d: 'M128.00106,0 C57.3172926,0 0,57.3066942 0,128.00106 C0,184.555281 36.6761997,232.535542 87.534937,249.460899 C93.9320223,250.645779 96.280588,246.684165 96.280588,243.303333 C96.280588,240.251045 96.1618878,230.167899 96.106777,219.472176 C60.4967585,227.215235 52.9826207,204.369712 52.9826207,204.369712 C47.1599584,189.574598 38.770408,185.640538 38.770408,185.640538 C27.1568785,177.696113 39.6458206,177.859325 39.6458206,177.859325 C52.4993419,178.762293 59.267365,191.04987 59.267365,191.04987 C70.6837675,210.618423 89.2115753,204.961093 96.5158685,201.690482 C97.6647155,193.417512 100.981959,187.77078 104.642583,184.574357 C76.211799,181.33766 46.324819,170.362144 46.324819,121.315702 C46.324819,107.340889 51.3250588,95.9223682 59.5132437,86.9583937 C58.1842268,83.7344152 53.8029229,70.715562 60.7532354,53.0843636 C60.7532354,53.0843636 71.5019501,49.6441813 95.9626412,66.2049595 C106.172967,63.368876 117.123047,61.9465949 128.00106,61.8978432 C138.879073,61.9465949 149.837632,63.368876 160.067033,66.2049595 C184.49805,49.6441813 195.231926,53.0843636 195.231926,53.0843636 C202.199197,70.715562 197.815773,83.7344152 196.486756,86.9583937 C204.694018,95.9223682 209.660343,107.340889 209.660343,121.315702 C209.660343,170.478725 179.716133,181.303747 151.213281,184.472614 C155.80443,188.444828 159.895342,196.234518 159.895342,208.176593 C159.895342,225.303317 159.746968,239.087361 159.746968,243.303333 C159.746968,246.709601 162.05102,250.70089 168.53925,249.443941 C219.370432,232.499507 256,184.536204 256,128.00106 C256,57.3066942 198.691187,0 128.00106,0 Z M47.9405593,182.340212 C47.6586465,182.976105 46.6581745,183.166873 45.7467277,182.730227 C44.8183235,182.312656 44.2968914,181.445722 44.5978808,180.80771 C44.8734344,180.152739 45.876026,179.97045 46.8023103,180.409216 C47.7328342,180.826786 48.2627451,181.702199 47.9405593,182.340212 Z M54.2367892,187.958254 C53.6263318,188.524199 52.4329723,188.261363 51.6232682,187.366874 C50.7860088,186.474504 50.6291553,185.281144 51.2480912,184.70672 C51.8776254,184.140775 53.0349512,184.405731 53.8743302,185.298101 C54.7115892,186.201069 54.8748019,187.38595 54.2367892,187.958254 Z M58.5562413,195.146347 C57.7719732,195.691096 56.4895886,195.180261 55.6968417,194.042013 C54.9125733,192.903764 54.9125733,191.538713 55.713799,190.991845 C56.5086651,190.444977 57.7719732,190.936735 58.5753181,192.066505 C59.3574669,193.22383 59.3574669,194.58888 58.5562413,195.146347 Z M65.8613592,203.471174 C65.1597571,204.244846 63.6654083,204.03712 62.5716717,202.981538 C61.4524999,201.94927 61.1409122,200.484596 61.8446341,199.710926 C62.5547146,198.935137 64.0575422,199.15346 65.1597571,200.200564 C66.2704506,201.230712 66.6095936,202.705984 65.8613592,203.471174 Z M75.3025151,206.281542 C74.9930474,207.284134 73.553809,207.739857 72.1039724,207.313809 C70.6562556,206.875043 69.7087748,205.700761 70.0012857,204.687571 C70.302275,203.678621 71.7478721,203.20382 73.2083069,203.659543 C74.6539041,204.09619 75.6035048,205.261994 75.3025151,206.281542 Z M86.046947,207.473627 C86.0829806,208.529209 84.8535871,209.404622 83.3316829,209.4237 C81.8013,209.457614 80.563428,208.603398 80.5464708,207.564772 C80.5464708,206.498591 81.7483088,205.631657 83.2786917,205.606221 C84.8005962,205.576546 86.046947,206.424403 86.046947,207.473627 Z M96.6021471,207.069023 C96.7844366,208.099171 95.7267341,209.156872 94.215428,209.438785 C92.7295577,209.710099 91.3539086,209.074206 91.1652603,208.052538 C90.9808515,206.996955 92.0576306,205.939253 93.5413813,205.66582 C95.054807,205.402984 96.4092596,206.021919 96.6021471,207.069023 Z',
                  fill: '#161614',
                })
              )
            )
          }
          var type_github_default = SvgTypeGitHub

          // src/components/eoi.tsx
          var EOI = ({ block, inline, className }) => {
            var _a, _b, _c
            const { components } = useNotionContext()
            const { original_url, attributes, domain } =
              (block == null ? void 0 : block.format) || {}
            if (!original_url || !attributes) {
              return null
            }
            const title =
              (_a = attributes.find((attr) => attr.id === 'title')) == null ? void 0 : _a.values[0]
            let owner =
              (_b = attributes.find((attr) => attr.id === 'owner')) == null ? void 0 : _b.values[0]
            const lastUpdatedAt =
              (_c = attributes.find((attr) => attr.id === 'updated_at')) == null
                ? void 0
                : _c.values[0]
            const lastUpdated = lastUpdatedAt
              ? (0, notion_utils__WEBPACK_IMPORTED_MODULE_1__.formatNotionDateTime)(lastUpdatedAt)
              : null
            let externalImage
            switch (domain) {
              case 'github.com':
                externalImage = /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  type_github_default,
                  null
                )
                if (owner) {
                  const parts = owner.split('/')
                  owner = parts[parts.length - 1]
                }
                break
              default:
                if (true) {
                  console.log(
                    `Unsupported external_object_instance domain "${domain}"`,
                    JSON.stringify(block, null, 2)
                  )
                }
                return null
            }
            return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              components.Link,
              {
                target: '_blank',
                rel: 'noopener noreferrer',
                href: original_url,
                className: cs(
                  'notion-external',
                  inline ? 'notion-external-mention' : 'notion-external-block notion-row',
                  className
                ),
              },
              externalImage &&
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  'div',
                  {
                    className: 'notion-external-image',
                  },
                  externalImage
                ),
              /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                'div',
                {
                  className: 'notion-external-description',
                },
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  'div',
                  {
                    className: 'notion-external-title',
                  },
                  title
                ),
                (owner || lastUpdated) &&
                  /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                    'div',
                    {
                      className: 'notion-external-subtitle',
                    },
                    owner &&
                      /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                        'span',
                        null,
                        owner
                      ),
                    owner &&
                      lastUpdated &&
                      /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                        'span',
                        null,
                        ' \u2022 '
                      ),
                    lastUpdated &&
                      /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                        'span',
                        null,
                        'Updated ',
                        lastUpdated
                      )
                  )
              )
            )
          }

          // src/components/graceful-image.tsx

          var GracefulImage = (props) => {
            if (isBrowser) {
              return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                react_image__WEBPACK_IMPORTED_MODULE_12__.Img,
                __spreadValues({}, props)
              )
            } else {
              return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                'img',
                __spreadValues({}, props)
              )
            }
          }

          // src/components/lazy-image.tsx

          var LazyImage = (_a) => {
            var _b = _a,
              { src, alt, className, style, zoomable = false, priority = false, height } = _b,
              rest = __objRest(_b, [
                'src',
                'alt',
                'className',
                'style',
                'zoomable',
                'priority',
                'height',
              ])
            var _a2, _b2, _c
            const { recordMap, zoom, previewImages, forceCustomImages, components } =
              useNotionContext()
            const zoomRef = react__WEBPACK_IMPORTED_MODULE_0__.useRef(zoom ? zoom.clone() : null)
            const previewImage = previewImages
              ? (_c =
                  (_a2 = recordMap == null ? void 0 : recordMap.preview_images) == null
                    ? void 0
                    : _a2[src]) != null
                ? _c
                : (_b2 = recordMap == null ? void 0 : recordMap.preview_images) == null
                ? void 0
                : _b2[(0, notion_utils__WEBPACK_IMPORTED_MODULE_1__.normalizeUrl)(src)]
              : null
            const onLoad = react__WEBPACK_IMPORTED_MODULE_0__.useCallback(
              (e) => {
                if (zoomable && (e.target.src || e.target.srcset)) {
                  if (zoomRef.current) {
                    zoomRef.current.attach(e.target)
                  }
                }
              },
              [zoomRef, zoomable]
            )
            const attachZoom = react__WEBPACK_IMPORTED_MODULE_0__.useCallback(
              (image) => {
                if (zoomRef.current && image) {
                  zoomRef.current.attach(image)
                }
              },
              [zoomRef]
            )
            const attachZoomRef = react__WEBPACK_IMPORTED_MODULE_0__.useMemo(
              () => (zoomable ? attachZoom : void 0),
              [zoomable, attachZoom]
            )
            if (previewImage) {
              const aspectRatio = previewImage.originalHeight / previewImage.originalWidth
              if (components.Image) {
                return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  components.Image,
                  {
                    src,
                    alt,
                    style,
                    className,
                    width: previewImage.originalWidth,
                    height: previewImage.originalHeight,
                    blurDataURL: previewImage.dataURIBase64,
                    placeholder: 'blur',
                    priority,
                    onLoad,
                  }
                )
              }
              return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                react_lazy_images__WEBPACK_IMPORTED_MODULE_13__ /* .LazyImageFull */.AZ,
                __spreadProps(
                  __spreadValues(
                    {
                      src,
                    },
                    rest
                  ),
                  {
                    experimentalDecode: true,
                  }
                ),
                ({ imageState, ref }) => {
                  const isLoaded =
                    imageState ===
                    react_lazy_images__WEBPACK_IMPORTED_MODULE_13__ /* .ImageState.LoadSuccess */.zl
                      .LoadSuccess
                  const wrapperStyle = {
                    width: '100%',
                  }
                  const imgStyle = {}
                  if (height) {
                    wrapperStyle.height = height
                  } else {
                    imgStyle.position = 'absolute'
                    wrapperStyle.paddingBottom = `${aspectRatio * 100}%`
                  }
                  return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                    'div',
                    {
                      className: cs(
                        'lazy-image-wrapper',
                        isLoaded && 'lazy-image-loaded',
                        className
                      ),
                      style: wrapperStyle,
                    },
                    /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('img', {
                      className: 'lazy-image-preview',
                      src: previewImage.dataURIBase64,
                      alt,
                      ref,
                      style,
                      decoding: 'async',
                    }),
                    /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('img', {
                      className: 'lazy-image-real',
                      src,
                      alt,
                      ref: attachZoomRef,
                      style: __spreadValues(__spreadValues({}, style), imgStyle),
                      width: previewImage.originalWidth,
                      height: previewImage.originalHeight,
                      decoding: 'async',
                      loading: 'lazy',
                    })
                  )
                }
              )
            } else {
              if (components.Image && forceCustomImages) {
                return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  components.Image,
                  {
                    src,
                    alt,
                    className,
                    style,
                    width: null,
                    height: height || null,
                    priority,
                    onLoad,
                  }
                )
              }
              return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                'img',
                __spreadValues(
                  {
                    className,
                    style,
                    src,
                    alt,
                    ref: attachZoomRef,
                    loading: 'lazy',
                    decoding: 'async',
                  },
                  rest
                )
              )
            }
          }

          // src/components/page-icon.tsx

          // src/icons/default-page-icon.tsx

          var DefaultPageIcon = (props) => {
            const _a = props,
              { className } = _a,
              rest = __objRest(_a, ['className'])
            return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              'svg',
              __spreadProps(
                __spreadValues(
                  {
                    className,
                  },
                  rest
                ),
                {
                  viewBox: '0 0 30 30',
                  width: '16',
                }
              ),
              /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('path', {
                d: 'M16,1H4v28h22V11L16,1z M16,3.828L23.172,11H16V3.828z M24,27H6V3h8v10h10V27z M8,17h14v-2H8V17z M8,21h14v-2H8V21z M8,25h14v-2H8V25z',
              })
            )
          }

          // src/components/page-icon.tsx
          var isIconBlock = (value) => {
            return (
              value.type === 'page' ||
              value.type === 'callout' ||
              value.type === 'collection_view' ||
              value.type === 'collection_view_page'
            )
          }
          var PageIconImpl = ({
            block,
            className,
            inline = true,
            hideDefaultIcon = false,
            defaultIcon,
          }) => {
            var _a
            const { mapImageUrl, recordMap, darkMode } = useNotionContext()
            let isImage = false
            let content = null
            if (isIconBlock(block)) {
              const icon =
                ((_a = (0, notion_utils__WEBPACK_IMPORTED_MODULE_1__.getBlockIcon)(
                  block,
                  recordMap
                )) == null
                  ? void 0
                  : _a.trim()) || defaultIcon
              const title = (0, notion_utils__WEBPACK_IMPORTED_MODULE_1__.getBlockTitle)(
                block,
                recordMap
              )
              if (icon && (0, notion_utils__WEBPACK_IMPORTED_MODULE_1__.isUrl)(icon)) {
                const url = mapImageUrl(icon, block)
                isImage = true
                content = /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  LazyImage,
                  {
                    src: url,
                    alt: title || 'page icon',
                    className: cs(className, 'notion-page-icon'),
                  }
                )
              } else if (icon && icon.startsWith('/icons/')) {
                const url =
                  'https://www.notion.so' + icon + '?mode=' + (darkMode ? 'dark' : 'light')
                content = /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  LazyImage,
                  {
                    src: url,
                    alt: title || 'page icon',
                    className: cs(className, 'notion-page-icon'),
                  }
                )
              } else if (!icon) {
                if (!hideDefaultIcon) {
                  isImage = true
                  content = /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                    DefaultPageIcon,
                    {
                      className: cs(className, 'notion-page-icon'),
                      alt: title ? title : 'page icon',
                    }
                  )
                }
              } else {
                isImage = false
                content = /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  'span',
                  {
                    className: cs(className, 'notion-page-icon'),
                    role: 'img',
                    'aria-label': icon,
                  },
                  icon
                )
              }
            }
            if (!content) {
              return null
            }
            return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              'div',
              {
                className: cs(
                  inline ? 'notion-page-icon-inline' : 'notion-page-icon-hero',
                  isImage ? 'notion-page-icon-image' : 'notion-page-icon-span'
                ),
              },
              content
            )
          }
          var PageIcon = react__WEBPACK_IMPORTED_MODULE_0__.memo(PageIconImpl)

          // src/components/page-title.tsx

          var PageTitleImpl = (_a) => {
            var _b = _a,
              { block, className, defaultIcon } = _b,
              rest = __objRest(_b, ['block', 'className', 'defaultIcon'])
            var _a2, _b2
            const { recordMap } = useNotionContext()
            if (!block) return null
            if (block.type === 'collection_view_page' || block.type === 'collection_view') {
              const title = (0, notion_utils__WEBPACK_IMPORTED_MODULE_1__.getBlockTitle)(
                block,
                recordMap
              )
              if (!title) {
                return null
              }
              const titleDecoration = [[title]]
              return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                'span',
                __spreadValues(
                  {
                    className: cs('notion-page-title', className),
                  },
                  rest
                ),
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(PageIcon, {
                  block,
                  defaultIcon,
                  className: 'notion-page-title-icon',
                }),
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  'span',
                  {
                    className: 'notion-page-title-text',
                  },
                  /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(Text, {
                    value: titleDecoration,
                    block,
                  })
                )
              )
            }
            if (!((_a2 = block.properties) == null ? void 0 : _a2.title)) {
              return null
            }
            return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              'span',
              __spreadValues(
                {
                  className: cs('notion-page-title', className),
                },
                rest
              ),
              /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(PageIcon, {
                block,
                defaultIcon,
                className: 'notion-page-title-icon',
              }),
              /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                'span',
                {
                  className: 'notion-page-title-text',
                },
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(Text, {
                  value: (_b2 = block.properties) == null ? void 0 : _b2.title,
                  block,
                })
              )
            )
          }
          var PageTitle = react__WEBPACK_IMPORTED_MODULE_0__.memo(PageTitleImpl)

          // src/components/header.tsx

          // src/icons/search-icon.tsx

          var SearchIcon = (props) => {
            const _a = props,
              { className } = _a,
              rest = __objRest(_a, ['className'])
            return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              'svg',
              __spreadValues(
                {
                  className: cs('notion-icon', className),
                  viewBox: '0 0 17 17',
                },
                rest
              ),
              /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('path', {
                d: 'M6.78027 13.6729C8.24805 13.6729 9.60156 13.1982 10.709 12.4072L14.875 16.5732C15.0684 16.7666 15.3232 16.8633 15.5957 16.8633C16.167 16.8633 16.5713 16.4238 16.5713 15.8613C16.5713 15.5977 16.4834 15.3516 16.29 15.1582L12.1504 11.0098C13.0205 9.86719 13.5391 8.45215 13.5391 6.91406C13.5391 3.19629 10.498 0.155273 6.78027 0.155273C3.0625 0.155273 0.0214844 3.19629 0.0214844 6.91406C0.0214844 10.6318 3.0625 13.6729 6.78027 13.6729ZM6.78027 12.2139C3.87988 12.2139 1.48047 9.81445 1.48047 6.91406C1.48047 4.01367 3.87988 1.61426 6.78027 1.61426C9.68066 1.61426 12.0801 4.01367 12.0801 6.91406C12.0801 9.81445 9.68066 12.2139 6.78027 12.2139Z',
              })
            )
          }

          // src/components/search-dialog.tsx
          var import_lodash = __toESM(require_lodash(), 1)

          // src/icons/clear-icon.tsx

          var ClearIcon = (props) => {
            const _a = props,
              { className } = _a,
              rest = __objRest(_a, ['className'])
            return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              'svg',
              __spreadProps(
                __spreadValues(
                  {
                    className: cs('notion-icon', className),
                  },
                  rest
                ),
                {
                  viewBox: '0 0 30 30',
                }
              ),
              /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('path', {
                d: 'M15,0C6.716,0,0,6.716,0,15s6.716,15,15,15s15-6.716,15-15S23.284,0,15,0z M22,20.6L20.6,22L15,16.4L9.4,22L8,20.6l5.6-5.6 L8,9.4L9.4,8l5.6,5.6L20.6,8L22,9.4L16.4,15L22,20.6z',
              })
            )
          }

          // src/icons/loading-icon.tsx

          var LoadingIcon = (props) => {
            const _a = props,
              { className } = _a,
              rest = __objRest(_a, ['className'])
            return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              'svg',
              __spreadProps(
                __spreadValues(
                  {
                    className: cs('notion-icon', className),
                  },
                  rest
                ),
                {
                  viewBox: '0 0 24 24',
                }
              ),
              /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                'defs',
                null,
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  'linearGradient',
                  {
                    x1: '28.1542969%',
                    y1: '63.7402344%',
                    x2: '74.6289062%',
                    y2: '17.7832031%',
                    id: 'linearGradient-1',
                  },
                  /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('stop', {
                    stopColor: 'rgba(164, 164, 164, 1)',
                    offset: '0%',
                  }),
                  /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('stop', {
                    stopColor: 'rgba(164, 164, 164, 0)',
                    stopOpacity: '0',
                    offset: '100%',
                  })
                )
              ),
              /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                'g',
                {
                  id: 'Page-1',
                  stroke: 'none',
                  strokeWidth: '1',
                  fill: 'none',
                },
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  'g',
                  {
                    transform: 'translate(-236.000000, -286.000000)',
                  },
                  /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                    'g',
                    {
                      transform: 'translate(238.000000, 286.000000)',
                    },
                    /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('circle', {
                      id: 'Oval-2',
                      stroke: 'url(#linearGradient-1)',
                      strokeWidth: '4',
                      cx: '10',
                      cy: '12',
                      r: '10',
                    }),
                    /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('path', {
                      d: 'M10,2 C4.4771525,2 0,6.4771525 0,12',
                      id: 'Oval-2',
                      stroke: 'rgba(164, 164, 164, 1)',
                      strokeWidth: '4',
                    }),
                    /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('rect', {
                      id: 'Rectangle-1',
                      fill: 'rgba(164, 164, 164, 1)',
                      x: '8',
                      y: '0',
                      width: '4',
                      height: '4',
                      rx: '8',
                    })
                  )
                )
              )
            )
          }

          // src/components/search-dialog.tsx
          var SearchDialog = class extends react__WEBPACK_IMPORTED_MODULE_0__.Component {
            constructor(props) {
              super(props)
              this.state = {
                isLoading: false,
                query: '',
                searchResult: null,
                searchError: null,
              }
              this._onAfterOpen = () => {
                if (this._inputRef.current) {
                  this._inputRef.current.focus()
                }
              }
              this._onChangeQuery = (e) => {
                const query = e.target.value
                this.setState({ query })
                if (!query.trim()) {
                  this.setState({ isLoading: false, searchResult: null, searchError: null })
                  return
                } else {
                  this._search()
                }
              }
              this._onClearQuery = () => {
                this._onChangeQuery({ target: { value: '' } })
              }
              this._warmupSearch = () =>
                __async(this, null, function* () {
                  const { searchNotion, rootBlockId } = this.props
                  yield searchNotion({
                    query: '',
                    ancestorId: rootBlockId,
                  })
                })
              this._searchImpl = () =>
                __async(this, null, function* () {
                  const { searchNotion, rootBlockId } = this.props
                  const { query } = this.state
                  if (!query.trim()) {
                    this.setState({ isLoading: false, searchResult: null, searchError: null })
                    return
                  }
                  this.setState({ isLoading: true })
                  const result = yield searchNotion({
                    query,
                    ancestorId: rootBlockId,
                  })
                  console.log('search', query, result)
                  let searchResult = null
                  let searchError = null
                  if (result.error || result.errorId) {
                    searchError = result
                  } else {
                    searchResult = __spreadValues({}, result)
                    const results = searchResult.results
                      .map((result2) => {
                        var _a, _b
                        const block =
                          (_a = searchResult.recordMap.block[result2.id]) == null
                            ? void 0
                            : _a.value
                        if (!block) return
                        const title = (0, notion_utils__WEBPACK_IMPORTED_MODULE_1__.getBlockTitle)(
                          block,
                          searchResult.recordMap
                        )
                        if (!title) {
                          return
                        }
                        result2.title = title
                        result2.block = block
                        result2.recordMap = searchResult.recordMap
                        result2.page =
                          (0, notion_utils__WEBPACK_IMPORTED_MODULE_1__.getBlockParentPage)(
                            block,
                            searchResult.recordMap,
                            {
                              inclusive: true,
                            }
                          ) || block
                        if (!result2.page.id) {
                          return
                        }
                        if ((_b = result2.highlight) == null ? void 0 : _b.text) {
                          result2.highlight.html = result2.highlight.text
                            .replace(/<gzkNfoUU>/gi, '<b>')
                            .replace(/<\/gzkNfoUU>/gi, '</b>')
                        }
                        return result2
                      })
                      .filter(Boolean)
                    const searchResultsMap = results.reduce(
                      (map, result2) =>
                        __spreadProps(__spreadValues({}, map), {
                          [result2.page.id]: result2,
                        }),
                      {}
                    )
                    searchResult.results = Object.values(searchResultsMap)
                  }
                  if (this.state.query === query) {
                    this.setState({ isLoading: false, searchResult, searchError })
                  }
                })
              this._inputRef = react__WEBPACK_IMPORTED_MODULE_0__.createRef()
            }
            componentDidMount() {
              this._search = (0, import_lodash.default)(this._searchImpl.bind(this), 1e3)
              this._warmupSearch()
            }
            render() {
              const { isOpen, onClose } = this.props
              const { isLoading, query, searchResult, searchError } = this.state
              const hasQuery = !!query.trim()
              return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                NotionContextConsumer,
                null,
                (ctx2) => {
                  const { components, defaultPageIcon, mapPageUrl } = ctx2
                  return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                    components.Modal,
                    {
                      isOpen,
                      contentLabel: 'Search',
                      className: 'notion-search',
                      overlayClassName: 'notion-search-overlay',
                      onRequestClose: onClose,
                      onAfterOpen: this._onAfterOpen,
                    },
                    /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                      'div',
                      {
                        className: 'quickFindMenu',
                      },
                      /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                        'div',
                        {
                          className: 'searchBar',
                        },
                        /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                          'div',
                          {
                            className: 'inlineIcon',
                          },
                          isLoading
                            ? /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                                LoadingIcon,
                                {
                                  className: 'loadingIcon',
                                }
                              )
                            : /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                                SearchIcon,
                                null
                              )
                        ),
                        /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('input', {
                          className: 'searchInput',
                          placeholder: 'Search',
                          value: query,
                          ref: this._inputRef,
                          onChange: this._onChangeQuery,
                        }),
                        query &&
                          /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                            'div',
                            {
                              role: 'button',
                              className: 'clearButton',
                              onClick: this._onClearQuery,
                            },
                            /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                              ClearIcon,
                              {
                                className: 'clearIcon',
                              }
                            )
                          )
                      ),
                      hasQuery &&
                        searchResult &&
                        /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                          react__WEBPACK_IMPORTED_MODULE_0__.Fragment,
                          null,
                          searchResult.results.length
                            ? /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                                NotionContextProvider,
                                __spreadProps(__spreadValues({}, ctx2), {
                                  recordMap: searchResult.recordMap,
                                }),
                                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                                  'div',
                                  {
                                    className: 'resultsPane',
                                  },
                                  searchResult.results.map((result) => {
                                    var _a
                                    return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                                      components.PageLink,
                                      {
                                        key: result.id,
                                        className: cs('result', 'notion-page-link'),
                                        href: mapPageUrl(result.page.id, searchResult.recordMap),
                                      },
                                      /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                                        PageTitle,
                                        {
                                          block: result.page,
                                          defaultIcon: defaultPageIcon,
                                        }
                                      ),
                                      ((_a = result.highlight) == null ? void 0 : _a.html) &&
                                        /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                                          'div',
                                          {
                                            className: 'notion-search-result-highlight',
                                            dangerouslySetInnerHTML: {
                                              __html: result.highlight.html,
                                            },
                                          }
                                        )
                                    )
                                  })
                                ),
                                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                                  'footer',
                                  {
                                    className: 'resultsFooter',
                                  },
                                  /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                                    'div',
                                    null,
                                    /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                                      'span',
                                      {
                                        className: 'resultsCount',
                                      },
                                      searchResult.total
                                    ),
                                    searchResult.total === 1 ? ' result' : ' results'
                                  )
                                )
                              )
                            : /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                                'div',
                                {
                                  className: 'noResultsPane',
                                },
                                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                                  'div',
                                  {
                                    className: 'noResults',
                                  },
                                  'No results'
                                ),
                                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                                  'div',
                                  {
                                    className: 'noResultsDetail',
                                  },
                                  'Try different search terms'
                                )
                              )
                        ),
                      hasQuery &&
                        !searchResult &&
                        searchError &&
                        /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                          'div',
                          {
                            className: 'noResultsPane',
                          },
                          /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                            'div',
                            {
                              className: 'noResults',
                            },
                            'Search error'
                          )
                        )
                    )
                  )
                }
              )
            }
          }

          // src/components/header.tsx
          var Header = ({ block }) => {
            return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              'header',
              {
                className: 'notion-header',
              },
              /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                'div',
                {
                  className: 'notion-nav-header',
                },
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(Breadcrumbs, {
                  block,
                }),
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(Search, {
                  block,
                })
              )
            )
          }
          var Breadcrumbs = ({ block, rootOnly = false }) => {
            const { recordMap, mapPageUrl, components } = useNotionContext()
            const breadcrumbs = react__WEBPACK_IMPORTED_MODULE_0__.useMemo(() => {
              const breadcrumbs2 = (0,
              notion_utils__WEBPACK_IMPORTED_MODULE_1__.getPageBreadcrumbs)(recordMap, block.id)
              if (rootOnly) {
                return [breadcrumbs2[0]].filter(Boolean)
              }
              return breadcrumbs2
            }, [recordMap, block.id, rootOnly])
            return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              'div',
              {
                className: 'breadcrumbs',
                key: 'breadcrumbs',
              },
              breadcrumbs.map((breadcrumb, index) => {
                if (!breadcrumb) {
                  return null
                }
                const pageLinkProps = {}
                const componentMap = {
                  pageLink: components.PageLink,
                }
                if (breadcrumb.active) {
                  componentMap.pageLink = (props) =>
                    /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                      'div',
                      __spreadValues({}, props)
                    )
                } else {
                  pageLinkProps.href = mapPageUrl(breadcrumb.pageId)
                }
                return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  react__WEBPACK_IMPORTED_MODULE_0__.Fragment,
                  {
                    key: breadcrumb.pageId,
                  },
                  /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                    componentMap.pageLink,
                    __spreadValues(
                      {
                        className: cs('breadcrumb', breadcrumb.active && 'active'),
                      },
                      pageLinkProps
                    ),
                    breadcrumb.icon &&
                      /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(PageIcon, {
                        className: 'icon',
                        block: breadcrumb.block,
                      }),
                    breadcrumb.title &&
                      /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                        'span',
                        {
                          className: 'title',
                        },
                        breadcrumb.title
                      )
                  ),
                  index < breadcrumbs.length - 1 &&
                    /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                      'span',
                      {
                        className: 'spacer',
                      },
                      '/'
                    )
                )
              })
            )
          }
          var Search = ({ block, search, title = 'Search' }) => {
            const { searchNotion, rootPageId, isShowingSearch, onHideSearch } = useNotionContext()
            const onSearchNotion = search || searchNotion
            const [isSearchOpen, setIsSearchOpen] =
              react__WEBPACK_IMPORTED_MODULE_0__.useState(isShowingSearch)
            react__WEBPACK_IMPORTED_MODULE_0__.useEffect(() => {
              setIsSearchOpen(isShowingSearch)
            }, [isShowingSearch])
            const onOpenSearch = react__WEBPACK_IMPORTED_MODULE_0__.useCallback(() => {
              setIsSearchOpen(true)
            }, [])
            const onCloseSearch = react__WEBPACK_IMPORTED_MODULE_0__.useCallback(() => {
              setIsSearchOpen(false)
              if (onHideSearch) {
                onHideSearch()
              }
            }, [onHideSearch])
            ;(0, react_hotkeys_hook__WEBPACK_IMPORTED_MODULE_14__.useHotkeys)('cmd+p', (event) => {
              onOpenSearch()
              event.preventDefault()
              event.stopPropagation()
            })
            ;(0, react_hotkeys_hook__WEBPACK_IMPORTED_MODULE_14__.useHotkeys)('cmd+k', (event) => {
              onOpenSearch()
              event.preventDefault()
              event.stopPropagation()
            })
            const hasSearch = !!onSearchNotion
            return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              react__WEBPACK_IMPORTED_MODULE_0__.Fragment,
              null,
              hasSearch &&
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  'div',
                  {
                    role: 'button',
                    className: cs('breadcrumb', 'button', 'notion-search-button'),
                    onClick: onOpenSearch,
                  },
                  /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(SearchIcon, {
                    className: 'searchIcon',
                  }),
                  title &&
                    /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                      'span',
                      {
                        className: 'title',
                      },
                      title
                    )
                ),
              isSearchOpen &&
                hasSearch &&
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(SearchDialog, {
                  isOpen: isSearchOpen,
                  rootBlockId: rootPageId || (block == null ? void 0 : block.id),
                  onClose: onCloseSearch,
                  searchNotion: onSearchNotion,
                })
            )
          }

          // src/components/asset.tsx

          // src/components/lite-youtube-embed.tsx

          var qs = (params) => {
            return Object.keys(params)
              .map((key) => `${encodeURIComponent(key)}=${encodeURIComponent(params[key])}`)
              .join('&')
          }
          var LiteYouTubeEmbed = ({
            id,
            defaultPlay = false,
            mute = false,
            lazyImage = false,
            iframeTitle = 'YouTube video',
            alt = 'Video preview',
            params = {},
            adLinksPreconnect = true,
            style,
            className,
          }) => {
            const muteParam = mute || defaultPlay ? '1' : '0'
            const queryString = react__WEBPACK_IMPORTED_MODULE_0__.useMemo(
              () => qs(__spreadValues({ autoplay: '1', mute: muteParam }, params)),
              [muteParam, params]
            )
            const resolution = 'hqdefault'
            const posterUrl = `https://i.ytimg.com/vi/${id}/${resolution}.jpg`
            const ytUrl = 'https://www.youtube-nocookie.com'
            const iframeSrc = `${ytUrl}/embed/${id}?${queryString}`
            const [isPreconnected, setIsPreconnected] =
              react__WEBPACK_IMPORTED_MODULE_0__.useState(false)
            const [iframeInitialized, setIframeInitialized] =
              react__WEBPACK_IMPORTED_MODULE_0__.useState(defaultPlay)
            const [isIframeLoaded, setIsIframeLoaded] =
              react__WEBPACK_IMPORTED_MODULE_0__.useState(false)
            const warmConnections = react__WEBPACK_IMPORTED_MODULE_0__.useCallback(() => {
              if (isPreconnected) return
              setIsPreconnected(true)
            }, [isPreconnected])
            const onLoadIframe = react__WEBPACK_IMPORTED_MODULE_0__.useCallback(() => {
              if (iframeInitialized) return
              setIframeInitialized(true)
            }, [iframeInitialized])
            const onIframeLoaded = react__WEBPACK_IMPORTED_MODULE_0__.useCallback(() => {
              setIsIframeLoaded(true)
            }, [])
            return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              react__WEBPACK_IMPORTED_MODULE_0__.Fragment,
              null,
              /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('link', {
                rel: 'preload',
                href: posterUrl,
                as: 'image',
              }),
              isPreconnected &&
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  react__WEBPACK_IMPORTED_MODULE_0__.Fragment,
                  null,
                  /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('link', {
                    rel: 'preconnect',
                    href: ytUrl,
                  }),
                  /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('link', {
                    rel: 'preconnect',
                    href: 'https://www.google.com',
                  })
                ),
              isPreconnected &&
                adLinksPreconnect &&
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  react__WEBPACK_IMPORTED_MODULE_0__.Fragment,
                  null,
                  /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('link', {
                    rel: 'preconnect',
                    href: 'https://static.doubleclick.net',
                  }),
                  /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('link', {
                    rel: 'preconnect',
                    href: 'https://googleads.g.doubleclick.net',
                  })
                ),
              /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                'div',
                {
                  onClick: onLoadIframe,
                  onPointerOver: warmConnections,
                  className: cs(
                    'notion-yt-lite',
                    isIframeLoaded && 'notion-yt-loaded',
                    iframeInitialized && 'notion-yt-initialized',
                    className
                  ),
                  style,
                },
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('img', {
                  src: posterUrl,
                  className: 'notion-yt-thumbnail',
                  loading: lazyImage ? 'lazy' : void 0,
                  alt,
                }),
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('div', {
                  className: 'notion-yt-playbtn',
                }),
                iframeInitialized &&
                  /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('iframe', {
                    width: '560',
                    height: '315',
                    frameBorder: '0',
                    allow:
                      'accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture',
                    allowFullScreen: true,
                    title: iframeTitle,
                    src: iframeSrc,
                    onLoad: onIframeLoaded,
                  })
              )
            )
          }

          // src/components/asset.tsx
          var isServer = typeof window === 'undefined'
          var supportedAssetTypes = [
            'video',
            'image',
            'embed',
            'figma',
            'typeform',
            'excalidraw',
            'maps',
            'tweet',
            'pdf',
            'gist',
            'codepen',
            'drive',
          ]
          var Asset = ({ block, zoomable = true, children }) => {
            var _a, _b, _c, _d, _e, _f, _g, _h, _i, _j
            const { recordMap, mapImageUrl, components } = useNotionContext()
            if (!block || !supportedAssetTypes.includes(block.type)) {
              return null
            }
            const style = {
              position: 'relative',
              display: 'flex',
              justifyContent: 'center',
              alignSelf: 'center',
              width: '100%',
              maxWidth: '100%',
              flexDirection: 'column',
            }
            const assetStyle = {}
            if (block.format) {
              const {
                block_aspect_ratio,
                block_height,
                block_width,
                block_full_width,
                block_page_width,
                block_preserve_scale,
              } = block.format
              if (block_full_width || block_page_width) {
                if (block_full_width) {
                  style.width = '100vw'
                } else {
                  style.width = '100%'
                }
                if (block.type === 'video') {
                  if (block_height) {
                    style.height = block_height
                  } else if (block_aspect_ratio) {
                    style.paddingBottom = `${block_aspect_ratio * 100}%`
                  } else if (block_preserve_scale) {
                    style.objectFit = 'contain'
                  }
                } else if (block_aspect_ratio && block.type !== 'image') {
                  style.paddingBottom = `${block_aspect_ratio * 100}%`
                } else if (block_height) {
                  style.height = block_height
                } else if (block_preserve_scale) {
                  if (block.type === 'image') {
                    style.height = '100%'
                  } else {
                    style.paddingBottom = '75%'
                    style.minHeight = 100
                  }
                }
              } else {
                switch ((_a = block.format) == null ? void 0 : _a.block_alignment) {
                  case 'center': {
                    style.alignSelf = 'center'
                    break
                  }
                  case 'left': {
                    style.alignSelf = 'start'
                    break
                  }
                  case 'right': {
                    style.alignSelf = 'end'
                    break
                  }
                }
                if (block_width) {
                  style.width = block_width
                }
                if (block_preserve_scale && block.type !== 'image') {
                  style.paddingBottom = '50%'
                  style.minHeight = 100
                } else {
                  if (block_height && block.type !== 'image') {
                    style.height = block_height
                  }
                }
              }
              if (block.type === 'image') {
                assetStyle.objectFit = 'cover'
              } else if (block_preserve_scale) {
                assetStyle.objectFit = 'contain'
              }
            }
            let source =
              ((_b = recordMap.signed_urls) == null ? void 0 : _b[block.id]) ||
              ((_e =
                (_d = (_c = block.properties) == null ? void 0 : _c.source) == null
                  ? void 0
                  : _d[0]) == null
                ? void 0
                : _e[0])
            let content = null
            if (!source) {
              return null
            }
            if (block.type === 'tweet') {
              const src = source
              if (!src) return null
              const id = src.split('?')[0].split('/').pop()
              if (!id) return null
              content = /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                'div',
                {
                  style: __spreadProps(__spreadValues({}, assetStyle), {
                    maxWidth: 420,
                    width: '100%',
                    marginLeft: 'auto',
                    marginRight: 'auto',
                  }),
                },
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(components.Tweet, {
                  id,
                })
              )
            } else if (block.type === 'pdf') {
              style.overflow = 'auto'
              style.background = 'rgb(226, 226, 226)'
              style.display = 'block'
              if (!style.padding) {
                style.padding = '8px 16px'
              }
              if (!isServer) {
                content = /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  components.Pdf,
                  {
                    file: source,
                  }
                )
              }
            } else if (
              block.type === 'embed' ||
              block.type === 'video' ||
              block.type === 'figma' ||
              block.type === 'typeform' ||
              block.type === 'gist' ||
              block.type === 'maps' ||
              block.type === 'excalidraw' ||
              block.type === 'codepen' ||
              block.type === 'drive'
            ) {
              if (
                block.type === 'video' &&
                source &&
                source.indexOf('youtube') < 0 &&
                source.indexOf('youtu.be') < 0 &&
                source.indexOf('vimeo') < 0 &&
                source.indexOf('wistia') < 0 &&
                source.indexOf('loom') < 0 &&
                source.indexOf('videoask') < 0 &&
                source.indexOf('getcloudapp') < 0
              ) {
                style.paddingBottom = void 0
                content = /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  'video',
                  {
                    playsInline: true,
                    controls: true,
                    preload: 'metadata',
                    style: assetStyle,
                    src: source,
                    title: block.type,
                  }
                )
              } else {
                let src = ((_f = block.format) == null ? void 0 : _f.display_source) || source
                if (src) {
                  const youtubeVideoId = block.type === 'video' ? getYoutubeId(src) : null
                  if (youtubeVideoId) {
                    content = /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                      LiteYouTubeEmbed,
                      {
                        id: youtubeVideoId,
                        style: assetStyle,
                        className: 'notion-asset-object-fit',
                      }
                    )
                  } else if (block.type === 'gist') {
                    if (!src.endsWith('.pibb')) {
                      src = `${src}.pibb`
                    }
                    assetStyle.width = '100%'
                    style.paddingBottom = '50%'
                    content = /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                      'iframe',
                      {
                        style: assetStyle,
                        className: 'notion-asset-object-fit',
                        src,
                        title: 'GitHub Gist',
                        frameBorder: '0',
                        loading: 'lazy',
                        scrolling: 'auto',
                      }
                    )
                  } else {
                    content = /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                      'iframe',
                      {
                        className: 'notion-asset-object-fit',
                        style: assetStyle,
                        src,
                        title: `iframe ${block.type}`,
                        frameBorder: '0',
                        allowFullScreen: true,
                        loading: 'lazy',
                        scrolling: 'auto',
                      }
                    )
                  }
                }
              }
            } else if (block.type === 'image') {
              if (source.includes('file.notion.so')) {
                source =
                  (_i =
                    (_h = (_g = block.properties) == null ? void 0 : _g.source) == null
                      ? void 0
                      : _h[0]) == null
                    ? void 0
                    : _i[0]
              }
              const src = mapImageUrl(source, block)
              const caption = (0, notion_utils__WEBPACK_IMPORTED_MODULE_1__.getTextContent)(
                (_j = block.properties) == null ? void 0 : _j.caption
              )
              const alt = caption || 'notion image'
              content = /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                LazyImage,
                {
                  src,
                  alt,
                  zoomable,
                  height: style.height,
                  style: assetStyle,
                }
              )
            }
            return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              react__WEBPACK_IMPORTED_MODULE_0__.Fragment,
              null,
              /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                'div',
                {
                  style,
                },
                content,
                block.type === 'image' && children
              ),
              block.type !== 'image' && children
            )
          }

          // src/components/asset-wrapper.tsx
          var urlStyle = { width: '100%' }
          var AssetWrapper = ({ blockId, block }) => {
            var _a, _b, _c, _d, _e, _f
            const value = block
            const { components, mapPageUrl, rootDomain, zoom } = useNotionContext()
            let isURL = false
            if (block.type === 'image') {
              const caption =
                (_c =
                  (_b =
                    (_a = value == null ? void 0 : value.properties) == null
                      ? void 0
                      : _a.caption) == null
                    ? void 0
                    : _b[0]) == null
                  ? void 0
                  : _c[0]
              if (caption) {
                const id = (0, notion_utils__WEBPACK_IMPORTED_MODULE_1__.parsePageId)(caption, {
                  uuid: true,
                })
                const isPage = caption.charAt(0) === '/' && id
                if (isPage || isValidURL(caption)) {
                  isURL = true
                }
              }
            }
            const figure = /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              'figure',
              {
                className: cs(
                  'notion-asset-wrapper',
                  `notion-asset-wrapper-${block.type}`,
                  ((_d = value.format) == null ? void 0 : _d.block_full_width) &&
                    'notion-asset-wrapper-full',
                  blockId
                ),
              },
              /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                Asset,
                {
                  block: value,
                  zoomable: zoom && !isURL,
                },
                ((_e = value == null ? void 0 : value.properties) == null ? void 0 : _e.caption) &&
                  !isURL &&
                  /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                    'figcaption',
                    {
                      className: 'notion-asset-caption',
                    },
                    /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(Text, {
                      value: value.properties.caption,
                      block,
                    })
                  )
              )
            )
            if (isURL) {
              const caption =
                (_f = value == null ? void 0 : value.properties) == null ? void 0 : _f.caption[0][0]
              const id = (0, notion_utils__WEBPACK_IMPORTED_MODULE_1__.parsePageId)(caption, {
                uuid: true,
              })
              const isPage = caption.charAt(0) === '/' && id
              const captionHostname = extractHostname(caption)
              return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                components.PageLink,
                {
                  style: urlStyle,
                  href: isPage ? mapPageUrl(id) : caption,
                  target:
                    captionHostname && captionHostname !== rootDomain && !caption.startsWith('/')
                      ? 'blank_'
                      : null,
                },
                figure
              )
            }
            return figure
          }
          function isValidURL(str) {
            const pattern = new RegExp(
              '^(https?:\\/\\/)?((([a-z\\d]([a-z\\d-]*[a-z\\d])*)\\.)+[a-z]{2,}|((\\d{1,3}\\.){3}\\d{1,3}))(\\:\\d+)?(\\/[-a-z\\d%_.~+]*)*(\\?[;&a-z\\d%_.~+=-]*)?(\\#[-a-z\\d_]*)?$',
              'i'
            )
            return !!pattern.test(str)
          }
          function extractHostname(url) {
            try {
              const hostname = new URL(url).hostname
              return hostname
            } catch (err) {
              return ''
            }
          }

          // src/components/checkbox.tsx

          // src/icons/check.tsx

          function SvgCheck(props) {
            return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              'svg',
              __spreadValues(
                {
                  viewBox: '0 0 14 14',
                },
                props
              ),
              /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('path', {
                d: 'M5.5 12L14 3.5 12.5 2l-7 7-4-4.003L0 6.499z',
              })
            )
          }
          var check_default = SvgCheck

          // src/components/checkbox.tsx
          var Checkbox = ({ isChecked }) => {
            let content = null
            if (isChecked) {
              content = /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                'div',
                {
                  className: 'notion-property-checkbox-checked',
                },
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  check_default,
                  null
                )
              )
            } else {
              content = /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('div', {
                className: 'notion-property-checkbox-unchecked',
              })
            }
            return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              'span',
              {
                className: 'notion-property notion-property-checkbox',
              },
              content
            )
          }

          // src/next.tsx

          var wrapNextImage = (NextImage) => {
            return react__WEBPACK_IMPORTED_MODULE_0__.memo(function ReactNotionXNextImage(_a) {
              var _b = _a,
                { src, alt, width, height, className, style, layout } = _b,
                rest = __objRest(_b, [
                  'src',
                  'alt',
                  'width',
                  'height',
                  'className',
                  'style',
                  'layout',
                ])
              if (!layout) {
                layout = width && height ? 'intrinsic' : 'fill'
              }
              return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                NextImage,
                __spreadValues(
                  {
                    className,
                    src,
                    alt,
                    width: layout === 'intrinsic' && width,
                    height: layout === 'intrinsic' && height,
                    objectFit: style == null ? void 0 : style.objectFit,
                    objectPosition: style == null ? void 0 : style.objectPosition,
                    layout,
                  },
                  rest
                )
              )
            }, react_fast_compare__WEBPACK_IMPORTED_MODULE_15__)
          }
          var wrapNextLink = (NextLink) =>
            function ReactNotionXNextLink(_a) {
              var _b = _a,
                { href, as, passHref, prefetch, replace, scroll, shallow, locale } = _b,
                linkProps = __objRest(_b, [
                  'href',
                  'as',
                  'passHref',
                  'prefetch',
                  'replace',
                  'scroll',
                  'shallow',
                  'locale',
                ])
              return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                NextLink,
                {
                  href,
                  as,
                  passHref,
                  prefetch,
                  replace,
                  scroll,
                  shallow,
                  locale,
                },
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  'a',
                  __spreadValues({}, linkProps)
                )
              )
            }

          // src/context.tsx
          var DefaultLink = (props) =>
            /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              'a',
              __spreadValues(
                {
                  target: '_blank',
                  rel: 'noopener noreferrer',
                },
                props
              )
            )
          var DefaultLinkMemo = react__WEBPACK_IMPORTED_MODULE_0__.memo(DefaultLink)
          var DefaultPageLink = (props) =>
            /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              'a',
              __spreadValues({}, props)
            )
          var DefaultPageLinkMemo = react__WEBPACK_IMPORTED_MODULE_0__.memo(DefaultPageLink)
          var DefaultEmbed = (props) =>
            /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              AssetWrapper,
              __spreadValues({}, props)
            )
          var DefaultHeader = Header
          var dummyComponent = (name) => () => {
            console.warn(
              `Warning: using empty component "${name}" (you should override this in NotionRenderer.components)`
            )
            return null
          }
          var dummyOverrideFn = (_, defaultValueFn) => defaultValueFn()
          var defaultComponents = {
            Image: null,
            Link: DefaultLinkMemo,
            PageLink: DefaultPageLinkMemo,
            Checkbox,
            Callout: void 0,
            Code: dummyComponent('Code'),
            Equation: dummyComponent('Equation'),
            Collection: dummyComponent('Collection'),
            Property: void 0,
            propertyTextValue: dummyOverrideFn,
            propertySelectValue: dummyOverrideFn,
            propertyRelationValue: dummyOverrideFn,
            propertyFormulaValue: dummyOverrideFn,
            propertyTitleValue: dummyOverrideFn,
            propertyPersonValue: dummyOverrideFn,
            propertyFileValue: dummyOverrideFn,
            propertyCheckboxValue: dummyOverrideFn,
            propertyUrlValue: dummyOverrideFn,
            propertyEmailValue: dummyOverrideFn,
            propertyPhoneNumberValue: dummyOverrideFn,
            propertyNumberValue: dummyOverrideFn,
            propertyLastEditedTimeValue: dummyOverrideFn,
            propertyCreatedTimeValue: dummyOverrideFn,
            propertyDateValue: dummyOverrideFn,
            Pdf: dummyComponent('Pdf'),
            Tweet: dummyComponent('Tweet'),
            Modal: dummyComponent('Modal'),
            Header: DefaultHeader,
            Embed: DefaultEmbed,
          }
          var defaultNotionContext = {
            recordMap: {
              block: {},
              collection: {},
              collection_view: {},
              collection_query: {},
              notion_user: {},
              signed_urls: {},
            },
            components: defaultComponents,
            mapPageUrl: defaultMapPageUrl(),
            mapImageUrl: defaultMapImageUrl,
            searchNotion: null,
            isShowingSearch: false,
            onHideSearch: null,
            fullPage: false,
            darkMode: false,
            previewImages: false,
            forceCustomImages: false,
            showCollectionViewDropdown: true,
            linkTableTitleProperties: true,
            isLinkCollectionToUrlProperty: false,
            showTableOfContents: false,
            minTableOfContentsItems: 3,
            defaultPageIcon: null,
            defaultPageCover: null,
            defaultPageCoverPosition: 0.5,
            zoom: null,
          }
          var ctx = react__WEBPACK_IMPORTED_MODULE_0__.createContext(defaultNotionContext)
          var NotionContextProvider = (_a) => {
            var _b = _a,
              {
                components: themeComponents = {},
                children,
                mapPageUrl,
                mapImageUrl,
                rootPageId,
              } = _b,
              rest = __objRest(_b, [
                'components',
                'children',
                'mapPageUrl',
                'mapImageUrl',
                'rootPageId',
              ])
            for (const key of Object.keys(rest)) {
              if (rest[key] === void 0) {
                delete rest[key]
              }
            }
            const wrappedThemeComponents = react__WEBPACK_IMPORTED_MODULE_0__.useMemo(
              () => __spreadValues({}, themeComponents),
              [themeComponents]
            )
            if (wrappedThemeComponents.nextImage) {
              wrappedThemeComponents.Image = wrapNextImage(themeComponents.nextImage)
            }
            if (wrappedThemeComponents.nextLink) {
              wrappedThemeComponents.nextLink = wrapNextLink(themeComponents.nextLink)
            }
            for (const key of Object.keys(wrappedThemeComponents)) {
              if (!wrappedThemeComponents[key]) {
                delete wrappedThemeComponents[key]
              }
            }
            const value = react__WEBPACK_IMPORTED_MODULE_0__.useMemo(
              () =>
                __spreadProps(__spreadValues(__spreadValues({}, defaultNotionContext), rest), {
                  rootPageId,
                  mapPageUrl: mapPageUrl != null ? mapPageUrl : defaultMapPageUrl(rootPageId),
                  mapImageUrl: mapImageUrl != null ? mapImageUrl : defaultMapImageUrl,
                  components: __spreadValues(
                    __spreadValues({}, defaultComponents),
                    wrappedThemeComponents
                  ),
                }),
              [mapImageUrl, mapPageUrl, wrappedThemeComponents, rootPageId, rest]
            )
            return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              ctx.Provider,
              {
                value,
              },
              children
            )
          }
          var NotionContextConsumer = ctx.Consumer
          var useNotionContext = () => {
            return react__WEBPACK_IMPORTED_MODULE_0__.useContext(ctx)
          }

          // src/components/text.tsx
          var Text = ({ value, block, linkProps, linkProtocol }) => {
            const { components, recordMap, mapPageUrl, mapImageUrl, rootDomain } =
              useNotionContext()
            return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              react__WEBPACK_IMPORTED_MODULE_0__.Fragment,
              null,
              value == null
                ? void 0
                : value.map(([text, decorations], index) => {
                    if (!decorations) {
                      if (text === ',') {
                        return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                          'span',
                          {
                            key: index,
                            style: { padding: '0.5em' },
                          }
                        )
                      } else {
                        return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                          react__WEBPACK_IMPORTED_MODULE_0__.Fragment,
                          {
                            key: index,
                          },
                          text
                        )
                      }
                    }
                    const formatted = decorations.reduce((element, decorator) => {
                      var _a, _b, _c, _d, _e
                      switch (decorator[0]) {
                        case 'p': {
                          const blockId = decorator[1]
                          const linkedBlock =
                            (_a = recordMap.block[blockId]) == null ? void 0 : _a.value
                          if (!linkedBlock) {
                            console.log('"p" missing block', blockId)
                            return null
                          }
                          return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                            components.PageLink,
                            {
                              className: 'notion-link',
                              href: mapPageUrl(blockId),
                            },
                            /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                              PageTitle,
                              {
                                block: linkedBlock,
                              }
                            )
                          )
                        }
                        case '\u2023': {
                          const linkType = decorator[1][0]
                          const id = decorator[1][1]
                          switch (linkType) {
                            case 'u': {
                              const user =
                                (_b = recordMap.notion_user[id]) == null ? void 0 : _b.value
                              if (!user) {
                                console.log('"\u2023" missing user', id)
                                return null
                              }
                              const name = [user.given_name, user.family_name]
                                .filter(Boolean)
                                .join(' ')
                              return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                                GracefulImage,
                                {
                                  className: 'notion-user',
                                  src: mapImageUrl(user.profile_photo, block),
                                  alt: name,
                                }
                              )
                            }
                            default: {
                              const linkedBlock =
                                (_c = recordMap.block[id]) == null ? void 0 : _c.value
                              if (!linkedBlock) {
                                console.log('"\u2023" missing block', linkType, id)
                                return null
                              }
                              return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                                components.PageLink,
                                __spreadProps(
                                  __spreadValues(
                                    {
                                      className: 'notion-link',
                                      href: mapPageUrl(id),
                                    },
                                    linkProps
                                  ),
                                  {
                                    target: '_blank',
                                    rel: 'noopener noreferrer',
                                  }
                                ),
                                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                                  PageTitle,
                                  {
                                    block: linkedBlock,
                                  }
                                )
                              )
                            }
                          }
                        }
                        case 'h':
                          return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                            'span',
                            {
                              className: `notion-${decorator[1]}`,
                            },
                            element
                          )
                        case 'c':
                          return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                            'code',
                            {
                              className: 'notion-inline-code',
                            },
                            element
                          )
                        case 'b':
                          return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                            'b',
                            null,
                            element
                          )
                        case 'i':
                          return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                            'em',
                            null,
                            element
                          )
                        case 's':
                          return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                            's',
                            null,
                            element
                          )
                        case '_':
                          return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                            'span',
                            {
                              className: 'notion-inline-underscore',
                            },
                            element
                          )
                        case 'e':
                          return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                            components.Equation,
                            {
                              math: decorator[1],
                              inline: true,
                            }
                          )
                        case 'm':
                          return element
                        case 'a': {
                          const v = decorator[1]
                          const pathname = v.substr(1)
                          const id = (0, notion_utils__WEBPACK_IMPORTED_MODULE_1__.parsePageId)(
                            pathname,
                            { uuid: true }
                          )
                          if ((v[0] === '/' || v.includes(rootDomain)) && id) {
                            const href = v.includes(rootDomain)
                              ? v
                              : `${mapPageUrl(id)}${getHashFragmentValue(v)}`
                            return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                              components.PageLink,
                              __spreadValues(
                                {
                                  className: 'notion-link',
                                  href,
                                },
                                linkProps
                              ),
                              element
                            )
                          } else {
                            return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                              components.Link,
                              __spreadValues(
                                {
                                  className: 'notion-link',
                                  href: linkProtocol
                                    ? `${linkProtocol}:${decorator[1]}`
                                    : decorator[1],
                                },
                                linkProps
                              ),
                              element
                            )
                          }
                        }
                        case 'd': {
                          const v = decorator[1]
                          const type = v == null ? void 0 : v.type
                          if (type === 'date') {
                            const startDate = v.start_date
                            return (0, notion_utils__WEBPACK_IMPORTED_MODULE_1__.formatDate)(
                              startDate
                            )
                          } else if (type === 'daterange') {
                            const startDate = v.start_date
                            const endDate = v.end_date
                            return `${(0, notion_utils__WEBPACK_IMPORTED_MODULE_1__.formatDate)(
                              startDate
                            )} \u2192 ${(0, notion_utils__WEBPACK_IMPORTED_MODULE_1__.formatDate)(
                              endDate
                            )}`
                          } else {
                            return element
                          }
                        }
                        case 'u': {
                          const userId = decorator[1]
                          const user =
                            (_d = recordMap.notion_user[userId]) == null ? void 0 : _d.value
                          if (!user) {
                            console.log('missing user', userId)
                            return null
                          }
                          const name = [user.given_name, user.family_name].filter(Boolean).join(' ')
                          return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                            GracefulImage,
                            {
                              className: 'notion-user',
                              src: mapImageUrl(user.profile_photo, block),
                              alt: name,
                            }
                          )
                        }
                        case 'eoi': {
                          const blockId = decorator[1]
                          const externalObjectInstance =
                            (_e = recordMap.block[blockId]) == null ? void 0 : _e.value
                          return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                            EOI,
                            {
                              block: externalObjectInstance,
                              inline: true,
                            }
                          )
                        }
                        default:
                          if (true) {
                            console.log('unsupported text format', decorator)
                          }
                          return element
                      }
                    }, /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(react__WEBPACK_IMPORTED_MODULE_0__.Fragment, null, text))
                    return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                      react__WEBPACK_IMPORTED_MODULE_0__.Fragment,
                      {
                        key: index,
                      },
                      formatted
                    )
                  })
            )
          }

          // src/icons/copy.tsx

          function SvgCopy(props) {
            return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              'svg',
              __spreadValues(
                {
                  fill: 'currentColor',
                  viewBox: '0 0 16 16',
                  width: '1em',
                  version: '1.1',
                },
                props
              ),
              /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('path', {
                fillRule: 'evenodd',
                d: 'M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z',
              }),
              /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement('path', {
                fillRule: 'evenodd',
                d: 'M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z',
              })
            )
          }
          var copy_default = SvgCopy

          // src/third-party/code.tsx
          var Code = ({ block, defaultLanguage = 'typescript', className }) => {
            var _a, _b, _c
            const [isCopied, setIsCopied] = react__WEBPACK_IMPORTED_MODULE_0__.useState(false)
            const copyTimeout = react__WEBPACK_IMPORTED_MODULE_0__.useRef()
            const { recordMap } = useNotionContext()
            const content = (0, notion_utils__WEBPACK_IMPORTED_MODULE_1__.getBlockTitle)(
              block,
              recordMap
            )
            const language = (
              ((_c =
                (_b = (_a = block.properties) == null ? void 0 : _a.language) == null
                  ? void 0
                  : _b[0]) == null
                ? void 0
                : _c[0]) || defaultLanguage
            ).toLowerCase()
            const caption = block.properties.caption
            const codeRef = react__WEBPACK_IMPORTED_MODULE_0__.useRef()
            react__WEBPACK_IMPORTED_MODULE_0__.useEffect(() => {
              if (codeRef.current) {
                try {
                  ;(0, prismjs__WEBPACK_IMPORTED_MODULE_2__.highlightElement)(codeRef.current)
                } catch (err) {
                  console.warn('prismjs highlight error', err)
                }
              }
            }, [codeRef])
            const onClickCopyToClipboard = react__WEBPACK_IMPORTED_MODULE_0__.useCallback(() => {
              ;(0, import_clipboard_copy.default)(content)
              setIsCopied(true)
              if (copyTimeout.current) {
                clearTimeout(copyTimeout.current)
                copyTimeout.current = null
              }
              copyTimeout.current = setTimeout(() => {
                setIsCopied(false)
              }, 1200)
            }, [content, copyTimeout])
            const copyButton = /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              'div',
              {
                className: 'notion-code-copy-button',
                onClick: onClickCopyToClipboard,
              },
              /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(copy_default, null)
            )
            return /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
              react__WEBPACK_IMPORTED_MODULE_0__.Fragment,
              null,
              /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                'pre',
                {
                  className: cs('notion-code', className),
                },
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  'div',
                  {
                    className: 'notion-code-copy',
                  },
                  copyButton,
                  isCopied &&
                    /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                      'div',
                      {
                        className: 'notion-code-copy-tooltip',
                      },
                      /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                        'div',
                        null,
                        isCopied ? 'Copied' : 'Copy'
                      )
                    )
                ),
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  'code',
                  {
                    className: `language-${language}`,
                    ref: codeRef,
                  },
                  content
                )
              ),
              caption &&
                /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(
                  'figcaption',
                  {
                    className: 'notion-asset-caption',
                  },
                  /* @__PURE__ */ react__WEBPACK_IMPORTED_MODULE_0__.createElement(Text, {
                    value: caption,
                    block,
                  })
                )
            )
          }

          /*! clipboard-copy. MIT License. Feross Aboukhadijeh <https://feross.org/opensource> */

          __webpack_async_result__()
        } catch (e) {
          __webpack_async_result__(e)
        }
      }
    )

    /***/
  },
}
