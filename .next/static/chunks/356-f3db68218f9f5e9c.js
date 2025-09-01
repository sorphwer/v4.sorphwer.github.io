;(self.webpackChunk_N_E = self.webpackChunk_N_E || []).push([
  [356],
  {
    9713: function (e) {
      ;(e.exports = function (e, t, n) {
        return (
          t in e
            ? Object.defineProperty(e, t, {
                value: n,
                enumerable: !0,
                configurable: !0,
                writable: !0,
              })
            : (e[t] = n),
          e
        )
      }),
        (e.exports.default = e.exports),
        (e.exports.__esModule = !0)
    },
    7316: function (e) {
      ;(e.exports = function (e, t) {
        if (null == e) return {}
        var n,
          o,
          r = {},
          a = Object.keys(e)
        for (o = 0; o < a.length; o++) (n = a[o]), t.indexOf(n) >= 0 || (r[n] = e[n])
        return r
      }),
        (e.exports.default = e.exports),
        (e.exports.__esModule = !0)
    },
    6729: function (e) {
      'use strict'
      var t = Object.prototype.hasOwnProperty,
        n = '~'
      function o() {}
      function r(e, t, n) {
        ;(this.fn = e), (this.context = t), (this.once = n || !1)
      }
      function a(e, t, o, a, i) {
        if ('function' !== typeof o) throw new TypeError('The listener must be a function')
        var u = new r(o, a || e, i),
          l = n ? n + t : t
        return (
          e._events[l]
            ? e._events[l].fn
              ? (e._events[l] = [e._events[l], u])
              : e._events[l].push(u)
            : ((e._events[l] = u), e._eventsCount++),
          e
        )
      }
      function i(e, t) {
        0 === --e._eventsCount ? (e._events = new o()) : delete e._events[t]
      }
      function u() {
        ;(this._events = new o()), (this._eventsCount = 0)
      }
      Object.create && ((o.prototype = Object.create(null)), new o().__proto__ || (n = !1)),
        (u.prototype.eventNames = function () {
          var e,
            o,
            r = []
          if (0 === this._eventsCount) return r
          for (o in (e = this._events)) t.call(e, o) && r.push(n ? o.slice(1) : o)
          return Object.getOwnPropertySymbols ? r.concat(Object.getOwnPropertySymbols(e)) : r
        }),
        (u.prototype.listeners = function (e) {
          var t = n ? n + e : e,
            o = this._events[t]
          if (!o) return []
          if (o.fn) return [o.fn]
          for (var r = 0, a = o.length, i = new Array(a); r < a; r++) i[r] = o[r].fn
          return i
        }),
        (u.prototype.listenerCount = function (e) {
          var t = n ? n + e : e,
            o = this._events[t]
          return o ? (o.fn ? 1 : o.length) : 0
        }),
        (u.prototype.emit = function (e, t, o, r, a, i) {
          var u = n ? n + e : e
          if (!this._events[u]) return !1
          var l,
            c,
            s = this._events[u],
            d = arguments.length
          if (s.fn) {
            switch ((s.once && this.removeListener(e, s.fn, void 0, !0), d)) {
              case 1:
                return s.fn.call(s.context), !0
              case 2:
                return s.fn.call(s.context, t), !0
              case 3:
                return s.fn.call(s.context, t, o), !0
              case 4:
                return s.fn.call(s.context, t, o, r), !0
              case 5:
                return s.fn.call(s.context, t, o, r, a), !0
              case 6:
                return s.fn.call(s.context, t, o, r, a, i), !0
            }
            for (c = 1, l = new Array(d - 1); c < d; c++) l[c - 1] = arguments[c]
            s.fn.apply(s.context, l)
          } else {
            var m,
              f = s.length
            for (c = 0; c < f; c++)
              switch ((s[c].once && this.removeListener(e, s[c].fn, void 0, !0), d)) {
                case 1:
                  s[c].fn.call(s[c].context)
                  break
                case 2:
                  s[c].fn.call(s[c].context, t)
                  break
                case 3:
                  s[c].fn.call(s[c].context, t, o)
                  break
                case 4:
                  s[c].fn.call(s[c].context, t, o, r)
                  break
                default:
                  if (!l) for (m = 1, l = new Array(d - 1); m < d; m++) l[m - 1] = arguments[m]
                  s[c].fn.apply(s[c].context, l)
              }
          }
          return !0
        }),
        (u.prototype.on = function (e, t, n) {
          return a(this, e, t, n, !1)
        }),
        (u.prototype.once = function (e, t, n) {
          return a(this, e, t, n, !0)
        }),
        (u.prototype.removeListener = function (e, t, o, r) {
          var a = n ? n + e : e
          if (!this._events[a]) return this
          if (!t) return i(this, a), this
          var u = this._events[a]
          if (u.fn) u.fn !== t || (r && !u.once) || (o && u.context !== o) || i(this, a)
          else {
            for (var l = 0, c = [], s = u.length; l < s; l++)
              (u[l].fn !== t || (r && !u[l].once) || (o && u[l].context !== o)) && c.push(u[l])
            c.length ? (this._events[a] = 1 === c.length ? c[0] : c) : i(this, a)
          }
          return this
        }),
        (u.prototype.removeAllListeners = function (e) {
          var t
          return (
            e
              ? ((t = n ? n + e : e), this._events[t] && i(this, t))
              : ((this._events = new o()), (this._eventsCount = 0)),
            this
          )
        }),
        (u.prototype.off = u.prototype.removeListener),
        (u.prototype.addListener = u.prototype.on),
        (u.prefixed = n),
        (u.EventEmitter = u),
        (e.exports = u)
    },
    9671: function (e, t, n) {
      const o = n(1701)
      e.exports = a
      const r = Object.hasOwnProperty
      function a() {
        if (!(this instanceof a)) return new a()
        this.reset()
      }
      function i(e, t) {
        return 'string' !== typeof e
          ? ''
          : (t || (e = e.toLowerCase()), e.replace(o, '').replace(/ /g, '-'))
      }
      ;(a.prototype.slug = function (e, t) {
        const n = this
        let o = i(e, !0 === t)
        const a = o
        for (; r.call(n.occurrences, o); ) n.occurrences[a]++, (o = a + '-' + n.occurrences[a])
        return (n.occurrences[o] = 0), o
      }),
        (a.prototype.reset = function () {
          this.occurrences = Object.create(null)
        }),
        (a.slug = i)
    },
    1701: function (e) {
      e.exports =
        /[\0-\x1F!-,\.\/:-@\[-\^`\{-\xA9\xAB-\xB4\xB6-\xB9\xBB-\xBF\xD7\xF7\u02C2-\u02C5\u02D2-\u02DF\u02E5-\u02EB\u02ED\u02EF-\u02FF\u0375\u0378\u0379\u037E\u0380-\u0385\u0387\u038B\u038D\u03A2\u03F6\u0482\u0530\u0557\u0558\u055A-\u055F\u0589-\u0590\u05BE\u05C0\u05C3\u05C6\u05C8-\u05CF\u05EB-\u05EE\u05F3-\u060F\u061B-\u061F\u066A-\u066D\u06D4\u06DD\u06DE\u06E9\u06FD\u06FE\u0700-\u070F\u074B\u074C\u07B2-\u07BF\u07F6-\u07F9\u07FB\u07FC\u07FE\u07FF\u082E-\u083F\u085C-\u085F\u086B-\u089F\u08B5\u08BE-\u08D2\u08E2\u0964\u0965\u0970\u0984\u098D\u098E\u0991\u0992\u09A9\u09B1\u09B3-\u09B5\u09BA\u09BB\u09C5\u09C6\u09C9\u09CA\u09CF-\u09D6\u09D8-\u09DB\u09DE\u09E4\u09E5\u09F2-\u09FB\u09FD\u09FF\u0A00\u0A04\u0A0B-\u0A0E\u0A11\u0A12\u0A29\u0A31\u0A34\u0A37\u0A3A\u0A3B\u0A3D\u0A43-\u0A46\u0A49\u0A4A\u0A4E-\u0A50\u0A52-\u0A58\u0A5D\u0A5F-\u0A65\u0A76-\u0A80\u0A84\u0A8E\u0A92\u0AA9\u0AB1\u0AB4\u0ABA\u0ABB\u0AC6\u0ACA\u0ACE\u0ACF\u0AD1-\u0ADF\u0AE4\u0AE5\u0AF0-\u0AF8\u0B00\u0B04\u0B0D\u0B0E\u0B11\u0B12\u0B29\u0B31\u0B34\u0B3A\u0B3B\u0B45\u0B46\u0B49\u0B4A\u0B4E-\u0B55\u0B58-\u0B5B\u0B5E\u0B64\u0B65\u0B70\u0B72-\u0B81\u0B84\u0B8B-\u0B8D\u0B91\u0B96-\u0B98\u0B9B\u0B9D\u0BA0-\u0BA2\u0BA5-\u0BA7\u0BAB-\u0BAD\u0BBA-\u0BBD\u0BC3-\u0BC5\u0BC9\u0BCE\u0BCF\u0BD1-\u0BD6\u0BD8-\u0BE5\u0BF0-\u0BFF\u0C0D\u0C11\u0C29\u0C3A-\u0C3C\u0C45\u0C49\u0C4E-\u0C54\u0C57\u0C5B-\u0C5F\u0C64\u0C65\u0C70-\u0C7F\u0C84\u0C8D\u0C91\u0CA9\u0CB4\u0CBA\u0CBB\u0CC5\u0CC9\u0CCE-\u0CD4\u0CD7-\u0CDD\u0CDF\u0CE4\u0CE5\u0CF0\u0CF3-\u0CFF\u0D04\u0D0D\u0D11\u0D45\u0D49\u0D4F-\u0D53\u0D58-\u0D5E\u0D64\u0D65\u0D70-\u0D79\u0D80\u0D81\u0D84\u0D97-\u0D99\u0DB2\u0DBC\u0DBE\u0DBF\u0DC7-\u0DC9\u0DCB-\u0DCE\u0DD5\u0DD7\u0DE0-\u0DE5\u0DF0\u0DF1\u0DF4-\u0E00\u0E3B-\u0E3F\u0E4F\u0E5A-\u0E80\u0E83\u0E85\u0E8B\u0EA4\u0EA6\u0EBE\u0EBF\u0EC5\u0EC7\u0ECE\u0ECF\u0EDA\u0EDB\u0EE0-\u0EFF\u0F01-\u0F17\u0F1A-\u0F1F\u0F2A-\u0F34\u0F36\u0F38\u0F3A-\u0F3D\u0F48\u0F6D-\u0F70\u0F85\u0F98\u0FBD-\u0FC5\u0FC7-\u0FFF\u104A-\u104F\u109E\u109F\u10C6\u10C8-\u10CC\u10CE\u10CF\u10FB\u1249\u124E\u124F\u1257\u1259\u125E\u125F\u1289\u128E\u128F\u12B1\u12B6\u12B7\u12BF\u12C1\u12C6\u12C7\u12D7\u1311\u1316\u1317\u135B\u135C\u1360-\u137F\u1390-\u139F\u13F6\u13F7\u13FE-\u1400\u166D\u166E\u1680\u169B-\u169F\u16EB-\u16ED\u16F9-\u16FF\u170D\u1715-\u171F\u1735-\u173F\u1754-\u175F\u176D\u1771\u1774-\u177F\u17D4-\u17D6\u17D8-\u17DB\u17DE\u17DF\u17EA-\u180A\u180E\u180F\u181A-\u181F\u1879-\u187F\u18AB-\u18AF\u18F6-\u18FF\u191F\u192C-\u192F\u193C-\u1945\u196E\u196F\u1975-\u197F\u19AC-\u19AF\u19CA-\u19CF\u19DA-\u19FF\u1A1C-\u1A1F\u1A5F\u1A7D\u1A7E\u1A8A-\u1A8F\u1A9A-\u1AA6\u1AA8-\u1AAF\u1ABF-\u1AFF\u1B4C-\u1B4F\u1B5A-\u1B6A\u1B74-\u1B7F\u1BF4-\u1BFF\u1C38-\u1C3F\u1C4A-\u1C4C\u1C7E\u1C7F\u1C89-\u1C8F\u1CBB\u1CBC\u1CC0-\u1CCF\u1CD3\u1CFB-\u1CFF\u1DFA\u1F16\u1F17\u1F1E\u1F1F\u1F46\u1F47\u1F4E\u1F4F\u1F58\u1F5A\u1F5C\u1F5E\u1F7E\u1F7F\u1FB5\u1FBD\u1FBF-\u1FC1\u1FC5\u1FCD-\u1FCF\u1FD4\u1FD5\u1FDC-\u1FDF\u1FED-\u1FF1\u1FF5\u1FFD-\u203E\u2041-\u2053\u2055-\u2070\u2072-\u207E\u2080-\u208F\u209D-\u20CF\u20F1-\u2101\u2103-\u2106\u2108\u2109\u2114\u2116-\u2118\u211E-\u2123\u2125\u2127\u2129\u212E\u213A\u213B\u2140-\u2144\u214A-\u214D\u214F-\u215F\u2189-\u24B5\u24EA-\u2BFF\u2C2F\u2C5F\u2CE5-\u2CEA\u2CF4-\u2CFF\u2D26\u2D28-\u2D2C\u2D2E\u2D2F\u2D68-\u2D6E\u2D70-\u2D7E\u2D97-\u2D9F\u2DA7\u2DAF\u2DB7\u2DBF\u2DC7\u2DCF\u2DD7\u2DDF\u2E00-\u2E2E\u2E30-\u3004\u3008-\u3020\u3030\u3036\u3037\u303D-\u3040\u3097\u3098\u309B\u309C\u30A0\u30FB\u3100-\u3104\u3130\u318F-\u319F\u31BB-\u31EF\u3200-\u33FF\u4DB6-\u4DFF\u9FF0-\u9FFF\uA48D-\uA4CF\uA4FE\uA4FF\uA60D-\uA60F\uA62C-\uA63F\uA673\uA67E\uA6F2-\uA716\uA720\uA721\uA789\uA78A\uA7C0\uA7C1\uA7C7-\uA7F6\uA828-\uA83F\uA874-\uA87F\uA8C6-\uA8CF\uA8DA-\uA8DF\uA8F8-\uA8FA\uA8FC\uA92E\uA92F\uA954-\uA95F\uA97D-\uA97F\uA9C1-\uA9CE\uA9DA-\uA9DF\uA9FF\uAA37-\uAA3F\uAA4E\uAA4F\uAA5A-\uAA5F\uAA77-\uAA79\uAAC3-\uAADA\uAADE\uAADF\uAAF0\uAAF1\uAAF7-\uAB00\uAB07\uAB08\uAB0F\uAB10\uAB17-\uAB1F\uAB27\uAB2F\uAB5B\uAB68-\uAB6F\uABEB\uABEE\uABEF\uABFA-\uABFF\uD7A4-\uD7AF\uD7C7-\uD7CA\uD7FC-\uD7FF\uE000-\uF8FF\uFA6E\uFA6F\uFADA-\uFAFF\uFB07-\uFB12\uFB18-\uFB1C\uFB29\uFB37\uFB3D\uFB3F\uFB42\uFB45\uFBB2-\uFBD2\uFD3E-\uFD4F\uFD90\uFD91\uFDC8-\uFDEF\uFDFC-\uFDFF\uFE10-\uFE1F\uFE30-\uFE32\uFE35-\uFE4C\uFE50-\uFE6F\uFE75\uFEFD-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF3E\uFF40\uFF5B-\uFF65\uFFBF-\uFFC1\uFFC8\uFFC9\uFFD0\uFFD1\uFFD8\uFFD9\uFFDD-\uFFFF]|\uD800[\uDC0C\uDC27\uDC3B\uDC3E\uDC4E\uDC4F\uDC5E-\uDC7F\uDCFB-\uDD3F\uDD75-\uDDFC\uDDFE-\uDE7F\uDE9D-\uDE9F\uDED1-\uDEDF\uDEE1-\uDEFF\uDF20-\uDF2C\uDF4B-\uDF4F\uDF7B-\uDF7F\uDF9E\uDF9F\uDFC4-\uDFC7\uDFD0\uDFD6-\uDFFF]|\uD801[\uDC9E\uDC9F\uDCAA-\uDCAF\uDCD4-\uDCD7\uDCFC-\uDCFF\uDD28-\uDD2F\uDD64-\uDDFF\uDF37-\uDF3F\uDF56-\uDF5F\uDF68-\uDFFF]|\uD802[\uDC06\uDC07\uDC09\uDC36\uDC39-\uDC3B\uDC3D\uDC3E\uDC56-\uDC5F\uDC77-\uDC7F\uDC9F-\uDCDF\uDCF3\uDCF6-\uDCFF\uDD16-\uDD1F\uDD3A-\uDD7F\uDDB8-\uDDBD\uDDC0-\uDDFF\uDE04\uDE07-\uDE0B\uDE14\uDE18\uDE36\uDE37\uDE3B-\uDE3E\uDE40-\uDE5F\uDE7D-\uDE7F\uDE9D-\uDEBF\uDEC8\uDEE7-\uDEFF\uDF36-\uDF3F\uDF56-\uDF5F\uDF73-\uDF7F\uDF92-\uDFFF]|\uD803[\uDC49-\uDC7F\uDCB3-\uDCBF\uDCF3-\uDCFF\uDD28-\uDD2F\uDD3A-\uDEFF\uDF1D-\uDF26\uDF28-\uDF2F\uDF51-\uDFDF\uDFF7-\uDFFF]|\uD804[\uDC47-\uDC65\uDC70-\uDC7E\uDCBB-\uDCCF\uDCE9-\uDCEF\uDCFA-\uDCFF\uDD35\uDD40-\uDD43\uDD47-\uDD4F\uDD74\uDD75\uDD77-\uDD7F\uDDC5-\uDDC8\uDDCD-\uDDCF\uDDDB\uDDDD-\uDDFF\uDE12\uDE38-\uDE3D\uDE3F-\uDE7F\uDE87\uDE89\uDE8E\uDE9E\uDEA9-\uDEAF\uDEEB-\uDEEF\uDEFA-\uDEFF\uDF04\uDF0D\uDF0E\uDF11\uDF12\uDF29\uDF31\uDF34\uDF3A\uDF45\uDF46\uDF49\uDF4A\uDF4E\uDF4F\uDF51-\uDF56\uDF58-\uDF5C\uDF64\uDF65\uDF6D-\uDF6F\uDF75-\uDFFF]|\uD805[\uDC4B-\uDC4F\uDC5A-\uDC5D\uDC60-\uDC7F\uDCC6\uDCC8-\uDCCF\uDCDA-\uDD7F\uDDB6\uDDB7\uDDC1-\uDDD7\uDDDE-\uDDFF\uDE41-\uDE43\uDE45-\uDE4F\uDE5A-\uDE7F\uDEB9-\uDEBF\uDECA-\uDEFF\uDF1B\uDF1C\uDF2C-\uDF2F\uDF3A-\uDFFF]|\uD806[\uDC3B-\uDC9F\uDCEA-\uDCFE\uDD00-\uDD9F\uDDA8\uDDA9\uDDD8\uDDD9\uDDE2\uDDE5-\uDDFF\uDE3F-\uDE46\uDE48-\uDE4F\uDE9A-\uDE9C\uDE9E-\uDEBF\uDEF9-\uDFFF]|\uD807[\uDC09\uDC37\uDC41-\uDC4F\uDC5A-\uDC71\uDC90\uDC91\uDCA8\uDCB7-\uDCFF\uDD07\uDD0A\uDD37-\uDD39\uDD3B\uDD3E\uDD48-\uDD4F\uDD5A-\uDD5F\uDD66\uDD69\uDD8F\uDD92\uDD99-\uDD9F\uDDAA-\uDEDF\uDEF7-\uDFFF]|\uD808[\uDF9A-\uDFFF]|\uD809[\uDC6F-\uDC7F\uDD44-\uDFFF]|[\uD80A\uD80B\uD80E-\uD810\uD812-\uD819\uD823-\uD82B\uD82D\uD82E\uD830-\uD833\uD837\uD839\uD83D-\uD83F\uD87B-\uD87D\uD87F-\uDB3F\uDB41-\uDBFF][\uDC00-\uDFFF]|\uD80D[\uDC2F-\uDFFF]|\uD811[\uDE47-\uDFFF]|\uD81A[\uDE39-\uDE3F\uDE5F\uDE6A-\uDECF\uDEEE\uDEEF\uDEF5-\uDEFF\uDF37-\uDF3F\uDF44-\uDF4F\uDF5A-\uDF62\uDF78-\uDF7C\uDF90-\uDFFF]|\uD81B[\uDC00-\uDE3F\uDE80-\uDEFF\uDF4B-\uDF4E\uDF88-\uDF8E\uDFA0-\uDFDF\uDFE2\uDFE4-\uDFFF]|\uD821[\uDFF8-\uDFFF]|\uD822[\uDEF3-\uDFFF]|\uD82C[\uDD1F-\uDD4F\uDD53-\uDD63\uDD68-\uDD6F\uDEFC-\uDFFF]|\uD82F[\uDC6B-\uDC6F\uDC7D-\uDC7F\uDC89-\uDC8F\uDC9A-\uDC9C\uDC9F-\uDFFF]|\uD834[\uDC00-\uDD64\uDD6A-\uDD6C\uDD73-\uDD7A\uDD83\uDD84\uDD8C-\uDDA9\uDDAE-\uDE41\uDE45-\uDFFF]|\uD835[\uDC55\uDC9D\uDCA0\uDCA1\uDCA3\uDCA4\uDCA7\uDCA8\uDCAD\uDCBA\uDCBC\uDCC4\uDD06\uDD0B\uDD0C\uDD15\uDD1D\uDD3A\uDD3F\uDD45\uDD47-\uDD49\uDD51\uDEA6\uDEA7\uDEC1\uDEDB\uDEFB\uDF15\uDF35\uDF4F\uDF6F\uDF89\uDFA9\uDFC3\uDFCC\uDFCD]|\uD836[\uDC00-\uDDFF\uDE37-\uDE3A\uDE6D-\uDE74\uDE76-\uDE83\uDE85-\uDE9A\uDEA0\uDEB0-\uDFFF]|\uD838[\uDC07\uDC19\uDC1A\uDC22\uDC25\uDC2B-\uDCFF\uDD2D-\uDD2F\uDD3E\uDD3F\uDD4A-\uDD4D\uDD4F-\uDEBF\uDEFA-\uDFFF]|\uD83A[\uDCC5-\uDCCF\uDCD7-\uDCFF\uDD4C-\uDD4F\uDD5A-\uDFFF]|\uD83B[\uDC00-\uDDFF\uDE04\uDE20\uDE23\uDE25\uDE26\uDE28\uDE33\uDE38\uDE3A\uDE3C-\uDE41\uDE43-\uDE46\uDE48\uDE4A\uDE4C\uDE50\uDE53\uDE55\uDE56\uDE58\uDE5A\uDE5C\uDE5E\uDE60\uDE63\uDE65\uDE66\uDE6B\uDE73\uDE78\uDE7D\uDE7F\uDE8A\uDE9C-\uDEA0\uDEA4\uDEAA\uDEBC-\uDFFF]|\uD83C[\uDC00-\uDD2F\uDD4A-\uDD4F\uDD6A-\uDD6F\uDD8A-\uDFFF]|\uD869[\uDED7-\uDEFF]|\uD86D[\uDF35-\uDF3F]|\uD86E[\uDC1E\uDC1F]|\uD873[\uDEA2-\uDEAF]|\uD87A[\uDFE1-\uDFFF]|\uD87E[\uDE1E-\uDFFF]|\uDB40[\uDC00-\uDCFF\uDDF0-\uDFFF]/g
    },
    1143: function (e) {
      'use strict'
      e.exports = function (e, t, n, o, r, a, i, u) {
        if (!e) {
          var l
          if (void 0 === t)
            l = new Error(
              'Minified exception occurred; use the non-minified dev environment for the full error message and additional helpful warnings.'
            )
          else {
            var c = [n, o, r, a, i, u],
              s = 0
            ;(l = new Error(
              t.replace(/%s/g, function () {
                return c[s++]
              })
            )).name = 'Invariant Violation'
          }
          throw ((l.framesToPop = 1), l)
        }
      }
    },
    8918: function (e, t, n) {
      'use strict'
      var o =
          (this && this.__awaiter) ||
          function (e, t, n, o) {
            return new (n || (n = Promise))(function (r, a) {
              function i(e) {
                try {
                  l(o.next(e))
                } catch (t) {
                  a(t)
                }
              }
              function u(e) {
                try {
                  l(o.throw(e))
                } catch (t) {
                  a(t)
                }
              }
              function l(e) {
                e.done
                  ? r(e.value)
                  : new n(function (t) {
                      t(e.value)
                    }).then(i, u)
              }
              l((o = o.apply(e, t || [])).next())
            })
          },
        r =
          (this && this.__importDefault) ||
          function (e) {
            return e && e.__esModule ? e : { default: e }
          }
      Object.defineProperty(t, '__esModule', { value: !0 })
      const a = r(n(3931))
      function i(e, t = 'maxAge') {
        let n, r, i
        const u = () =>
            o(this, void 0, void 0, function* () {
              if (void 0 !== n) return
              const u = (u) =>
                o(this, void 0, void 0, function* () {
                  i = a.default()
                  const o = u[1][t] - Date.now()
                  return o <= 0
                    ? (e.delete(u[0]), void i.resolve())
                    : ((n = u[0]),
                      (r = setTimeout(() => {
                        e.delete(u[0]), i && i.resolve()
                      }, o)),
                      'function' === typeof r.unref && r.unref(),
                      i.promise)
                })
              try {
                for (const t of e) yield u(t)
              } catch (l) {}
              n = void 0
            }),
          l = e.set.bind(e)
        return (
          (e.set = (t, o) => {
            e.has(t) && e.delete(t)
            const a = l(t, o)
            return (
              n &&
                n === t &&
                ((n = void 0),
                void 0 !== r && (clearTimeout(r), (r = void 0)),
                void 0 !== i && (i.reject(void 0), (i = void 0))),
              u(),
              a
            )
          }),
          u(),
          e
        )
      }
      ;(t.default = i), (e.exports = i), (e.exports.default = i)
    },
    7645: function (e, t, n) {
      'use strict'
      function o(e, t, n) {
        return (
          t in e
            ? Object.defineProperty(e, t, {
                value: n,
                enumerable: !0,
                configurable: !0,
                writable: !0,
              })
            : (e[t] = n),
          e
        )
      }
      function r(e) {
        for (var t = 1; t < arguments.length; t++) {
          var n = null != arguments[t] ? arguments[t] : {},
            r = Object.keys(n)
          'function' === typeof Object.getOwnPropertySymbols &&
            (r = r.concat(
              Object.getOwnPropertySymbols(n).filter(function (e) {
                return Object.getOwnPropertyDescriptor(n, e).enumerable
              })
            )),
            r.forEach(function (t) {
              o(e, t, n[t])
            })
        }
        return e
      }
      t.default = function (e, t) {
        var n = a.default,
          o = {
            loading: function (e) {
              e.error, e.isLoading
              return e.pastDelay, null
            },
          }
        ;(i = e),
          (l = Promise),
          (
            null != l && 'undefined' !== typeof Symbol && l[Symbol.hasInstance]
              ? l[Symbol.hasInstance](i)
              : i instanceof l
          )
            ? (o.loader = function () {
                return e
              })
            : 'function' === typeof e
            ? (o.loader = e)
            : 'object' === typeof e && (o = r({}, o, e))
        var i, l
        var c = (o = r({}, o, t))
        if (c.suspense)
          throw new Error(
            'Invalid suspense option usage in next/dynamic. Read more: https://nextjs.org/docs/messages/invalid-dynamic-suspense'
          )
        if (c.suspense) return n(c)
        o.loadableGenerated && delete (o = r({}, o, o.loadableGenerated)).loadableGenerated
        if ('boolean' === typeof o.ssr) {
          if (!o.ssr) return delete o.ssr, u(n, o)
          delete o.ssr
        }
        return n(o)
      }
      i(n(1720))
      var a = i(n(4588))
      function i(e) {
        return e && e.__esModule ? e : { default: e }
      }
      function u(e, t) {
        return delete t.webpack, delete t.modules, e(t)
      }
    },
    3644: function (e, t, n) {
      'use strict'
      var o
      Object.defineProperty(t, '__esModule', { value: !0 }), (t.LoadableContext = void 0)
      var r = ((o = n(1720)) && o.__esModule ? o : { default: o }).default.createContext(null)
      t.LoadableContext = r
    },
    4588: function (e, t, n) {
      'use strict'
      function o(e, t) {
        for (var n = 0; n < t.length; n++) {
          var o = t[n]
          ;(o.enumerable = o.enumerable || !1),
            (o.configurable = !0),
            'value' in o && (o.writable = !0),
            Object.defineProperty(e, o.key, o)
        }
      }
      function r(e, t, n) {
        return (
          t in e
            ? Object.defineProperty(e, t, {
                value: n,
                enumerable: !0,
                configurable: !0,
                writable: !0,
              })
            : (e[t] = n),
          e
        )
      }
      function a(e) {
        for (var t = 1; t < arguments.length; t++) {
          var n = null != arguments[t] ? arguments[t] : {},
            o = Object.keys(n)
          'function' === typeof Object.getOwnPropertySymbols &&
            (o = o.concat(
              Object.getOwnPropertySymbols(n).filter(function (e) {
                return Object.getOwnPropertyDescriptor(n, e).enumerable
              })
            )),
            o.forEach(function (t) {
              r(e, t, n[t])
            })
        }
        return e
      }
      Object.defineProperty(t, '__esModule', { value: !0 }), (t.default = void 0)
      var i,
        u = (i = n(1720)) && i.__esModule ? i : { default: i },
        l = n(2021),
        c = n(3644)
      var s = [],
        d = [],
        m = !1
      function f(e) {
        var t = e(),
          n = { loading: !0, loaded: null, error: null }
        return (
          (n.promise = t
            .then(function (e) {
              return (n.loading = !1), (n.loaded = e), e
            })
            .catch(function (e) {
              throw ((n.loading = !1), (n.error = e), e)
            })),
          n
        )
      }
      var p = (function () {
        function e(t, n) {
          !(function (e, t) {
            if (!(e instanceof t)) throw new TypeError('Cannot call a class as a function')
          })(this, e),
            (this._loadFn = t),
            (this._opts = n),
            (this._callbacks = new Set()),
            (this._delay = null),
            (this._timeout = null),
            this.retry()
        }
        var t, n, r
        return (
          (t = e),
          (n = [
            {
              key: 'promise',
              value: function () {
                return this._res.promise
              },
            },
            {
              key: 'retry',
              value: function () {
                var e = this
                this._clearTimeouts(),
                  (this._res = this._loadFn(this._opts.loader)),
                  (this._state = { pastDelay: !1, timedOut: !1 })
                var t = this._res,
                  n = this._opts
                if (t.loading) {
                  if ('number' === typeof n.delay)
                    if (0 === n.delay) this._state.pastDelay = !0
                    else {
                      var o = this
                      this._delay = setTimeout(function () {
                        o._update({ pastDelay: !0 })
                      }, n.delay)
                    }
                  if ('number' === typeof n.timeout) {
                    var r = this
                    this._timeout = setTimeout(function () {
                      r._update({ timedOut: !0 })
                    }, n.timeout)
                  }
                }
                this._res.promise
                  .then(function () {
                    e._update({}), e._clearTimeouts()
                  })
                  .catch(function (t) {
                    e._update({}), e._clearTimeouts()
                  }),
                  this._update({})
              },
            },
            {
              key: '_update',
              value: function (e) {
                ;(this._state = a(
                  {},
                  this._state,
                  { error: this._res.error, loaded: this._res.loaded, loading: this._res.loading },
                  e
                )),
                  this._callbacks.forEach(function (e) {
                    return e()
                  })
              },
            },
            {
              key: '_clearTimeouts',
              value: function () {
                clearTimeout(this._delay), clearTimeout(this._timeout)
              },
            },
            {
              key: 'getCurrentValue',
              value: function () {
                return this._state
              },
            },
            {
              key: 'subscribe',
              value: function (e) {
                var t = this
                return (
                  this._callbacks.add(e),
                  function () {
                    t._callbacks.delete(e)
                  }
                )
              },
            },
          ]) && o(t.prototype, n),
          r && o(t, r),
          e
        )
      })()
      function h(e) {
        return (function (e, t) {
          var n = function () {
              if (!r) {
                var t = new p(e, o)
                r = {
                  getCurrentValue: t.getCurrentValue.bind(t),
                  subscribe: t.subscribe.bind(t),
                  retry: t.retry.bind(t),
                  promise: t.promise.bind(t),
                }
              }
              return r.promise()
            },
            o = Object.assign(
              {
                loader: null,
                loading: null,
                delay: 200,
                timeout: null,
                webpack: null,
                modules: null,
                suspense: !1,
              },
              t
            )
          o.suspense && (o.lazy = u.default.lazy(o.loader))
          var r = null
          if (!m && !o.suspense) {
            var i = o.webpack ? o.webpack() : o.modules
            i &&
              d.push(function (e) {
                var t = !0,
                  o = !1,
                  r = void 0
                try {
                  for (var a, u = i[Symbol.iterator](); !(t = (a = u.next()).done); t = !0) {
                    var l = a.value
                    if (-1 !== e.indexOf(l)) return n()
                  }
                } catch (c) {
                  ;(o = !0), (r = c)
                } finally {
                  try {
                    t || null == u.return || u.return()
                  } finally {
                    if (o) throw r
                  }
                }
              })
          }
          var s = o.suspense
            ? function (e, t) {
                return u.default.createElement(o.lazy, a({}, e, { ref: t }))
              }
            : function (e, t) {
                n()
                var a = u.default.useContext(c.LoadableContext),
                  i = l.useSubscription(r)
                return (
                  u.default.useImperativeHandle(
                    t,
                    function () {
                      return { retry: r.retry }
                    },
                    []
                  ),
                  a &&
                    Array.isArray(o.modules) &&
                    o.modules.forEach(function (e) {
                      a(e)
                    }),
                  u.default.useMemo(
                    function () {
                      return i.loading || i.error
                        ? u.default.createElement(o.loading, {
                            isLoading: i.loading,
                            pastDelay: i.pastDelay,
                            timedOut: i.timedOut,
                            error: i.error,
                            retry: r.retry,
                          })
                        : i.loaded
                        ? u.default.createElement(
                            (function (e) {
                              return e && e.__esModule ? e.default : e
                            })(i.loaded),
                            e
                          )
                        : null
                    },
                    [e, i]
                  )
                )
              }
          return (
            (s.preload = function () {
              return !o.suspense && n()
            }),
            (s.displayName = 'LoadableComponent'),
            u.default.forwardRef(s)
          )
        })(f, e)
      }
      function v(e, t) {
        for (var n = []; e.length; ) {
          var o = e.pop()
          n.push(o(t))
        }
        return Promise.all(n).then(function () {
          if (e.length) return v(e, t)
        })
      }
      ;(h.preloadAll = function () {
        return new Promise(function (e, t) {
          v(s).then(e, t)
        })
      }),
        (h.preloadReady = function () {
          var e = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : []
          return new Promise(function (t) {
            var n = function () {
              return (m = !0), t()
            }
            v(d, e).then(n, n)
          })
        }),
        (window.__NEXT_PRELOADREADY = h.preloadReady)
      var D = h
      t.default = D
    },
    2021: function (e, t, n) {
      ;(() => {
        'use strict'
        var t = {
            800: (e) => {
              var t = Object.getOwnPropertySymbols,
                n = Object.prototype.hasOwnProperty,
                o = Object.prototype.propertyIsEnumerable
              function r(e) {
                if (null === e || void 0 === e)
                  throw new TypeError('Object.assign cannot be called with null or undefined')
                return Object(e)
              }
              e.exports = (function () {
                try {
                  if (!Object.assign) return !1
                  var e = new String('abc')
                  if (((e[5] = 'de'), '5' === Object.getOwnPropertyNames(e)[0])) return !1
                  for (var t = {}, n = 0; n < 10; n++) t['_' + String.fromCharCode(n)] = n
                  var o = Object.getOwnPropertyNames(t).map(function (e) {
                    return t[e]
                  })
                  if ('0123456789' !== o.join('')) return !1
                  var r = {}
                  return (
                    'abcdefghijklmnopqrst'.split('').forEach(function (e) {
                      r[e] = e
                    }),
                    'abcdefghijklmnopqrst' === Object.keys(Object.assign({}, r)).join('')
                  )
                } catch (e) {
                  return !1
                }
              })()
                ? Object.assign
                : function (e, a) {
                    for (var i, u, l = r(e), c = 1; c < arguments.length; c++) {
                      for (var s in (i = Object(arguments[c]))) n.call(i, s) && (l[s] = i[s])
                      if (t) {
                        u = t(i)
                        for (var d = 0; d < u.length; d++) o.call(i, u[d]) && (l[u[d]] = i[u[d]])
                      }
                    }
                    return l
                  }
            },
            569: (e, t, n) => {
              0
            },
            403: (e, t, n) => {
              var o = n(800),
                r = n(522)
              t.useSubscription = function (e) {
                var t = e.getCurrentValue,
                  n = e.subscribe,
                  a = r.useState(function () {
                    return { getCurrentValue: t, subscribe: n, value: t() }
                  })
                e = a[0]
                var i = a[1]
                return (
                  (a = e.value),
                  (e.getCurrentValue === t && e.subscribe === n) ||
                    ((a = t()), i({ getCurrentValue: t, subscribe: n, value: a })),
                  r.useDebugValue(a),
                  r.useEffect(
                    function () {
                      function e() {
                        if (!r) {
                          var e = t()
                          i(function (r) {
                            return r.getCurrentValue !== t || r.subscribe !== n || r.value === e
                              ? r
                              : o({}, r, { value: e })
                          })
                        }
                      }
                      var r = !1,
                        a = n(e)
                      return (
                        e(),
                        function () {
                          ;(r = !0), a()
                        }
                      )
                    },
                    [t, n]
                  ),
                  a
                )
              }
            },
            138: (e, t, n) => {
              e.exports = n(403)
            },
            522: (e) => {
              e.exports = n(1720)
            },
          },
          o = {}
        function r(e) {
          var n = o[e]
          if (void 0 !== n) return n.exports
          var a = (o[e] = { exports: {} }),
            i = !0
          try {
            t[e](a, a.exports, r), (i = !1)
          } finally {
            i && delete o[e]
          }
          return a.exports
        }
        r.ab = '//'
        var a = r(138)
        e.exports = a
      })()
    },
    5152: function (e, t, n) {
      e.exports = n(7645)
    },
    3931: function (e) {
      'use strict'
      e.exports = () => {
        const e = {}
        return (
          (e.promise = new Promise((t, n) => {
            ;(e.resolve = t), (e.reject = n)
          })),
          e
        )
      }
    },
    1032: function (e, t, n) {
      e.exports = n(6584)
    },
    9590: function (e) {
      var t = 'undefined' !== typeof Element,
        n = 'function' === typeof Map,
        o = 'function' === typeof Set,
        r = 'function' === typeof ArrayBuffer && !!ArrayBuffer.isView
      function a(e, i) {
        if (e === i) return !0
        if (e && i && 'object' == typeof e && 'object' == typeof i) {
          if (e.constructor !== i.constructor) return !1
          var u, l, c, s
          if (Array.isArray(e)) {
            if ((u = e.length) != i.length) return !1
            for (l = u; 0 !== l--; ) if (!a(e[l], i[l])) return !1
            return !0
          }
          if (n && e instanceof Map && i instanceof Map) {
            if (e.size !== i.size) return !1
            for (s = e.entries(); !(l = s.next()).done; ) if (!i.has(l.value[0])) return !1
            for (s = e.entries(); !(l = s.next()).done; )
              if (!a(l.value[1], i.get(l.value[0]))) return !1
            return !0
          }
          if (o && e instanceof Set && i instanceof Set) {
            if (e.size !== i.size) return !1
            for (s = e.entries(); !(l = s.next()).done; ) if (!i.has(l.value[0])) return !1
            return !0
          }
          if (r && ArrayBuffer.isView(e) && ArrayBuffer.isView(i)) {
            if ((u = e.length) != i.length) return !1
            for (l = u; 0 !== l--; ) if (e[l] !== i[l]) return !1
            return !0
          }
          if (e.constructor === RegExp) return e.source === i.source && e.flags === i.flags
          if (
            e.valueOf !== Object.prototype.valueOf &&
            'function' === typeof e.valueOf &&
            'function' === typeof i.valueOf
          )
            return e.valueOf() === i.valueOf()
          if (
            e.toString !== Object.prototype.toString &&
            'function' === typeof e.toString &&
            'function' === typeof i.toString
          )
            return e.toString() === i.toString()
          if ((u = (c = Object.keys(e)).length) !== Object.keys(i).length) return !1
          for (l = u; 0 !== l--; ) if (!Object.prototype.hasOwnProperty.call(i, c[l])) return !1
          if (t && e instanceof Element) return !1
          for (l = u; 0 !== l--; )
            if (
              (('_owner' !== c[l] && '__v' !== c[l] && '__o' !== c[l]) || !e.$$typeof) &&
              !a(e[c[l]], i[c[l]])
            )
              return !1
          return !0
        }
        return e !== e && i !== i
      }
      e.exports = function (e, t) {
        try {
          return a(e, t)
        } catch (n) {
          if ((n.message || '').match(/stack|recursion/i))
            return console.warn('react-fast-compare cannot handle circular refs'), !1
          throw n
        }
      }
    },
    9218: function (e, t, n) {
      'use strict'
      n.d(t, {
        y1: function () {
          return B
        },
      })
      var o =
        'undefined' !== typeof navigator && navigator.userAgent.toLowerCase().indexOf('firefox') > 0
      function r(e, t, n, o) {
        e.addEventListener
          ? e.addEventListener(t, n, o)
          : e.attachEvent &&
            e.attachEvent('on'.concat(t), function () {
              n(window.event)
            })
      }
      function a(e, t) {
        for (var n = t.slice(0, t.length - 1), o = 0; o < n.length; o++)
          n[o] = e[n[o].toLowerCase()]
        return n
      }
      function i(e) {
        'string' !== typeof e && (e = '')
        for (var t = (e = e.replace(/\s/g, '')).split(','), n = t.lastIndexOf(''); n >= 0; )
          (t[n - 1] += ','), t.splice(n, 1), (n = t.lastIndexOf(''))
        return t
      }
      for (
        var u = {
            backspace: 8,
            tab: 9,
            clear: 12,
            enter: 13,
            return: 13,
            esc: 27,
            escape: 27,
            space: 32,
            left: 37,
            up: 38,
            right: 39,
            down: 40,
            del: 46,
            delete: 46,
            ins: 45,
            insert: 45,
            home: 36,
            end: 35,
            pageup: 33,
            pagedown: 34,
            capslock: 20,
            num_0: 96,
            num_1: 97,
            num_2: 98,
            num_3: 99,
            num_4: 100,
            num_5: 101,
            num_6: 102,
            num_7: 103,
            num_8: 104,
            num_9: 105,
            num_multiply: 106,
            num_add: 107,
            num_enter: 108,
            num_subtract: 109,
            num_decimal: 110,
            num_divide: 111,
            '\u21ea': 20,
            ',': 188,
            '.': 190,
            '/': 191,
            '`': 192,
            '-': o ? 173 : 189,
            '=': o ? 61 : 187,
            ';': o ? 59 : 186,
            "'": 222,
            '[': 219,
            ']': 221,
            '\\': 220,
          },
          l = {
            '\u21e7': 16,
            shift: 16,
            '\u2325': 18,
            alt: 18,
            option: 18,
            '\u2303': 17,
            ctrl: 17,
            control: 17,
            '\u2318': 91,
            cmd: 91,
            command: 91,
          },
          c = {
            16: 'shiftKey',
            18: 'altKey',
            17: 'ctrlKey',
            91: 'metaKey',
            shiftKey: 16,
            ctrlKey: 17,
            altKey: 18,
            metaKey: 91,
          },
          s = { 16: !1, 18: !1, 17: !1, 91: !1 },
          d = {},
          m = 1;
        m < 20;
        m++
      )
        u['f'.concat(m)] = 111 + m
      var f = [],
        p = !1,
        h = 'all',
        v = [],
        D = function (e) {
          return u[e.toLowerCase()] || l[e.toLowerCase()] || e.toUpperCase().charCodeAt(0)
        }
      function g(e) {
        h = e || 'all'
      }
      function E() {
        return h || 'all'
      }
      var y = function (e) {
        var t = e.key,
          n = e.scope,
          o = e.method,
          r = e.splitKey,
          u = void 0 === r ? '+' : r
        i(t).forEach(function (e) {
          var t = e.split(u),
            r = t.length,
            i = t[r - 1],
            c = '*' === i ? '*' : D(i)
          if (d[c]) {
            n || (n = E())
            var s = r > 1 ? a(l, t) : []
            d[c] = d[c].filter(function (e) {
              return !(
                (!o || e.method === o) &&
                e.scope === n &&
                (function (e, t) {
                  for (
                    var n = e.length >= t.length ? e : t,
                      o = e.length >= t.length ? t : e,
                      r = !0,
                      a = 0;
                    a < n.length;
                    a++
                  )
                    -1 === o.indexOf(n[a]) && (r = !1)
                  return r
                })(e.mods, s)
              )
            })
          }
        })
      }
      function b(e, t, n, o) {
        var r
        if (t.element === o && (t.scope === n || 'all' === t.scope)) {
          for (var a in ((r = t.mods.length > 0), s))
            Object.prototype.hasOwnProperty.call(s, a) &&
              ((!s[a] && t.mods.indexOf(+a) > -1) || (s[a] && -1 === t.mods.indexOf(+a))) &&
              (r = !1)
          ;((0 !== t.mods.length || s[16] || s[18] || s[17] || s[91]) &&
            !r &&
            '*' !== t.shortcut) ||
            (!1 === t.method(e, t) &&
              (e.preventDefault ? e.preventDefault() : (e.returnValue = !1),
              e.stopPropagation && e.stopPropagation(),
              e.cancelBubble && (e.cancelBubble = !0)))
        }
      }
      function F(e, t) {
        var n = d['*'],
          o = e.keyCode || e.which || e.charCode
        if (C.filter.call(this, e)) {
          if (
            ((93 !== o && 224 !== o) || (o = 91),
            -1 === f.indexOf(o) && 229 !== o && f.push(o),
            ['ctrlKey', 'altKey', 'shiftKey', 'metaKey'].forEach(function (t) {
              var n = c[t]
              e[t] && -1 === f.indexOf(n)
                ? f.push(n)
                : !e[t] && f.indexOf(n) > -1
                ? f.splice(f.indexOf(n), 1)
                : 'metaKey' === t &&
                  e[t] &&
                  3 === f.length &&
                  (e.ctrlKey || e.shiftKey || e.altKey || (f = f.slice(f.indexOf(n))))
            }),
            o in s)
          ) {
            for (var r in ((s[o] = !0), l)) l[r] === o && (C[r] = !0)
            if (!n) return
          }
          for (var a in s) Object.prototype.hasOwnProperty.call(s, a) && (s[a] = e[c[a]])
          e.getModifierState &&
            (!e.altKey || e.ctrlKey) &&
            e.getModifierState('AltGraph') &&
            (-1 === f.indexOf(17) && f.push(17),
            -1 === f.indexOf(18) && f.push(18),
            (s[17] = !0),
            (s[18] = !0))
          var i = E()
          if (n)
            for (var u = 0; u < n.length; u++)
              n[u].scope === i &&
                (('keydown' === e.type && n[u].keydown) || ('keyup' === e.type && n[u].keyup)) &&
                b(e, n[u], i, t)
          if (o in d)
            for (var m = 0; m < d[o].length; m++)
              if (
                (('keydown' === e.type && d[o][m].keydown) ||
                  ('keyup' === e.type && d[o][m].keyup)) &&
                d[o][m].key
              ) {
                for (
                  var p = d[o][m], h = p.splitKey, v = p.key.split(h), g = [], y = 0;
                  y < v.length;
                  y++
                )
                  g.push(D(v[y]))
                g.sort().join('') === f.sort().join('') && b(e, p, i, t)
              }
        }
      }
      function C(e, t, n) {
        f = []
        var o = i(e),
          u = [],
          c = 'all',
          m = document,
          h = 0,
          g = !1,
          E = !0,
          y = '+',
          b = !1
        for (
          void 0 === n && 'function' === typeof t && (n = t),
            '[object Object]' === Object.prototype.toString.call(t) &&
              (t.scope && (c = t.scope),
              t.element && (m = t.element),
              t.keyup && (g = t.keyup),
              void 0 !== t.keydown && (E = t.keydown),
              void 0 !== t.capture && (b = t.capture),
              'string' === typeof t.splitKey && (y = t.splitKey)),
            'string' === typeof t && (c = t);
          h < o.length;
          h++
        )
          (u = []),
            (e = o[h].split(y)).length > 1 && (u = a(l, e)),
            (e = '*' === (e = e[e.length - 1]) ? '*' : D(e)) in d || (d[e] = []),
            d[e].push({
              keyup: g,
              keydown: E,
              scope: c,
              mods: u,
              shortcut: o[h],
              method: n,
              key: o[h],
              splitKey: y,
              element: m,
            })
        'undefined' !== typeof m &&
          !(function (e) {
            return v.indexOf(e) > -1
          })(m) &&
          window &&
          (v.push(m),
          r(
            m,
            'keydown',
            function (e) {
              F(e, m)
            },
            b
          ),
          p ||
            ((p = !0),
            r(
              window,
              'focus',
              function () {
                f = []
              },
              b
            )),
          r(
            m,
            'keyup',
            function (e) {
              F(e, m),
                (function (e) {
                  var t = e.keyCode || e.which || e.charCode,
                    n = f.indexOf(t)
                  if (
                    (n >= 0 && f.splice(n, 1),
                    e.key && 'meta' === e.key.toLowerCase() && f.splice(0, f.length),
                    (93 !== t && 224 !== t) || (t = 91),
                    t in s)
                  )
                    for (var o in ((s[t] = !1), l)) l[o] === t && (C[o] = !1)
                })(e)
            },
            b
          ))
      }
      var w = {
        setScope: g,
        getScope: E,
        deleteScope: function (e, t) {
          var n, o
          for (var r in (e || (e = E()), d))
            if (Object.prototype.hasOwnProperty.call(d, r))
              for (n = d[r], o = 0; o < n.length; ) n[o].scope === e ? n.splice(o, 1) : o++
          E() === e && g(t || 'all')
        },
        getPressedKeyCodes: function () {
          return f.slice(0)
        },
        isPressed: function (e) {
          return 'string' === typeof e && (e = D(e)), -1 !== f.indexOf(e)
        },
        filter: function (e) {
          var t = e.target || e.srcElement,
            n = t.tagName,
            o = !0
          return (
            (!t.isContentEditable &&
              (('INPUT' !== n && 'TEXTAREA' !== n && 'SELECT' !== n) || t.readOnly)) ||
              (o = !1),
            o
          )
        },
        trigger: function (e) {
          var t = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : 'all'
          Object.keys(d).forEach(function (n) {
            var o = d[n].find(function (n) {
              return n.scope === t && n.shortcut === e
            })
            o && o.method && o.method()
          })
        },
        unbind: function (e) {
          if ('undefined' === typeof e)
            Object.keys(d).forEach(function (e) {
              return delete d[e]
            })
          else if (Array.isArray(e))
            e.forEach(function (e) {
              e.key && y(e)
            })
          else if ('object' === typeof e) e.key && y(e)
          else if ('string' === typeof e) {
            for (var t = arguments.length, n = new Array(t > 1 ? t - 1 : 0), o = 1; o < t; o++)
              n[o - 1] = arguments[o]
            var r = n[0],
              a = n[1]
            'function' === typeof r && ((a = r), (r = '')),
              y({ key: e, scope: r, method: a, splitKey: '+' })
          }
        },
        keyMap: u,
        modifier: l,
        modifierMap: c,
      }
      for (var k in w) Object.prototype.hasOwnProperty.call(w, k) && (C[k] = w[k])
      if ('undefined' !== typeof window) {
        var A = window.hotkeys
        ;(C.noConflict = function (e) {
          return e && window.hotkeys === C && (window.hotkeys = A), C
        }),
          (window.hotkeys = C)
      }
      var _ = n(1720)
      C.filter = function () {
        return !0
      }
      var O = function (e, t) {
        var n = e.target,
          o = n && n.tagName
        return Boolean(o && t && t.includes(o))
      }
      function B(e, t, n, o) {
        n instanceof Array && ((o = n), (n = void 0))
        var r = n || {},
          a = r.enableOnTags,
          i = r.filter,
          u = r.keyup,
          l = r.keydown,
          c = r.filterPreventDefault,
          s = void 0 === c || c,
          d = r.enabled,
          m = void 0 === d || d,
          f = r.enableOnContentEditable,
          p = void 0 !== f && f,
          h = (0, _.useRef)(null),
          v = (0, _.useCallback)(
            function (e, n) {
              var o, r
              return i && !i(e)
                ? !s
                : !!(
                    (O(e, ['INPUT', 'TEXTAREA', 'SELECT']) && !O(e, a)) ||
                    (null != (o = e.target) && o.isContentEditable && !p)
                  ) ||
                    (!!(
                      null === h.current ||
                      document.activeElement === h.current ||
                      (null != (r = h.current) && r.contains(document.activeElement))
                    ) &&
                      (t(e, n), !0))
            },
            o ? [h, a, i].concat(o) : [h, a, i]
          )
        return (
          (0, _.useEffect)(
            function () {
              if (m)
                return (
                  u && !0 !== l && (n.keydown = !1),
                  C(e, n || {}, v),
                  function () {
                    return C.unbind(e, v)
                  }
                )
              C.unbind(e, v)
            },
            [v, e, m]
          ),
          h
        )
      }
      C.isPressed
    },
    3324: function (e, t, n) {
      !(function (e, t, n, o) {
        'use strict'
        function r(e) {
          return e && 'object' == typeof e && 'default' in e ? e : { default: e }
        }
        function a(e, t) {
          var n = Object.keys(e)
          if (Object.getOwnPropertySymbols) {
            var o = Object.getOwnPropertySymbols(e)
            t &&
              (o = o.filter(function (t) {
                return Object.getOwnPropertyDescriptor(e, t).enumerable
              })),
              n.push.apply(n, o)
          }
          return n
        }
        function i(e) {
          for (var t, n = 1; n < arguments.length; n++)
            (t = null == arguments[n] ? {} : arguments[n]),
              n % 2
                ? a(Object(t), !0).forEach(function (n) {
                    s.default(e, n, t[n])
                  })
                : Object.getOwnPropertyDescriptors
                ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t))
                : a(Object(t)).forEach(function (n) {
                    Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(t, n))
                  })
          return e
        }
        function u(e) {
          var t = e.srcList,
            n = e.imgPromise,
            r = void 0 === n ? f({ decode: !0 }) : n,
            a = e.useSuspense,
            u = void 0 === a || a,
            l = o.useState(!1)[1],
            c = p(h(t)),
            s = c.join('')
          if (
            (v[s] || (v[s] = { promise: D(c, r), cache: 'pending', error: null }),
            'resolved' === v[s].cache)
          )
            return { src: v[s].src, isLoading: !1, error: null }
          if ('rejected' === v[s].cache) {
            if (u) throw v[s].error
            return { isLoading: !1, error: v[s].error, src: void 0 }
          }
          if (
            (v[s].promise
              .then(function (e) {
                ;(v[s] = i(i({}, v[s]), {}, { cache: 'resolved', src: e })), u || l(s)
              })
              .catch(function (e) {
                ;(v[s] = i(i({}, v[s]), {}, { cache: 'rejected', error: e })), u || l(s)
              }),
            u)
          )
            throw v[s].promise
          return { isLoading: !0, src: void 0, error: null }
        }
        function l(e, t) {
          var n = Object.keys(e)
          if (Object.getOwnPropertySymbols) {
            var o = Object.getOwnPropertySymbols(e)
            t &&
              (o = o.filter(function (t) {
                return Object.getOwnPropertyDescriptor(e, t).enumerable
              })),
              n.push.apply(n, o)
          }
          return n
        }
        function c(e) {
          for (var t, n = 1; n < arguments.length; n++)
            (t = null == arguments[n] ? {} : arguments[n]),
              n % 2
                ? l(Object(t), !0).forEach(function (n) {
                    s.default(e, n, t[n])
                  })
                : Object.getOwnPropertyDescriptors
                ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t))
                : l(Object(t)).forEach(function (n) {
                    Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(t, n))
                  })
          return e
        }
        var s = r(t),
          d = r(n),
          m = r(o),
          f = function (e) {
            var t = e.decode,
              n = e.crossOrigin,
              o = void 0 === n ? '' : n
            return function (e) {
              return new Promise(function (n, r) {
                var a = new Image()
                o && (a.crossOrigin = o),
                  (a.onload = function () {
                    ;(void 0 === t || t) && a.decode ? a.decode().then(n).catch(r) : n()
                  }),
                  (a.onerror = r),
                  (a.src = e)
              })
            }
          },
          p = function (e) {
            return e.filter(function (e) {
              return e
            })
          },
          h = function (e) {
            return Array.isArray(e) ? e : [e]
          },
          v = {},
          D = function (e, t) {
            var n = !1
            return new Promise(function (o, r) {
              var a = function (e) {
                return t(e).then(function () {
                  ;(n = !0), o(e)
                })
              }
              e.reduce(function (e, t) {
                return e.catch(function () {
                  if (!n) return a(t)
                })
              }, a(e.shift())).catch(r)
            })
          },
          g = [
            'decode',
            'src',
            'loader',
            'unloader',
            'container',
            'loaderContainer',
            'unloaderContainer',
            'imgPromise',
            'crossorigin',
            'useSuspense',
          ],
          E = function (e) {
            return e
          }
        ;(e.Img = function (e) {
          var t = e.decode,
            n = e.src,
            o = void 0 === n ? [] : n,
            r = e.loader,
            a = void 0 === r ? null : r,
            i = e.unloader,
            l = void 0 === i ? null : i,
            s = e.container,
            p = void 0 === s ? E : s,
            h = e.loaderContainer,
            v = void 0 === h ? E : h,
            D = e.unloaderContainer,
            y = void 0 === D ? E : D,
            b = e.imgPromise,
            F = e.crossorigin,
            C = e.useSuspense,
            w = void 0 !== C && C,
            k = d.default(e, g),
            A = u({
              srcList: o,
              imgPromise: (b = b || f({ decode: !(void 0 !== t) || t, crossOrigin: F })),
              useSuspense: w,
            }),
            _ = A.src,
            O = A.isLoading
          return _
            ? p(m.default.createElement('img', c({ src: _ }, k)))
            : !w && O
            ? v(a)
            : !w && l
            ? y(l)
            : null
        }),
          (e.useImage = u),
          Object.defineProperty(e, '__esModule', { value: !0 })
      })(t, n(9713), n(7316), n(1720))
    },
    8577: function (e, t, n) {
      'use strict'
      n.d(t, {
        zl: function () {
          return D
        },
        AZ: function () {
          return O
        },
      })
      var o = n(1720),
        r = n(7462),
        a = n(3366),
        i = n(9611)
      var u = n(7326),
        l = n(4942),
        c = n(1143),
        s = n.n(c),
        d = new Map(),
        m = new Map()
      function f(e, t, n, o) {
        void 0 === n && (n = {}),
          s()(
            !d.has(e),
            "react-intersection-observer: Trying to observe %s, but it's already being observed by another instance.\nMake sure the `ref` is only used by a single <Observer /> instance.\n\n%s",
            e
          ),
          n.threshold || (n.threshold = 0)
        var r = n,
          a = r.root,
          i = r.rootMargin,
          u = r.threshold
        if (e && t) {
          var l = i ? u.toString() + '_' + i : u.toString()
          a && (l = o ? o + '_' + l : null)
          var c = l ? m.get(l) : null
          c || ((c = new IntersectionObserver(h, n)), l && m.set(l, c))
          var f = { callback: t, visible: !1, options: n, observerId: l, observer: l ? void 0 : c }
          return d.set(e, f), c.observe(e), f
        }
      }
      function p(e) {
        if (e) {
          var t = d.get(e)
          if (t) {
            var n = t.observerId,
              o = t.observer,
              r = n ? m.get(n) : o
            r && r.unobserve(e)
            var a = !1
            n &&
              d.forEach(function (t, o) {
                t && t.observerId === n && o !== e && (a = !0)
              }),
              r && !a && (r.disconnect(), m.delete(n)),
              d.delete(e)
          }
        }
      }
      function h(e) {
        e.forEach(function (e) {
          var t = e.isIntersecting,
            n = e.intersectionRatio,
            o = e.target,
            r = d.get(o)
          if (r && n >= 0) {
            var a = r.options,
              i = !1
            Array.isArray(a.threshold)
              ? (i = a.threshold.some(function (e) {
                  return r.visible ? n > e : n >= e
                }))
              : void 0 !== a.threshold && (i = r.visible ? n > a.threshold : n >= a.threshold),
              void 0 !== t && (i = i && t),
              (r.visible = i),
              r.callback(i, n)
          }
        })
      }
      var v = (function (e) {
        var t, n
        function c() {
          for (var t, n = arguments.length, o = new Array(n), r = 0; r < n; r++) o[r] = arguments[r]
          return (
            (t = e.call.apply(e, [this].concat(o)) || this),
            (0, l.Z)((0, u.Z)((0, u.Z)(t)), 'state', { inView: !1, intersectionRatio: 0 }),
            (0, l.Z)((0, u.Z)((0, u.Z)(t)), 'node', null),
            (0, l.Z)((0, u.Z)((0, u.Z)(t)), 'handleNode', function (e) {
              t.node && p(t.node), (t.node = e), t.observeNode()
            }),
            (0, l.Z)((0, u.Z)((0, u.Z)(t)), 'handleChange', function (e, n) {
              t.setState({ inView: e, intersectionRatio: n }),
                t.props.onChange && t.props.onChange(e, n)
            }),
            t
          )
        }
        ;(n = e),
          ((t = c).prototype = Object.create(n.prototype)),
          (t.prototype.constructor = t),
          (0, i.Z)(t, n)
        var s = c.prototype
        return (
          (s.componentDidMount = function () {
            0
          }),
          (s.componentDidUpdate = function (e, t) {
            ;(e.rootMargin === this.props.rootMargin &&
              e.root === this.props.root &&
              e.threshold === this.props.threshold) ||
              (p(this.node), this.observeNode()),
              t.inView !== this.state.inView &&
                this.state.inView &&
                this.props.triggerOnce &&
                (p(this.node), (this.node = null))
          }),
          (s.componentWillUnmount = function () {
            this.node && (p(this.node), (this.node = null))
          }),
          (s.observeNode = function () {
            if (this.node) {
              var e = this.props,
                t = e.threshold,
                n = e.root,
                o = e.rootMargin,
                r = e.rootId
              f(this.node, this.handleChange, { threshold: t, root: n, rootMargin: o }, r)
            }
          }),
          (s.render = function () {
            var e = this.props,
              t = e.children,
              n = e.render,
              i = e.tag,
              u =
                (e.triggerOnce,
                e.threshold,
                e.root,
                e.rootId,
                e.rootMargin,
                (0, a.Z)(e, [
                  'children',
                  'render',
                  'tag',
                  'triggerOnce',
                  'threshold',
                  'root',
                  'rootId',
                  'rootMargin',
                ])),
              l = this.state,
              c = l.inView,
              s = l.intersectionRatio,
              d = t || n
            return 'function' === typeof d
              ? d({ inView: c, intersectionRatio: s, ref: this.handleNode })
              : (0, o.createElement)(i || 'div', (0, r.Z)({ ref: this.handleNode }, u), t)
          }),
          c
        )
      })(o.Component)
      ;(0, l.Z)(v, 'defaultProps', { threshold: 0, triggerOnce: !1 })
      var D,
        g,
        E = v,
        y = n(3904),
        b = function (e, t) {
          return (b =
            Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array &&
              function (e, t) {
                e.__proto__ = t
              }) ||
            function (e, t) {
              for (var n in t) t.hasOwnProperty(n) && (e[n] = t[n])
            })(e, t)
        },
        F = function () {
          return (F =
            Object.assign ||
            function (e) {
              for (var t, n = 1, o = arguments.length; n < o; n++)
                for (var r in (t = arguments[n]))
                  Object.prototype.hasOwnProperty.call(t, r) && (e[r] = t[r])
              return e
            }).apply(this, arguments)
        }
      function C(e, t) {
        var n = {}
        for (var o in e)
          Object.prototype.hasOwnProperty.call(e, o) && t.indexOf(o) < 0 && (n[o] = e[o])
        if (null != e && 'function' == typeof Object.getOwnPropertySymbols) {
          var r = 0
          for (o = Object.getOwnPropertySymbols(e); r < o.length; r++)
            t.indexOf(o[r]) < 0 && (n[o[r]] = e[o[r]])
        }
        return n
      }
      ;((g = D || (D = {})).NotAsked = 'NotAsked'),
        (g.Loading = 'Loading'),
        (g.LoadSuccess = 'LoadSuccess'),
        (g.LoadError = 'LoadError')
      var w = (0, y.unionize)({
          NotAsked: {},
          Buffering: {},
          Loading: {},
          LoadSuccess: {},
          LoadError: (0, y.ofType)(),
        }),
        k = (0, y.unionize)({
          ViewChanged: (0, y.ofType)(),
          BufferingEnded: {},
          LoadSuccess: {},
          LoadError: (0, y.ofType)(),
        }),
        A = function (e, t) {
          return function (n) {
            var o = x(B(e, t))
            o.promise
              .then(function (e) {
                return n.update(k.LoadSuccess({}))
              })
              .catch(function (e) {
                e.isCanceled || n.update(k.LoadError({ msg: 'Failed to load' }))
              }),
              (n.promiseCache.loading = o)
          }
        },
        _ = function (e) {
          e.promiseCache.buffering.cancel()
        },
        O = (function (e) {
          function t(t) {
            var n = e.call(this, t) || this
            return (
              (n.promiseCache = {}),
              (n.initialState = w.NotAsked()),
              (n.state = n.initialState),
              (n.update = n.update.bind(n)),
              n
            )
          }
          return (
            (function (e, t) {
              function n() {
                this.constructor = e
              }
              b(e, t),
                (e.prototype =
                  null === t ? Object.create(t) : ((n.prototype = t.prototype), new n()))
            })(t, e),
            (t.reducer = function (e, t, n) {
              return k.match(e, {
                ViewChanged: function (e) {
                  return !0 === e.inView
                    ? n.src
                      ? w.match(t, {
                          NotAsked: function () {
                            return n.debounceDurationMs
                              ? {
                                  nextState: w.Buffering(),
                                  cmd:
                                    ((e = n.debounceDurationMs),
                                    function (t) {
                                      var n = x(N(e))
                                      n.promise
                                        .then(function () {
                                          return t.update(k.BufferingEnded())
                                        })
                                        .catch(function (e) {}),
                                        (t.promiseCache.buffering = n)
                                    }),
                                }
                              : { nextState: w.Loading(), cmd: A(n, n.experimentalDecode) }
                            var e
                          },
                          default: function () {
                            return { nextState: t }
                          },
                        })
                      : { nextState: w.LoadSuccess() }
                    : w.match(t, {
                        Buffering: function () {
                          return { nextState: w.NotAsked(), cmd: _ }
                        },
                        default: function () {
                          return { nextState: t }
                        },
                      })
                },
                BufferingEnded: function () {
                  return { nextState: w.Loading(), cmd: A(n, n.experimentalDecode) }
                },
                LoadSuccess: function () {
                  return { nextState: w.LoadSuccess() }
                },
                LoadError: function (e) {
                  return { nextState: w.LoadError(e) }
                },
              })
            }),
            (t.prototype.update = function (e) {
              var n = this,
                o = t.reducer(e, this.state, this.props),
                r = o.nextState,
                a = o.cmd
              this.props.debugActions &&
                (console.warn(
                  'You are running LazyImage with debugActions="true" in production. This might have performance implications.'
                ),
                console.log({ action: e, prevState: this.state, nextState: r })),
                this.setState(r, function () {
                  return a && a(n)
                })
            }),
            (t.prototype.componentWillUnmount = function () {
              this.promiseCache.loading && this.promiseCache.loading.cancel(),
                this.promiseCache.buffering && this.promiseCache.buffering.cancel(),
                (this.promiseCache = {})
            }),
            (t.prototype.render = function () {
              var e = this,
                t = this.props,
                n = t.children,
                r = t.loadEagerly,
                a = t.observerProps,
                i = C(t, [
                  'children',
                  'loadEagerly',
                  'observerProps',
                  'experimentalDecode',
                  'debounceDurationMs',
                  'debugActions',
                ])
              return r
                ? n({ imageState: w.LoadSuccess().tag, imageProps: i })
                : o.default.createElement(
                    E,
                    F({ rootMargin: '50px 0px', threshold: 0.01 }, a, {
                      onChange: function (t) {
                        return e.update(k.ViewChanged({ inView: t }))
                      },
                    }),
                    function (t) {
                      return n({
                        imageState: 'Buffering' === e.state.tag ? D.Loading : e.state.tag,
                        imageProps: i,
                        ref: t.ref,
                      })
                    }
                  )
            }),
            (t.displayName = 'LazyImageFull'),
            t
          )
        })(o.default.Component),
        B = function (e, t) {
          var n = e.src,
            o = e.srcSet,
            r = e.alt,
            a = e.sizes
          return (
            void 0 === t && (t = !1),
            new Promise(function (e, i) {
              var u = new Image()
              if (
                (o && (u.srcset = o),
                r && (u.alt = r),
                a && (u.sizes = a),
                (u.src = n),
                t && 'decode' in u)
              )
                return u
                  .decode()
                  .then(function (t) {
                    return e(t)
                  })
                  .catch(function (e) {
                    return i(e)
                  })
              ;(u.onload = e), (u.onerror = i)
            })
          )
        },
        N = function (e) {
          return new Promise(function (t) {
            return setTimeout(t, e)
          })
        },
        x = function (e) {
          var t = !1
          return {
            promise: new Promise(function (n, o) {
              e.then(function (e) {
                return t ? o({ isCanceled: !0 }) : n(e)
              }),
                e.catch(function (e) {
                  return o(t ? { isCanceled: !0 } : e)
                })
            }),
            cancel: function () {
              t = !0
            },
          }
        }
    },
    3904: function (e, t) {
      'use strict'
      var n =
        (this && this.__assign) ||
        function () {
          return (
            (n =
              Object.assign ||
              function (e) {
                for (var t, n = 1, o = arguments.length; n < o; n++)
                  for (var r in (t = arguments[n]))
                    Object.prototype.hasOwnProperty.call(t, r) && (e[r] = t[r])
                return e
              }),
            n.apply(this, arguments)
          )
        }
      function o(e, t) {
        var o = t || {},
          r = o.value,
          a = void 0 === r ? void 0 : r,
          i = o.tag,
          u = void 0 === i ? 'tag' : i,
          l = {},
          c = function (e) {
            l[e] = function (t) {
              var o, r
              return (
                void 0 === t && (t = {}),
                a ? (((o = {})[u] = e), (o[a] = t), o) : n({}, t, (((r = {})[u] = e), r))
              )
            }
          }
        for (var s in e) c(s)
        var d = {},
          m = function (e) {
            d[e] = function (t) {
              return t[u] === e
            }
          }
        for (var s in e) m(s)
        function f(e, t, n) {
          void 0 === n && (n = t.default)
          var o = t[e[u]]
          return o ? o(a ? e[a] : e) : n(e)
        }
        var p = function (e, t) {
            return t
              ? f(e, t)
              : function (t) {
                  return f(t, e)
                }
          },
          h = function (e) {
            return e
          },
          v = {},
          D = function (e) {
            var t
            v[e] = p(
              (((t = {})[e] = function (e) {
                return e
              }),
              (t.default = function (t) {
                throw new Error('Attempted to cast ' + t[u] + ' as ' + e)
              }),
              t)
            )
          }
        for (var g in e) D(g)
        return Object.assign(
          {
            is: d,
            as: v,
            match: p,
            transform: function (e, t) {
              return t
                ? f(e, t, h)
                : function (t) {
                    return f(t, e, h)
                  }
            },
            _Record: e,
          },
          l
        )
      }
      Object.defineProperty(t, '__esModule', { value: !0 }),
        (t.unionize = o),
        (t.ofType = function () {}),
        (t.default = o)
    },
    3194: function (e, t, n) {
      e.exports = n(8773)
    },
    8773: function (e, t, n) {
      'use strict'
      t.getMDXComponent = function (e, t) {
        return l(e, t).default
      }
      var o = u(n(1720)),
        r = u(n(1032)),
        a = u(n(1720))
      function i(e) {
        if ('function' !== typeof WeakMap) return null
        var t = new WeakMap(),
          n = new WeakMap()
        return (i = function (e) {
          return e ? n : t
        })(e)
      }
      function u(e, t) {
        if (!t && e && e.__esModule) return e
        if (null === e || ('object' !== typeof e && 'function' !== typeof e)) return { default: e }
        var n = i(t)
        if (n && n.has(e)) return n.get(e)
        var o = {},
          r = Object.defineProperty && Object.getOwnPropertyDescriptor
        for (var a in e)
          if ('default' !== a && Object.prototype.hasOwnProperty.call(e, a)) {
            var u = r ? Object.getOwnPropertyDescriptor(e, a) : null
            u && (u.get || u.set) ? Object.defineProperty(o, a, u) : (o[a] = e[a])
          }
        return (o.default = e), n && n.set(e, o), o
      }
      function l(e, t) {
        const n = { React: o, ReactDOM: a, _jsx_runtime: r, ...t }
        return new Function(...Object.keys(n), e)(...Object.values(n))
      }
    },
    7326: function (e, t, n) {
      'use strict'
      function o(e) {
        if (void 0 === e)
          throw new ReferenceError("this hasn't been initialised - super() hasn't been called")
        return e
      }
      n.d(t, {
        Z: function () {
          return o
        },
      })
    },
    4942: function (e, t, n) {
      'use strict'
      function o(e, t, n) {
        return (
          t in e
            ? Object.defineProperty(e, t, {
                value: n,
                enumerable: !0,
                configurable: !0,
                writable: !0,
              })
            : (e[t] = n),
          e
        )
      }
      n.d(t, {
        Z: function () {
          return o
        },
      })
    },
    7462: function (e, t, n) {
      'use strict'
      function o() {
        return (
          (o =
            Object.assign ||
            function (e) {
              for (var t = 1; t < arguments.length; t++) {
                var n = arguments[t]
                for (var o in n) Object.prototype.hasOwnProperty.call(n, o) && (e[o] = n[o])
              }
              return e
            }),
          o.apply(this, arguments)
        )
      }
      n.d(t, {
        Z: function () {
          return o
        },
      })
    },
    3366: function (e, t, n) {
      'use strict'
      function o(e, t) {
        if (null == e) return {}
        var n,
          o,
          r = {},
          a = Object.keys(e)
        for (o = 0; o < a.length; o++) (n = a[o]), t.indexOf(n) >= 0 || (r[n] = e[n])
        return r
      }
      n.d(t, {
        Z: function () {
          return o
        },
      })
    },
    9611: function (e, t, n) {
      'use strict'
      function o(e, t) {
        return (
          (o =
            Object.setPrototypeOf ||
            function (e, t) {
              return (e.__proto__ = t), e
            }),
          o(e, t)
        )
      }
      n.d(t, {
        Z: function () {
          return o
        },
      })
    },
    8110: function (e, t, n) {
      'use strict'
      function o(e, { lenient: t = !1 } = {}) {
        if ('string' !== typeof e) throw new TypeError('Expected a string')
        if ((e = e.trim()).includes(' ')) return !1
        try {
          return new URL(e), !0
        } catch {
          return !!t && o(`https://${e}`)
        }
      }
      n.d(t, {
        Z: function () {
          return o
        },
      })
    },
    9253: function (e, t, n) {
      'use strict'
      n.d(t, {
        p6: function () {
          return J
        },
        c8: function () {
          return ee
        },
        c5: function () {
          return T
        },
        Ck: function () {
          return H
        },
        cj: function () {
          return W
        },
        Ho: function () {
          return $
        },
        Co: function () {
          return U
        },
        Kl: function () {
          return Y
        },
        Ru: function () {
          return Z
        },
        pz: function () {
          return R
        },
        FB: function () {
          return M
        },
        D5: function () {
          return X
        },
        q5: function () {
          return G
        },
        Gw: function () {
          return Q
        },
      })
      n(6729)
      Error
      Error
      new WeakMap()
      var o,
        r,
        a,
        i,
        u,
        l,
        c,
        s,
        d,
        m,
        f,
        p,
        h,
        v,
        D,
        g,
        E,
        y,
        b,
        F,
        C,
        w = function (e, t, n, o, r) {
          if ('m' === o) throw new TypeError('Private method is not writable')
          if ('a' === o && !r) throw new TypeError('Private accessor was defined without a setter')
          if ('function' === typeof t ? e !== t || !r : !t.has(e))
            throw new TypeError(
              'Cannot write private member to an object whose class did not declare it'
            )
          return 'a' === o ? r.call(e, n) : r ? (r.value = n) : t.set(e, n), n
        },
        k = function (e, t, n, o) {
          if ('a' === n && !o) throw new TypeError('Private accessor was defined without a getter')
          if ('function' === typeof t ? e !== t || !o : !t.has(e))
            throw new TypeError(
              'Cannot read private member from an object whose class did not declare it'
            )
          return 'm' === n ? o : 'a' === n ? o.call(e) : o ? o.value : t.get(e)
        }
      class A extends Error {}
      ;(r = new WeakMap()),
        (a = new WeakMap()),
        (i = new WeakMap()),
        (u = new WeakMap()),
        (l = new WeakMap()),
        (c = new WeakMap()),
        (s = new WeakMap()),
        (d = new WeakMap()),
        (m = new WeakMap()),
        new WeakMap(),
        (f = new WeakMap()),
        (p = new WeakMap()),
        (h = new WeakMap()),
        new WeakMap(),
        (o = new WeakSet()),
        (v = function () {
          return k(this, a, 'f') || k(this, i, 'f') < k(this, u, 'f')
        }),
        (D = function () {
          return k(this, f, 'f') < k(this, p, 'f')
        }),
        (g = function () {
          k(this, o, 'm', F).call(this), k(this, o, 'm', b).call(this), w(this, d, void 0, 'f')
        }),
        (E = function () {
          const e = Date.now()
          if (void 0 === k(this, s, 'f')) {
            const t = k(this, c, 'f') - e
            if (!(t < 0))
              return (
                void 0 === k(this, d, 'f') &&
                  w(
                    this,
                    d,
                    setTimeout(() => {
                      k(this, o, 'm', g).call(this)
                    }, t),
                    'f'
                  ),
                !0
              )
            w(this, i, k(this, r, 'f') ? k(this, f, 'f') : 0, 'f')
          }
          return !1
        }),
        (y = function () {
          if (0 === k(this, m, 'f').size)
            return (
              k(this, s, 'f') && clearInterval(k(this, s, 'f')),
              w(this, s, void 0, 'f'),
              this.emit('empty'),
              0 === k(this, f, 'f') && this.emit('idle'),
              !1
            )
          if (!k(this, h, 'f')) {
            const e = !k(this, o, 'a', E)
            if (k(this, o, 'a', v) && k(this, o, 'a', D)) {
              const t = k(this, m, 'f').dequeue()
              return !!t && (this.emit('active'), t(), e && k(this, o, 'm', b).call(this), !0)
            }
          }
          return !1
        }),
        (b = function () {
          k(this, a, 'f') ||
            void 0 !== k(this, s, 'f') ||
            (w(
              this,
              s,
              setInterval(() => {
                k(this, o, 'm', F).call(this)
              }, k(this, l, 'f')),
              'f'
            ),
            w(this, c, Date.now() + k(this, l, 'f'), 'f'))
        }),
        (F = function () {
          0 === k(this, i, 'f') &&
            0 === k(this, f, 'f') &&
            k(this, s, 'f') &&
            (clearInterval(k(this, s, 'f')), w(this, s, void 0, 'f')),
            w(this, i, k(this, r, 'f') ? k(this, f, 'f') : 0, 'f'),
            k(this, o, 'm', C).call(this)
        }),
        (C = function () {
          for (; k(this, o, 'm', y).call(this); );
        })
      const _ = (e, t, n, o) => {
          if ('length' === n || 'prototype' === n) return
          if ('arguments' === n || 'caller' === n) return
          const r = Object.getOwnPropertyDescriptor(e, n),
            a = Object.getOwnPropertyDescriptor(t, n)
          ;(!O(r, a) && o) || Object.defineProperty(e, n, a)
        },
        O = function (e, t) {
          return (
            void 0 === e ||
            e.configurable ||
            (e.writable === t.writable &&
              e.enumerable === t.enumerable &&
              e.configurable === t.configurable &&
              (e.writable || e.value === t.value))
          )
        },
        B = (e, t) => `/* Wrapped ${e}*/\n${t}`,
        N = Object.getOwnPropertyDescriptor(Function.prototype, 'toString'),
        x = Object.getOwnPropertyDescriptor(Function.prototype.toString, 'name')
      function P(e, t, { ignoreNonConfigurable: n = !1 } = {}) {
        const { name: o } = e
        for (const r of Reflect.ownKeys(t)) _(e, t, r, n)
        return (
          ((e, t) => {
            const n = Object.getPrototypeOf(t)
            n !== Object.getPrototypeOf(e) && Object.setPrototypeOf(e, n)
          })(e, t),
          ((e, t, n) => {
            const o = '' === n ? '' : `with ${n.trim()}() `,
              r = B.bind(null, o, t.toString())
            Object.defineProperty(r, 'name', x),
              Object.defineProperty(e, 'toString', { ...N, value: r })
          })(e, t, o),
          e
        )
      }
      var j = n(8918)
      const L = new WeakMap()
      function z(e, { cacheKey: t, cache: n = new Map(), maxAge: o } = {}) {
        'number' === typeof o && j(n)
        const r = function (...r) {
          const a = t ? t(r) : r[0],
            i = n.get(a)
          if (i) return i.data
          const u = e.apply(this, r)
          return n.set(a, { data: u, maxAge: o ? Date.now() + o : Number.POSITIVE_INFINITY }), u
        }
        return P(r, e, { ignoreNonConfigurable: !0 }), L.set(r, n), r
      }
      const S = (e, t) => t.some((t) => (t instanceof RegExp ? t.test(e) : t === e))
      function I(e, t) {
        if (
          ((t = {
            defaultProtocol: 'http:',
            normalizeProtocol: !0,
            forceHttp: !1,
            forceHttps: !1,
            stripAuthentication: !0,
            stripHash: !1,
            stripTextFragment: !0,
            stripWWW: !0,
            removeQueryParameters: [/^utm_\w+/i],
            removeTrailingSlash: !0,
            removeSingleSlash: !0,
            removeDirectoryIndex: !1,
            removeExplicitPort: !1,
            sortQueryParameters: !0,
            ...t,
          }),
          (e = e.trim()),
          /^data:/i.test(e))
        )
          return ((e, { stripHash: t }) => {
            const n = /^data:(?<type>[^,]*?),(?<data>[^#]*?)(?:#(?<hash>.*))?$/.exec(e)
            if (!n) throw new Error(`Invalid URL: ${e}`)
            let { type: o, data: r, hash: a } = n.groups
            const i = o.split(';')
            a = t ? '' : a
            let u = !1
            'base64' === i[i.length - 1] && (i.pop(), (u = !0))
            const l = (i.shift() || '').toLowerCase(),
              c = [
                ...i
                  .map((e) => {
                    let [t, n = ''] = e.split('=').map((e) => e.trim())
                    return 'charset' === t && ((n = n.toLowerCase()), 'us-ascii' === n)
                      ? ''
                      : `${t}${n ? `=${n}` : ''}`
                  })
                  .filter(Boolean),
              ]
            return (
              u && c.push('base64'),
              (c.length > 0 || (l && 'text/plain' !== l)) && c.unshift(l),
              `data:${c.join(';')},${u ? r.trim() : r}${a ? `#${a}` : ''}`
            )
          })(e, t)
        if (/^view-source:/i.test(e))
          throw new Error('`view-source:` is not supported as it is a non-standard protocol')
        const n = e.startsWith('//')
        ;(!n && /^\.*\//.test(e)) || (e = e.replace(/^(?!(?:\w+:)?\/\/)|^\/\//, t.defaultProtocol))
        const o = new URL(e)
        if (t.forceHttp && t.forceHttps)
          throw new Error('The `forceHttp` and `forceHttps` options cannot be used together')
        if (
          (t.forceHttp && 'https:' === o.protocol && (o.protocol = 'http:'),
          t.forceHttps && 'http:' === o.protocol && (o.protocol = 'https:'),
          t.stripAuthentication && ((o.username = ''), (o.password = '')),
          t.stripHash
            ? (o.hash = '')
            : t.stripTextFragment && (o.hash = o.hash.replace(/#?:~:text.*?$/i, '')),
          o.pathname)
        ) {
          const e = /\b[a-z][a-z\d+\-.]{1,50}:\/\//g
          let t = 0,
            n = ''
          for (;;) {
            const r = e.exec(o.pathname)
            if (!r) break
            const a = r[0],
              i = r.index
            ;(n += o.pathname.slice(t, i).replace(/\/{2,}/g, '/')), (n += a), (t = i + a.length)
          }
          ;(n += o.pathname.slice(t, o.pathname.length).replace(/\/{2,}/g, '/')), (o.pathname = n)
        }
        if (o.pathname)
          try {
            o.pathname = decodeURI(o.pathname)
          } catch {}
        if (
          (!0 === t.removeDirectoryIndex && (t.removeDirectoryIndex = [/^index\.[a-z]+$/]),
          Array.isArray(t.removeDirectoryIndex) && t.removeDirectoryIndex.length > 0)
        ) {
          let e = o.pathname.split('/')
          const n = e[e.length - 1]
          S(n, t.removeDirectoryIndex) &&
            ((e = e.slice(0, -1)), (o.pathname = e.slice(1).join('/') + '/'))
        }
        if (
          (o.hostname &&
            ((o.hostname = o.hostname.replace(/\.$/, '')),
            t.stripWWW &&
              /^www\.(?!www\.)[a-z\-\d]{1,63}\.[a-z.\-\d]{2,63}$/.test(o.hostname) &&
              (o.hostname = o.hostname.replace(/^www\./, ''))),
          Array.isArray(t.removeQueryParameters))
        )
          for (const a of [...o.searchParams.keys()])
            S(a, t.removeQueryParameters) && o.searchParams.delete(a)
        if (
          (Array.isArray(t.keepQueryParameters) ||
            !0 !== t.removeQueryParameters ||
            (o.search = ''),
          Array.isArray(t.keepQueryParameters) && t.keepQueryParameters.length > 0)
        )
          for (const a of [...o.searchParams.keys()])
            S(a, t.keepQueryParameters) || o.searchParams.delete(a)
        if (t.sortQueryParameters) {
          o.searchParams.sort()
          try {
            o.search = decodeURIComponent(o.search)
          } catch {}
        }
        t.removeTrailingSlash && (o.pathname = o.pathname.replace(/\/$/, '')),
          t.removeExplicitPort && o.port && (o.port = '')
        const r = e
        return (
          (e = o.toString()),
          t.removeSingleSlash ||
            '/' !== o.pathname ||
            r.endsWith('/') ||
            '' !== o.hash ||
            (e = e.replace(/\/$/, '')),
          (t.removeTrailingSlash || '/' === o.pathname) &&
            '' === o.hash &&
            t.removeSingleSlash &&
            (e = e.replace(/\/$/, '')),
          n && !t.normalizeProtocol && (e = e.replace(/^http:\/\//, '//')),
          t.stripProtocol && (e = e.replace(/^(?:https?:)?\/\//, '')),
          e
        )
      }
      Object.defineProperty,
        Object.defineProperties,
        Object.getOwnPropertyDescriptors,
        Object.getOwnPropertySymbols,
        Object.prototype.hasOwnProperty,
        Object.prototype.propertyIsEnumerable
      var M = (e) => {
        var t
        return e
          ? Array.isArray(e)
            ? null !=
              (t =
                null == e
                  ? void 0
                  : e.reduce(
                      (e, t) => e + ('\u204d' !== t[0] && '\u2023' !== t[0] ? t[0] : ''),
                      ''
                    ))
              ? t
              : ''
            : e
          : ''
      }
      function T(e, t) {
        var n, o, r, a, i, u, l
        let c =
          e.collection_id ||
          (null == (o = null == (n = e.format) ? void 0 : n.collection_pointer) ? void 0 : o.id)
        if (c) return c
        let s = null == (r = null == e ? void 0 : e.view_ids) ? void 0 : r[0]
        if (s) {
          let e = null == (i = null == (a = t.collection_view) ? void 0 : a[s]) ? void 0 : i.value
          if (e)
            return null == (l = null == (u = e.format) ? void 0 : u.collection_pointer)
              ? void 0
              : l.id
        }
        return null
      }
      function $(e, t) {
        var n, o
        if (null != (n = e.properties) && n.title) return M(e.properties.title)
        if ('collection_view_page' === e.type || 'collection_view' === e.type) {
          let n = T(e, t)
          if (n) {
            let e = null == (o = t.collection[n]) ? void 0 : o.value
            if (e) return M(e.name)
          }
        }
        return ''
      }
      function H(e, t) {
        var n, o, r
        if (null != (n = e.format) && n.page_icon)
          return null == (o = e.format) ? void 0 : o.page_icon
        if ('collection_view_page' === e.type || 'collection_view' === e.type) {
          let n = T(e, t)
          if (n) {
            let e = null == (r = t.collection[n]) ? void 0 : r.value
            if (e) return e.icon
          }
        }
        return null
      }
      function R(e) {
        var t
        let n = null == (t = e.block[Object.keys(e.block)[0]]) ? void 0 : t.value
        return n ? $(n, e) : null
      }
      var U = (e) => {
          if (e && Array.isArray(e)) {
            if ('d' === e[0]) return e[1]
            for (let t of e) {
              let e = U(t)
              if (e) return e
            }
          }
          return null
        },
        W = (e, t, { inclusive: n = !1 } = {}) => {
          var o, r
          let a = e
          for (; null != a; ) {
            if (n && 'page' === (null == a ? void 0 : a.type)) return a
            let e = a.parent_id,
              i = a.parent_table
            if (!e) break
            if ('collection' === i) a = null == (o = t.collection[e]) ? void 0 : o.value
            else if (
              ((a = null == (r = t.block[e]) ? void 0 : r.value),
              'page' === (null == a ? void 0 : a.type))
            )
              return a
          }
          return null
        },
        V = { header: 0, sub_header: 1, sub_sub_header: 2 },
        Z = (e, t) => {
          var n
          let o = (null != (n = e.content) ? n : [])
              .map((e) => {
                var n, o
                let r = null == (n = t.block[e]) ? void 0 : n.value
                if (r) {
                  let { type: t } = r
                  if ('header' === t || 'sub_header' === t || 'sub_sub_header' === t)
                    return {
                      id: e,
                      type: t,
                      text: M(null == (o = r.properties) ? void 0 : o.title),
                      indentLevel: V[t],
                    }
                }
                return null
              })
              .filter(Boolean),
            r = [{ actual: -1, effective: -1 }]
          for (let a of o) {
            let { indentLevel: e } = a,
              t = e
            for (;;) {
              let e = r[r.length - 1],
                { actual: n, effective: o } = e
              if (t > n) (a.indentLevel = o + 1), r.push({ actual: t, effective: a.indentLevel })
              else {
                if (t === n) {
                  a.indentLevel = o
                  break
                }
                r.pop()
              }
            }
          }
          return o
        },
        q = /\b([a-f0-9]{32})\b/,
        K = /\b([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})\b/,
        G = (e = '', { uuid: t = !0 } = {}) => {
          if (!e) return null
          let n = (e = e.split('?')[0]).match(q)
          if (n)
            return t
              ? ((e = '') =>
                  `${e.substr(0, 8)}-${e.substr(8, 4)}-${e.substr(12, 4)}-${e.substr(
                    16,
                    4
                  )}-${e.substr(20)}`)(n[1])
              : n[1]
          let o = e.match(K)
          return o ? (t ? o[1] : o[1].replace(/-/g, '')) : null
        },
        Q = (e) => e.replace(/-/g, '')
      var Y = (e, t) => {
          var n
          let o = e.block,
            r = [],
            a = t
          for (;;) {
            let i = null == (n = o[a]) ? void 0 : n.value
            if (!i) break
            let u = $(i, e),
              l = H(i, e)
            if (!u && !l) break
            r.push({ block: i, active: a === t, pageId: a, title: u, icon: l })
            let c = W(i, e),
              s = null == c ? void 0 : c.id
            if (!s) break
            a = s
          }
          return r.reverse(), r
        },
        X = z((e) => {
          if (!e) return ''
          try {
            if (e.startsWith('https://www.notion.so/image/')) {
              let t = new URL(e),
                n = decodeURIComponent(t.pathname.substr(7)),
                o = X(n)
              ;(t.pathname = `/image/${encodeURIComponent(o)}`), (e = t.toString())
            }
            return I(e, {
              stripProtocol: !0,
              stripWWW: !0,
              stripHash: !0,
              stripTextFragment: !0,
              removeQueryParameters: !0,
            })
          } catch (t) {
            return ''
          }
        })
      var J = (e, { month: t = 'short' } = {}) => {
          let n = new Date(e)
          return `${n.toLocaleString('en-US', {
            month: t,
          })} ${n.getUTCDate()}, ${n.getUTCFullYear()}`
        },
        ee = (e) => {
          let t = `${e.start_time || ''} ${e.start_date} ${e.time_zone || ''}`
          return J(t)
        }
    },
    4793: function (e, t, n) {
      'use strict'
      n.d(t, {
        cp: function () {
          return Pe
        },
      })
      var o = n(1720),
        r =
          Object.assign ||
          function (e) {
            for (var t = 1; t < arguments.length; t++) {
              var n = arguments[t]
              for (var o in n) Object.prototype.hasOwnProperty.call(n, o) && (e[o] = n[o])
            }
            return e
          },
        a = function (e) {
          return 'IMG' === e.tagName
        },
        i = function (e) {
          return e && 1 === e.nodeType
        },
        u = function (e) {
          return '.svg' === (e.currentSrc || e.src).substr(-4).toLowerCase()
        },
        l = function (e) {
          try {
            return Array.isArray(e)
              ? e.filter(a)
              : (function (e) {
                  return NodeList.prototype.isPrototypeOf(e)
                })(e)
              ? [].slice.call(e).filter(a)
              : i(e)
              ? [e].filter(a)
              : 'string' === typeof e
              ? [].slice.call(document.querySelectorAll(e)).filter(a)
              : []
          } catch (t) {
            throw new TypeError(
              'The provided selector is invalid.\nExpects a CSS selector, a Node element, a NodeList or an array.\nSee: https://github.com/francoischalifour/medium-zoom'
            )
          }
        },
        c = function (e) {
          var t = document.createElement('div')
          return t.classList.add('medium-zoom-overlay'), (t.style.background = e), t
        },
        s = function (e) {
          var t = e.getBoundingClientRect(),
            n = t.top,
            o = t.left,
            r = t.width,
            a = t.height,
            i = e.cloneNode(),
            u =
              window.pageYOffset ||
              document.documentElement.scrollTop ||
              document.body.scrollTop ||
              0,
            l =
              window.pageXOffset ||
              document.documentElement.scrollLeft ||
              document.body.scrollLeft ||
              0
          return (
            i.removeAttribute('id'),
            (i.style.position = 'absolute'),
            (i.style.top = n + u + 'px'),
            (i.style.left = o + l + 'px'),
            (i.style.width = r + 'px'),
            (i.style.height = a + 'px'),
            (i.style.transform = ''),
            i
          )
        },
        d = function (e, t) {
          var n = r({ bubbles: !1, cancelable: !1, detail: void 0 }, t)
          if ('function' === typeof window.CustomEvent) return new CustomEvent(e, n)
          var o = document.createEvent('CustomEvent')
          return o.initCustomEvent(e, n.bubbles, n.cancelable, n.detail), o
        }
      !(function (e, t) {
        void 0 === t && (t = {})
        var n = t.insertAt
        if (e && 'undefined' !== typeof document) {
          var o = document.head || document.getElementsByTagName('head')[0],
            r = document.createElement('style')
          ;(r.type = 'text/css'),
            'top' === n && o.firstChild ? o.insertBefore(r, o.firstChild) : o.appendChild(r),
            r.styleSheet ? (r.styleSheet.cssText = e) : r.appendChild(document.createTextNode(e))
        }
      })(
        '.medium-zoom-overlay{position:fixed;top:0;right:0;bottom:0;left:0;opacity:0;transition:opacity .3s;will-change:opacity}.medium-zoom--opened .medium-zoom-overlay{cursor:pointer;cursor:zoom-out;opacity:1}.medium-zoom-image{cursor:pointer;cursor:zoom-in;transition:transform .3s cubic-bezier(.2,0,.2,1)!important}.medium-zoom-image--hidden{visibility:hidden}.medium-zoom-image--opened{position:relative;cursor:pointer;cursor:zoom-out;will-change:transform}'
      )
      var m,
        f,
        p = function e(t) {
          var n = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : {},
            o =
              window.Promise ||
              function (e) {
                function t() {}
                e(t, t)
              },
            a = function (e) {
              var t = e.target
              t !== P ? -1 !== A.indexOf(t) && F({ target: t }) : b()
            },
            m = function () {
              if (!O && x.original) {
                var e =
                  window.pageYOffset ||
                  document.documentElement.scrollTop ||
                  document.body.scrollTop ||
                  0
                Math.abs(B - e) > N.scrollOffset && setTimeout(b, 150)
              }
            },
            f = function (e) {
              var t = e.key || e.keyCode
              ;('Escape' !== t && 'Esc' !== t && 27 !== t) || b()
            },
            p = function () {
              var e = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {},
                t = e
              if (
                (e.background && (P.style.background = e.background),
                e.container &&
                  e.container instanceof Object &&
                  (t.container = r({}, N.container, e.container)),
                e.template)
              ) {
                var n = i(e.template) ? e.template : document.querySelector(e.template)
                t.template = n
              }
              return (
                (N = r({}, N, t)),
                A.forEach(function (e) {
                  e.dispatchEvent(d('medium-zoom:update', { detail: { zoom: j } }))
                }),
                j
              )
            },
            h = function () {
              var t = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {}
              return e(r({}, N, t))
            },
            v = function () {
              for (var e = arguments.length, t = Array(e), n = 0; n < e; n++) t[n] = arguments[n]
              var o = t.reduce(function (e, t) {
                return [].concat(e, l(t))
              }, [])
              return (
                o
                  .filter(function (e) {
                    return -1 === A.indexOf(e)
                  })
                  .forEach(function (e) {
                    A.push(e), e.classList.add('medium-zoom-image')
                  }),
                _.forEach(function (e) {
                  var t = e.type,
                    n = e.listener,
                    r = e.options
                  o.forEach(function (e) {
                    e.addEventListener(t, n, r)
                  })
                }),
                j
              )
            },
            D = function () {
              for (var e = arguments.length, t = Array(e), n = 0; n < e; n++) t[n] = arguments[n]
              x.zoomed && b()
              var o =
                t.length > 0
                  ? t.reduce(function (e, t) {
                      return [].concat(e, l(t))
                    }, [])
                  : A
              return (
                o.forEach(function (e) {
                  e.classList.remove('medium-zoom-image'),
                    e.dispatchEvent(d('medium-zoom:detach', { detail: { zoom: j } }))
                }),
                (A = A.filter(function (e) {
                  return -1 === o.indexOf(e)
                })),
                j
              )
            },
            g = function (e, t) {
              var n = arguments.length > 2 && void 0 !== arguments[2] ? arguments[2] : {}
              return (
                A.forEach(function (o) {
                  o.addEventListener('medium-zoom:' + e, t, n)
                }),
                _.push({ type: 'medium-zoom:' + e, listener: t, options: n }),
                j
              )
            },
            E = function (e, t) {
              var n = arguments.length > 2 && void 0 !== arguments[2] ? arguments[2] : {}
              return (
                A.forEach(function (o) {
                  o.removeEventListener('medium-zoom:' + e, t, n)
                }),
                (_ = _.filter(function (n) {
                  return !(n.type === 'medium-zoom:' + e && n.listener.toString() === t.toString())
                })),
                j
              )
            },
            y = function () {
              var e = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {},
                t = e.target,
                n = function () {
                  var e = {
                      width: document.documentElement.clientWidth,
                      height: document.documentElement.clientHeight,
                      left: 0,
                      top: 0,
                      right: 0,
                      bottom: 0,
                    },
                    t = void 0,
                    n = void 0
                  if (N.container)
                    if (N.container instanceof Object)
                      (t = (e = r({}, e, N.container)).width - e.left - e.right - 2 * N.margin),
                        (n = e.height - e.top - e.bottom - 2 * N.margin)
                    else {
                      var o = (
                          i(N.container) ? N.container : document.querySelector(N.container)
                        ).getBoundingClientRect(),
                        a = o.width,
                        l = o.height,
                        c = o.left,
                        s = o.top
                      e = r({}, e, { width: a, height: l, left: c, top: s })
                    }
                  ;(t = t || e.width - 2 * N.margin), (n = n || e.height - 2 * N.margin)
                  var d = N.minZoomScale || 1,
                    m = x.zoomedHd || x.original,
                    f = u(m) ? t : m.naturalWidth || t,
                    p = u(m) ? n : m.naturalHeight || n,
                    h = m.getBoundingClientRect(),
                    v = h.top,
                    D = h.left,
                    g = h.width,
                    E = h.height,
                    y = Math.min(Math.max(f * d, g), t) / g,
                    b = Math.min(Math.max(p * d, E), n) / E,
                    F = Math.min(y, b),
                    C =
                      'scale(' +
                      F +
                      ') translate3d(' +
                      ((t - g) / 2 - D + N.margin + e.left) / F +
                      'px, ' +
                      ((n - E) / 2 - v + N.margin + e.top) / F +
                      'px, 0)'
                  ;(x.zoomed.style.transform = C), x.zoomedHd && (x.zoomedHd.style.transform = C)
                }
              return new o(function (e) {
                if (t && -1 === A.indexOf(t)) e(j)
                else {
                  if (x.zoomed) e(j)
                  else {
                    if (t) x.original = t
                    else {
                      if (!(A.length > 0)) return void e(j)
                      var o = A
                      x.original = o[0]
                    }
                    if (
                      (x.original.dispatchEvent(d('medium-zoom:open', { detail: { zoom: j } })),
                      (B =
                        window.pageYOffset ||
                        document.documentElement.scrollTop ||
                        document.body.scrollTop ||
                        0),
                      (O = !0),
                      (x.zoomed = s(x.original)),
                      document.body.appendChild(P),
                      N.template)
                    ) {
                      var r = i(N.template) ? N.template : document.querySelector(N.template)
                      ;(x.template = document.createElement('div')),
                        x.template.appendChild(r.content.cloneNode(!0)),
                        document.body.appendChild(x.template)
                    }
                    if (
                      (document.body.appendChild(x.zoomed),
                      window.requestAnimationFrame(function () {
                        document.body.classList.add('medium-zoom--opened')
                      }),
                      x.original.classList.add('medium-zoom-image--hidden'),
                      x.zoomed.classList.add('medium-zoom-image--opened'),
                      x.zoomed.addEventListener('click', b),
                      x.zoomed.addEventListener('transitionend', function t() {
                        ;(O = !1),
                          x.zoomed.removeEventListener('transitionend', t),
                          x.original.dispatchEvent(
                            d('medium-zoom:opened', { detail: { zoom: j } })
                          ),
                          e(j)
                      }),
                      x.original.getAttribute('data-zoom-src'))
                    ) {
                      ;(x.zoomedHd = x.zoomed.cloneNode()),
                        x.zoomedHd.removeAttribute('srcset'),
                        x.zoomedHd.removeAttribute('sizes'),
                        x.zoomedHd.removeAttribute('loading'),
                        (x.zoomedHd.src = x.zoomed.getAttribute('data-zoom-src')),
                        (x.zoomedHd.onerror = function () {
                          clearInterval(a),
                            console.warn('Unable to reach the zoom image target ' + x.zoomedHd.src),
                            (x.zoomedHd = null),
                            n()
                        })
                      var a = setInterval(function () {
                        x.zoomedHd.complete &&
                          (clearInterval(a),
                          x.zoomedHd.classList.add('medium-zoom-image--opened'),
                          x.zoomedHd.addEventListener('click', b),
                          document.body.appendChild(x.zoomedHd),
                          n())
                      }, 10)
                    } else if (x.original.hasAttribute('srcset')) {
                      ;(x.zoomedHd = x.zoomed.cloneNode()),
                        x.zoomedHd.removeAttribute('sizes'),
                        x.zoomedHd.removeAttribute('loading')
                      var u = x.zoomedHd.addEventListener('load', function () {
                        x.zoomedHd.removeEventListener('load', u),
                          x.zoomedHd.classList.add('medium-zoom-image--opened'),
                          x.zoomedHd.addEventListener('click', b),
                          document.body.appendChild(x.zoomedHd),
                          n()
                      })
                    } else n()
                  }
                }
              })
            },
            b = function () {
              return new o(function (e) {
                if (!O && x.original) {
                  ;(O = !0),
                    document.body.classList.remove('medium-zoom--opened'),
                    (x.zoomed.style.transform = ''),
                    x.zoomedHd && (x.zoomedHd.style.transform = ''),
                    x.template &&
                      ((x.template.style.transition = 'opacity 150ms'),
                      (x.template.style.opacity = 0)),
                    x.original.dispatchEvent(d('medium-zoom:close', { detail: { zoom: j } })),
                    x.zoomed.addEventListener('transitionend', function t() {
                      x.original.classList.remove('medium-zoom-image--hidden'),
                        document.body.removeChild(x.zoomed),
                        x.zoomedHd && document.body.removeChild(x.zoomedHd),
                        document.body.removeChild(P),
                        x.zoomed.classList.remove('medium-zoom-image--opened'),
                        x.template && document.body.removeChild(x.template),
                        (O = !1),
                        x.zoomed.removeEventListener('transitionend', t),
                        x.original.dispatchEvent(d('medium-zoom:closed', { detail: { zoom: j } })),
                        (x.original = null),
                        (x.zoomed = null),
                        (x.zoomedHd = null),
                        (x.template = null),
                        e(j)
                    })
                } else e(j)
              })
            },
            F = function () {
              var e = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {},
                t = e.target
              return x.original ? b() : y({ target: t })
            },
            C = function () {
              return N
            },
            w = function () {
              return A
            },
            k = function () {
              return x.original
            },
            A = [],
            _ = [],
            O = !1,
            B = 0,
            N = n,
            x = { original: null, zoomed: null, zoomedHd: null, template: null }
          '[object Object]' === Object.prototype.toString.call(t)
            ? (N = t)
            : (t || 'string' === typeof t) && v(t),
            (N = r(
              {
                margin: 0,
                background: '#fff',
                scrollOffset: 40,
                minZoomScale: 1,
                container: null,
                template: null,
              },
              N
            ))
          var P = c(N.background)
          document.addEventListener('click', a),
            document.addEventListener('keyup', f),
            document.addEventListener('scroll', m),
            window.addEventListener('resize', b)
          var j = {
            open: y,
            close: b,
            toggle: F,
            update: p,
            clone: h,
            attach: v,
            detach: D,
            on: g,
            off: E,
            getOptions: C,
            getImages: w,
            getZoomedImage: k,
          }
          return j
        },
        h = n(9253),
        v = n(8110),
        D = n(9218),
        g = n(8577),
        E = n(3324),
        y = n(9590),
        b = Object.create,
        F = Object.defineProperty,
        C = Object.defineProperties,
        w = Object.getOwnPropertyDescriptor,
        k = Object.getOwnPropertyDescriptors,
        A = Object.getOwnPropertyNames,
        _ = Object.getOwnPropertySymbols,
        O = Object.getPrototypeOf,
        B = Object.prototype.hasOwnProperty,
        N = Object.prototype.propertyIsEnumerable,
        x = (e, t, n) =>
          t in e
            ? F(e, t, { enumerable: !0, configurable: !0, writable: !0, value: n })
            : (e[t] = n),
        P = (e, t) => {
          for (var n in t || (t = {})) B.call(t, n) && x(e, n, t[n])
          if (_) for (var n of _(t)) N.call(t, n) && x(e, n, t[n])
          return e
        },
        j = (e, t) => C(e, k(t)),
        L = (e, t) => {
          var n = {}
          for (var o in e) B.call(e, o) && t.indexOf(o) < 0 && (n[o] = e[o])
          if (null != e && _) for (var o of _(e)) t.indexOf(o) < 0 && N.call(e, o) && (n[o] = e[o])
          return n
        },
        z = (e, t, n) => (
          (n = null != e ? b(O(e)) : {}),
          ((e, t, n, o) => {
            if ((t && 'object' === typeof t) || 'function' === typeof t)
              for (let r of A(t))
                B.call(e, r) ||
                  r === n ||
                  F(e, r, { get: () => t[r], enumerable: !(o = w(t, r)) || o.enumerable })
            return e
          })(!t && e && e.__esModule ? n : F(n, 'default', { value: e, enumerable: !0 }), e)
        ),
        S = (e, t, n) =>
          new Promise((o, r) => {
            var a = (e) => {
                try {
                  u(n.next(e))
                } catch (t) {
                  r(t)
                }
              },
              i = (e) => {
                try {
                  u(n.throw(e))
                } catch (t) {
                  r(t)
                }
              },
              u = (e) => (e.done ? o(e.value) : Promise.resolve(e.value).then(a, i))
            u((n = n.apply(e, t)).next())
          }),
        I =
          ((m = {
            '../../node_modules/lodash.throttle/index.js'(e, t) {
              var n = 'Expected a function',
                o = /^\s+|\s+$/g,
                r = /^[-+]0x[0-9a-f]+$/i,
                a = /^0b[01]+$/i,
                i = /^0o[0-7]+$/i,
                u = parseInt,
                l = 'object' == typeof global && global && global.Object === Object && global,
                c = 'object' == typeof self && self && self.Object === Object && self,
                s = l || c || Function('return this')(),
                d = Object.prototype.toString,
                m = Math.max,
                f = Math.min,
                p = function () {
                  return s.Date.now()
                }
              function h(e, t, o) {
                var r,
                  a,
                  i,
                  u,
                  l,
                  c,
                  s = 0,
                  d = !1,
                  h = !1,
                  g = !0
                if ('function' != typeof e) throw new TypeError(n)
                function E(t) {
                  var n = r,
                    o = a
                  return (r = a = void 0), (s = t), (u = e.apply(o, n))
                }
                function y(e) {
                  return (s = e), (l = setTimeout(F, t)), d ? E(e) : u
                }
                function b(e) {
                  var n = e - c
                  return void 0 === c || n >= t || n < 0 || (h && e - s >= i)
                }
                function F() {
                  var e = p()
                  if (b(e)) return C(e)
                  l = setTimeout(
                    F,
                    (function (e) {
                      var n = t - (e - c)
                      return h ? f(n, i - (e - s)) : n
                    })(e)
                  )
                }
                function C(e) {
                  return (l = void 0), g && r ? E(e) : ((r = a = void 0), u)
                }
                function w() {
                  var e = p(),
                    n = b(e)
                  if (((r = arguments), (a = this), (c = e), n)) {
                    if (void 0 === l) return y(c)
                    if (h) return (l = setTimeout(F, t)), E(c)
                  }
                  return void 0 === l && (l = setTimeout(F, t)), u
                }
                return (
                  (t = D(t) || 0),
                  v(o) &&
                    ((d = !!o.leading),
                    (i = (h = 'maxWait' in o) ? m(D(o.maxWait) || 0, t) : i),
                    (g = 'trailing' in o ? !!o.trailing : g)),
                  (w.cancel = function () {
                    void 0 !== l && clearTimeout(l), (s = 0), (r = c = a = l = void 0)
                  }),
                  (w.flush = function () {
                    return void 0 === l ? u : C(p())
                  }),
                  w
                )
              }
              function v(e) {
                var t = typeof e
                return !!e && ('object' == t || 'function' == t)
              }
              function D(e) {
                if ('number' == typeof e) return e
                if (
                  (function (e) {
                    return (
                      'symbol' == typeof e ||
                      ((function (e) {
                        return !!e && 'object' == typeof e
                      })(e) &&
                        '[object Symbol]' == d.call(e))
                    )
                  })(e)
                )
                  return NaN
                if (v(e)) {
                  var t = 'function' == typeof e.valueOf ? e.valueOf() : e
                  e = v(t) ? t + '' : t
                }
                if ('string' != typeof e) return 0 === e ? e : +e
                e = e.replace(o, '')
                var n = a.test(e)
                return n || i.test(e) ? u(e.slice(2), n ? 2 : 8) : r.test(e) ? NaN : +e
              }
              t.exports = function (e, t, o) {
                var r = !0,
                  a = !0
                if ('function' != typeof e) throw new TypeError(n)
                return (
                  v(o) &&
                    ((r = 'leading' in o ? !!o.leading : r),
                    (a = 'trailing' in o ? !!o.trailing : a)),
                  h(e, t, { leading: r, maxWait: t, trailing: a })
                )
              }
            },
          }),
          function () {
            return f || (0, m[A(m)[0]])((f = { exports: {} }).exports, f), f.exports
          }),
        M = (e, t) => {
          if (!e) return null
          if (e.startsWith('data:')) return e
          if (e.startsWith('https://images.unsplash.com')) return e
          try {
            const t = new URL(e)
            if (
              t.pathname.startsWith('/secure.notion-static.com') &&
              t.hostname.endsWith('.amazonaws.com') &&
              t.searchParams.has('X-Amz-Credential') &&
              t.searchParams.has('X-Amz-Signature') &&
              t.searchParams.has('X-Amz-Algorithm')
            )
              return e
          } catch (r) {}
          e.startsWith('/images') && (e = `https://www.notion.so${e}`),
            (e = `https://www.notion.so${
              e.startsWith('/image') ? e : `/image/${encodeURIComponent(e)}`
            }`)
          const n = new URL(e)
          let o = 'space' === t.parent_table ? 'block' : t.parent_table
          return (
            ('collection' !== o && 'team' !== o) || (o = 'block'),
            n.searchParams.set('table', o),
            n.searchParams.set('id', t.id),
            n.searchParams.set('cache', 'v2'),
            (e = n.toString())
          )
        },
        T = (e) => (t) => ((t = (t || '').replace(/-/g, '')), e && t === e ? '/' : `/${t}`),
        $ = (...e) => e.filter((e) => !!e).join(' '),
        H = (e, t) => {
          const n = ((e) => {
              const t = []
              let n,
                o = -1
              return (
                Object.keys(e).forEach((r) => {
                  var a, i
                  const u = null == (a = e[r]) ? void 0 : a.value
                  u &&
                    (null == (i = u.content) ||
                      i.forEach((r) => {
                        var a, i
                        const u =
                          null == (i = null == (a = e[r]) ? void 0 : a.value) ? void 0 : i.type
                        u && u !== n && (o++, (n = u), (t[o] = [])), o > -1 && t[o].push(r)
                      })),
                    (n = void 0)
                }),
                t
              )
            })(t),
            o = n.find((t) => t.includes(e))
          if (o) return o.indexOf(e) + 1
        },
        R = 'undefined' !== typeof window,
        U = new Set([
          'youtu.be',
          'youtube.com',
          'www.youtube.com',
          'youtube-nocookie.com',
          'www.youtube-nocookie.com',
        ])
      var W = function (e) {
          return o.createElement(
            'svg',
            P({ viewBox: '0 0 14 14' }, e),
            o.createElement('path', { d: 'M5.5 12L14 3.5 12.5 2l-7 7-4-4.003L0 6.499z' })
          )
        },
        V = (e) => {
          const t = e,
            { className: n } = t,
            r = L(t, ['className'])
          return o.createElement(
            'svg',
            P({ className: $('notion-icon', n), viewBox: '0 0 17 17' }, r),
            o.createElement('path', {
              d: 'M6.78027 13.6729C8.24805 13.6729 9.60156 13.1982 10.709 12.4072L14.875 16.5732C15.0684 16.7666 15.3232 16.8633 15.5957 16.8633C16.167 16.8633 16.5713 16.4238 16.5713 15.8613C16.5713 15.5977 16.4834 15.3516 16.29 15.1582L12.1504 11.0098C13.0205 9.86719 13.5391 8.45215 13.5391 6.91406C13.5391 3.19629 10.498 0.155273 6.78027 0.155273C3.0625 0.155273 0.0214844 3.19629 0.0214844 6.91406C0.0214844 10.6318 3.0625 13.6729 6.78027 13.6729ZM6.78027 12.2139C3.87988 12.2139 1.48047 9.81445 1.48047 6.91406C1.48047 4.01367 3.87988 1.61426 6.78027 1.61426C9.68066 1.61426 12.0801 4.01367 12.0801 6.91406C12.0801 9.81445 9.68066 12.2139 6.78027 12.2139Z',
            })
          )
        },
        Z = (e) => {
          const t = e,
            { className: n } = t,
            r = L(t, ['className'])
          return o.createElement(
            'svg',
            j(P({ className: n }, r), { viewBox: '0 0 30 30', width: '16' }),
            o.createElement('path', {
              d: 'M16,1H4v28h22V11L16,1z M16,3.828L23.172,11H16V3.828z M24,27H6V3h8v10h10V27z M8,17h14v-2H8V17z M8,21h14v-2H8V21z M8,25h14v-2H8V25z',
            })
          )
        },
        q = (e) => {
          var t,
            n,
            r,
            a = e,
            {
              src: i,
              alt: u,
              className: l,
              style: c,
              zoomable: s = !1,
              priority: d = !1,
              height: m,
            } = a,
            f = L(a, ['src', 'alt', 'className', 'style', 'zoomable', 'priority', 'height'])
          const {
              recordMap: p,
              zoom: v,
              previewImages: D,
              forceCustomImages: E,
              components: y,
            } = pe(),
            b = o.useRef(v ? v.clone() : null),
            F = D
              ? null != (r = null == (t = null == p ? void 0 : p.preview_images) ? void 0 : t[i])
                ? r
                : null == (n = null == p ? void 0 : p.preview_images)
                ? void 0
                : n[(0, h.D5)(i)]
              : null,
            C = o.useCallback(
              (e) => {
                s && (e.target.src || e.target.srcset) && b.current && b.current.attach(e.target)
              },
              [b, s]
            ),
            w = o.useCallback(
              (e) => {
                b.current && e && b.current.attach(e)
              },
              [b]
            ),
            k = o.useMemo(() => (s ? w : void 0), [s, w])
          if (F) {
            const e = F.originalHeight / F.originalWidth
            return y.Image
              ? o.createElement(y.Image, {
                  src: i,
                  alt: u,
                  style: c,
                  className: l,
                  width: F.originalWidth,
                  height: F.originalHeight,
                  blurDataURL: F.dataURIBase64,
                  placeholder: 'blur',
                  priority: d,
                  onLoad: C,
                })
              : o.createElement(
                  g.AZ,
                  j(P({ src: i }, f), { experimentalDecode: !0 }),
                  ({ imageState: t, ref: n }) => {
                    const r = t === g.zl.LoadSuccess,
                      a = { width: '100%' },
                      s = {}
                    return (
                      m
                        ? (a.height = m)
                        : ((s.position = 'absolute'), (a.paddingBottom = 100 * e + '%')),
                      o.createElement(
                        'div',
                        {
                          className: $('lazy-image-wrapper', r && 'lazy-image-loaded', l),
                          style: a,
                        },
                        o.createElement('img', {
                          className: 'lazy-image-preview',
                          src: F.dataURIBase64,
                          alt: u,
                          ref: n,
                          style: c,
                          decoding: 'async',
                        }),
                        o.createElement('img', {
                          className: 'lazy-image-real',
                          src: i,
                          alt: u,
                          ref: k,
                          style: P(P({}, c), s),
                          width: F.originalWidth,
                          height: F.originalHeight,
                          decoding: 'async',
                          loading: 'lazy',
                        })
                      )
                    )
                  }
                )
          }
          return y.Image && E
            ? o.createElement(y.Image, {
                src: i,
                alt: u,
                className: l,
                style: c,
                width: null,
                height: m || null,
                priority: d,
                onLoad: C,
              })
            : o.createElement(
                'img',
                P(
                  {
                    className: l,
                    style: c,
                    src: i,
                    alt: u,
                    ref: k,
                    loading: 'lazy',
                    decoding: 'async',
                  },
                  f
                )
              )
        },
        K = o.memo(
          ({ block: e, className: t, inline: n = !0, hideDefaultIcon: r = !1, defaultIcon: a }) => {
            var i
            const { mapImageUrl: u, recordMap: l, darkMode: c } = pe()
            let s = !1,
              d = null
            if (
              'page' === (m = e).type ||
              'callout' === m.type ||
              'collection_view' === m.type ||
              'collection_view_page' === m.type
            ) {
              const n = (null == (i = (0, h.Ck)(e, l)) ? void 0 : i.trim()) || a,
                m = (0, h.Ho)(e, l)
              if (n && (0, v.Z)(n)) {
                const r = u(n, e)
                ;(s = !0),
                  (d = o.createElement(q, {
                    src: r,
                    alt: m || 'page icon',
                    className: $(t, 'notion-page-icon'),
                  }))
              } else if (n && n.startsWith('/icons/')) {
                const e = 'https://www.notion.so' + n + '?mode=' + (c ? 'dark' : 'light')
                d = o.createElement(q, {
                  src: e,
                  alt: m || 'page icon',
                  className: $(t, 'notion-page-icon'),
                })
              } else
                n
                  ? ((s = !1),
                    (d = o.createElement(
                      'span',
                      { className: $(t, 'notion-page-icon'), role: 'img', 'aria-label': n },
                      n
                    )))
                  : r ||
                    ((s = !0),
                    (d = o.createElement(Z, {
                      className: $(t, 'notion-page-icon'),
                      alt: m || 'page icon',
                    })))
            }
            var m
            return d
              ? o.createElement(
                  'div',
                  {
                    className: $(
                      n ? 'notion-page-icon-inline' : 'notion-page-icon-hero',
                      s ? 'notion-page-icon-image' : 'notion-page-icon-span'
                    ),
                  },
                  d
                )
              : null
          }
        ),
        G = z(I(), 1),
        Q = (e) => {
          const t = e,
            { className: n } = t,
            r = L(t, ['className'])
          return o.createElement(
            'svg',
            j(P({ className: $('notion-icon', n) }, r), { viewBox: '0 0 30 30' }),
            o.createElement('path', {
              d: 'M15,0C6.716,0,0,6.716,0,15s6.716,15,15,15s15-6.716,15-15S23.284,0,15,0z M22,20.6L20.6,22L15,16.4L9.4,22L8,20.6l5.6-5.6 L8,9.4L9.4,8l5.6,5.6L20.6,8L22,9.4L16.4,15L22,20.6z',
            })
          )
        },
        Y = (e) => {
          const t = e,
            { className: n } = t,
            r = L(t, ['className'])
          return o.createElement(
            'svg',
            j(P({ className: $('notion-icon', n) }, r), { viewBox: '0 0 24 24' }),
            o.createElement(
              'defs',
              null,
              o.createElement(
                'linearGradient',
                {
                  x1: '28.1542969%',
                  y1: '63.7402344%',
                  x2: '74.6289062%',
                  y2: '17.7832031%',
                  id: 'linearGradient-1',
                },
                o.createElement('stop', { stopColor: 'rgba(164, 164, 164, 1)', offset: '0%' }),
                o.createElement('stop', {
                  stopColor: 'rgba(164, 164, 164, 0)',
                  stopOpacity: '0',
                  offset: '100%',
                })
              )
            ),
            o.createElement(
              'g',
              { id: 'Page-1', stroke: 'none', strokeWidth: '1', fill: 'none' },
              o.createElement(
                'g',
                { transform: 'translate(-236.000000, -286.000000)' },
                o.createElement(
                  'g',
                  { transform: 'translate(238.000000, 286.000000)' },
                  o.createElement('circle', {
                    id: 'Oval-2',
                    stroke: 'url(#linearGradient-1)',
                    strokeWidth: '4',
                    cx: '10',
                    cy: '12',
                    r: '10',
                  }),
                  o.createElement('path', {
                    d: 'M10,2 C4.4771525,2 0,6.4771525 0,12',
                    id: 'Oval-2',
                    stroke: 'rgba(164, 164, 164, 1)',
                    strokeWidth: '4',
                  }),
                  o.createElement('rect', {
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
      var X = function (e) {
          return o.createElement(
            'svg',
            P({ viewBox: '0 0 260 260' }, e),
            o.createElement(
              'g',
              null,
              o.createElement('path', {
                d: 'M128.00106,0 C57.3172926,0 0,57.3066942 0,128.00106 C0,184.555281 36.6761997,232.535542 87.534937,249.460899 C93.9320223,250.645779 96.280588,246.684165 96.280588,243.303333 C96.280588,240.251045 96.1618878,230.167899 96.106777,219.472176 C60.4967585,227.215235 52.9826207,204.369712 52.9826207,204.369712 C47.1599584,189.574598 38.770408,185.640538 38.770408,185.640538 C27.1568785,177.696113 39.6458206,177.859325 39.6458206,177.859325 C52.4993419,178.762293 59.267365,191.04987 59.267365,191.04987 C70.6837675,210.618423 89.2115753,204.961093 96.5158685,201.690482 C97.6647155,193.417512 100.981959,187.77078 104.642583,184.574357 C76.211799,181.33766 46.324819,170.362144 46.324819,121.315702 C46.324819,107.340889 51.3250588,95.9223682 59.5132437,86.9583937 C58.1842268,83.7344152 53.8029229,70.715562 60.7532354,53.0843636 C60.7532354,53.0843636 71.5019501,49.6441813 95.9626412,66.2049595 C106.172967,63.368876 117.123047,61.9465949 128.00106,61.8978432 C138.879073,61.9465949 149.837632,63.368876 160.067033,66.2049595 C184.49805,49.6441813 195.231926,53.0843636 195.231926,53.0843636 C202.199197,70.715562 197.815773,83.7344152 196.486756,86.9583937 C204.694018,95.9223682 209.660343,107.340889 209.660343,121.315702 C209.660343,170.478725 179.716133,181.303747 151.213281,184.472614 C155.80443,188.444828 159.895342,196.234518 159.895342,208.176593 C159.895342,225.303317 159.746968,239.087361 159.746968,243.303333 C159.746968,246.709601 162.05102,250.70089 168.53925,249.443941 C219.370432,232.499507 256,184.536204 256,128.00106 C256,57.3066942 198.691187,0 128.00106,0 Z M47.9405593,182.340212 C47.6586465,182.976105 46.6581745,183.166873 45.7467277,182.730227 C44.8183235,182.312656 44.2968914,181.445722 44.5978808,180.80771 C44.8734344,180.152739 45.876026,179.97045 46.8023103,180.409216 C47.7328342,180.826786 48.2627451,181.702199 47.9405593,182.340212 Z M54.2367892,187.958254 C53.6263318,188.524199 52.4329723,188.261363 51.6232682,187.366874 C50.7860088,186.474504 50.6291553,185.281144 51.2480912,184.70672 C51.8776254,184.140775 53.0349512,184.405731 53.8743302,185.298101 C54.7115892,186.201069 54.8748019,187.38595 54.2367892,187.958254 Z M58.5562413,195.146347 C57.7719732,195.691096 56.4895886,195.180261 55.6968417,194.042013 C54.9125733,192.903764 54.9125733,191.538713 55.713799,190.991845 C56.5086651,190.444977 57.7719732,190.936735 58.5753181,192.066505 C59.3574669,193.22383 59.3574669,194.58888 58.5562413,195.146347 Z M65.8613592,203.471174 C65.1597571,204.244846 63.6654083,204.03712 62.5716717,202.981538 C61.4524999,201.94927 61.1409122,200.484596 61.8446341,199.710926 C62.5547146,198.935137 64.0575422,199.15346 65.1597571,200.200564 C66.2704506,201.230712 66.6095936,202.705984 65.8613592,203.471174 Z M75.3025151,206.281542 C74.9930474,207.284134 73.553809,207.739857 72.1039724,207.313809 C70.6562556,206.875043 69.7087748,205.700761 70.0012857,204.687571 C70.302275,203.678621 71.7478721,203.20382 73.2083069,203.659543 C74.6539041,204.09619 75.6035048,205.261994 75.3025151,206.281542 Z M86.046947,207.473627 C86.0829806,208.529209 84.8535871,209.404622 83.3316829,209.4237 C81.8013,209.457614 80.563428,208.603398 80.5464708,207.564772 C80.5464708,206.498591 81.7483088,205.631657 83.2786917,205.606221 C84.8005962,205.576546 86.046947,206.424403 86.046947,207.473627 Z M96.6021471,207.069023 C96.7844366,208.099171 95.7267341,209.156872 94.215428,209.438785 C92.7295577,209.710099 91.3539086,209.074206 91.1652603,208.052538 C90.9808515,206.996955 92.0576306,205.939253 93.5413813,205.66582 C95.054807,205.402984 96.4092596,206.021919 96.6021471,207.069023 Z',
                fill: '#161614',
              })
            )
          )
        },
        J = ({ block: e, inline: t, className: n }) => {
          var r, a, i
          const { components: u } = pe(),
            { original_url: l, attributes: c, domain: s } = (null == e ? void 0 : e.format) || {}
          if (!l || !c) return null
          const d = null == (r = c.find((e) => 'title' === e.id)) ? void 0 : r.values[0]
          let m = null == (a = c.find((e) => 'owner' === e.id)) ? void 0 : a.values[0]
          const f = null == (i = c.find((e) => 'updated_at' === e.id)) ? void 0 : i.values[0],
            p = f ? (0, h.c8)(f) : null
          let v
          if ('github.com' !== s)
            return (
              console.log(
                `Unsupported external_object_instance domain "${s}"`,
                JSON.stringify(e, null, 2)
              ),
              null
            )
          if (((v = o.createElement(X, null)), m)) {
            const e = m.split('/')
            m = e[e.length - 1]
          }
          return o.createElement(
            u.Link,
            {
              target: '_blank',
              rel: 'noopener noreferrer',
              href: l,
              className: $(
                'notion-external',
                t ? 'notion-external-mention' : 'notion-external-block notion-row',
                n
              ),
            },
            v && o.createElement('div', { className: 'notion-external-image' }, v),
            o.createElement(
              'div',
              { className: 'notion-external-description' },
              o.createElement('div', { className: 'notion-external-title' }, d),
              (m || p) &&
                o.createElement(
                  'div',
                  { className: 'notion-external-subtitle' },
                  m && o.createElement('span', null, m),
                  m && p && o.createElement('span', null, ' \u2022 '),
                  p && o.createElement('span', null, 'Updated ', p)
                )
            )
          )
        },
        ee = (e) => (R ? o.createElement(E.Img, P({}, e)) : o.createElement('img', P({}, e))),
        te = ({ value: e, block: t, linkProps: n, linkProtocol: r }) => {
          const { components: a, recordMap: i, mapPageUrl: u, mapImageUrl: l, rootDomain: c } = pe()
          return o.createElement(
            o.Fragment,
            null,
            null == e
              ? void 0
              : e.map(([e, s], d) => {
                  if (!s)
                    return ',' === e
                      ? o.createElement('span', { key: d, style: { padding: '0.5em' } })
                      : o.createElement(o.Fragment, { key: d }, e)
                  const m = s.reduce((e, s) => {
                    var d, m, f, p, v, D
                    switch (s[0]) {
                      case 'p': {
                        const e = s[1],
                          t = null == (d = i.block[e]) ? void 0 : d.value
                        return t
                          ? o.createElement(
                              a.PageLink,
                              { className: 'notion-link', href: u(e) },
                              o.createElement(ne, { block: t })
                            )
                          : (console.log('"p" missing block', e), null)
                      }
                      case '\u2023': {
                        const e = s[1][0],
                          r = s[1][1]
                        if ('u' === e) {
                          const e = null == (m = i.notion_user[r]) ? void 0 : m.value
                          if (!e) return console.log('"\u2023" missing user', r), null
                          const n = [e.given_name, e.family_name].filter(Boolean).join(' ')
                          return o.createElement(ee, {
                            className: 'notion-user',
                            src: l(e.profile_photo, t),
                            alt: n,
                          })
                        }
                        {
                          const t = null == (f = i.block[r]) ? void 0 : f.value
                          return t
                            ? o.createElement(
                                a.PageLink,
                                j(P({ className: 'notion-link', href: u(r) }, n), {
                                  target: '_blank',
                                  rel: 'noopener noreferrer',
                                }),
                                o.createElement(ne, { block: t })
                              )
                            : (console.log('"\u2023" missing block', e, r), null)
                        }
                      }
                      case 'h':
                        return o.createElement('span', { className: `notion-${s[1]}` }, e)
                      case 'c':
                        return o.createElement('code', { className: 'notion-inline-code' }, e)
                      case 'b':
                        return o.createElement('b', null, e)
                      case 'i':
                        return o.createElement('em', null, e)
                      case 's':
                        return o.createElement('s', null, e)
                      case '_':
                        return o.createElement('span', { className: 'notion-inline-underscore' }, e)
                      case 'e':
                        return o.createElement(a.Equation, { math: s[1], inline: !0 })
                      case 'm':
                        return e
                      case 'a': {
                        const t = s[1],
                          i = t.substr(1),
                          l = (0, h.q5)(i, { uuid: !0 })
                        if (('/' === t[0] || t.includes(c)) && l) {
                          const r = t.includes(c)
                            ? t
                            : `${u(l)}${
                                ((D = t), D.includes('#') ? D.replace(/^.+(#.+)$/, '$1') : '')
                              }`
                          return o.createElement(
                            a.PageLink,
                            P({ className: 'notion-link', href: r }, n),
                            e
                          )
                        }
                        return o.createElement(
                          a.Link,
                          P({ className: 'notion-link', href: r ? `${r}:${s[1]}` : s[1] }, n),
                          e
                        )
                      }
                      case 'd': {
                        const t = s[1],
                          n = null == t ? void 0 : t.type
                        if ('date' === n) {
                          const e = t.start_date
                          return (0, h.p6)(e)
                        }
                        if ('daterange' === n) {
                          const e = t.start_date,
                            n = t.end_date
                          return `${(0, h.p6)(e)} \u2192 ${(0, h.p6)(n)}`
                        }
                        return e
                      }
                      case 'u': {
                        const e = s[1],
                          n = null == (p = i.notion_user[e]) ? void 0 : p.value
                        if (!n) return console.log('missing user', e), null
                        const r = [n.given_name, n.family_name].filter(Boolean).join(' ')
                        return o.createElement(ee, {
                          className: 'notion-user',
                          src: l(n.profile_photo, t),
                          alt: r,
                        })
                      }
                      case 'eoi': {
                        const e = s[1],
                          t = null == (v = i.block[e]) ? void 0 : v.value
                        return o.createElement(J, { block: t, inline: !0 })
                      }
                      default:
                        return console.log('unsupported text format', s), e
                    }
                  }, o.createElement(o.Fragment, null, e))
                  return o.createElement(o.Fragment, { key: d }, m)
                })
          )
        },
        ne = o.memo((e) => {
          var t,
            n,
            r = e,
            { block: a, className: i, defaultIcon: u } = r,
            l = L(r, ['block', 'className', 'defaultIcon'])
          const { recordMap: c } = pe()
          if (!a) return null
          if ('collection_view_page' === a.type || 'collection_view' === a.type) {
            const e = (0, h.Ho)(a, c)
            if (!e) return null
            const t = [[e]]
            return o.createElement(
              'span',
              P({ className: $('notion-page-title', i) }, l),
              o.createElement(K, { block: a, defaultIcon: u, className: 'notion-page-title-icon' }),
              o.createElement(
                'span',
                { className: 'notion-page-title-text' },
                o.createElement(te, { value: t, block: a })
              )
            )
          }
          return (null == (t = a.properties) ? void 0 : t.title)
            ? o.createElement(
                'span',
                P({ className: $('notion-page-title', i) }, l),
                o.createElement(K, {
                  block: a,
                  defaultIcon: u,
                  className: 'notion-page-title-icon',
                }),
                o.createElement(
                  'span',
                  { className: 'notion-page-title-text' },
                  o.createElement(te, {
                    value: null == (n = a.properties) ? void 0 : n.title,
                    block: a,
                  })
                )
              )
            : null
        }),
        oe = class extends o.Component {
          constructor(e) {
            super(e),
              (this.state = { isLoading: !1, query: '', searchResult: null, searchError: null }),
              (this._onAfterOpen = () => {
                this._inputRef.current && this._inputRef.current.focus()
              }),
              (this._onChangeQuery = (e) => {
                const t = e.target.value
                this.setState({ query: t }),
                  t.trim()
                    ? this._search()
                    : this.setState({ isLoading: !1, searchResult: null, searchError: null })
              }),
              (this._onClearQuery = () => {
                this._onChangeQuery({ target: { value: '' } })
              }),
              (this._warmupSearch = () =>
                S(this, null, function* () {
                  const { searchNotion: e, rootBlockId: t } = this.props
                  yield e({ query: '', ancestorId: t })
                })),
              (this._searchImpl = () =>
                S(this, null, function* () {
                  const { searchNotion: e, rootBlockId: t } = this.props,
                    { query: n } = this.state
                  if (!n.trim())
                    return void this.setState({
                      isLoading: !1,
                      searchResult: null,
                      searchError: null,
                    })
                  this.setState({ isLoading: !0 })
                  const o = yield e({ query: n, ancestorId: t })
                  console.log('search', n, o)
                  let r = null,
                    a = null
                  if (o.error || o.errorId) a = o
                  else {
                    r = P({}, o)
                    const e = r.results
                      .map((e) => {
                        var t, n
                        const o = null == (t = r.recordMap.block[e.id]) ? void 0 : t.value
                        if (!o) return
                        const a = (0, h.Ho)(o, r.recordMap)
                        return a &&
                          ((e.title = a),
                          (e.block = o),
                          (e.recordMap = r.recordMap),
                          (e.page = (0, h.cj)(o, r.recordMap, { inclusive: !0 }) || o),
                          e.page.id)
                          ? ((null == (n = e.highlight) ? void 0 : n.text) &&
                              (e.highlight.html = e.highlight.text
                                .replace(/<gzkNfoUU>/gi, '<b>')
                                .replace(/<\/gzkNfoUU>/gi, '</b>')),
                            e)
                          : void 0
                      })
                      .filter(Boolean)
                      .reduce((e, t) => j(P({}, e), { [t.page.id]: t }), {})
                    r.results = Object.values(e)
                  }
                  this.state.query === n &&
                    this.setState({ isLoading: !1, searchResult: r, searchError: a })
                })),
              (this._inputRef = o.createRef())
          }
          componentDidMount() {
            ;(this._search = (0, G.default)(this._searchImpl.bind(this), 1e3)), this._warmupSearch()
          }
          render() {
            const { isOpen: e, onClose: t } = this.props,
              { isLoading: n, query: r, searchResult: a, searchError: i } = this.state,
              u = !!r.trim()
            return o.createElement(fe, null, (l) => {
              const { components: c, defaultPageIcon: s, mapPageUrl: d } = l
              return o.createElement(
                c.Modal,
                {
                  isOpen: e,
                  contentLabel: 'Search',
                  className: 'notion-search',
                  overlayClassName: 'notion-search-overlay',
                  onRequestClose: t,
                  onAfterOpen: this._onAfterOpen,
                },
                o.createElement(
                  'div',
                  { className: 'quickFindMenu' },
                  o.createElement(
                    'div',
                    { className: 'searchBar' },
                    o.createElement(
                      'div',
                      { className: 'inlineIcon' },
                      n
                        ? o.createElement(Y, { className: 'loadingIcon' })
                        : o.createElement(V, null)
                    ),
                    o.createElement('input', {
                      className: 'searchInput',
                      placeholder: 'Search',
                      value: r,
                      ref: this._inputRef,
                      onChange: this._onChangeQuery,
                    }),
                    r &&
                      o.createElement(
                        'div',
                        { role: 'button', className: 'clearButton', onClick: this._onClearQuery },
                        o.createElement(Q, { className: 'clearIcon' })
                      )
                  ),
                  u &&
                    a &&
                    o.createElement(
                      o.Fragment,
                      null,
                      a.results.length
                        ? o.createElement(
                            me,
                            j(P({}, l), { recordMap: a.recordMap }),
                            o.createElement(
                              'div',
                              { className: 'resultsPane' },
                              a.results.map((e) => {
                                var t
                                return o.createElement(
                                  c.PageLink,
                                  {
                                    key: e.id,
                                    className: $('result', 'notion-page-link'),
                                    href: d(e.page.id, a.recordMap),
                                  },
                                  o.createElement(ne, { block: e.page, defaultIcon: s }),
                                  (null == (t = e.highlight) ? void 0 : t.html) &&
                                    o.createElement('div', {
                                      className: 'notion-search-result-highlight',
                                      dangerouslySetInnerHTML: { __html: e.highlight.html },
                                    })
                                )
                              })
                            ),
                            o.createElement(
                              'footer',
                              { className: 'resultsFooter' },
                              o.createElement(
                                'div',
                                null,
                                o.createElement('span', { className: 'resultsCount' }, a.total),
                                1 === a.total ? ' result' : ' results'
                              )
                            )
                          )
                        : o.createElement(
                            'div',
                            { className: 'noResultsPane' },
                            o.createElement('div', { className: 'noResults' }, 'No results'),
                            o.createElement(
                              'div',
                              { className: 'noResultsDetail' },
                              'Try different search terms'
                            )
                          )
                    ),
                  u &&
                    !a &&
                    i &&
                    o.createElement(
                      'div',
                      { className: 'noResultsPane' },
                      o.createElement('div', { className: 'noResults' }, 'Search error')
                    )
                )
              )
            })
          }
        },
        re = ({ block: e, rootOnly: t = !1 }) => {
          const { recordMap: n, mapPageUrl: r, components: a } = pe(),
            i = o.useMemo(() => {
              const o = (0, h.Kl)(n, e.id)
              return t ? [o[0]].filter(Boolean) : o
            }, [n, e.id, t])
          return o.createElement(
            'div',
            { className: 'breadcrumbs', key: 'breadcrumbs' },
            i.map((e, t) => {
              if (!e) return null
              const n = {},
                u = { pageLink: a.PageLink }
              return (
                e.active
                  ? (u.pageLink = (e) => o.createElement('div', P({}, e)))
                  : (n.href = r(e.pageId)),
                o.createElement(
                  o.Fragment,
                  { key: e.pageId },
                  o.createElement(
                    u.pageLink,
                    P({ className: $('breadcrumb', e.active && 'active') }, n),
                    e.icon && o.createElement(K, { className: 'icon', block: e.block }),
                    e.title && o.createElement('span', { className: 'title' }, e.title)
                  ),
                  t < i.length - 1 && o.createElement('span', { className: 'spacer' }, '/')
                )
              )
            })
          )
        },
        ae = ({ block: e, search: t, title: n = 'Search' }) => {
          const { searchNotion: r, rootPageId: a, isShowingSearch: i, onHideSearch: u } = pe(),
            l = t || r,
            [c, s] = o.useState(i)
          o.useEffect(() => {
            s(i)
          }, [i])
          const d = o.useCallback(() => {
              s(!0)
            }, []),
            m = o.useCallback(() => {
              s(!1), u && u()
            }, [u])
          ;(0, D.y1)('cmd+p', (e) => {
            d(), e.preventDefault(), e.stopPropagation()
          }),
            (0, D.y1)('cmd+k', (e) => {
              d(), e.preventDefault(), e.stopPropagation()
            })
          const f = !!l
          return o.createElement(
            o.Fragment,
            null,
            f &&
              o.createElement(
                'div',
                {
                  role: 'button',
                  className: $('breadcrumb', 'button', 'notion-search-button'),
                  onClick: d,
                },
                o.createElement(V, { className: 'searchIcon' }),
                n && o.createElement('span', { className: 'title' }, n)
              ),
            c &&
              f &&
              o.createElement(oe, {
                isOpen: c,
                rootBlockId: a || (null == e ? void 0 : e.id),
                onClose: m,
                searchNotion: l,
              })
          )
        },
        ie = ({ block: e }) =>
          o.createElement(
            'header',
            { className: 'notion-header' },
            o.createElement(
              'div',
              { className: 'notion-nav-header' },
              o.createElement(re, { block: e }),
              o.createElement(ae, { block: e })
            )
          ),
        ue = (e) => () => (
          console.warn(
            `Warning: using empty component "${e}" (you should override this in NotionRenderer.components)`
          ),
          null
        ),
        le = (e, t) => t(),
        ce = {
          Image: null,
          Link: o.memo((e) =>
            o.createElement('a', P({ target: '_blank', rel: 'noopener noreferrer' }, e))
          ),
          PageLink: o.memo((e) => o.createElement('a', P({}, e))),
          Checkbox: ({ isChecked: e }) => {
            let t = null
            return (
              (t = e
                ? o.createElement(
                    'div',
                    { className: 'notion-property-checkbox-checked' },
                    o.createElement(W, null)
                  )
                : o.createElement('div', { className: 'notion-property-checkbox-unchecked' })),
              o.createElement('span', { className: 'notion-property notion-property-checkbox' }, t)
            )
          },
          Callout: void 0,
          Code: ue('Code'),
          Equation: ue('Equation'),
          Collection: ue('Collection'),
          Property: void 0,
          propertyTextValue: le,
          propertySelectValue: le,
          propertyRelationValue: le,
          propertyFormulaValue: le,
          propertyTitleValue: le,
          propertyPersonValue: le,
          propertyFileValue: le,
          propertyCheckboxValue: le,
          propertyUrlValue: le,
          propertyEmailValue: le,
          propertyPhoneNumberValue: le,
          propertyNumberValue: le,
          propertyLastEditedTimeValue: le,
          propertyCreatedTimeValue: le,
          propertyDateValue: le,
          Pdf: ue('Pdf'),
          Tweet: ue('Tweet'),
          Modal: ue('Modal'),
          Header: ie,
          Embed: (e) => o.createElement(ye, P({}, e)),
        },
        se = {
          recordMap: {
            block: {},
            collection: {},
            collection_view: {},
            collection_query: {},
            notion_user: {},
            signed_urls: {},
          },
          components: ce,
          mapPageUrl: T(),
          mapImageUrl: M,
          searchNotion: null,
          isShowingSearch: !1,
          onHideSearch: null,
          fullPage: !1,
          darkMode: !1,
          previewImages: !1,
          forceCustomImages: !1,
          showCollectionViewDropdown: !0,
          linkTableTitleProperties: !0,
          isLinkCollectionToUrlProperty: !1,
          showTableOfContents: !1,
          minTableOfContentsItems: 3,
          defaultPageIcon: null,
          defaultPageCover: null,
          defaultPageCoverPosition: 0.5,
          zoom: null,
        },
        de = o.createContext(se),
        me = (e) => {
          var t = e,
            { components: n = {}, children: r, mapPageUrl: a, mapImageUrl: i, rootPageId: u } = t,
            l = L(t, ['components', 'children', 'mapPageUrl', 'mapImageUrl', 'rootPageId'])
          for (const o of Object.keys(l)) void 0 === l[o] && delete l[o]
          const c = o.useMemo(() => P({}, n), [n])
          var s, d
          c.nextImage &&
            (c.Image =
              ((s = n.nextImage),
              o.memo(function (e) {
                var t = e,
                  { src: n, alt: r, width: a, height: i, className: u, style: l, layout: c } = t,
                  d = L(t, ['src', 'alt', 'width', 'height', 'className', 'style', 'layout'])
                return (
                  c || (c = a && i ? 'intrinsic' : 'fill'),
                  o.createElement(
                    s,
                    P(
                      {
                        className: u,
                        src: n,
                        alt: r,
                        width: 'intrinsic' === c && a,
                        height: 'intrinsic' === c && i,
                        objectFit: null == l ? void 0 : l.objectFit,
                        objectPosition: null == l ? void 0 : l.objectPosition,
                        layout: c,
                      },
                      d
                    )
                  )
                )
              }, y))),
            c.nextLink &&
              (c.nextLink =
                ((d = n.nextLink),
                function (e) {
                  var t = e,
                    {
                      href: n,
                      as: r,
                      passHref: a,
                      prefetch: i,
                      replace: u,
                      scroll: l,
                      shallow: c,
                      locale: s,
                    } = t,
                    m = L(t, [
                      'href',
                      'as',
                      'passHref',
                      'prefetch',
                      'replace',
                      'scroll',
                      'shallow',
                      'locale',
                    ])
                  return o.createElement(
                    d,
                    {
                      href: n,
                      as: r,
                      passHref: a,
                      prefetch: i,
                      replace: u,
                      scroll: l,
                      shallow: c,
                      locale: s,
                    },
                    o.createElement('a', P({}, m))
                  )
                }))
          for (const o of Object.keys(c)) c[o] || delete c[o]
          const m = o.useMemo(
            () =>
              j(P(P({}, se), l), {
                rootPageId: u,
                mapPageUrl: null != a ? a : T(u),
                mapImageUrl: null != i ? i : M,
                components: P(P({}, ce), c),
              }),
            [i, a, c, u, l]
          )
          return o.createElement(de.Provider, { value: m }, r)
        },
        fe = de.Consumer,
        pe = () => o.useContext(de),
        he = ({
          id: e,
          defaultPlay: t = !1,
          mute: n = !1,
          lazyImage: r = !1,
          iframeTitle: a = 'YouTube video',
          alt: i = 'Video preview',
          params: u = {},
          adLinksPreconnect: l = !0,
          style: c,
          className: s,
        }) => {
          const d = n || t ? '1' : '0',
            m = `https://i.ytimg.com/vi/${e}/hqdefault.jpg`,
            f = 'https://www.youtube-nocookie.com',
            p = `${f}/embed/${e}?${o.useMemo(
              () =>
                ((e) =>
                  Object.keys(e)
                    .map((t) => `${encodeURIComponent(t)}=${encodeURIComponent(e[t])}`)
                    .join('&'))(P({ autoplay: '1', mute: d }, u)),
              [d, u]
            )}`,
            [h, v] = o.useState(!1),
            [D, g] = o.useState(t),
            [E, y] = o.useState(!1),
            b = o.useCallback(() => {
              h || v(!0)
            }, [h]),
            F = o.useCallback(() => {
              D || g(!0)
            }, [D]),
            C = o.useCallback(() => {
              y(!0)
            }, [])
          return o.createElement(
            o.Fragment,
            null,
            o.createElement('link', { rel: 'preload', href: m, as: 'image' }),
            h &&
              o.createElement(
                o.Fragment,
                null,
                o.createElement('link', { rel: 'preconnect', href: f }),
                o.createElement('link', { rel: 'preconnect', href: 'https://www.google.com' })
              ),
            h &&
              l &&
              o.createElement(
                o.Fragment,
                null,
                o.createElement('link', {
                  rel: 'preconnect',
                  href: 'https://static.doubleclick.net',
                }),
                o.createElement('link', {
                  rel: 'preconnect',
                  href: 'https://googleads.g.doubleclick.net',
                })
              ),
            o.createElement(
              'div',
              {
                onClick: F,
                onPointerOver: b,
                className: $(
                  'notion-yt-lite',
                  E && 'notion-yt-loaded',
                  D && 'notion-yt-initialized',
                  s
                ),
                style: c,
              },
              o.createElement('img', {
                src: m,
                className: 'notion-yt-thumbnail',
                loading: r ? 'lazy' : void 0,
                alt: i,
              }),
              o.createElement('div', { className: 'notion-yt-playbtn' }),
              D &&
                o.createElement('iframe', {
                  width: '560',
                  height: '315',
                  frameBorder: '0',
                  allow: 'accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture',
                  allowFullScreen: !0,
                  title: a,
                  src: p,
                  onLoad: C,
                })
            )
          )
        },
        ve = 'undefined' === typeof window,
        De = [
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
        ],
        ge = ({ block: e, zoomable: t = !0, children: n }) => {
          var r, a, i, u, l, c, s, d, m, f
          const { recordMap: p, mapImageUrl: v, components: D } = pe()
          if (!e || !De.includes(e.type)) return null
          const g = {
              position: 'relative',
              display: 'flex',
              justifyContent: 'center',
              alignSelf: 'center',
              width: '100%',
              maxWidth: '100%',
              flexDirection: 'column',
            },
            E = {}
          if (e.format) {
            const {
              block_aspect_ratio: t,
              block_height: n,
              block_width: o,
              block_full_width: a,
              block_page_width: i,
              block_preserve_scale: u,
            } = e.format
            if (a || i)
              (g.width = a ? '100vw' : '100%'),
                'video' === e.type
                  ? n
                    ? (g.height = n)
                    : t
                    ? (g.paddingBottom = 100 * t + '%')
                    : u && (g.objectFit = 'contain')
                  : t && 'image' !== e.type
                  ? (g.paddingBottom = 100 * t + '%')
                  : n
                  ? (g.height = n)
                  : u &&
                    ('image' === e.type
                      ? (g.height = '100%')
                      : ((g.paddingBottom = '75%'), (g.minHeight = 100)))
            else {
              switch (null == (r = e.format) ? void 0 : r.block_alignment) {
                case 'center':
                  g.alignSelf = 'center'
                  break
                case 'left':
                  g.alignSelf = 'start'
                  break
                case 'right':
                  g.alignSelf = 'end'
              }
              o && (g.width = o),
                u && 'image' !== e.type
                  ? ((g.paddingBottom = '50%'), (g.minHeight = 100))
                  : n && 'image' !== e.type && (g.height = n)
            }
            'image' === e.type ? (E.objectFit = 'cover') : u && (E.objectFit = 'contain')
          }
          let y =
              (null == (a = p.signed_urls) ? void 0 : a[e.id]) ||
              (null ==
              (l = null == (u = null == (i = e.properties) ? void 0 : i.source) ? void 0 : u[0])
                ? void 0
                : l[0]),
            b = null
          if (!y) return null
          if ('tweet' === e.type) {
            if (!y) return null
            const e = y.split('?')[0].split('/').pop()
            if (!e) return null
            b = o.createElement(
              'div',
              {
                style: j(P({}, E), {
                  maxWidth: 420,
                  width: '100%',
                  marginLeft: 'auto',
                  marginRight: 'auto',
                }),
              },
              o.createElement(D.Tweet, { id: e })
            )
          } else if ('pdf' === e.type)
            (g.overflow = 'auto'),
              (g.background = 'rgb(226, 226, 226)'),
              (g.display = 'block'),
              g.padding || (g.padding = '8px 16px'),
              ve || (b = o.createElement(D.Pdf, { file: y }))
          else if (
            'embed' === e.type ||
            'video' === e.type ||
            'figma' === e.type ||
            'typeform' === e.type ||
            'gist' === e.type ||
            'maps' === e.type ||
            'excalidraw' === e.type ||
            'codepen' === e.type ||
            'drive' === e.type
          )
            if (
              'video' === e.type &&
              y &&
              y.indexOf('youtube') < 0 &&
              y.indexOf('youtu.be') < 0 &&
              y.indexOf('vimeo') < 0 &&
              y.indexOf('wistia') < 0 &&
              y.indexOf('loom') < 0 &&
              y.indexOf('videoask') < 0 &&
              y.indexOf('getcloudapp') < 0
            )
              (g.paddingBottom = void 0),
                (b = o.createElement('video', {
                  playsInline: !0,
                  controls: !0,
                  preload: 'metadata',
                  style: E,
                  src: y,
                  title: e.type,
                }))
            else {
              let t = (null == (c = e.format) ? void 0 : c.display_source) || y
              if (t) {
                const n =
                  'video' === e.type
                    ? ((e) => {
                        try {
                          const { hostname: t } = new URL(e)
                          if (!U.has(t)) return null
                          const n =
                              /^.*(youtu\.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/i,
                            o = e.match(n)
                          if (o && 11 == o[2].length) return o[2]
                        } catch (t) {}
                        return null
                      })(t)
                    : null
                n
                  ? (b = o.createElement(he, {
                      id: n,
                      style: E,
                      className: 'notion-asset-object-fit',
                    }))
                  : 'gist' === e.type
                  ? (t.endsWith('.pibb') || (t = `${t}.pibb`),
                    (E.width = '100%'),
                    (g.paddingBottom = '50%'),
                    (b = o.createElement('iframe', {
                      style: E,
                      className: 'notion-asset-object-fit',
                      src: t,
                      title: 'GitHub Gist',
                      frameBorder: '0',
                      loading: 'lazy',
                      scrolling: 'auto',
                    })))
                  : (b = o.createElement('iframe', {
                      className: 'notion-asset-object-fit',
                      style: E,
                      src: t,
                      title: `iframe ${e.type}`,
                      frameBorder: '0',
                      allowFullScreen: !0,
                      loading: 'lazy',
                      scrolling: 'auto',
                    }))
              }
            }
          else if ('image' === e.type) {
            y.includes('file.notion.so') &&
              (y =
                null ==
                (m = null == (d = null == (s = e.properties) ? void 0 : s.source) ? void 0 : d[0])
                  ? void 0
                  : m[0])
            const n = v(y, e),
              r = (0, h.FB)(null == (f = e.properties) ? void 0 : f.caption) || 'notion image'
            b = o.createElement(q, { src: n, alt: r, zoomable: t, height: g.height, style: E })
          }
          return o.createElement(
            o.Fragment,
            null,
            o.createElement('div', { style: g }, b, 'image' === e.type && n),
            'image' !== e.type && n
          )
        },
        Ee = { width: '100%' },
        ye = ({ blockId: e, block: t }) => {
          var n, r, a, i, u, l
          const c = t,
            { components: s, mapPageUrl: d, rootDomain: m, zoom: f } = pe()
          let p = !1
          if ('image' === t.type) {
            const e =
              null ==
              (a =
                null == (r = null == (n = null == c ? void 0 : c.properties) ? void 0 : n.caption)
                  ? void 0
                  : r[0])
                ? void 0
                : a[0]
            if (e) {
              const t = (0, h.q5)(e, { uuid: !0 })
              ;(('/' === e.charAt(0) && t) ||
                ((v = e),
                new RegExp(
                  '^(https?:\\/\\/)?((([a-z\\d]([a-z\\d-]*[a-z\\d])*)\\.)+[a-z]{2,}|((\\d{1,3}\\.){3}\\d{1,3}))(\\:\\d+)?(\\/[-a-z\\d%_.~+]*)*(\\?[;&a-z\\d%_.~+=-]*)?(\\#[-a-z\\d_]*)?$',
                  'i'
                ).test(v))) &&
                (p = !0)
            }
          }
          var v
          const D = o.createElement(
            'figure',
            {
              className: $(
                'notion-asset-wrapper',
                `notion-asset-wrapper-${t.type}`,
                (null == (i = c.format) ? void 0 : i.block_full_width) &&
                  'notion-asset-wrapper-full',
                e
              ),
            },
            o.createElement(
              ge,
              { block: c, zoomable: f && !p },
              (null == (u = null == c ? void 0 : c.properties) ? void 0 : u.caption) &&
                !p &&
                o.createElement(
                  'figcaption',
                  { className: 'notion-asset-caption' },
                  o.createElement(te, { value: c.properties.caption, block: t })
                )
            )
          )
          if (p) {
            const e = null == (l = null == c ? void 0 : c.properties) ? void 0 : l.caption[0][0],
              t = (0, h.q5)(e, { uuid: !0 }),
              n = '/' === e.charAt(0) && t,
              r = (function (e) {
                try {
                  return new URL(e).hostname
                } catch (t) {
                  return ''
                }
              })(e)
            return o.createElement(
              s.PageLink,
              {
                style: Ee,
                href: n ? d(t) : e,
                target: r && r !== m && !e.startsWith('/') ? 'blank_' : null,
              },
              D
            )
          }
          return D
        }
      var be = ({ block: e, className: t }) => {
          var n, r, a
          const { recordMap: i } = pe(),
            u =
              i.signed_urls[e.id] ||
              (null ==
              (a = null == (r = null == (n = e.properties) ? void 0 : n.source) ? void 0 : r[0])
                ? void 0
                : a[0])
          return o.createElement(
            'div',
            { className: $('notion-audio', t) },
            o.createElement('audio', { controls: !0, preload: 'none', src: u })
          )
        },
        Fe = (e) => {
          const t = e,
            { className: n } = t,
            r = L(t, ['className'])
          return o.createElement(
            'svg',
            j(P({ className: n }, r), { viewBox: '0 0 30 30' }),
            o.createElement('path', {
              d: 'M22,8v12c0,3.866-3.134,7-7,7s-7-3.134-7-7V8c0-2.762,2.238-5,5-5s5,2.238,5,5v12c0,1.657-1.343,3-3,3s-3-1.343-3-3V8h-2v12c0,2.762,2.238,5,5,5s5-2.238,5-5V8c0-3.866-3.134-7-7-7S6,4.134,6,8v12c0,4.971,4.029,9,9,9s9-4.029,9-9V8H22z',
            })
          )
        },
        Ce = ({ block: e, className: t }) => {
          var n, r, a, i, u
          const { components: l, recordMap: c } = pe(),
            s =
              c.signed_urls[e.id] ||
              (null ==
              (a = null == (r = null == (n = e.properties) ? void 0 : n.source) ? void 0 : r[0])
                ? void 0
                : a[0])
          return o.createElement(
            'div',
            { className: $('notion-file', t) },
            o.createElement(
              l.Link,
              {
                className: 'notion-file-link',
                href: s,
                target: '_blank',
                rel: 'noopener noreferrer',
              },
              o.createElement(Fe, { className: 'notion-file-icon' }),
              o.createElement(
                'div',
                { className: 'notion-file-info' },
                o.createElement(
                  'div',
                  { className: 'notion-file-title' },
                  o.createElement(te, {
                    value: (null == (i = e.properties) ? void 0 : i.title) || [['File']],
                    block: e,
                  })
                ),
                (null == (u = e.properties) ? void 0 : u.size) &&
                  o.createElement(
                    'div',
                    { className: 'notion-file-size' },
                    o.createElement(te, { value: e.properties.size, block: e })
                  )
              )
            )
          )
        },
        we = ({ block: e, className: t }) => {
          var n
          const { components: r, mapImageUrl: a } = pe(),
            i = null == (n = e.format) ? void 0 : n.drive_properties
          if (!i) return null
          let u
          try {
            u = new URL(i.url).hostname
          } catch (l) {}
          return o.createElement(
            'div',
            { className: $('notion-google-drive', t) },
            o.createElement(
              r.Link,
              {
                className: 'notion-google-drive-link',
                href: i.url,
                target: '_blank',
                rel: 'noopener noreferrer',
              },
              o.createElement(
                'div',
                { className: 'notion-google-drive-preview' },
                o.createElement(ee, {
                  src: a(i.thumbnail, e),
                  alt: i.title || 'Google Drive Document',
                  loading: 'lazy',
                })
              ),
              o.createElement(
                'div',
                { className: 'notion-google-drive-body' },
                i.title &&
                  o.createElement('div', { className: 'notion-google-drive-body-title' }, i.title),
                i.icon &&
                  u &&
                  o.createElement(
                    'div',
                    { className: 'notion-google-drive-body-source' },
                    i.icon &&
                      o.createElement('div', {
                        className: 'notion-google-drive-body-source-icon',
                        style: { backgroundImage: `url(${i.icon})` },
                      }),
                    u &&
                      o.createElement(
                        'div',
                        { className: 'notion-google-drive-body-source-domain' },
                        u
                      )
                  )
              )
            )
          )
        },
        ke = z(I(), 1),
        Ae = ({
          toc: e,
          activeSection: t,
          setActiveSection: n,
          pageAside: r,
          hasToc: a,
          hasAside: i,
          className: u,
        }) => {
          const l = o.useMemo(
            () =>
              (0, ke.default)(() => {
                const e = document.getElementsByClassName('notion-h')
                let o = null,
                  r = t
                for (let t = 0; t < e.length; ++t) {
                  const n = e[t]
                  if (!n || !(n instanceof Element)) continue
                  r || (r = n.getAttribute('data-id'))
                  const a = n.getBoundingClientRect(),
                    i = o ? a.top - o.bottom : 0,
                    u = Math.max(150, i / 4)
                  if (!(a.top - u < 0)) break
                  ;(r = n.getAttribute('data-id')), (o = a)
                }
                n(r)
              }, 100),
            [n]
          )
          return (
            o.useEffect(() => {
              if (a)
                return (
                  window.addEventListener('scroll', l),
                  l(),
                  () => {
                    window.removeEventListener('scroll', l)
                  }
                )
            }, [a, l]),
            i
              ? o.createElement(
                  'aside',
                  { className: $('notion-aside', u) },
                  a &&
                    o.createElement(
                      'div',
                      { className: 'notion-aside-table-of-contents' },
                      o.createElement(
                        'div',
                        { className: 'notion-aside-table-of-contents-header' },
                        'Table of Contents'
                      ),
                      o.createElement(
                        'nav',
                        { className: 'notion-table-of-contents' },
                        e.map((e) => {
                          const n = (0, h.Gw)(e.id)
                          return o.createElement(
                            'a',
                            {
                              key: n,
                              href: `#${n}`,
                              className: $(
                                'notion-table-of-contents-item',
                                `notion-table-of-contents-item-indent-level-${e.indentLevel}`,
                                t === n && 'notion-table-of-contents-active-item'
                              ),
                            },
                            o.createElement(
                              'span',
                              {
                                className: 'notion-table-of-contents-item-body',
                                style: { display: 'inline-block', marginLeft: 16 * e.indentLevel },
                              },
                              e.text
                            )
                          )
                        })
                      )
                    ),
                  r
                )
              : null
          )
        },
        _e = ({ block: e, level: t }) => {
          var n, r
          if (!e) return console.warn('missing sync pointer block', e.id), null
          const a =
            null ==
            (r =
              null == (n = null == e ? void 0 : e.format)
                ? void 0
                : n.transclusion_reference_pointer)
              ? void 0
              : r.id
          return a ? o.createElement(je, { key: a, level: t, blockId: a }) : null
        },
        Oe = (e) => {
          const t = e,
            { className: n } = t,
            r = L(t, ['className'])
          return o.createElement(
            'svg',
            j(P({ className: n }, r), { viewBox: '0 0 16 16', width: '16', height: '16' }),
            o.createElement('path', {
              fillRule: 'evenodd',
              d: 'M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z',
            })
          )
        },
        Be = {},
        Ne = {},
        xe = (e) => {
          var t,
            n,
            r,
            a,
            i,
            u,
            l,
            c,
            s,
            d,
            m,
            f,
            p,
            D,
            g,
            E,
            y,
            b,
            F,
            C,
            w,
            k,
            A,
            _,
            O,
            B,
            N,
            x,
            j,
            L,
            z,
            S,
            I,
            M,
            T,
            R,
            U,
            W,
            V,
            Z,
            G,
            Q,
            Y
          const X = pe(),
            {
              components: ee,
              fullPage: oe,
              darkMode: re,
              recordMap: ae,
              mapPageUrl: ie,
              mapImageUrl: ue,
              showTableOfContents: le,
              minTableOfContentsItems: ce,
              defaultPageIcon: se,
              defaultPageCover: de,
              defaultPageCoverPosition: me,
            } = X,
            [fe, he] = o.useState(null),
            {
              block: ve,
              children: De,
              level: ge,
              className: Ee,
              bodyClassName: Fe,
              header: ke,
              footer: xe,
              pageHeader: Pe,
              pageFooter: je,
              pageTitle: Le,
              pageAside: ze,
              pageCover: Se,
              hideBlockId: Ie,
              disableHeader: Me,
            } = e
          if (!ve) return null
          0 === ge && 'collection_view' === ve.type && (ve.type = 'collection_view_page')
          const Te = Ie ? 'notion-block' : `notion-block-${(0, h.Gw)(ve.id)}`
          switch (ve.type) {
            case 'collection_view_page':
            case 'page':
              if (0 === ge) {
                const {
                  page_icon: e = se,
                  page_cover: a = de,
                  page_cover_position: i = me,
                  page_full_width: u,
                  page_small_text: l,
                } = ve.format || {}
                if (oe) {
                  const c =
                      'page' === ve.type
                        ? ve.properties
                        : {
                            title:
                              null ==
                              (n =
                                null == (t = ae.collection[(0, h.c5)(ve, ae)]) ? void 0 : t.value)
                                ? void 0
                                : n.name,
                          },
                    s = `center ${100 * (1 - (i || 0.5))}%`
                  let d = Ne[s]
                  d || (d = Ne[s] = { objectPosition: s })
                  const m = null != (r = (0, h.Ck)(ve, ae)) ? r : se,
                    f = m && (0, v.Z)(m),
                    p = (0, h.Ru)(ve, ae),
                    D = le && p.length >= ce,
                    g = (D || ze) && !u,
                    E = Se || a
                  return o.createElement(
                    'div',
                    {
                      className: $('notion', 'notion-app', re ? 'dark-mode' : 'light-mode', Te, Ee),
                    },
                    o.createElement('div', { className: 'notion-viewport' }),
                    o.createElement(
                      'div',
                      { className: 'notion-frame' },
                      !Me && o.createElement(ee.Header, { block: ve }),
                      ke,
                      o.createElement(
                        'div',
                        { className: 'notion-page-scroller' },
                        E &&
                          (Se ||
                            o.createElement(
                              'div',
                              { className: 'notion-page-cover-wrapper' },
                              o.createElement(q, {
                                src: ue(a, ve),
                                alt: (0, h.FB)(null == c ? void 0 : c.title),
                                priority: !0,
                                className: 'notion-page-cover',
                                style: d,
                              })
                            )),
                        o.createElement(
                          'main',
                          {
                            className: $(
                              'notion-page',
                              E ? 'notion-page-has-cover' : 'notion-page-no-cover',
                              e ? 'notion-page-has-icon' : 'notion-page-no-icon',
                              f ? 'notion-page-has-image-icon' : 'notion-page-has-text-icon',
                              'notion-full-page',
                              u && 'notion-full-width',
                              l && 'notion-small-text',
                              Fe
                            ),
                          },
                          e && o.createElement(K, { block: ve, defaultIcon: se, inline: !1 }),
                          Pe,
                          o.createElement(
                            'h1',
                            { className: 'notion-title' },
                            null != Le
                              ? Le
                              : o.createElement(te, {
                                  value: null == c ? void 0 : c.title,
                                  block: ve,
                                })
                          ),
                          ('collection_view_page' === ve.type ||
                            ('page' === ve.type && 'collection' === ve.parent_table)) &&
                            o.createElement(ee.Collection, { block: ve, ctx: X }),
                          'collection_view_page' !== ve.type &&
                            o.createElement(
                              'div',
                              {
                                className: $(
                                  'notion-page-content',
                                  g && 'notion-page-content-has-aside',
                                  D && 'notion-page-content-has-toc'
                                ),
                              },
                              o.createElement(
                                'article',
                                { className: 'notion-page-content-inner' },
                                De
                              ),
                              g &&
                                o.createElement(Ae, {
                                  toc: p,
                                  activeSection: fe,
                                  setActiveSection: he,
                                  hasToc: D,
                                  hasAside: g,
                                  pageAside: ze,
                                })
                            ),
                          je
                        ),
                        xe
                      )
                    )
                  )
                }
                return o.createElement(
                  'main',
                  {
                    className: $(
                      'notion',
                      re ? 'dark-mode' : 'light-mode',
                      'notion-page',
                      u && 'notion-full-width',
                      l && 'notion-small-text',
                      Te,
                      Ee,
                      Fe
                    ),
                  },
                  o.createElement('div', { className: 'notion-viewport' }),
                  Pe,
                  ('collection_view_page' === ve.type ||
                    ('page' === ve.type && 'collection' === ve.parent_table)) &&
                    o.createElement(ee.Collection, { block: ve, ctx: X }),
                  'collection_view_page' !== ve.type && De,
                  je
                )
              }
              {
                const e = null == (a = ve.format) ? void 0 : a.block_color
                return o.createElement(
                  ee.PageLink,
                  { className: $('notion-page-link', e && `notion-${e}`, Te), href: ie(ve.id) },
                  o.createElement(ne, { block: ve })
                )
              }
            case 'header':
            case 'sub_header':
            case 'sub_sub_header': {
              if (!ve.properties) return null
              const e = null == (i = ve.format) ? void 0 : i.block_color,
                t = (0, h.Gw)(ve.id),
                n = (0, h.FB)(ve.properties.title) || `Notion Header ${t}`
              let r,
                a = Be[ve.id]
              if (void 0 === a) {
                const e = (0, h.cj)(ve, ae)
                if (e) {
                  const t = (0, h.Ru)(e, ae).find((e) => e.id === ve.id)
                  t && ((a = t.indentLevel), (Be[ve.id] = a))
                }
              }
              void 0 !== a && (r = `notion-h-indent-${a}`)
              const c = 'header' === ve.type,
                s = 'sub_header' === ve.type,
                d = 'sub_sub_header' === ve.type,
                m = $(
                  c && 'notion-h notion-h1',
                  s && 'notion-h notion-h2',
                  d && 'notion-h notion-h3',
                  e && `notion-${e}`,
                  r,
                  Te
                ),
                f = o.createElement(
                  'span',
                  null,
                  o.createElement('div', { id: t, className: 'notion-header-anchor' }),
                  !(null == (u = ve.format) ? void 0 : u.toggleable) &&
                    o.createElement(
                      'a',
                      { className: 'notion-hash-link', href: `#${t}`, title: n },
                      o.createElement(Oe, null)
                    ),
                  o.createElement(
                    'span',
                    { className: 'notion-h-title' },
                    o.createElement(te, { value: ve.properties.title, block: ve })
                  )
                )
              let p = null
              return (
                (p = c
                  ? o.createElement('h2', { className: m, 'data-id': t }, f)
                  : s
                  ? o.createElement('h3', { className: m, 'data-id': t }, f)
                  : o.createElement('h4', { className: m, 'data-id': t }, f)),
                (null == (l = ve.format) ? void 0 : l.toggleable)
                  ? o.createElement(
                      'details',
                      { className: $('notion-toggle', Te) },
                      o.createElement('summary', null, p),
                      o.createElement('div', null, De)
                    )
                  : p
              )
            }
            case 'divider':
              return o.createElement('hr', { className: $('notion-hr', Te) })
            case 'text': {
              if (!ve.properties && !(null == (c = ve.content) ? void 0 : c.length))
                return o.createElement('div', { className: $('notion-blank', Te) }, '\xa0')
              const e = null == (s = ve.format) ? void 0 : s.block_color
              return o.createElement(
                'div',
                { className: $('notion-text', e && `notion-${e}`, Te) },
                (null == (d = ve.properties) ? void 0 : d.title) &&
                  o.createElement(te, { value: ve.properties.title, block: ve }),
                De && o.createElement('div', { className: 'notion-text-children' }, De)
              )
            }
            case 'bulleted_list':
            case 'numbered_list': {
              const e = (e, t) =>
                'bulleted_list' === ve.type
                  ? o.createElement(
                      'ul',
                      { className: $('notion-list', 'notion-list-disc', Te) },
                      e
                    )
                  : o.createElement(
                      'ol',
                      { start: t, className: $('notion-list', 'notion-list-numbered', Te) },
                      e
                    )
              let t = null
              t = ve.content
                ? o.createElement(
                    o.Fragment,
                    null,
                    ve.properties &&
                      o.createElement(
                        'li',
                        null,
                        o.createElement(te, { value: ve.properties.title, block: ve })
                      ),
                    e(De)
                  )
                : ve.properties
                ? o.createElement(
                    'li',
                    null,
                    o.createElement(te, { value: ve.properties.title, block: ve })
                  )
                : null
              const n =
                  ve.type !==
                  (null == (f = null == (m = ae.block[ve.parent_id]) ? void 0 : m.value)
                    ? void 0
                    : f.type),
                r = H(ve.id, ae.block)
              return n ? e(t, r) : t
            }
            case 'embed':
              return o.createElement(ee.Embed, { blockId: Te, block: ve })
            case 'tweet':
            case 'maps':
            case 'pdf':
            case 'figma':
            case 'typeform':
            case 'codepen':
            case 'excalidraw':
            case 'image':
            case 'gist':
            case 'video':
              return o.createElement(ye, { blockId: Te, block: ve })
            case 'drive':
              return !(null == (p = ve.format) ? void 0 : p.drive_properties) &&
                (null == (D = ve.format) ? void 0 : D.display_source)
                ? o.createElement(ye, { blockId: Te, block: ve })
                : o.createElement(we, { block: ve, className: Te })
            case 'audio':
              return o.createElement(be, { block: ve, className: Te })
            case 'file':
              return o.createElement(Ce, { block: ve, className: Te })
            case 'equation':
              return o.createElement(ee.Equation, { block: ve, inline: !1, className: Te })
            case 'code':
              return o.createElement(ee.Code, { block: ve })
            case 'column_list':
              return o.createElement('div', { className: $('notion-row', Te) }, De)
            case 'column': {
              const e = 'min(32px, 4vw)',
                t = (null == (g = ve.format) ? void 0 : g.column_ratio) || 0.5,
                n = null == (E = ae.block[ve.parent_id]) ? void 0 : E.value,
                r = {
                  width: `calc((100% - (${
                    ((null == (y = null == n ? void 0 : n.content) ? void 0 : y.length) ||
                      Math.max(2, Math.ceil(1 / t))) - 1
                  } * ${e})) * ${t})`,
                }
              return o.createElement(
                o.Fragment,
                null,
                o.createElement('div', { className: $('notion-column', Te), style: r }, De),
                o.createElement('div', { className: 'notion-spacer' })
              )
            }
            case 'quote': {
              if (!ve.properties) return null
              const e = null == (b = ve.format) ? void 0 : b.block_color
              return o.createElement(
                'blockquote',
                { className: $('notion-quote', e && `notion-${e}`, Te) },
                o.createElement(
                  'div',
                  null,
                  o.createElement(te, { value: ve.properties.title, block: ve })
                ),
                De
              )
            }
            case 'collection_view':
              return o.createElement(ee.Collection, { block: ve, className: Te, ctx: X })
            case 'callout':
              return ee.Callout
                ? o.createElement(ee.Callout, { block: ve, className: Te })
                : o.createElement(
                    'div',
                    {
                      className: $(
                        'notion-callout',
                        (null == (F = ve.format) ? void 0 : F.block_color) &&
                          `notion-${null == (C = ve.format) ? void 0 : C.block_color}_co`,
                        Te
                      ),
                    },
                    o.createElement(K, { block: ve }),
                    o.createElement(
                      'div',
                      { className: 'notion-callout-text' },
                      o.createElement(te, {
                        value: null == (w = ve.properties) ? void 0 : w.title,
                        block: ve,
                      }),
                      De
                    )
                  )
            case 'bookmark': {
              if (!ve.properties) return null
              const e = ve.properties.link
              if (!e || !(null == (k = e[0]) ? void 0 : k[0])) return null
              let t = (0, h.FB)(ve.properties.title)
              if ((t || (t = (0, h.FB)(e)), t && t.startsWith('http')))
                try {
                  t = new URL(t).hostname
                } catch ($e) {}
              return o.createElement(
                'div',
                { className: 'notion-row' },
                o.createElement(
                  ee.Link,
                  {
                    target: '_blank',
                    rel: 'noopener noreferrer',
                    className: $(
                      'notion-bookmark',
                      (null == (A = ve.format) ? void 0 : A.block_color) &&
                        `notion-${ve.format.block_color}`,
                      Te
                    ),
                    href: e[0][0],
                  },
                  o.createElement(
                    'div',
                    null,
                    t &&
                      o.createElement(
                        'div',
                        { className: 'notion-bookmark-title' },
                        o.createElement(te, { value: [[t]], block: ve })
                      ),
                    (null == (_ = ve.properties) ? void 0 : _.description) &&
                      o.createElement(
                        'div',
                        { className: 'notion-bookmark-description' },
                        o.createElement(te, {
                          value: null == (O = ve.properties) ? void 0 : O.description,
                          block: ve,
                        })
                      ),
                    o.createElement(
                      'div',
                      { className: 'notion-bookmark-link' },
                      (null == (B = ve.format) ? void 0 : B.bookmark_icon) &&
                        o.createElement(
                          'div',
                          { className: 'notion-bookmark-link-icon' },
                          o.createElement(q, {
                            src: ue(null == (N = ve.format) ? void 0 : N.bookmark_icon, ve),
                            alt: t,
                          })
                        ),
                      o.createElement(
                        'div',
                        { className: 'notion-bookmark-link-text' },
                        o.createElement(te, { value: e, block: ve })
                      )
                    )
                  ),
                  (null == (x = ve.format) ? void 0 : x.bookmark_cover) &&
                    o.createElement(
                      'div',
                      { className: 'notion-bookmark-image' },
                      o.createElement(q, {
                        src: ue(null == (j = ve.format) ? void 0 : j.bookmark_cover, ve),
                        alt: (0, h.FB)(null == (L = ve.properties) ? void 0 : L.title),
                        style: { objectFit: 'cover' },
                      })
                    )
                )
              )
            }
            case 'toggle':
              return o.createElement(
                'details',
                { className: $('notion-toggle', Te) },
                o.createElement(
                  'summary',
                  null,
                  o.createElement(te, {
                    value: null == (z = ve.properties) ? void 0 : z.title,
                    block: ve,
                  })
                ),
                o.createElement('div', null, De)
              )
            case 'table_of_contents': {
              const e = (0, h.cj)(ve, ae)
              if (!e) return null
              const t = (0, h.Ru)(e, ae),
                n = null == (S = ve.format) ? void 0 : S.block_color
              return o.createElement(
                'div',
                { className: $('notion-table-of-contents', n && `notion-${n}`, Te) },
                t.map((e) =>
                  o.createElement(
                    'a',
                    {
                      key: e.id,
                      href: `#${(0, h.Gw)(e.id)}`,
                      className: 'notion-table-of-contents-item',
                    },
                    o.createElement(
                      'span',
                      {
                        className: 'notion-table-of-contents-item-body',
                        style: { display: 'inline-block', marginLeft: 24 * e.indentLevel },
                      },
                      e.text
                    )
                  )
                )
              )
            }
            case 'to_do': {
              const e =
                'Yes' ===
                (null ==
                (T = null == (M = null == (I = ve.properties) ? void 0 : I.checked) ? void 0 : M[0])
                  ? void 0
                  : T[0])
              return o.createElement(
                'div',
                { className: $('notion-to-do', Te) },
                o.createElement(
                  'div',
                  { className: 'notion-to-do-item' },
                  o.createElement(ee.Checkbox, { blockId: Te, isChecked: e }),
                  o.createElement(
                    'div',
                    { className: $('notion-to-do-body', e && 'notion-to-do-checked') },
                    o.createElement(te, {
                      value: null == (R = ve.properties) ? void 0 : R.title,
                      block: ve,
                    })
                  )
                ),
                o.createElement('div', { className: 'notion-to-do-children' }, De)
              )
            }
            case 'transclusion_container':
              return o.createElement('div', { className: $('notion-sync-block', Te) }, De)
            case 'transclusion_reference':
              return o.createElement(_e, P({ block: ve, level: ge + 1 }, e))
            case 'alias': {
              const e =
                  null ==
                  (W = null == (U = null == ve ? void 0 : ve.format) ? void 0 : U.alias_pointer)
                    ? void 0
                    : W.id,
                t = null == (V = ae.block[e]) ? void 0 : V.value
              return t
                ? o.createElement(
                    ee.PageLink,
                    { className: $('notion-page-link', e), href: ie(e) },
                    o.createElement(ne, { block: t })
                  )
                : (console.log('"alias" missing block', e), null)
            }
            case 'table':
              return o.createElement(
                'table',
                { className: $('notion-simple-table', Te) },
                o.createElement('tbody', null, De)
              )
            case 'table_row': {
              const e = null == (Z = ae.block[ve.parent_id]) ? void 0 : Z.value,
                t = null == (G = e.format) ? void 0 : G.table_block_column_order,
                n = null == (Q = e.format) ? void 0 : Q.table_block_column_format,
                r = null == (Y = ve.format) ? void 0 : Y.block_color
              return e && t
                ? o.createElement(
                    'tr',
                    { className: $('notion-simple-table-row', r && `notion-${r}`, Te) },
                    t.map((e) => {
                      var t, r, a
                      const i = null == (t = null == n ? void 0 : n[e]) ? void 0 : t.color
                      return o.createElement(
                        'td',
                        {
                          key: e,
                          className: i ? `notion-${i}` : '',
                          style: {
                            width:
                              (null == (r = null == n ? void 0 : n[e]) ? void 0 : r.width) || 120,
                          },
                        },
                        o.createElement(
                          'div',
                          { className: 'notion-simple-table-cell' },
                          o.createElement(te, {
                            value: (null == (a = ve.properties) ? void 0 : a[e]) || [['\u3164']],
                            block: ve,
                          })
                        )
                      )
                    })
                  )
                : null
            }
            case 'external_object_instance':
              return o.createElement(J, { block: ve, className: Te })
            default:
              return (
                console.log('Unsupported type ' + ve.type, JSON.stringify(ve, null, 2)),
                o.createElement('div', null)
              )
          }
          return null
        },
        Pe = (e) => {
          var t = e,
            {
              components: n,
              recordMap: r,
              mapPageUrl: a,
              mapImageUrl: i,
              searchNotion: u,
              isShowingSearch: l,
              onHideSearch: c,
              fullPage: s,
              rootPageId: d,
              rootDomain: m,
              darkMode: f,
              previewImages: h,
              forceCustomImages: v,
              showCollectionViewDropdown: D,
              linkTableTitleProperties: g,
              isLinkCollectionToUrlProperty: E,
              isImageZoomable: y = !0,
              showTableOfContents: b,
              minTableOfContentsItems: F,
              defaultPageIcon: C,
              defaultPageCover: w,
              defaultPageCoverPosition: k,
            } = t,
            A = L(t, [
              'components',
              'recordMap',
              'mapPageUrl',
              'mapImageUrl',
              'searchNotion',
              'isShowingSearch',
              'onHideSearch',
              'fullPage',
              'rootPageId',
              'rootDomain',
              'darkMode',
              'previewImages',
              'forceCustomImages',
              'showCollectionViewDropdown',
              'linkTableTitleProperties',
              'isLinkCollectionToUrlProperty',
              'isImageZoomable',
              'showTableOfContents',
              'minTableOfContentsItems',
              'defaultPageIcon',
              'defaultPageCover',
              'defaultPageCoverPosition',
            ])
          const _ = o.useMemo(
            () =>
              'undefined' !== typeof window &&
              p({ background: 'rgba(0, 0, 0, 0.8)', minZoomScale: 2, margin: Le() }),
            []
          )
          return o.createElement(
            me,
            {
              components: n,
              recordMap: r,
              mapPageUrl: a,
              mapImageUrl: i,
              searchNotion: u,
              isShowingSearch: l,
              onHideSearch: c,
              fullPage: s,
              rootPageId: d,
              rootDomain: m,
              darkMode: f,
              previewImages: h,
              forceCustomImages: v,
              showCollectionViewDropdown: D,
              linkTableTitleProperties: g,
              isLinkCollectionToUrlProperty: E,
              showTableOfContents: b,
              minTableOfContentsItems: F,
              defaultPageIcon: C,
              defaultPageCover: w,
              defaultPageCoverPosition: k,
              zoom: y ? _ : null,
            },
            o.createElement(je, P({}, A))
          )
        },
        je = (e) => {
          var t,
            n,
            r = e,
            { level: a = 0, blockId: i } = r,
            u = L(r, ['level', 'blockId'])
          const { recordMap: l } = pe(),
            c = i || Object.keys(l.block)[0],
            s = null == (t = l.block[c]) ? void 0 : t.value
          return s
            ? o.createElement(
                xe,
                P({ key: c, level: a, block: s }, u),
                null == (n = null == s ? void 0 : s.content)
                  ? void 0
                  : n.map((e) => o.createElement(je, P({ key: e, blockId: e, level: a + 1 }, u)))
              )
            : (console.warn('missing block', i), null)
        }
      function Le() {
        const e = window.innerWidth
        return e < 500 ? 8 : e < 800 ? 20 : e < 1280 ? 30 : e < 1600 ? 40 : e < 1920 ? 48 : 72
      }
    },
  },
])
