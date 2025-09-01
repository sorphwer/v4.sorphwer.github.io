;(self.webpackChunk_N_E = self.webpackChunk_N_E || []).push([
  [509],
  {
    1987: function (e, t, n) {
      !(function () {
        var t = {
            477: function (e) {
              'use strict'
              e.exports = n(7673)
            },
          },
          r = {}
        function o(e) {
          var n = r[e]
          if (void 0 !== n) return n.exports
          var a = (r[e] = { exports: {} }),
            i = !0
          try {
            t[e](a, a.exports, o), (i = !1)
          } finally {
            i && delete r[e]
          }
          return a.exports
        }
        o.ab = '//'
        var a = {}
        !(function () {
          var e,
            t = a,
            n = (e = o(477)) && 'object' == typeof e && 'default' in e ? e.default : e,
            r = /https?|ftp|gopher|file/
          function i(e) {
            'string' == typeof e && (e = b(e))
            var t = (function (e, t, n) {
              var r = e.auth,
                o = e.hostname,
                a = e.protocol || '',
                i = e.pathname || '',
                c = e.hash || '',
                u = e.query || '',
                s = !1
              ;(r = r ? encodeURIComponent(r).replace(/%3A/i, ':') + '@' : ''),
                e.host
                  ? (s = r + e.host)
                  : o &&
                    ((s = r + (~o.indexOf(':') ? '[' + o + ']' : o)),
                    e.port && (s += ':' + e.port)),
                u && 'object' == typeof u && (u = t.encode(u))
              var l = e.search || (u && '?' + u) || ''
              return (
                a && ':' !== a.substr(-1) && (a += ':'),
                e.slashes || ((!a || n.test(a)) && !1 !== s)
                  ? ((s = '//' + (s || '')), i && '/' !== i[0] && (i = '/' + i))
                  : s || (s = ''),
                c && '#' !== c[0] && (c = '#' + c),
                l && '?' !== l[0] && (l = '?' + l),
                {
                  protocol: a,
                  host: s,
                  pathname: (i = i.replace(/[?#]/g, encodeURIComponent)),
                  search: (l = l.replace('#', '%23')),
                  hash: c,
                }
              )
            })(e, n, r)
            return '' + t.protocol + t.host + t.pathname + t.search + t.hash
          }
          var c = 'http://',
            u = 'w.w',
            s = c + u,
            l = /^([a-z0-9.+-]*:\/\/\/)([a-z0-9.+-]:\/*)?/i,
            f = /https?|ftp|gopher|file/
          function p(e, t) {
            var n = 'string' == typeof e ? b(e) : e
            e = 'object' == typeof e ? i(e) : e
            var r = b(t),
              o = ''
            n.protocol &&
              !n.slashes &&
              ((o = n.protocol),
              (e = e.replace(n.protocol, '')),
              (o += '/' === t[0] || '/' === e[0] ? '/' : '')),
              o &&
                r.protocol &&
                ((o = ''), r.slashes || ((o = r.protocol), (t = t.replace(r.protocol, ''))))
            var a = e.match(l)
            a &&
              !r.protocol &&
              ((e = e.substr((o = a[1] + (a[2] || '')).length)),
              /^\/\/[^/]/.test(t) && (o = o.slice(0, -1)))
            var u = new URL(e, s + '/'),
              p = new URL(t, u).toString().replace(s, ''),
              d = r.protocol || n.protocol
            return (
              (d += n.slashes || r.slashes ? '//' : ''),
              !o && d ? (p = p.replace(c, d)) : o && (p = p.replace(c, '')),
              f.test(p) ||
                ~t.indexOf('.') ||
                '/' === e.slice(-1) ||
                '/' === t.slice(-1) ||
                '/' !== p.slice(-1) ||
                (p = p.slice(0, -1)),
              o && (p = o + ('/' === p[0] ? p.substr(1) : p)),
              p
            )
          }
          function d() {}
          ;(d.prototype.parse = b),
            (d.prototype.format = i),
            (d.prototype.resolve = p),
            (d.prototype.resolveObject = p)
          var h = /^https?|ftp|gopher|file/,
            g = /^(.*?)([#?].*)/,
            v = /^([a-z0-9.+-]*:)(\/{0,3})(.*)/i,
            y = /^([a-z0-9.+-]*:)?\/\/\/*/i,
            m = /^([a-z0-9.+-]*:)(\/{0,2})\[(.*)\]$/i
          function b(e, t, r) {
            if (
              (void 0 === t && (t = !1),
              void 0 === r && (r = !1),
              e && 'object' == typeof e && e instanceof d)
            )
              return e
            var o = (e = e.trim()).match(g)
            ;(e = o ? o[1].replace(/\\/g, '/') + o[2] : e.replace(/\\/g, '/')),
              m.test(e) && '/' !== e.slice(-1) && (e += '/')
            var a = !/(^javascript)/.test(e) && e.match(v),
              c = y.test(e),
              l = ''
            a &&
              (h.test(a[1]) || ((l = a[1].toLowerCase()), (e = '' + a[2] + a[3])),
              a[2] || ((c = !1), h.test(a[1]) ? ((l = a[1]), (e = '' + a[3])) : (e = '//' + a[3])),
              (3 !== a[2].length && 1 !== a[2].length) || ((l = a[1]), (e = '/' + a[3])))
            var f,
              p = (o ? o[1] : e).match(/^https?:\/\/[^/]+(:[0-9]+)(?=\/|$)/),
              b = p && p[1],
              k = new d(),
              w = '',
              E = ''
            try {
              f = new URL(e)
            } catch (t) {
              ;(w = t),
                l ||
                  r ||
                  !/^\/\//.test(e) ||
                  /^\/\/.+[@.]/.test(e) ||
                  ((E = '/'), (e = e.substr(1)))
              try {
                f = new URL(e, s)
              } catch (e) {
                return (k.protocol = l), (k.href = l), k
              }
            }
            ;(k.slashes = c && !E),
              (k.host = f.host === u ? '' : f.host),
              (k.hostname = f.hostname === u ? '' : f.hostname.replace(/(\[|\])/g, '')),
              (k.protocol = w ? l || null : f.protocol),
              (k.search = f.search.replace(/\\/g, '%5C')),
              (k.hash = f.hash.replace(/\\/g, '%5C'))
            var R = e.split('#')
            !k.search && ~R[0].indexOf('?') && (k.search = '?'),
              k.hash || '' !== R[1] || (k.hash = '#'),
              (k.query = t ? n.decode(f.search.substr(1)) : k.search.substr(1)),
              (k.pathname =
                E +
                (a
                  ? (function (e) {
                      return e
                        .replace(/['^|`]/g, function (e) {
                          return '%' + e.charCodeAt().toString(16).toUpperCase()
                        })
                        .replace(/((?:%[0-9A-F]{2})+)/g, function (e, t) {
                          try {
                            return decodeURIComponent(t)
                              .split('')
                              .map(function (e) {
                                var t = e.charCodeAt()
                                return t > 256 || /^[a-z0-9]$/i.test(e)
                                  ? e
                                  : '%' + t.toString(16).toUpperCase()
                              })
                              .join('')
                          } catch (e) {
                            return t
                          }
                        })
                    })(f.pathname)
                  : f.pathname)),
              'about:' === k.protocol &&
                'blank' === k.pathname &&
                ((k.protocol = ''), (k.pathname = '')),
              w && '/' !== e[0] && (k.pathname = k.pathname.substr(1)),
              l && !h.test(l) && '/' !== e.slice(-1) && '/' === k.pathname && (k.pathname = ''),
              (k.path = k.pathname + k.search),
              (k.auth = [f.username, f.password].map(decodeURIComponent).filter(Boolean).join(':')),
              (k.port = f.port),
              b && !k.host.endsWith(b) && ((k.host += b), (k.port = b.slice(1))),
              (k.href = E ? '' + k.pathname + k.search + k.hash : i(k))
            var P = /^(file)/.test(k.href) ? ['host', 'hostname'] : []
            return (
              Object.keys(k).forEach(function (e) {
                ~P.indexOf(e) || (k[e] = k[e] || null)
              }),
              k
            )
          }
          ;(t.parse = b),
            (t.format = i),
            (t.resolve = p),
            (t.resolveObject = function (e, t) {
              return b(p(e, t))
            }),
            (t.Url = d)
        })(),
          (e.exports = a)
      })()
    },
    2587: function (e) {
      'use strict'
      function t(e, t) {
        return Object.prototype.hasOwnProperty.call(e, t)
      }
      e.exports = function (e, n, r, o) {
        ;(n = n || '&'), (r = r || '=')
        var a = {}
        if ('string' !== typeof e || 0 === e.length) return a
        var i = /\+/g
        e = e.split(n)
        var c = 1e3
        o && 'number' === typeof o.maxKeys && (c = o.maxKeys)
        var u = e.length
        c > 0 && u > c && (u = c)
        for (var s = 0; s < u; ++s) {
          var l,
            f,
            p,
            d,
            h = e[s].replace(i, '%20'),
            g = h.indexOf(r)
          g >= 0 ? ((l = h.substr(0, g)), (f = h.substr(g + 1))) : ((l = h), (f = '')),
            (p = decodeURIComponent(l)),
            (d = decodeURIComponent(f)),
            t(a, p) ? (Array.isArray(a[p]) ? a[p].push(d) : (a[p] = [a[p], d])) : (a[p] = d)
        }
        return a
      }
    },
    2361: function (e) {
      'use strict'
      var t = function (e) {
        switch (typeof e) {
          case 'string':
            return e
          case 'boolean':
            return e ? 'true' : 'false'
          case 'number':
            return isFinite(e) ? e : ''
          default:
            return ''
        }
      }
      e.exports = function (e, n, r, o) {
        return (
          (n = n || '&'),
          (r = r || '='),
          null === e && (e = void 0),
          'object' === typeof e
            ? Object.keys(e)
                .map(function (o) {
                  var a = encodeURIComponent(t(o)) + r
                  return Array.isArray(e[o])
                    ? e[o]
                        .map(function (e) {
                          return a + encodeURIComponent(t(e))
                        })
                        .join(n)
                    : a + encodeURIComponent(t(e[o]))
                })
                .filter(Boolean)
                .join(n)
            : o
            ? encodeURIComponent(t(o)) + r + encodeURIComponent(t(e))
            : ''
        )
      }
    },
    7673: function (e, t, n) {
      'use strict'
      ;(t.decode = t.parse = n(2587)), (t.encode = t.stringify = n(2361))
    },
    6402: function (e, t, n) {
      'use strict'
      n.r(t),
        n.d(t, {
          Pdf: function () {
            return rt
          },
        })
      var r = n(1720),
        o = n(8783),
        a = n(7462),
        i = n(3366)
      function c(e, t) {
        if (null == e) return {}
        var n,
          r,
          o = (0, i.Z)(e, t)
        if (Object.getOwnPropertySymbols) {
          var a = Object.getOwnPropertySymbols(e)
          for (r = 0; r < a.length; r++)
            (n = a[r]),
              t.indexOf(n) >= 0 ||
                (Object.prototype.propertyIsEnumerable.call(e, n) && (o[n] = e[n]))
        }
        return o
      }
      function u(e) {
        return (
          (u =
            'function' === typeof Symbol && 'symbol' === typeof Symbol.iterator
              ? function (e) {
                  return typeof e
                }
              : function (e) {
                  return e &&
                    'function' === typeof Symbol &&
                    e.constructor === Symbol &&
                    e !== Symbol.prototype
                    ? 'symbol'
                    : typeof e
                }),
          u(e)
        )
      }
      function s(e, t) {
        if (!(e instanceof t)) throw new TypeError('Cannot call a class as a function')
      }
      function l(e, t) {
        for (var n = 0; n < t.length; n++) {
          var r = t[n]
          ;(r.enumerable = r.enumerable || !1),
            (r.configurable = !0),
            'value' in r && (r.writable = !0),
            Object.defineProperty(e, r.key, r)
        }
      }
      function f(e, t, n) {
        return t && l(e.prototype, t), n && l(e, n), e
      }
      var p = n(7326),
        d = n(9611)
      function h(e, t) {
        if ('function' !== typeof t && null !== t)
          throw new TypeError('Super expression must either be null or a function')
        ;(e.prototype = Object.create(t && t.prototype, {
          constructor: { value: e, writable: !0, configurable: !0 },
        })),
          t && (0, d.Z)(e, t)
      }
      function g(e, t) {
        if (t && ('object' === u(t) || 'function' === typeof t)) return t
        if (void 0 !== t)
          throw new TypeError('Derived constructors may only return object or undefined')
        return (0, p.Z)(e)
      }
      function v(e) {
        return (
          (v = Object.setPrototypeOf
            ? Object.getPrototypeOf
            : function (e) {
                return e.__proto__ || Object.getPrototypeOf(e)
              }),
          v(e)
        )
      }
      var y = n(4942),
        m = n(5697),
        b = n.n(m),
        k = function (e, t, n) {
          if (n || 2 === arguments.length)
            for (var r, o = 0, a = t.length; o < a; o++)
              (!r && o in t) || (r || (r = Array.prototype.slice.call(t, 0, o)), (r[o] = t[o]))
          return e.concat(r || Array.prototype.slice.call(t))
        },
        w = ['onKeyDown', 'onKeyPress', 'onKeyUp'],
        E = [
          'onClick',
          'onContextMenu',
          'onDoubleClick',
          'onMouseDown',
          'onMouseEnter',
          'onMouseLeave',
          'onMouseMove',
          'onMouseOut',
          'onMouseOver',
          'onMouseUp',
        ],
        R = ['onTouchCancel', 'onTouchEnd', 'onTouchMove', 'onTouchStart'],
        P = k(
          k(
            k(
              k(
                k(
                  k(
                    k(
                      k(
                        k(
                          k(
                            k(
                              k(
                                k(
                                  k(
                                    k(
                                      k(
                                        k(
                                          k([], ['onCopy', 'onCut', 'onPaste'], !0),
                                          [
                                            'onCompositionEnd',
                                            'onCompositionStart',
                                            'onCompositionUpdate',
                                          ],
                                          !0
                                        ),
                                        ['onFocus', 'onBlur'],
                                        !0
                                      ),
                                      ['onInput', 'onInvalid', 'onReset', 'onSubmit'],
                                      !0
                                    ),
                                    ['onLoad', 'onError'],
                                    !0
                                  ),
                                  w,
                                  !0
                                ),
                                [
                                  'onAbort',
                                  'onCanPlay',
                                  'onCanPlayThrough',
                                  'onDurationChange',
                                  'onEmptied',
                                  'onEncrypted',
                                  'onEnded',
                                  'onError',
                                  'onLoadedData',
                                  'onLoadedMetadata',
                                  'onLoadStart',
                                  'onPause',
                                  'onPlay',
                                  'onPlaying',
                                  'onProgress',
                                  'onRateChange',
                                  'onSeeked',
                                  'onSeeking',
                                  'onStalled',
                                  'onSuspend',
                                  'onTimeUpdate',
                                  'onVolumeChange',
                                  'onWaiting',
                                ],
                                !0
                              ),
                              E,
                              !0
                            ),
                            [
                              'onDrag',
                              'onDragEnd',
                              'onDragEnter',
                              'onDragExit',
                              'onDragLeave',
                              'onDragOver',
                              'onDragStart',
                              'onDrop',
                            ],
                            !0
                          ),
                          ['onSelect'],
                          !0
                        ),
                        R,
                        !0
                      ),
                      [
                        'onPointerDown',
                        'onPointerMove',
                        'onPointerUp',
                        'onPointerCancel',
                        'onGotPointerCapture',
                        'onLostPointerCapture',
                        'onPointerEnter',
                        'onPointerLeave',
                        'onPointerOver',
                        'onPointerOut',
                      ],
                      !0
                    ),
                    ['onScroll'],
                    !0
                  ),
                  ['onWheel'],
                  !0
                ),
                ['onAnimationStart', 'onAnimationEnd', 'onAnimationIteration'],
                !0
              ),
              ['onTransitionEnd'],
              !0
            ),
            ['onChange'],
            !0
          ),
          ['onToggle'],
          !0
        )
      function O(e, t) {
        var n = {}
        return (
          P.forEach(function (r) {
            var o = e[r]
            o &&
              (n[r] = t
                ? function (e) {
                    return o(e, t(r))
                  }
                : o)
          }),
          n
        )
      }
      function S(e) {
        var t = !1
        return {
          promise: new Promise(function (n, r) {
            e.then(function (e) {
              return !t && n(e)
            }).catch(function (e) {
              return !t && r(e)
            })
          }),
          cancel: function () {
            t = !0
          },
        }
      }
      function x() {
        return Array.prototype.slice
          .call(arguments)
          .reduce(function (e, t) {
            return e.concat(t)
          }, [])
          .filter(function (e) {
            return 'string' === typeof e
          })
          .join(' ')
      }
      var Z = 'Invariant failed'
      function C(e, t) {
        if (!e) throw new Error(Z)
      }
      var T = function (e, t) {},
        L = (0, r.createContext)(null)
      function D(e) {
        var t = e.children,
          n = e.type
        return r.default.createElement(
          'div',
          { className: 'react-pdf__message react-pdf__message--'.concat(n) },
          t
        )
      }
      D.propTypes = {
        children: b().node,
        type: b().oneOf(['error', 'loading', 'no-data']).isRequired,
      }
      var A = (function () {
          function e() {
            s(this, e), (this.externalLinkTarget = null), (this.externalLinkRel = null)
          }
          return (
            f(e, [
              {
                key: 'setDocument',
                value: function (e) {
                  this.pdfDocument = e
                },
              },
              {
                key: 'setViewer',
                value: function (e) {
                  this.pdfViewer = e
                },
              },
              {
                key: 'setExternalLinkRel',
                value: function (e) {
                  this.externalLinkRel = e
                },
              },
              {
                key: 'setExternalLinkTarget',
                value: function (e) {
                  this.externalLinkTarget = e
                },
              },
              { key: 'setHistory', value: function () {} },
              {
                key: 'pagesCount',
                get: function () {
                  return this.pdfDocument ? this.pdfDocument.numPages : 0
                },
              },
              {
                key: 'page',
                get: function () {
                  return this.pdfViewer.currentPageNumber
                },
                set: function (e) {
                  this.pdfViewer.currentPageNumber = e
                },
              },
              {
                key: 'rotation',
                get: function () {
                  return 0
                },
                set: function (e) {},
              },
              {
                key: 'goToDestination',
                value: function (e) {
                  var t = this
                  new Promise(function (n) {
                    'string' === typeof e
                      ? t.pdfDocument.getDestination(e).then(n)
                      : Array.isArray(e)
                      ? n(e)
                      : e.then(n)
                  }).then(function (n) {
                    C(Array.isArray(n), '"'.concat(n, '" is not a valid destination array.'))
                    var r = n[0]
                    new Promise(function (e) {
                      r instanceof Object
                        ? t.pdfDocument
                            .getPageIndex(r)
                            .then(function (t) {
                              e(t)
                            })
                            .catch(function () {
                              C(!1, '"'.concat(r, '" is not a valid page reference.'))
                            })
                        : 'number' === typeof r
                        ? e(r)
                        : C(!1, '"'.concat(r, '" is not a valid destination reference.'))
                    }).then(function (n) {
                      var r = n + 1
                      C(
                        r >= 1 && r <= t.pagesCount,
                        '"'.concat(r, '" is not a valid page number.')
                      ),
                        t.pdfViewer.scrollPageIntoView({ dest: e, pageIndex: n, pageNumber: r })
                    })
                  })
                },
              },
              {
                key: 'navigateTo',
                value: function (e) {
                  this.goToDestination(e)
                },
              },
              { key: 'goToPage', value: function () {} },
              {
                key: 'addLinkAttributes',
                value: function (e, t, n) {
                  ;(e.href = t),
                    (e.rel = this.externalLinkRel || 'noopener noreferrer nofollow'),
                    (e.target = n ? '_blank' : this.externalLinkTarget || '')
                },
              },
              {
                key: 'getDestinationHash',
                value: function () {
                  return '#'
                },
              },
              {
                key: 'getAnchorUrl',
                value: function () {
                  return '#'
                },
              },
              { key: 'setHash', value: function () {} },
              { key: 'executeNamedAction', value: function () {} },
              { key: 'cachePageRef', value: function () {} },
              {
                key: 'isPageVisible',
                value: function () {
                  return !0
                },
              },
              {
                key: 'isPageCached',
                value: function () {
                  return !0
                },
              },
            ]),
            e
          )
        })(),
        j = { NEED_PASSWORD: 1, INCORRECT_PASSWORD: 2 }
      function I(e, t) {
        ;(null == t || t > e.length) && (t = e.length)
        for (var n = 0, r = new Array(t); n < t; n++) r[n] = e[n]
        return r
      }
      function N(e, t) {
        if (e) {
          if ('string' === typeof e) return I(e, t)
          var n = Object.prototype.toString.call(e).slice(8, -1)
          return (
            'Object' === n && e.constructor && (n = e.constructor.name),
            'Map' === n || 'Set' === n
              ? Array.from(e)
              : 'Arguments' === n || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)
              ? I(e, t)
              : void 0
          )
        }
      }
      function _(e, t) {
        return (
          (function (e) {
            if (Array.isArray(e)) return e
          })(e) ||
          (function (e, t) {
            var n =
              null == e
                ? null
                : ('undefined' !== typeof Symbol && e[Symbol.iterator]) || e['@@iterator']
            if (null != n) {
              var r,
                o,
                a = [],
                i = !0,
                c = !1
              try {
                for (
                  n = n.call(e);
                  !(i = (r = n.next()).done) && (a.push(r.value), !t || a.length !== t);
                  i = !0
                );
              } catch (u) {
                ;(c = !0), (o = u)
              } finally {
                try {
                  i || null == n.return || n.return()
                } finally {
                  if (c) throw o
                }
              }
              return a
            }
          })(e, t) ||
          N(e, t) ||
          (function () {
            throw new TypeError(
              'Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.'
            )
          })()
        )
      }
      var B = 'undefined' !== typeof window,
        U = B && 'file:' === window.location.protocol
      function F(e) {
        return 'undefined' !== typeof e
      }
      function M(e) {
        return F(e) && null !== e
      }
      function q(e) {
        return e instanceof ArrayBuffer
      }
      function V(e) {
        return C(B), e instanceof Blob
      }
      function G(e) {
        return C(B), e instanceof File
      }
      function W(e) {
        return 'string' === typeof e && /^data:/.test(e)
      }
      function K(e) {
        C(W(e))
        var t = _(e.split(','), 2),
          n = t[0],
          r = t[1]
        return -1 !== n.split(';').indexOf('base64') ? atob(r) : unescape(r)
      }
      var H =
        'On Chromium based browsers, you can use --allow-file-access-from-files flag for debugging purposes.'
      function z() {
        T(
          !U,
          'Loading PDF as base64 strings/URLs may not work on protocols other than HTTP/HTTPS. '.concat(
            H
          )
        )
      }
      function $(e) {
        e && e.cancel && e.cancel()
      }
      function Y(e, t) {
        return (
          Object.defineProperty(e, 'width', {
            get: function () {
              return this.view[2] * t
            },
            configurable: !0,
          }),
          Object.defineProperty(e, 'height', {
            get: function () {
              return this.view[3] * t
            },
            configurable: !0,
          }),
          Object.defineProperty(e, 'originalWidth', {
            get: function () {
              return this.view[2]
            },
            configurable: !0,
          }),
          Object.defineProperty(e, 'originalHeight', {
            get: function () {
              return this.view[3]
            },
            configurable: !0,
          }),
          e
        )
      }
      function X(e) {
        return 'RenderingCancelledException' === e.name
      }
      function J(e) {
        return new Promise(function (t, n) {
          var r = new FileReader()
          return (
            (r.onload = function () {
              return t(new Uint8Array(r.result))
            }),
            (r.onerror = function (e) {
              switch (e.target.error.code) {
                case e.target.error.NOT_FOUND_ERR:
                  return n(new Error('Error while reading a file: File not found.'))
                case e.target.error.NOT_READABLE_ERR:
                  return n(new Error('Error while reading a file: File not readable.'))
                case e.target.error.SECURITY_ERR:
                  return n(new Error('Error while reading a file: Security error.'))
                case e.target.error.ABORT_ERR:
                  return n(new Error('Error while reading a file: Aborted.'))
                default:
                  return n(new Error('Error while reading a file.'))
              }
            }),
            r.readAsArrayBuffer(e),
            null
          )
        })
      }
      function Q(e) {
        return (
          (function (e) {
            if (Array.isArray(e)) return I(e)
          })(e) ||
          (function (e) {
            if (
              ('undefined' !== typeof Symbol && null != e[Symbol.iterator]) ||
              null != e['@@iterator']
            )
              return Array.from(e)
          })(e) ||
          N(e) ||
          (function () {
            throw new TypeError(
              'Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.'
            )
          })()
        )
      }
      var ee = (function () {
          var e = {}
          return (
            [].concat(Q(E), Q(R), Q(w)).forEach(function (t) {
              e[t] = b().func
            }),
            e
          )
        })(),
        te = [
          b().string,
          b().instanceOf(ArrayBuffer),
          b().shape({
            data: b().oneOfType([b().object, b().string]),
            httpHeaders: b().object,
            range: b().object,
            url: b().string,
            withCredentials: b().bool,
          }),
        ]
      'undefined' !== typeof File && te.push(b().instanceOf(File)),
        'undefined' !== typeof Blob && te.push(b().instanceOf(Blob))
      var ne = b().oneOfType([b().string, b().arrayOf(b().string)]),
        re = b().oneOfType(te),
        oe = b().instanceOf(A),
        ae =
          (b().oneOf(['_self', '_blank', '_parent', '_top']),
          b().shape({
            _transport: b().shape({ fontLoader: b().object.isRequired }).isRequired,
            commonObjs: b().shape({ _objs: b().object.isRequired }).isRequired,
            getAnnotations: b().func.isRequired,
            getTextContent: b().func.isRequired,
            getViewport: b().func.isRequired,
            render: b().func.isRequired,
          })),
        ie = b().oneOfType([
          b().shape({
            getDestination: b().func.isRequired,
            getOutline: b().func.isRequired,
            getPage: b().func.isRequired,
            numPages: b().number.isRequired,
          }),
          b().bool,
        ]),
        ce = b().oneOfType([b().func, b().shape({ current: b().any })]),
        ue = b().oneOf(['canvas', 'none', 'svg']),
        se = b().oneOf([0, 90, 180, 270]),
        le = ['url']
      function fe(e, t) {
        var n = Object.keys(e)
        if (Object.getOwnPropertySymbols) {
          var r = Object.getOwnPropertySymbols(e)
          t &&
            (r = r.filter(function (t) {
              return Object.getOwnPropertyDescriptor(e, t).enumerable
            })),
            n.push.apply(n, r)
        }
        return n
      }
      function pe(e) {
        for (var t = 1; t < arguments.length; t++) {
          var n = null != arguments[t] ? arguments[t] : {}
          t % 2
            ? fe(Object(n), !0).forEach(function (t) {
                ;(0, y.Z)(e, t, n[t])
              })
            : Object.getOwnPropertyDescriptors
            ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n))
            : fe(Object(n)).forEach(function (t) {
                Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(n, t))
              })
        }
        return e
      }
      function de(e) {
        var t = (function () {
          if ('undefined' === typeof Reflect || !Reflect.construct) return !1
          if (Reflect.construct.sham) return !1
          if ('function' === typeof Proxy) return !0
          try {
            return (
              Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})), !0
            )
          } catch (e) {
            return !1
          }
        })()
        return function () {
          var n,
            r = v(e)
          if (t) {
            var o = v(this).constructor
            n = Reflect.construct(r, arguments, o)
          } else n = r.apply(this, arguments)
          return g(this, n)
        }
      }
      var he = o.PDFDataRangeTransport,
        ge = (function (e) {
          h(n, e)
          var t = de(n)
          function n() {
            var e
            s(this, n)
            for (var r = arguments.length, a = new Array(r), i = 0; i < r; i++) a[i] = arguments[i]
            return (
              (e = t.call.apply(t, [this].concat(a))),
              (0, y.Z)((0, p.Z)(e), 'state', { pdf: null }),
              (0, y.Z)((0, p.Z)(e), 'viewer', {
                scrollPageIntoView: function (t) {
                  var n = t.dest,
                    r = t.pageIndex,
                    o = t.pageNumber,
                    a = e.props.onItemClick
                  if (a) a({ dest: n, pageIndex: r, pageNumber: o })
                  else {
                    var i = e.pages[r]
                    i
                      ? i.scrollIntoView()
                      : T(
                          !1,
                          'An internal link leading to page '.concat(
                            o,
                            ' was clicked, but neither <Document> was provided with onItemClick nor it was able to find the page within itself. Either provide onItemClick to <Document> and handle navigating by yourself or ensure that all pages are rendered within <Document>.'
                          )
                        )
                  }
                },
              }),
              (0, y.Z)((0, p.Z)(e), 'linkService', new A()),
              (0, y.Z)((0, p.Z)(e), 'loadDocument', function () {
                $(e.runningTask), e.loadingTask && e.loadingTask.destroy()
                var t = S(e.findDocumentSource())
                ;(e.runningTask = t),
                  t.promise
                    .then(function (t) {
                      if ((e.onSourceSuccess(), t)) {
                        e.setState(function (e) {
                          return e.pdf ? { pdf: null } : null
                        })
                        var n = e.props,
                          r = n.options,
                          a = n.onLoadProgress,
                          i = n.onPassword
                        ;(e.loadingTask = o.getDocument(pe(pe({}, t), r))),
                          (e.loadingTask.onPassword = i),
                          a && (e.loadingTask.onProgress = a)
                        var c = S(e.loadingTask.promise)
                        ;(e.runningTask = c),
                          c.promise
                            .then(function (t) {
                              e.setState(function (e) {
                                return e.pdf && e.pdf.fingerprint === t.fingerprint
                                  ? null
                                  : { pdf: t }
                              }, e.onLoadSuccess)
                            })
                            .catch(function (t) {
                              e.onLoadError(t)
                            })
                      }
                    })
                    .catch(function (t) {
                      e.onSourceError(t)
                    })
              }),
              (0, y.Z)((0, p.Z)(e), 'setupLinkService', function () {
                var t = e.props,
                  n = t.externalLinkRel,
                  r = t.externalLinkTarget
                e.linkService.setViewer(e.viewer),
                  e.linkService.setExternalLinkRel(n),
                  e.linkService.setExternalLinkTarget(r)
              }),
              (0, y.Z)((0, p.Z)(e), 'onSourceSuccess', function () {
                var t = e.props.onSourceSuccess
                t && t()
              }),
              (0, y.Z)((0, p.Z)(e), 'onSourceError', function (t) {
                T(t)
                var n = e.props.onSourceError
                n && n(t)
              }),
              (0, y.Z)((0, p.Z)(e), 'onLoadSuccess', function () {
                var t = e.props.onLoadSuccess,
                  n = e.state.pdf
                t && t(n), (e.pages = new Array(n.numPages)), e.linkService.setDocument(n)
              }),
              (0, y.Z)((0, p.Z)(e), 'onLoadError', function (t) {
                e.setState({ pdf: !1 }), T(t)
                var n = e.props.onLoadError
                n && n(t)
              }),
              (0, y.Z)((0, p.Z)(e), 'findDocumentSource', function () {
                return new Promise(function (t) {
                  var n = e.props.file
                  if ((n || t(null), 'string' === typeof n)) {
                    if (W(n)) {
                      var r = K(n)
                      t({ data: r })
                    }
                    z(), t({ url: n })
                  }
                  if (
                    (n instanceof he && t({ range: n }),
                    q(n) && t({ data: n }),
                    B && (V(n) || G(n)))
                  )
                    J(n).then(function (e) {
                      t({ data: e })
                    })
                  else {
                    if (
                      (C('object' === u(n)),
                      C(n.url || n.data || n.range),
                      'string' === typeof n.url)
                    ) {
                      if (W(n.url)) {
                        var o = n.url,
                          a = c(n, le),
                          i = K(o)
                        t(pe({ data: i }, a))
                      }
                      z()
                    }
                    t(n)
                  }
                })
              }),
              (0, y.Z)((0, p.Z)(e), 'registerPage', function (t, n) {
                e.pages[t] = n
              }),
              (0, y.Z)((0, p.Z)(e), 'unregisterPage', function (t) {
                delete e.pages[t]
              }),
              e
            )
          }
          return (
            f(n, [
              {
                key: 'componentDidMount',
                value: function () {
                  this.loadDocument(), this.setupLinkService()
                },
              },
              {
                key: 'componentDidUpdate',
                value: function (e) {
                  this.props.file !== e.file && this.loadDocument()
                },
              },
              {
                key: 'componentWillUnmount',
                value: function () {
                  $(this.runningTask), this.loadingTask && this.loadingTask.destroy()
                },
              },
              {
                key: 'childContext',
                get: function () {
                  var e = this.linkService,
                    t = this.registerPage,
                    n = this.unregisterPage,
                    r = this.props,
                    o = r.imageResourcesPath,
                    a = r.renderMode,
                    i = r.rotate
                  return {
                    imageResourcesPath: o,
                    linkService: e,
                    pdf: this.state.pdf,
                    registerPage: t,
                    renderMode: a,
                    rotate: i,
                    unregisterPage: n,
                  }
                },
              },
              {
                key: 'eventProps',
                get: function () {
                  var e = this
                  return O(this.props, function () {
                    return e.state.pdf
                  })
                },
              },
              {
                key: 'renderChildren',
                value: function () {
                  var e = this.props.children
                  return r.default.createElement(L.Provider, { value: this.childContext }, e)
                },
              },
              {
                key: 'renderContent',
                value: function () {
                  var e = this.props.file,
                    t = this.state.pdf
                  if (!e) {
                    var n = this.props.noData
                    return r.default.createElement(
                      D,
                      { type: 'no-data' },
                      'function' === typeof n ? n() : n
                    )
                  }
                  if (null === t) {
                    var o = this.props.loading
                    return r.default.createElement(
                      D,
                      { type: 'loading' },
                      'function' === typeof o ? o() : o
                    )
                  }
                  if (!1 === t) {
                    var a = this.props.error
                    return r.default.createElement(
                      D,
                      { type: 'error' },
                      'function' === typeof a ? a() : a
                    )
                  }
                  return this.renderChildren()
                },
              },
              {
                key: 'render',
                value: function () {
                  var e = this.props,
                    t = e.className,
                    n = e.inputRef
                  return r.default.createElement(
                    'div',
                    (0, a.Z)({ className: x('react-pdf__Document', t), ref: n }, this.eventProps),
                    this.renderContent()
                  )
                },
              },
            ]),
            n
          )
        })(r.PureComponent)
      ge.defaultProps = {
        error: 'Failed to load PDF file.',
        loading: 'Loading PDF\u2026',
        noData: 'No PDF file specified.',
        onPassword: function (e, t) {
          switch (t) {
            case j.NEED_PASSWORD:
              e(prompt('Enter the password to open this PDF file.'))
              break
            case j.INCORRECT_PASSWORD:
              e(prompt('Invalid password. Please try again.'))
          }
        },
      }
      var ve = b().oneOfType([b().func, b().node])
      ge.propTypes = pe(
        pe({}, ee),
        {},
        {
          children: b().node,
          className: ne,
          error: ve,
          externalLinkRel: b().string,
          externalLinkTarget: b().string,
          file: re,
          imageResourcesPath: b().string,
          inputRef: ce,
          loading: ve,
          noData: ve,
          onItemClick: b().func,
          onLoadError: b().func,
          onLoadProgress: b().func,
          onLoadSuccess: b().func,
          onPassword: b().func,
          onSourceError: b().func,
          onSourceSuccess: b().func,
          rotate: b().number,
        }
      )
      var ye = (0, r.createContext)(null),
        me = (function () {
          function e(t) {
            var n = t.num,
              r = t.gen
            s(this, e), (this.num = n), (this.gen = r)
          }
          return (
            f(e, [
              {
                key: 'toString',
                value: function () {
                  var e = ''.concat(this.num, 'R')
                  return 0 !== this.gen && (e += this.gen), e
                },
              },
            ]),
            e
          )
        })(),
        be = ['item']
      function ke(e) {
        var t = (function () {
          if ('undefined' === typeof Reflect || !Reflect.construct) return !1
          if (Reflect.construct.sham) return !1
          if ('function' === typeof Proxy) return !0
          try {
            return (
              Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})), !0
            )
          } catch (e) {
            return !1
          }
        })()
        return function () {
          var n,
            r = v(e)
          if (t) {
            var o = v(this).constructor
            n = Reflect.construct(r, arguments, o)
          } else n = r.apply(this, arguments)
          return g(this, n)
        }
      }
      var we = (function (e) {
          h(n, e)
          var t = ke(n)
          function n() {
            var e
            s(this, n)
            for (var r = arguments.length, o = new Array(r), a = 0; a < r; a++) o[a] = arguments[a]
            return (
              (e = t.call.apply(t, [this].concat(o))),
              (0, y.Z)((0, p.Z)(e), 'getDestination', function () {
                return new Promise(function (t, n) {
                  var r = e.props,
                    o = r.item,
                    a = r.pdf
                  return (
                    F(e.destination) ||
                      ('string' === typeof o.dest
                        ? a.getDestination(o.dest).then(t).catch(n)
                        : t(o.dest)),
                    e.destination
                  )
                }).then(function (t) {
                  return (e.destination = t), t
                })
              }),
              (0, y.Z)((0, p.Z)(e), 'getPageIndex', function () {
                return new Promise(function (t, n) {
                  var r = e.props.pdf
                  F(e.pageIndex) && t(e.pageIndex),
                    e.getDestination().then(function (e) {
                      if (e) {
                        var o = _(e, 1)[0]
                        r.getPageIndex(new me(o)).then(t).catch(n)
                      }
                    })
                }).then(function (t) {
                  return (e.pageIndex = t), e.pageIndex
                })
              }),
              (0, y.Z)((0, p.Z)(e), 'getPageNumber', function () {
                return new Promise(function (t, n) {
                  F(e.pageNumber) && t(e.pageNumber),
                    e
                      .getPageIndex()
                      .then(function (e) {
                        t(e + 1)
                      })
                      .catch(n)
                }).then(function (t) {
                  return (e.pageNumber = t), t
                })
              }),
              (0, y.Z)((0, p.Z)(e), 'onClick', function (t) {
                var n = e.props.onClick
                return (
                  t.preventDefault(),
                  !!n &&
                    Promise.all([e.getDestination(), e.getPageIndex(), e.getPageNumber()]).then(
                      function (e) {
                        var t = _(e, 3),
                          r = t[0],
                          o = t[1],
                          a = t[2]
                        n({ dest: r, pageIndex: o, pageNumber: a })
                      }
                    )
                )
              }),
              e
            )
          }
          return (
            f(n, [
              {
                key: 'renderSubitems',
                value: function () {
                  var e = this.props,
                    t = e.item,
                    o = c(e, be)
                  if (!t.items || !t.items.length) return null
                  var i = t.items
                  return r.default.createElement(
                    'ul',
                    null,
                    i.map(function (e, t) {
                      return r.default.createElement(
                        n,
                        (0, a.Z)(
                          { key: 'string' === typeof e.destination ? e.destination : t, item: e },
                          o
                        )
                      )
                    })
                  )
                },
              },
              {
                key: 'render',
                value: function () {
                  var e = this.props.item
                  return r.default.createElement(
                    'li',
                    null,
                    r.default.createElement('a', { href: '#', onClick: this.onClick }, e.title),
                    this.renderSubitems()
                  )
                },
              },
            ]),
            n
          )
        })(r.PureComponent),
        Ee = b().oneOfType([b().string, b().arrayOf(b().any)])
      we.propTypes = {
        item: b().shape({
          dest: Ee,
          items: b().arrayOf(b().shape({ dest: Ee, title: b().string })),
          title: b().string,
        }).isRequired,
        onClick: b().func,
        pdf: ie.isRequired,
      }
      var Re = function (e) {
        return r.default.createElement(L.Consumer, null, function (t) {
          return r.default.createElement(ye.Consumer, null, function (n) {
            return r.default.createElement(we, (0, a.Z)({}, t, n, e))
          })
        })
      }
      function Pe(e, t) {
        var n = Object.keys(e)
        if (Object.getOwnPropertySymbols) {
          var r = Object.getOwnPropertySymbols(e)
          t &&
            (r = r.filter(function (t) {
              return Object.getOwnPropertyDescriptor(e, t).enumerable
            })),
            n.push.apply(n, r)
        }
        return n
      }
      function Oe(e) {
        var t = (function () {
          if ('undefined' === typeof Reflect || !Reflect.construct) return !1
          if (Reflect.construct.sham) return !1
          if ('function' === typeof Proxy) return !0
          try {
            return (
              Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})), !0
            )
          } catch (e) {
            return !1
          }
        })()
        return function () {
          var n,
            r = v(e)
          if (t) {
            var o = v(this).constructor
            n = Reflect.construct(r, arguments, o)
          } else n = r.apply(this, arguments)
          return g(this, n)
        }
      }
      var Se = (function (e) {
        h(n, e)
        var t = Oe(n)
        function n() {
          var e
          s(this, n)
          for (var r = arguments.length, o = new Array(r), a = 0; a < r; a++) o[a] = arguments[a]
          return (
            (e = t.call.apply(t, [this].concat(o))),
            (0, y.Z)((0, p.Z)(e), 'state', { outline: null }),
            (0, y.Z)((0, p.Z)(e), 'loadOutline', function () {
              var t = e.props.pdf
              e.setState(function (e) {
                return e.outline ? { outline: null } : null
              })
              var n = S(t.getOutline())
              ;(e.runningTask = n),
                n.promise
                  .then(function (t) {
                    e.setState({ outline: t }, e.onLoadSuccess)
                  })
                  .catch(function (t) {
                    e.onLoadError(t)
                  })
            }),
            (0, y.Z)((0, p.Z)(e), 'onLoadSuccess', function () {
              var t = e.props.onLoadSuccess,
                n = e.state.outline
              t && t(n)
            }),
            (0, y.Z)((0, p.Z)(e), 'onLoadError', function (t) {
              e.setState({ outline: !1 }), T(t)
              var n = e.props.onLoadError
              n && n(t)
            }),
            (0, y.Z)((0, p.Z)(e), 'onItemClick', function (t) {
              var n = t.dest,
                r = t.pageIndex,
                o = t.pageNumber,
                a = e.props.onItemClick
              a && a({ dest: n, pageIndex: r, pageNumber: o })
            }),
            e
          )
        }
        return (
          f(n, [
            {
              key: 'componentDidMount',
              value: function () {
                C(this.props.pdf), this.loadOutline()
              },
            },
            {
              key: 'componentDidUpdate',
              value: function (e) {
                var t = this.props.pdf
                e.pdf && t !== e.pdf && this.loadOutline()
              },
            },
            {
              key: 'componentWillUnmount',
              value: function () {
                $(this.runningTask)
              },
            },
            {
              key: 'childContext',
              get: function () {
                return { onClick: this.onItemClick }
              },
            },
            {
              key: 'eventProps',
              get: function () {
                var e = this
                return O(this.props, function () {
                  return e.state.outline
                })
              },
            },
            {
              key: 'renderOutline',
              value: function () {
                var e = this.state.outline
                return r.default.createElement(
                  'ul',
                  null,
                  e.map(function (e, t) {
                    return r.default.createElement(Re, {
                      key: 'string' === typeof e.destination ? e.destination : t,
                      item: e,
                    })
                  })
                )
              },
            },
            {
              key: 'render',
              value: function () {
                var e = this.props.pdf,
                  t = this.state.outline
                if (!e || !t) return null
                var n = this.props,
                  o = n.className,
                  i = n.inputRef
                return r.default.createElement(
                  'div',
                  (0, a.Z)({ className: x('react-pdf__Outline', o), ref: i }, this.eventProps),
                  r.default.createElement(
                    ye.Provider,
                    { value: this.childContext },
                    this.renderOutline()
                  )
                )
              },
            },
          ]),
          n
        )
      })(r.PureComponent)
      Se.propTypes = (function (e) {
        for (var t = 1; t < arguments.length; t++) {
          var n = null != arguments[t] ? arguments[t] : {}
          t % 2
            ? Pe(Object(n), !0).forEach(function (t) {
                ;(0, y.Z)(e, t, n[t])
              })
            : Object.getOwnPropertyDescriptors
            ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n))
            : Pe(Object(n)).forEach(function (t) {
                Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(n, t))
              })
        }
        return e
      })(
        {
          className: ne,
          inputRef: ce,
          onItemClick: b().func,
          onLoadError: b().func,
          onLoadSuccess: b().func,
          pdf: ie,
        },
        ee
      )
      function xe() {
        for (var e = [], t = 0; t < arguments.length; t++) e[t] = arguments[t]
        var n = e.filter(Boolean)
        if (n.length <= 1) {
          var r = n[0]
          return r || null
        }
        return function (e) {
          n.forEach(function (t) {
            'function' === typeof t ? t(e) : t && (t.current = e)
          })
        }
      }
      var Ze = (0, r.createContext)(null)
      function Ce(e) {
        var t = (function () {
          if ('undefined' === typeof Reflect || !Reflect.construct) return !1
          if (Reflect.construct.sham) return !1
          if ('function' === typeof Proxy) return !0
          try {
            return (
              Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})), !0
            )
          } catch (e) {
            return !1
          }
        })()
        return function () {
          var n,
            r = v(e)
          if (t) {
            var o = v(this).constructor
            n = Reflect.construct(r, arguments, o)
          } else n = r.apply(this, arguments)
          return g(this, n)
        }
      }
      var Te = o.AnnotationMode,
        Le = (function (e) {
          h(n, e)
          var t = Ce(n)
          function n() {
            var e
            s(this, n)
            for (var o = arguments.length, a = new Array(o), i = 0; i < o; i++) a[i] = arguments[i]
            return (
              (e = t.call.apply(t, [this].concat(a))),
              (0, y.Z)((0, p.Z)(e), 'canvasElement', (0, r.createRef)()),
              (0, y.Z)((0, p.Z)(e), 'onRenderSuccess', function () {
                e.renderer = null
                var t = e.props,
                  n = t.onRenderSuccess,
                  r = t.page,
                  o = t.scale
                n && n(Y(r, o))
              }),
              (0, y.Z)((0, p.Z)(e), 'onRenderError', function (t) {
                if (!X(t)) {
                  T(t)
                  var n = e.props.onRenderError
                  n && n(t)
                }
              }),
              (0, y.Z)((0, p.Z)(e), 'drawPageOnCanvas', function () {
                var t = e.canvasElement.current
                if (!t) return null
                var n = (0, p.Z)(e),
                  r = n.renderViewport,
                  o = n.viewport,
                  a = e.props,
                  i = a.canvasBackground,
                  c = a.page,
                  u = a.renderForms
                ;(t.width = r.width),
                  (t.height = r.height),
                  (t.style.width = ''.concat(Math.floor(o.width), 'px')),
                  (t.style.height = ''.concat(Math.floor(o.height), 'px'))
                var s = {
                  annotationMode: u ? Te.ENABLE_FORMS : Te.ENABLE,
                  get canvasContext() {
                    return t.getContext('2d')
                  },
                  viewport: r,
                }
                return (
                  i && (s.background = i),
                  e.cancelRenderingTask(),
                  (e.renderer = c.render(s)),
                  e.renderer.promise.then(e.onRenderSuccess).catch(e.onRenderError)
                )
              }),
              e
            )
          }
          return (
            f(n, [
              {
                key: 'componentDidMount',
                value: function () {
                  this.drawPageOnCanvas()
                },
              },
              {
                key: 'componentDidUpdate',
                value: function (e) {
                  var t = this.props,
                    n = t.canvasBackground,
                    r = t.page,
                    o = t.renderForms
                  ;(n === e.canvasBackground && o === e.renderForms) ||
                    (r.cleanup(), this.drawPageOnCanvas())
                },
              },
              {
                key: 'componentWillUnmount',
                value: function () {
                  this.cancelRenderingTask()
                  var e = this.canvasElement.current
                  e && ((e.width = 0), (e.height = 0))
                },
              },
              {
                key: 'cancelRenderingTask',
                value: function () {
                  this.renderer && (this.renderer.cancel(), (this.renderer = null))
                },
              },
              {
                key: 'renderViewport',
                get: function () {
                  var e = this.props,
                    t = e.page,
                    n = e.rotate,
                    r = e.scale,
                    o = (B && window.devicePixelRatio) || 1
                  return t.getViewport({ scale: r * o, rotation: n })
                },
              },
              {
                key: 'viewport',
                get: function () {
                  var e = this.props,
                    t = e.page,
                    n = e.rotate,
                    r = e.scale
                  return t.getViewport({ scale: r, rotation: n })
                },
              },
              {
                key: 'render',
                value: function () {
                  var e = this.props.canvasRef
                  return r.default.createElement('canvas', {
                    className: 'react-pdf__Page__canvas',
                    dir: 'ltr',
                    ref: xe(e, this.canvasElement),
                    style: { display: 'block', userSelect: 'none' },
                  })
                },
              },
            ]),
            n
          )
        })(r.PureComponent)
      function De(e) {
        return r.default.createElement(Ze.Consumer, null, function (t) {
          return r.default.createElement(Le, (0, a.Z)({}, t, e))
        })
      }
      function Ae(e) {
        var t = (function () {
          if ('undefined' === typeof Reflect || !Reflect.construct) return !1
          if (Reflect.construct.sham) return !1
          if ('function' === typeof Proxy) return !0
          try {
            return (
              Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})), !0
            )
          } catch (e) {
            return !1
          }
        })()
        return function () {
          var n,
            r = v(e)
          if (t) {
            var o = v(this).constructor
            n = Reflect.construct(r, arguments, o)
          } else n = r.apply(this, arguments)
          return g(this, n)
        }
      }
      Le.propTypes = {
        canvasBackground: b().string,
        canvasRef: ce,
        onRenderError: b().func,
        onRenderSuccess: b().func,
        page: ae.isRequired,
        renderForms: b().bool,
        rotate: se,
        scale: b().number.isRequired,
      }
      var je = (function (e) {
        h(n, e)
        var t = Ae(n)
        function n() {
          var e
          s(this, n)
          for (var r = arguments.length, a = new Array(r), i = 0; i < r; i++) a[i] = arguments[i]
          return (
            (e = t.call.apply(t, [this].concat(a))),
            (0, y.Z)((0, p.Z)(e), 'state', { svg: null }),
            (0, y.Z)((0, p.Z)(e), 'onRenderSuccess', function () {
              e.renderer = null
              var t = e.props,
                n = t.onRenderSuccess,
                r = t.page,
                o = t.scale
              n && n(Y(r, o))
            }),
            (0, y.Z)((0, p.Z)(e), 'onRenderError', function (t) {
              if (!X(t)) {
                T(t)
                var n = e.props.onRenderError
                n && n(t)
              }
            }),
            (0, y.Z)((0, p.Z)(e), 'renderSVG', function () {
              var t = e.props.page
              return (
                (e.renderer = t.getOperatorList()),
                e.renderer
                  .then(function (n) {
                    var r = new o.SVGGraphics(t.commonObjs, t.objs)
                    e.renderer = r
                      .getSVG(n, e.viewport)
                      .then(function (t) {
                        e.setState({ svg: t }, e.onRenderSuccess)
                      })
                      .catch(e.onRenderError)
                  })
                  .catch(e.onRenderError)
              )
            }),
            (0, y.Z)((0, p.Z)(e), 'drawPageOnContainer', function (t) {
              var n = e.state.svg
              if (t && n) {
                t.firstElementChild || t.appendChild(n)
                var r = e.viewport,
                  o = r.width,
                  a = r.height
                n.setAttribute('width', o), n.setAttribute('height', a)
              }
            }),
            e
          )
        }
        return (
          f(n, [
            {
              key: 'componentDidMount',
              value: function () {
                this.renderSVG()
              },
            },
            {
              key: 'viewport',
              get: function () {
                var e = this.props,
                  t = e.page,
                  n = e.rotate,
                  r = e.scale
                return t.getViewport({ scale: r, rotation: n })
              },
            },
            {
              key: 'render',
              value: function () {
                var e = this,
                  t = this.viewport,
                  n = t.width,
                  o = t.height
                return r.default.createElement('div', {
                  className: 'react-pdf__Page__svg',
                  ref: function (t) {
                    return e.drawPageOnContainer(t)
                  },
                  style: {
                    display: 'block',
                    backgroundColor: 'white',
                    overflow: 'hidden',
                    width: n,
                    height: o,
                    userSelect: 'none',
                  },
                })
              },
            },
          ]),
          n
        )
      })(r.PureComponent)
      function Ie(e) {
        return r.default.createElement(Ze.Consumer, null, function (t) {
          return r.default.createElement(je, (0, a.Z)({}, t, e))
        })
      }
      function Ne(e) {
        var t = (function () {
          if ('undefined' === typeof Reflect || !Reflect.construct) return !1
          if (Reflect.construct.sham) return !1
          if ('function' === typeof Proxy) return !0
          try {
            return (
              Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})), !0
            )
          } catch (e) {
            return !1
          }
        })()
        return function () {
          var n,
            r = v(e)
          if (t) {
            var o = v(this).constructor
            n = Reflect.construct(r, arguments, o)
          } else n = r.apply(this, arguments)
          return g(this, n)
        }
      }
      je.propTypes = {
        onRenderError: b().func,
        onRenderSuccess: b().func,
        page: ae.isRequired,
        rotate: se,
        scale: b().number.isRequired,
      }
      var _e = (function (e) {
        h(n, e)
        var t = Ne(n)
        function n() {
          var e
          s(this, n)
          for (var o = arguments.length, a = new Array(o), i = 0; i < o; i++) a[i] = arguments[i]
          return (
            (e = t.call.apply(t, [this].concat(a))),
            (0, y.Z)((0, p.Z)(e), 'itemElement', (0, r.createRef)()),
            (0, y.Z)((0, p.Z)(e), 'getElementWidth', function (t) {
              var n = (0, p.Z)(e).sideways
              return t.getBoundingClientRect()[n ? 'height' : 'width']
            }),
            e
          )
        }
        return (
          f(n, [
            {
              key: 'componentDidMount',
              value: function () {
                this.alignTextItem()
              },
            },
            {
              key: 'componentDidUpdate',
              value: function () {
                this.alignTextItem()
              },
            },
            {
              key: 'unrotatedViewport',
              get: function () {
                var e = this.props,
                  t = e.page,
                  n = e.scale
                return t.getViewport({ scale: n })
              },
            },
            {
              key: 'rotate',
              get: function () {
                var e = this.props,
                  t = e.page
                return e.rotate - t.rotate
              },
            },
            {
              key: 'sideways',
              get: function () {
                return this.rotate % 180 !== 0
              },
            },
            {
              key: 'defaultSideways',
              get: function () {
                return this.unrotatedViewport.rotation % 180 !== 0
              },
            },
            {
              key: 'fontSize',
              get: function () {
                var e = this.props.transform,
                  t = this.defaultSideways,
                  n = _(e, 2),
                  r = n[0],
                  o = n[1]
                return t ? o : r
              },
            },
            {
              key: 'top',
              get: function () {
                var e = this.props.transform,
                  t = this.unrotatedViewport,
                  n = this.defaultSideways,
                  r = _(e, 6),
                  o = r[2],
                  a = r[3],
                  i = r[4],
                  c = r[5],
                  u = _(t.viewBox, 4),
                  s = u[1],
                  l = u[3]
                return n ? i + o + s : l - (c + a)
              },
            },
            {
              key: 'left',
              get: function () {
                var e = this.props.transform,
                  t = this.unrotatedViewport,
                  n = this.defaultSideways,
                  r = _(e, 6),
                  o = r[4],
                  a = r[5],
                  i = _(t.viewBox, 1)[0]
                return n ? a - i : o - i
              },
            },
            {
              key: 'getFontData',
              value: function (e) {
                var t = this.props.page
                return new Promise(function (n) {
                  t.commonObjs.get(e, n)
                })
              },
            },
            {
              key: 'alignTextItem',
              value: function () {
                var e = this,
                  t = this.itemElement.current
                if (t) {
                  t.style.transform = ''
                  var n = this.props,
                    r = n.fontName,
                    o = n.scale,
                    a = n.width
                  ;(t.style.fontFamily = ''.concat(r, ', sans-serif')),
                    this.getFontData(r).then(function (n) {
                      var i = n ? n.fallbackName : 'sans-serif'
                      t.style.fontFamily = ''.concat(r, ', ').concat(i)
                      var c = a * o,
                        u = e.getElementWidth(t),
                        s = 'scaleX('.concat(c / u, ')'),
                        l = n ? n.ascent : 0
                      l && (s += ' translateY('.concat(100 * (1 - l), '%)')),
                        (t.style.transform = s),
                        (t.style.WebkitTransform = s)
                    })
                }
              },
            },
            {
              key: 'render',
              value: function () {
                var e = this.fontSize,
                  t = this.top,
                  n = this.left,
                  o = this.props,
                  a = o.customTextRenderer,
                  i = o.scale,
                  c = o.str
                return r.default.createElement(
                  'span',
                  {
                    ref: this.itemElement,
                    style: {
                      height: '1em',
                      fontFamily: 'sans-serif',
                      fontSize: ''.concat(e * i, 'px'),
                      position: 'absolute',
                      top: ''.concat(t * i, 'px'),
                      left: ''.concat(n * i, 'px'),
                      transformOrigin: 'left bottom',
                      whiteSpace: 'pre',
                      pointerEvents: 'all',
                    },
                  },
                  a ? a(this.props) : c
                )
              },
            },
          ]),
          n
        )
      })(r.PureComponent)
      function Be(e) {
        return r.default.createElement(Ze.Consumer, null, function (t) {
          return r.default.createElement(_e, (0, a.Z)({}, t, e))
        })
      }
      function Ue(e) {
        var t = (function () {
          if ('undefined' === typeof Reflect || !Reflect.construct) return !1
          if (Reflect.construct.sham) return !1
          if ('function' === typeof Proxy) return !0
          try {
            return (
              Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})), !0
            )
          } catch (e) {
            return !1
          }
        })()
        return function () {
          var n,
            r = v(e)
          if (t) {
            var o = v(this).constructor
            n = Reflect.construct(r, arguments, o)
          } else n = r.apply(this, arguments)
          return g(this, n)
        }
      }
      _e.propTypes = {
        customTextRenderer: b().func,
        fontName: b().string.isRequired,
        itemIndex: b().number.isRequired,
        page: ae.isRequired,
        rotate: se,
        scale: b().number,
        str: b().string.isRequired,
        transform: b().arrayOf(b().number).isRequired,
        width: b().number.isRequired,
      }
      var Fe = (function (e) {
        h(n, e)
        var t = Ue(n)
        function n() {
          var e
          s(this, n)
          for (var r = arguments.length, o = new Array(r), a = 0; a < r; a++) o[a] = arguments[a]
          return (
            (e = t.call.apply(t, [this].concat(o))),
            (0, y.Z)((0, p.Z)(e), 'state', { textItems: null }),
            (0, y.Z)((0, p.Z)(e), 'loadTextItems', function () {
              var t = S(e.props.page.getTextContent())
              ;(e.runningTask = t),
                t.promise
                  .then(function (t) {
                    var n = t.items
                    e.setState({ textItems: n }, e.onLoadSuccess)
                  })
                  .catch(function (t) {
                    e.onLoadError(t)
                  })
            }),
            (0, y.Z)((0, p.Z)(e), 'onLoadSuccess', function () {
              var t = e.props.onGetTextSuccess,
                n = e.state.textItems
              t && t(n)
            }),
            (0, y.Z)((0, p.Z)(e), 'onLoadError', function (t) {
              e.setState({ textItems: !1 }), T(t)
              var n = e.props.onGetTextError
              n && n(t)
            }),
            e
          )
        }
        return (
          f(n, [
            {
              key: 'componentDidMount',
              value: function () {
                C(this.props.page), this.loadTextItems()
              },
            },
            {
              key: 'componentDidUpdate',
              value: function (e) {
                var t = this.props.page
                e.page && t !== e.page && this.loadTextItems()
              },
            },
            {
              key: 'componentWillUnmount',
              value: function () {
                $(this.runningTask)
              },
            },
            {
              key: 'unrotatedViewport',
              get: function () {
                var e = this.props,
                  t = e.page,
                  n = e.scale
                return t.getViewport({ scale: n })
              },
            },
            {
              key: 'rotate',
              get: function () {
                var e = this.props,
                  t = e.page
                return e.rotate - t.rotate
              },
            },
            {
              key: 'renderTextItems',
              value: function () {
                var e = this.state.textItems
                return e
                  ? e.map(function (e, t) {
                      return r.default.createElement(Be, (0, a.Z)({ key: t, itemIndex: t }, e))
                    })
                  : null
              },
            },
            {
              key: 'render',
              value: function () {
                var e = this.unrotatedViewport,
                  t = this.rotate
                return r.default.createElement(
                  'div',
                  {
                    className: 'react-pdf__Page__textContent',
                    style: {
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      width: ''.concat(e.width, 'px'),
                      height: ''.concat(e.height, 'px'),
                      color: 'transparent',
                      transform: 'translate(-50%, -50%) rotate('.concat(t, 'deg)'),
                      WebkitTransform: 'translate(-50%, -50%) rotate('.concat(t, 'deg)'),
                      pointerEvents: 'none',
                    },
                  },
                  this.renderTextItems()
                )
              },
            },
          ]),
          n
        )
      })(r.PureComponent)
      function Me(e) {
        return r.default.createElement(Ze.Consumer, null, function (t) {
          return r.default.createElement(Fe, (0, a.Z)({}, t, e))
        })
      }
      function qe(e) {
        var t = (function () {
          if ('undefined' === typeof Reflect || !Reflect.construct) return !1
          if (Reflect.construct.sham) return !1
          if ('function' === typeof Proxy) return !0
          try {
            return (
              Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})), !0
            )
          } catch (e) {
            return !1
          }
        })()
        return function () {
          var n,
            r = v(e)
          if (t) {
            var o = v(this).constructor
            n = Reflect.construct(r, arguments, o)
          } else n = r.apply(this, arguments)
          return g(this, n)
        }
      }
      Fe.propTypes = {
        onGetTextError: b().func,
        onGetTextSuccess: b().func,
        page: ae.isRequired,
        rotate: se,
        scale: b().number,
      }
      var Ve = (function (e) {
        h(n, e)
        var t = qe(n)
        function n() {
          var e
          s(this, n)
          for (var o = arguments.length, a = new Array(o), i = 0; i < o; i++) a[i] = arguments[i]
          return (
            (e = t.call.apply(t, [this].concat(a))),
            (0, y.Z)((0, p.Z)(e), 'state', { annotations: null }),
            (0, y.Z)((0, p.Z)(e), 'layerElement', (0, r.createRef)()),
            (0, y.Z)((0, p.Z)(e), 'loadAnnotations', function () {
              var t = S(e.props.page.getAnnotations())
              ;(e.runningTask = t),
                t.promise
                  .then(function (t) {
                    e.setState({ annotations: t }, e.onLoadSuccess)
                  })
                  .catch(function (t) {
                    e.onLoadError(t)
                  })
            }),
            (0, y.Z)((0, p.Z)(e), 'onLoadSuccess', function () {
              var t = e.props.onGetAnnotationsSuccess,
                n = e.state.annotations
              t && t(n)
            }),
            (0, y.Z)((0, p.Z)(e), 'onLoadError', function (t) {
              e.setState({ annotations: !1 }), T(t)
              var n = e.props.onGetAnnotationsError
              n && n(t)
            }),
            (0, y.Z)((0, p.Z)(e), 'onRenderSuccess', function () {
              var t = e.props.onRenderAnnotationLayerSuccess
              t && t()
            }),
            (0, y.Z)((0, p.Z)(e), 'onRenderError', function (t) {
              T(t)
              var n = e.props.onRenderAnnotationLayerError
              n && n(t)
            }),
            e
          )
        }
        return (
          f(n, [
            {
              key: 'componentDidMount',
              value: function () {
                C(this.props.page), this.loadAnnotations()
              },
            },
            {
              key: 'componentDidUpdate',
              value: function (e) {
                var t = this.props,
                  n = t.page,
                  r = t.renderForms
                ;((e.page && n !== e.page) || r !== e.renderForms) && this.loadAnnotations()
              },
            },
            {
              key: 'componentWillUnmount',
              value: function () {
                $(this.runningTask)
              },
            },
            {
              key: 'viewport',
              get: function () {
                var e = this.props,
                  t = e.page,
                  n = e.rotate,
                  r = e.scale
                return t.getViewport({ scale: r, rotation: n })
              },
            },
            {
              key: 'renderAnnotationLayer',
              value: function () {
                var e = this.state.annotations
                if (e) {
                  var t = this.props,
                    n = t.imageResourcesPath,
                    r = t.linkService,
                    a = t.page,
                    i = t.renderForms,
                    c = this.viewport.clone({ dontFlip: !0 }),
                    u = {
                      annotations: e,
                      div: this.layerElement.current,
                      imageResourcesPath: n,
                      linkService: r,
                      page: a,
                      renderForms: i,
                      viewport: c,
                    }
                  this.layerElement.current.innerHTML = ''
                  try {
                    o.AnnotationLayer.render(u), this.onRenderSuccess()
                  } catch (s) {
                    this.onRenderError(s)
                  }
                }
              },
            },
            {
              key: 'render',
              value: function () {
                return r.default.createElement(
                  'div',
                  {
                    className: 'react-pdf__Page__annotations annotationLayer',
                    ref: this.layerElement,
                  },
                  this.renderAnnotationLayer()
                )
              },
            },
          ]),
          n
        )
      })(r.PureComponent)
      Ve.propTypes = {
        imageResourcesPath: b().string,
        linkService: oe.isRequired,
        onGetAnnotationsError: b().func,
        onGetAnnotationsSuccess: b().func,
        onRenderAnnotationLayerError: b().func,
        onRenderAnnotationLayerSuccess: b().func,
        page: ae,
        renderForms: b().bool,
        rotate: se,
        scale: b().number,
      }
      var Ge = function (e) {
        return r.default.createElement(L.Consumer, null, function (t) {
          return r.default.createElement(Ze.Consumer, null, function (n) {
            return r.default.createElement(Ve, (0, a.Z)({}, t, n, e))
          })
        })
      }
      function We(e, t) {
        var n = Object.keys(e)
        if (Object.getOwnPropertySymbols) {
          var r = Object.getOwnPropertySymbols(e)
          t &&
            (r = r.filter(function (t) {
              return Object.getOwnPropertyDescriptor(e, t).enumerable
            })),
            n.push.apply(n, r)
        }
        return n
      }
      function Ke(e) {
        for (var t = 1; t < arguments.length; t++) {
          var n = null != arguments[t] ? arguments[t] : {}
          t % 2
            ? We(Object(n), !0).forEach(function (t) {
                ;(0, y.Z)(e, t, n[t])
              })
            : Object.getOwnPropertyDescriptors
            ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n))
            : We(Object(n)).forEach(function (t) {
                Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(n, t))
              })
        }
        return e
      }
      function He(e) {
        var t = (function () {
          if ('undefined' === typeof Reflect || !Reflect.construct) return !1
          if (Reflect.construct.sham) return !1
          if ('function' === typeof Proxy) return !0
          try {
            return (
              Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})), !0
            )
          } catch (e) {
            return !1
          }
        })()
        return function () {
          var n,
            r = v(e)
          if (t) {
            var o = v(this).constructor
            n = Reflect.construct(r, arguments, o)
          } else n = r.apply(this, arguments)
          return g(this, n)
        }
      }
      var ze = (function (e) {
        h(n, e)
        var t = He(n)
        function n() {
          var e
          s(this, n)
          for (var o = arguments.length, a = new Array(o), i = 0; i < o; i++) a[i] = arguments[i]
          return (
            (e = t.call.apply(t, [this].concat(a))),
            (0, y.Z)((0, p.Z)(e), 'state', { page: null }),
            (0, y.Z)((0, p.Z)(e), 'pageElement', (0, r.createRef)()),
            (0, y.Z)((0, p.Z)(e), 'onLoadSuccess', function () {
              var t = e.props,
                n = t.onLoadSuccess,
                r = t.registerPage,
                o = e.state.page
              n && n(Y(o, e.scale)), r && r(e.pageIndex, e.pageElement.current)
            }),
            (0, y.Z)((0, p.Z)(e), 'onLoadError', function (t) {
              e.setState({ page: !1 }), T(t)
              var n = e.props.onLoadError
              n && n(t)
            }),
            (0, y.Z)((0, p.Z)(e), 'loadPage', function () {
              var t = e.props.pdf,
                n = e.getPageNumber()
              if (n) {
                e.setState(function (e) {
                  return e.page ? { page: null } : null
                })
                var r = S(t.getPage(n))
                ;(e.runningTask = r),
                  r.promise
                    .then(function (t) {
                      e.setState({ page: t }, e.onLoadSuccess)
                    })
                    .catch(function (t) {
                      e.onLoadError(t)
                    })
              }
            }),
            e
          )
        }
        return (
          f(n, [
            {
              key: 'componentDidMount',
              value: function () {
                C(this.props.pdf), this.loadPage()
              },
            },
            {
              key: 'componentDidUpdate',
              value: function (e) {
                var t = this.props.pdf
                if ((e.pdf && t !== e.pdf) || this.getPageNumber() !== this.getPageNumber(e)) {
                  var n = this.props.unregisterPage
                  n && n(this.getPageIndex(e)), this.loadPage()
                }
              },
            },
            {
              key: 'componentWillUnmount',
              value: function () {
                var e = this.props.unregisterPage
                e && e(this.pageIndex), $(this.runningTask)
              },
            },
            {
              key: 'childContext',
              get: function () {
                var e = this.state.page
                if (!e) return {}
                var t = this.props,
                  n = t.canvasBackground,
                  r = t.customTextRenderer,
                  o = t.onGetAnnotationsError,
                  a = t.onGetAnnotationsSuccess,
                  i = t.onGetTextError,
                  c = t.onGetTextSuccess,
                  u = t.onRenderAnnotationLayerError,
                  s = t.onRenderAnnotationLayerSuccess,
                  l = t.onRenderError,
                  f = t.onRenderSuccess,
                  p = t.renderForms,
                  d = t.renderInteractiveForms
                return {
                  canvasBackground: n,
                  customTextRenderer: r,
                  onGetAnnotationsError: o,
                  onGetAnnotationsSuccess: a,
                  onGetTextError: i,
                  onGetTextSuccess: c,
                  onRenderAnnotationLayerError: u,
                  onRenderAnnotationLayerSuccess: s,
                  onRenderError: l,
                  onRenderSuccess: f,
                  page: e,
                  renderForms: null !== p && void 0 !== p ? p : d,
                  rotate: this.rotate,
                  scale: this.scale,
                }
              },
            },
            {
              key: 'getPageIndex',
              value: function () {
                var e = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : this.props
                return M(e.pageNumber) ? e.pageNumber - 1 : M(e.pageIndex) ? e.pageIndex : null
              },
            },
            {
              key: 'getPageNumber',
              value: function () {
                var e = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : this.props
                return M(e.pageNumber) ? e.pageNumber : M(e.pageIndex) ? e.pageIndex + 1 : null
              },
            },
            {
              key: 'pageIndex',
              get: function () {
                return this.getPageIndex()
              },
            },
            {
              key: 'pageNumber',
              get: function () {
                return this.getPageNumber()
              },
            },
            {
              key: 'rotate',
              get: function () {
                var e = this.props.rotate
                if (M(e)) return e
                var t = this.state.page
                return t ? t.rotate : null
              },
            },
            {
              key: 'scale',
              get: function () {
                var e = this.state.page
                if (!e) return null
                var t = this.props,
                  n = t.scale,
                  r = t.width,
                  o = t.height,
                  a = this.rotate,
                  i = 1,
                  c = null === n ? 1 : n
                if (r || o) {
                  var u = e.getViewport({ scale: 1, rotation: a })
                  i = r ? r / u.width : o / u.height
                }
                return c * i
              },
            },
            {
              key: 'eventProps',
              get: function () {
                var e = this
                return O(this.props, function () {
                  var t = e.state.page
                  return t ? Y(t, e.scale) : t
                })
              },
            },
            {
              key: 'pageKey',
              get: function () {
                var e = this.state.page
                return ''.concat(e.pageIndex, '@').concat(this.scale, '/').concat(this.rotate)
              },
            },
            {
              key: 'pageKeyNoScale',
              get: function () {
                var e = this.state.page
                return ''.concat(e.pageIndex, '/').concat(this.rotate)
              },
            },
            {
              key: 'renderMainLayer',
              value: function () {
                var e = this.props,
                  t = e.canvasRef
                switch (e.renderMode) {
                  case 'none':
                    return null
                  case 'svg':
                    return r.default.createElement(Ie, {
                      key: ''.concat(this.pageKeyNoScale, '_svg'),
                    })
                  default:
                    return r.default.createElement(De, {
                      key: ''.concat(this.pageKey, '_canvas'),
                      canvasRef: t,
                    })
                }
              },
            },
            {
              key: 'renderTextLayer',
              value: function () {
                var e = this.props.renderTextLayer
                return e
                  ? r.default.createElement(Me, { key: ''.concat(this.pageKey, '_text') })
                  : null
              },
            },
            {
              key: 'renderAnnotationLayer',
              value: function () {
                var e = this.props.renderAnnotationLayer
                return e
                  ? r.default.createElement(Ge, { key: ''.concat(this.pageKey, '_annotations') })
                  : null
              },
            },
            {
              key: 'renderChildren',
              value: function () {
                var e = this.props.children
                return r.default.createElement(
                  Ze.Provider,
                  { value: this.childContext },
                  this.renderMainLayer(),
                  this.renderTextLayer(),
                  this.renderAnnotationLayer(),
                  e
                )
              },
            },
            {
              key: 'renderContent',
              value: function () {
                var e = this.pageNumber,
                  t = this.props.pdf,
                  n = this.state.page
                if (!e) {
                  var o = this.props.noData
                  return r.default.createElement(
                    D,
                    { type: 'no-data' },
                    'function' === typeof o ? o() : o
                  )
                }
                if (null === t || null === n) {
                  var a = this.props.loading
                  return r.default.createElement(
                    D,
                    { type: 'loading' },
                    'function' === typeof a ? a() : a
                  )
                }
                if (!1 === t || !1 === n) {
                  var i = this.props.error
                  return r.default.createElement(
                    D,
                    { type: 'error' },
                    'function' === typeof i ? i() : i
                  )
                }
                return this.renderChildren()
              },
            },
            {
              key: 'render',
              value: function () {
                var e = this.pageNumber,
                  t = this.props,
                  n = t.className,
                  o = t.inputRef
                return r.default.createElement(
                  'div',
                  (0, a.Z)(
                    {
                      className: x('react-pdf__Page', n),
                      'data-page-number': e,
                      ref: xe(o, this.pageElement),
                      style: { position: 'relative' },
                    },
                    this.eventProps
                  ),
                  this.renderContent()
                )
              },
            },
          ]),
          n
        )
      })(r.PureComponent)
      ze.defaultProps = {
        error: 'Failed to load the page.',
        loading: 'Loading page\u2026',
        noData: 'No page specified.',
        renderAnnotationLayer: !0,
        renderForms: !1,
        renderMode: 'canvas',
        renderTextLayer: !0,
        scale: 1,
      }
      var $e = b().oneOfType([b().func, b().node])
      function Ye(e, t) {
        return r.default.createElement(L.Consumer, null, function (n) {
          return r.default.createElement(ze, (0, a.Z)({ ref: t }, n, e))
        })
      }
      ze.propTypes = Ke(
        Ke({}, ee),
        {},
        {
          canvasBackground: b().string,
          children: b().node,
          className: ne,
          customTextRenderer: b().func,
          error: $e,
          height: b().number,
          imageResourcesPath: b().string,
          inputRef: ce,
          loading: $e,
          noData: $e,
          onGetTextError: b().func,
          onGetTextSuccess: b().func,
          onLoadError: b().func,
          onLoadSuccess: b().func,
          onRenderError: b().func,
          onRenderSuccess: b().func,
          pageIndex: function (e, t, n) {
            var r = e[t],
              o = e.pageNumber,
              a = e.pdf
            if (!F(a)) return null
            if (F(r)) {
              if ('number' !== typeof r)
                return new Error(
                  '`'
                    .concat(t, '` of type `')
                    .concat(u(r), '` supplied to `')
                    .concat(n, '`, expected `number`.')
                )
              if (r < 0) return new Error('Expected `'.concat(t, '` to be greater or equal to 0.'))
              var i = a.numPages
              if (r + 1 > i)
                return new Error(
                  'Expected `'.concat(t, '` to be less or equal to ').concat(i - 1, '.')
                )
            } else if (!F(o))
              return new Error(
                '`'
                  .concat(t, '` not supplied. Either pageIndex or pageNumber must be supplied to `')
                  .concat(n, '`.')
              )
            return null
          },
          pageNumber: function (e, t, n) {
            var r = e[t],
              o = e.pageIndex,
              a = e.pdf
            if (!F(a)) return null
            if (F(r)) {
              if ('number' !== typeof r)
                return new Error(
                  '`'
                    .concat(t, '` of type `')
                    .concat(u(r), '` supplied to `')
                    .concat(n, '`, expected `number`.')
                )
              if (r < 1) return new Error('Expected `'.concat(t, '` to be greater or equal to 1.'))
              var i = a.numPages
              if (r > i)
                return new Error('Expected `'.concat(t, '` to be less or equal to ').concat(i, '.'))
            } else if (!F(o))
              return new Error(
                '`'
                  .concat(t, '` not supplied. Either pageIndex or pageNumber must be supplied to `')
                  .concat(n, '`.')
              )
            return null
          },
          pdf: ie,
          registerPage: b().func,
          renderAnnotationLayer: b().bool,
          renderForms: b().bool,
          renderInteractiveForms: b().bool,
          renderMode: ue,
          renderTextLayer: b().bool,
          rotate: se,
          scale: b().number,
          unregisterPage: b().func,
          width: b().number,
        }
      )
      var Xe = r.default.forwardRef(Ye)
      T(!U, 'Loading PDF.js worker may not work on protocols other than HTTP/HTTPS. '.concat(H)),
        (o.GlobalWorkerOptions.workerSrc = 'pdf.worker.js')
      var Je = Object.defineProperty,
        Qe = Object.getOwnPropertySymbols,
        et = Object.prototype.hasOwnProperty,
        tt = Object.prototype.propertyIsEnumerable,
        nt = (e, t, n) =>
          t in e
            ? Je(e, t, { enumerable: !0, configurable: !0, writable: !0, value: n })
            : (e[t] = n)
      o.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${o.version}/legacy/build/pdf.worker.min.js`
      var rt = (e) => {
        var t = e,
          { file: n } = t,
          o = ((e, t) => {
            var n = {}
            for (var r in e) et.call(e, r) && t.indexOf(r) < 0 && (n[r] = e[r])
            if (null != e && Qe)
              for (var r of Qe(e)) t.indexOf(r) < 0 && tt.call(e, r) && (n[r] = e[r])
            return n
          })(t, ['file'])
        const [a, i] = r.useState(null)
        return r.createElement(
          ge,
          ((e, t) => {
            for (var n in t || (t = {})) et.call(t, n) && nt(e, n, t[n])
            if (Qe) for (var n of Qe(t)) tt.call(t, n) && nt(e, n, t[n])
            return e
          })(
            {
              file: n,
              onLoadSuccess: function ({ numPages: e }) {
                i(e)
              },
            },
            o
          ),
          Array.from(new Array(a), (e, t) =>
            r.createElement(Xe, { key: `page_${t + 1}`, pageNumber: t + 1 })
          )
        )
      }
    },
  },
])
