;(self.webpackChunk_N_E = self.webpackChunk_N_E || []).push([
  [712],
  {
    9618: function (e, t, r) {
      var n = {
        './AuthorLayout': 4856,
        './AuthorLayout.js': 4856,
        './ListLayout': 6055,
        './ListLayout.js': 6055,
        './PostLayout': 5067,
        './PostLayout.js': 5067,
        './PostSimple': 3168,
        './PostSimple.js': 3168,
      }
      function a(e) {
        var t = i(e)
        return r(t)
      }
      function i(e) {
        if (!r.o(n, e)) {
          var t = new Error("Cannot find module '" + e + "'")
          throw ((t.code = 'MODULE_NOT_FOUND'), t)
        }
        return n[e]
      }
      ;(a.keys = function () {
        return Object.keys(n)
      }),
        (a.resolve = i),
        (e.exports = a),
        (a.id = 9618)
    },
    1712: function (e, t, r) {
      'use strict'
      r.d(t, {
        J: function () {
          return k
        },
      })
      var n = r(7320),
        a = r(1720),
        i = r(3194),
        o = r(8100),
        l = r(7233),
        c = function (e) {
          var t = e.toc,
            r = e.indentDepth,
            a = void 0 === r ? 3 : r,
            i = e.fromHeading,
            o = void 0 === i ? 1 : i,
            l = e.toHeading,
            c = void 0 === l ? 6 : l,
            d = e.asDisclosure,
            s = void 0 !== d && d,
            u = e.exclude,
            m = void 0 === u ? '' : u,
            f = Array.isArray(m)
              ? new RegExp('^(' + m.join('|') + ')$', 'i')
              : new RegExp('^(' + m + ')$', 'i'),
            h = t.filter(function (e) {
              return e.depth >= o && e.depth <= c && !f.test(e.value)
            }),
            p = (0, n.tZ)('ul', {
              children: h.map(function (e) {
                return (0,
                n.tZ)('li', { className: ''.concat(e.depth >= a && 'ml-6'), children: (0, n.tZ)('a', { href: e.url, children: e.value }) }, e.value)
              }),
            })
          return (0, n.tZ)(n.HY, {
            children: s
              ? (0, n.BX)('details', {
                  open: !0,
                  children: [
                    (0, n.tZ)('summary', {
                      className: 'ml-6 pt-2 pb-2 text-xl font-bold',
                      children: 'Table of Contents',
                    }),
                    (0, n.tZ)('div', { className: 'ml-6', children: p }),
                  ],
                })
              : p,
          })
        },
        d = function (e) {
          var t = (0, a.useRef)(null),
            r = (0, a.useState)(!1),
            i = r[0],
            o = r[1],
            l = (0, a.useState)(!1),
            c = l[0],
            d = l[1]
          return (0, n.BX)('div', {
            ref: t,
            onMouseEnter: function () {
              o(!0)
            },
            onMouseLeave: function () {
              o(!1), d(!1)
            },
            className: 'relative',
            children: [
              i &&
                (0, n.tZ)('button', {
                  'aria-label': 'Copy code',
                  type: 'button',
                  className:
                    'absolute right-2 top-2 h-8 w-8 rounded border-2 bg-gray-700 p-1 dark:bg-gray-800 '.concat(
                      c
                        ? 'border-green-400 focus:border-green-400 focus:outline-none'
                        : 'border-gray-300'
                    ),
                  onClick: function () {
                    d(!0),
                      navigator.clipboard.writeText(t.current.textContent),
                      setTimeout(function () {
                        d(!1)
                      }, 2e3)
                  },
                  children: (0, n.tZ)('svg', {
                    xmlns: 'http://www.w3.org/2000/svg',
                    viewBox: '0 0 24 24',
                    stroke: 'currentColor',
                    fill: 'none',
                    className: c ? 'text-green-400' : 'text-gray-300',
                    children: c
                      ? (0, n.tZ)(n.HY, {
                          children: (0, n.tZ)('path', {
                            strokeLinecap: 'round',
                            strokeLinejoin: 'round',
                            strokeWidth: 2,
                            d: 'M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4',
                          }),
                        })
                      : (0, n.tZ)(n.HY, {
                          children: (0, n.tZ)('path', {
                            strokeLinecap: 'round',
                            strokeLinejoin: 'round',
                            strokeWidth: 2,
                            d: 'M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2',
                          }),
                        }),
                  }),
                }),
              (0, n.tZ)('pre', { children: e.children }),
            ],
          })
        },
        s = r(7726),
        u = r(4793),
        m = r(9253),
        f = r(5152)
      function h(e, t, r) {
        return (
          t in e
            ? Object.defineProperty(e, t, {
                value: r,
                enumerable: !0,
                configurable: !0,
                writable: !0,
              })
            : (e[t] = r),
          e
        )
      }
      function p(e) {
        for (var t = 1; t < arguments.length; t++) {
          var r = null != arguments[t] ? arguments[t] : {},
            n = Object.keys(r)
          'function' === typeof Object.getOwnPropertySymbols &&
            (n = n.concat(
              Object.getOwnPropertySymbols(r).filter(function (e) {
                return Object.getOwnPropertyDescriptor(r, e).enumerable
              })
            )),
            n.forEach(function (t) {
              h(e, t, r[t])
            })
        }
        return e
      }
      function v(e, t) {
        if (null == e) return {}
        var r,
          n,
          a = (function (e, t) {
            if (null == e) return {}
            var r,
              n,
              a = {},
              i = Object.keys(e)
            for (n = 0; n < i.length; n++) (r = i[n]), t.indexOf(r) >= 0 || (a[r] = e[r])
            return a
          })(e, t)
        if (Object.getOwnPropertySymbols) {
          var i = Object.getOwnPropertySymbols(e)
          for (n = 0; n < i.length; n++)
            (r = i[n]),
              t.indexOf(r) >= 0 ||
                (Object.prototype.propertyIsEnumerable.call(e, r) && (a[r] = e[r]))
        }
        return a
      }
      var y = (0, f.default)(
          function () {
            return r
              .e(794)
              .then(r.bind(r, 1794))
              .then(function (e) {
                return e.Code
              })
          },
          {
            loadableGenerated: {
              webpack: function () {
                return [1794]
              },
            },
          }
        ),
        b = (0, f.default)(
          function () {
            return Promise.all([r.e(276), r.e(853)])
              .then(r.bind(r, 635))
              .then(function (e) {
                return e.Collection
              })
          },
          {
            loadableGenerated: {
              webpack: function () {
                return [635]
              },
            },
          }
        ),
        g = (0, f.default)(
          function () {
            return Promise.all([r.e(265), r.e(274)])
              .then(r.bind(r, 7274))
              .then(function (e) {
                return e.Equation
              })
          },
          {
            loadableGenerated: {
              webpack: function () {
                return [7274]
              },
            },
          }
        ),
        x = (0, f.default)(
          function () {
            return Promise.all([r.e(824), r.e(764), r.e(509), r.e(738)])
              .then(r.bind(r, 6402))
              .then(function (e) {
                return e.Pdf
              })
          },
          {
            loadableGenerated: {
              webpack: function () {
                return [6402]
              },
            },
            ssr: !1,
          }
        ),
        Z = (0, f.default)(
          function () {
            return r
              .e(873)
              .then(r.bind(r, 6873))
              .then(function (e) {
                return e.Modal
              })
          },
          {
            loadableGenerated: {
              webpack: function () {
                return [6873]
              },
            },
            ssr: !1,
          }
        ),
        w = {
          Image: o.Z,
          TOCInline: c,
          a: l.Z,
          pre: d,
          BlogNewsletterForm: s.w,
          wrapper: function (e) {
            e.components
            var t = e.layout,
              a = v(e, ['components', 'layout']),
              i = r(9618)('./'.concat(t)).default
            return (0, n.tZ)(i, p({}, a))
          },
        },
        k = function (e) {
          var t = e.layout,
            r = e.mdxSource,
            o = e.recordMap,
            l = v(e, ['layout', 'mdxSource', 'recordMap']),
            c = (0, a.useMemo)(
              function () {
                return (0, i.getMDXComponent)(r)
              },
              [r]
            ),
            d = o
              ? (0, n.tZ)(u.cp, {
                  recordMap: o,
                  fullPage: !0,
                  darkMode: !0,
                  components: { Code: y, Collection: b, Equation: g, Modal: Z, Pdf: x },
                })
              : (0, n.tZ)('span', { className: 'noNotion' }),
            s = o ? (0, m.pz)(o) : null
          return (0, n.tZ)(n.HY, {
            children: (0, n.tZ)(
              c,
              p({ layout: t, components: w, NotionJsx: d, NotionTitle: s }, l)
            ),
          })
        }
    },
    7726: function (e, t, r) {
      'use strict'
      r.d(t, {
        w: function () {
          return u
        },
      })
      var n = r(4051),
        a = r.n(n),
        i = r(7320),
        o = r(1720),
        l = r(1576),
        c = r.n(l)
      function d(e, t, r, n, a, i, o) {
        try {
          var l = e[i](o),
            c = l.value
        } catch (d) {
          return void r(d)
        }
        l.done ? t(c) : Promise.resolve(c).then(n, a)
      }
      var s = function (e) {
        var t = e.title,
          r = void 0 === t ? 'Subscribe to the newsletter' : t,
          n = (0, o.useRef)(null),
          l = (0, o.useState)(!1),
          s = l[0],
          u = l[1],
          m = (0, o.useState)(''),
          f = m[0],
          h = m[1],
          p = (0, o.useState)(!1),
          v = p[0],
          y = p[1],
          b = (function () {
            var e,
              t =
                ((e = a().mark(function e(t) {
                  var r
                  return a().wrap(function (e) {
                    for (;;)
                      switch ((e.prev = e.next)) {
                        case 0:
                          return (
                            t.preventDefault(),
                            (e.next = 3),
                            fetch('/api/'.concat(c().newsletter.provider), {
                              body: JSON.stringify({ email: n.current.value }),
                              headers: { 'Content-Type': 'application/json' },
                              method: 'POST',
                            })
                          )
                        case 3:
                          return (r = e.sent), (e.next = 6), r.json()
                        case 6:
                          if (!e.sent.error) {
                            e.next = 11
                            break
                          }
                          return (
                            u(!0),
                            h('Your e-mail address is invalid or you are already subscribed!'),
                            e.abrupt('return')
                          )
                        case 11:
                          ;(n.current.value = ''),
                            u(!1),
                            y(!0),
                            h('Successfully! \ud83c\udf89 You are now subscribed.')
                        case 15:
                        case 'end':
                          return e.stop()
                      }
                  }, e)
                })),
                function () {
                  var t = this,
                    r = arguments
                  return new Promise(function (n, a) {
                    var i = e.apply(t, r)
                    function o(e) {
                      d(i, n, a, o, l, 'next', e)
                    }
                    function l(e) {
                      d(i, n, a, o, l, 'throw', e)
                    }
                    o(void 0)
                  })
                })
            return function (e) {
              return t.apply(this, arguments)
            }
          })()
        return (0, i.BX)('div', {
          children: [
            (0, i.tZ)('div', {
              className: 'pb-1 text-lg font-semibold text-gray-800 dark:text-gray-100',
              children: r,
            }),
            (0, i.BX)('form', {
              className: 'flex flex-col sm:flex-row',
              onSubmit: b,
              children: [
                (0, i.BX)('div', {
                  children: [
                    (0, i.tZ)('label', {
                      className: 'sr-only',
                      htmlFor: 'email-input',
                      children: 'Email address',
                    }),
                    (0, i.tZ)('input', {
                      autoComplete: 'email',
                      className:
                        'w-72 rounded-md px-4 focus:border-transparent focus:outline-none focus:ring-2 focus:ring-primary-600 dark:bg-black',
                      id: 'email-input',
                      name: 'email',
                      placeholder: v ? "You're subscribed !  \ud83c\udf89" : 'Enter your email',
                      ref: n,
                      required: !0,
                      type: 'email',
                      disabled: v,
                    }),
                  ],
                }),
                (0, i.tZ)('div', {
                  className: 'mt-2 flex w-full rounded-md shadow-sm sm:mt-0 sm:ml-3',
                  children: (0, i.tZ)('button', {
                    className:
                      'w-full rounded-md bg-primary-500 py-2 px-4 font-medium text-white sm:py-0 '.concat(
                        v ? 'cursor-default' : 'hover:bg-primary-700 dark:hover:bg-primary-400',
                        ' focus:outline-none focus:ring-2 focus:ring-primary-600 focus:ring-offset-2 dark:ring-offset-black'
                      ),
                    type: 'submit',
                    disabled: v,
                    children: v ? 'Thank you!' : 'Sign up',
                  }),
                }),
              ],
            }),
            s &&
              (0, i.tZ)('div', {
                className: 'w-72 pt-2 text-sm text-red-500 dark:text-red-400 sm:w-96',
                children: f,
              }),
          ],
        })
      }
      t.Z = s
      var u = function (e) {
        var t = e.title
        return (0, i.tZ)('div', {
          className: 'flex items-center justify-center',
          children: (0, i.tZ)('div', {
            className: 'bg-gray-100 p-6 dark:bg-gray-800 sm:px-14 sm:py-8',
            children: (0, i.tZ)(s, { title: t }),
          }),
        })
      }
    },
    920: function (e, t, r) {
      'use strict'
      r.d(t, {
        Z: function () {
          return a
        },
      })
      var n = r(7320)
      function a(e) {
        var t = e.children
        return (0, n.tZ)('h1', {
          className:
            'text-left font-rs text-4xl font-medium leading-9 tracking-tight text-gray-900 dark:text-gray-100 sm:text-4xl sm:leading-10 md:text-5xl md:leading-14',
          children: t,
        })
      }
    },
    7175: function (e, t, r) {
      'use strict'
      var n = r(7320),
        a = r(1576),
        i = r.n(a),
        o = r(1720)
      t.Z = function () {
        var e = (0, o.useState)(!1),
          t = e[0],
          r = e[1]
        ;(0, o.useEffect)(function () {
          var e = function () {
            window.scrollY > 50 ? r(!0) : r(!1)
          }
          return (
            window.addEventListener('scroll', e),
            function () {
              return window.removeEventListener('scroll', e)
            }
          )
        }, [])
        return (0, n.BX)('div', {
          className: 'fixed right-8 bottom-8 hidden flex-col gap-3 '.concat(
            t ? 'md:flex' : 'md:hidden'
          ),
          children: [
            i().comment.provider &&
              (0, n.tZ)('button', {
                'aria-label': 'Scroll To Comment',
                type: 'button',
                onClick: function () {
                  document.getElementById('comment').scrollIntoView()
                },
                className:
                  'rounded-full bg-gray-200 p-2 text-gray-500 transition-all hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-400 dark:hover:bg-gray-600',
                children: (0, n.tZ)('svg', {
                  className: 'h-5 w-5',
                  viewBox: '0 0 20 20',
                  fill: 'currentColor',
                  children: (0, n.tZ)('path', {
                    fillRule: 'evenodd',
                    d: 'M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9H5v2h2V9zm8 0h-2v2h2V9zM9 9h2v2H9V9z',
                    clipRule: 'evenodd',
                  }),
                }),
              }),
            (0, n.tZ)('button', {
              'aria-label': 'Scroll To Top',
              type: 'button',
              onClick: function () {
                window.scrollTo({ top: 0 })
              },
              className:
                'rounded-full bg-gray-200 p-2 text-gray-500 transition-all hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-400 dark:hover:bg-gray-600',
              children: (0, n.tZ)('svg', {
                className: 'h-5 w-5',
                viewBox: '0 0 20 20',
                fill: 'currentColor',
                children: (0, n.tZ)('path', {
                  fillRule: 'evenodd',
                  d: 'M3.293 9.707a1 1 0 010-1.414l6-6a1 1 0 011.414 0l6 6a1 1 0 01-1.414 1.414L11 5.414V17a1 1 0 11-2 0V5.414L4.707 9.707a1 1 0 01-1.414 0z',
                  clipRule: 'evenodd',
                }),
              }),
            }),
          ],
        })
      }
    },
    896: function (e, t, r) {
      'use strict'
      var n = r(7320),
        a = r(1576),
        i = r.n(a),
        o = r(5152),
        l = (0, o.default)(
          function () {
            return r.e(806).then(r.bind(r, 8806))
          },
          {
            loadableGenerated: {
              webpack: function () {
                return [8806]
              },
            },
            ssr: !1,
          }
        ),
        c = (0, o.default)(
          function () {
            return r.e(732).then(r.bind(r, 732))
          },
          {
            loadableGenerated: {
              webpack: function () {
                return [732]
              },
            },
            ssr: !1,
          }
        ),
        d = (0, o.default)(
          function () {
            return r.e(257).then(r.bind(r, 257))
          },
          {
            loadableGenerated: {
              webpack: function () {
                return [257]
              },
            },
            ssr: !1,
          }
        )
      t.Z = function (e) {
        var t = e.frontMatter,
          r = null === i() || void 0 === i() ? void 0 : i().comment
        return r && 0 !== Object.keys(r).length
          ? (0, n.BX)('div', {
              id: 'comment',
              children: [
                i().comment && 'giscus' === i().comment.provider && (0, n.tZ)(c, {}),
                i().comment && 'utterances' === i().comment.provider && (0, n.tZ)(l, {}),
                i().comment &&
                  'disqus' === i().comment.provider &&
                  (0, n.tZ)(d, { frontMatter: t }),
              ],
            })
          : (0, n.tZ)(n.HY, {})
      }
    },
    4856: function (e, t, r) {
      'use strict'
      r.r(t),
        r.d(t, {
          default: function () {
            return i
          },
        })
      var n = r(7320),
        a = (r(9159), r(8100), r(9831))
      function i(e) {
        var t = e.children,
          r = e.NotionJsx,
          i = e.frontMatter,
          o = i.name
        i.avatar, i.occupation, i.company, i.email, i.twitter, i.linkedin, i.github
        return (0, n.BX)(n.HY, {
          children: [
            (0, n.tZ)(a.TQ, { title: 'About - '.concat(o), description: 'About me - '.concat(o) }),
            (0, n.tZ)('div', {
              className: 'divide-y divide-gray-200 dark:divide-gray-700',
              children: (0, n.BX)('div', {
                className: 'prose max-w-none pt-8 pb-8 dark:prose-dark xl:col-span-2',
                children: [t, r],
              }),
            }),
          ],
        })
      }
    },
    5067: function (e, t, r) {
      'use strict'
      r.r(t),
        r.d(t, {
          default: function () {
            return y
          },
        })
      var n = r(7320),
        a = r(7233),
        i = r(920),
        o = r(890),
        l = r(9831),
        c = (r(8100), r(9019)),
        d = r(1576),
        s = r.n(d),
        u = (r(896), r(7175)),
        m = r(7814)
      function f(e, t, r) {
        return (
          t in e
            ? Object.defineProperty(e, t, {
                value: r,
                enumerable: !0,
                configurable: !0,
                writable: !0,
              })
            : (e[t] = r),
          e
        )
      }
      function h(e) {
        for (var t = 1; t < arguments.length; t++) {
          var r = null != arguments[t] ? arguments[t] : {},
            n = Object.keys(r)
          'function' === typeof Object.getOwnPropertySymbols &&
            (n = n.concat(
              Object.getOwnPropertySymbols(r).filter(function (e) {
                return Object.getOwnPropertyDescriptor(r, e).enumerable
              })
            )),
            n.forEach(function (t) {
              f(e, t, r[t])
            })
        }
        return e
      }
      var p = function (e) {
          return ''.concat(s().siteRepo, '/blob/master/data/blog/').concat(e)
        },
        v = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' }
      function y(e) {
        var t = e.frontMatter,
          r = e.authorDetails,
          d = e.next,
          f = e.prev,
          y = e.NotionJsx,
          b = e.NotionTitle,
          g = e.children,
          x = t.slug,
          Z = t.fileName,
          w = t.date,
          k = t.title,
          N = (t.images, t.tags)
        return (0, n.BX)(o.Z, {
          children: [
            (0, n.tZ)(
              l.Uy,
              h({ url: ''.concat(s().siteUrl, '/blog/').concat(x), authorDetails: r }, t)
            ),
            (0, n.tZ)(u.Z, {}),
            (0, n.tZ)('article', {
              children: (0, n.BX)('div', {
                className: 'xl:divide-y xl:divide-gray-200 xl:dark:divide-gray-700',
                children: [
                  (0, n.tZ)('header', {
                    className: 'pt-6 xl:pb-6',
                    children: (0, n.BX)('div', {
                      className: 'space-y-1 ',
                      children: [
                        (0, n.tZ)('div', {
                          children: (0, n.BX)(i.Z, {
                            children: [
                              b || k,
                              (0, n.BX)('span', {
                                className: 'text-base',
                                children: [
                                  (0, n.tZ)('span', { children: ' ' }),
                                  !t.notion &&
                                    (0, n.tZ)(a.Z, {
                                      href: p(Z),
                                      className:
                                        'hover:text-primary-600 dark:hover:text-primary-400',
                                      children: (0, n.tZ)(m.G, { icon: 'edit', className: 'ml-2' }),
                                    }),
                                ],
                              }),
                            ],
                          }),
                        }),
                        (0, n.tZ)('dl', {
                          className: 'space-y-10',
                          children: (0, n.tZ)('div', {
                            children: (0, n.BX)('dd', {
                              className: 'text-base font-medium leading-6',
                              children: [
                                (0, n.tZ)('span', { children: 'Posted by ' }),
                                (0, n.tZ)('span', {
                                  className: 'font-rs italic',
                                  children: r.map(function (e) {
                                    return e.name + ' '
                                  }),
                                }),
                                (0, n.tZ)('span', { children: 'on ' }),
                                (0, n.tZ)('time', {
                                  dateTime: w,
                                  children: new Date(w).toLocaleDateString(s().locale, v),
                                }),
                              ],
                            }),
                          }),
                        }),
                      ],
                    }),
                  }),
                  (0, n.BX)('div', {
                    className:
                      'divide-y divide-gray-200 pb-8 dark:divide-gray-700 xl:grid xl:grid-cols-4 xl:gap-x-6 xl:divide-y-0',
                    style: { gridTemplateRows: 'auto 1fr' },
                    children: [
                      (0, n.tZ)('div', {
                        className:
                          'divide-y divide-gray-200 dark:divide-gray-700 xl:col-span-3 xl:row-span-2 xl:pb-0',
                        children: (0, n.BX)('div', {
                          className: 'prose max-w-none pb-8 dark:prose-dark',
                          children: [g, y],
                        }),
                      }),
                      (0, n.BX)('footer', {
                        children: [
                          (0, n.BX)('div', {
                            className:
                              'divide-gray-200 text-sm font-medium leading-5 dark:divide-gray-700 xl:col-start-1 xl:row-start-2 xl:divide-y',
                            children: [
                              N &&
                                (0, n.tZ)('div', {
                                  className: 'py-4 xl:py-8',
                                  children: (0, n.tZ)('div', {
                                    className: 'flex flex-wrap',
                                    children: N.map(function (e) {
                                      return (0, n.tZ)(c.Z, { text: e }, e)
                                    }),
                                  }),
                                }),
                              (d || f) &&
                                (0, n.BX)('div', {
                                  className:
                                    'flex justify-between py-4 xl:block xl:space-y-8 xl:py-8',
                                  children: [
                                    f &&
                                      (0, n.BX)('div', {
                                        children: [
                                          (0, n.tZ)('h2', {
                                            className:
                                              'text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400',
                                            children: 'Previous Article',
                                          }),
                                          (0, n.tZ)('div', {
                                            className:
                                              'text-primary-500 hover:text-primary-600 dark:hover:text-primary-400',
                                            children: (0, n.tZ)(a.Z, {
                                              href: '/blog/'.concat(f.slug),
                                              children: f.title,
                                            }),
                                          }),
                                        ],
                                      }),
                                    d &&
                                      (0, n.BX)('div', {
                                        children: [
                                          (0, n.tZ)('h2', {
                                            className:
                                              'text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400',
                                            children: 'Next Article',
                                          }),
                                          (0, n.tZ)('div', {
                                            className:
                                              'text-primary-500 hover:text-primary-600 dark:hover:text-primary-400',
                                            children: (0, n.tZ)(a.Z, {
                                              href: '/blog/'.concat(d.slug),
                                              children: d.title,
                                            }),
                                          }),
                                        ],
                                      }),
                                  ],
                                }),
                            ],
                          }),
                          (0, n.tZ)('div', {
                            className: 'pt-4 xl:pt-8',
                            children: (0, n.tZ)(a.Z, {
                              href: '/blog',
                              className:
                                'text-primary-500 hover:text-primary-600 dark:hover:text-primary-400',
                              children: '\u2190 Back to the blog',
                            }),
                          }),
                        ],
                      }),
                    ],
                  }),
                ],
              }),
            }),
          ],
        })
      }
    },
    3168: function (e, t, r) {
      'use strict'
      r.r(t),
        r.d(t, {
          default: function () {
            return p
          },
        })
      var n = r(7320),
        a = r(7233),
        i = r(920),
        o = r(890),
        l = r(9831),
        c = r(1576),
        d = r.n(c),
        s = r(6232),
        u = r(896),
        m = r(7175)
      function f(e, t, r) {
        return (
          t in e
            ? Object.defineProperty(e, t, {
                value: r,
                enumerable: !0,
                configurable: !0,
                writable: !0,
              })
            : (e[t] = r),
          e
        )
      }
      function h(e) {
        for (var t = 1; t < arguments.length; t++) {
          var r = null != arguments[t] ? arguments[t] : {},
            n = Object.keys(r)
          'function' === typeof Object.getOwnPropertySymbols &&
            (n = n.concat(
              Object.getOwnPropertySymbols(r).filter(function (e) {
                return Object.getOwnPropertyDescriptor(r, e).enumerable
              })
            )),
            n.forEach(function (t) {
              f(e, t, r[t])
            })
        }
        return e
      }
      function p(e) {
        var t = e.frontMatter,
          r = (e.authorDetails, e.next),
          c = e.prev,
          f = e.children,
          p = t.date,
          v = t.title
        return (0, n.BX)(o.Z, {
          children: [
            (0, n.tZ)(l.Uy, h({ url: ''.concat(d().siteUrl, '/blog/').concat(t.slug) }, t)),
            (0, n.tZ)(m.Z, {}),
            (0, n.tZ)('article', {
              children: (0, n.BX)('div', {
                children: [
                  (0, n.tZ)('header', {
                    children: (0, n.BX)('div', {
                      className:
                        'space-y-1 border-b border-gray-200 pb-10 text-center dark:border-gray-700',
                      children: [
                        (0, n.tZ)('dl', {
                          children: (0, n.BX)('div', {
                            children: [
                              (0, n.tZ)('dt', { className: 'sr-only', children: 'Published on' }),
                              (0, n.tZ)('dd', {
                                className:
                                  'text-base font-medium leading-6 text-gray-500 dark:text-gray-400',
                                children: (0, n.tZ)('time', { dateTime: p, children: (0, s.Z)(p) }),
                              }),
                            ],
                          }),
                        }),
                        (0, n.tZ)('div', { children: (0, n.tZ)(i.Z, { children: v }) }),
                      ],
                    }),
                  }),
                  (0, n.BX)('div', {
                    className: 'divide-y divide-gray-200 pb-8 dark:divide-gray-700 xl:divide-y-0 ',
                    style: { gridTemplateRows: 'auto 1fr' },
                    children: [
                      (0, n.tZ)('div', {
                        className:
                          'divide-y divide-gray-200 dark:divide-gray-700 xl:col-span-3 xl:row-span-2 xl:pb-0',
                        children: (0, n.BX)('div', {
                          className: 'prose max-w-none pt-10 pb-8 dark:prose-dark',
                          children: ['??', f],
                        }),
                      }),
                      (0, n.tZ)(u.Z, { frontMatter: t }),
                      (0, n.tZ)('footer', {
                        children: (0, n.BX)('div', {
                          className:
                            'flex flex-col text-sm font-medium sm:flex-row sm:justify-between sm:text-base',
                          children: [
                            c &&
                              (0, n.tZ)('div', {
                                className: 'pt-4 xl:pt-8',
                                children: (0, n.BX)(a.Z, {
                                  href: '/blog/'.concat(c.slug),
                                  className:
                                    'text-primary-500 hover:text-primary-600 dark:hover:text-primary-400',
                                  children: ['\u2190 ', c.title],
                                }),
                              }),
                            r &&
                              (0, n.tZ)('div', {
                                className: 'pt-4 xl:pt-8',
                                children: (0, n.BX)(a.Z, {
                                  href: '/blog/'.concat(r.slug),
                                  className:
                                    'text-primary-500 hover:text-primary-600 dark:hover:text-primary-400',
                                  children: [r.title, ' \u2192'],
                                }),
                              }),
                          ],
                        }),
                      }),
                    ],
                  }),
                ],
              }),
            }),
          ],
        })
      }
    },
  },
])
