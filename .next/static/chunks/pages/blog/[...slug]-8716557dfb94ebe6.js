;(self.webpackChunk_N_E = self.webpackChunk_N_E || []).push([
  [94],
  {
    6082: function (t, r, n) {
      ;(window.__NEXT_P = window.__NEXT_P || []).push([
        '/blog/[...slug]',
        function () {
          return n(9662)
        },
      ])
    },
    9662: function (t, r, n) {
      'use strict'
      n.r(r),
        n.d(r, {
          __N_SSG: function () {
            return a
          },
          default: function () {
            return c
          },
        })
      var o = n(7320),
        e = n(920),
        u = n(1712),
        a = !0
      function c(t) {
        var r = t.post,
          n = t.recordMap,
          a = t.authorDetails,
          c = t.prev,
          i = t.next,
          s = r.mdxSource,
          l = r.toc,
          d = r.frontMatter
        return (0, o.tZ)(o.HY, {
          children:
            !0 !== d.draft
              ? (0, o.tZ)(u.J, {
                  layout: ('post' === d.layout ? 'PostLayout' : d.layout) || 'PostLayout',
                  toc: l,
                  mdxSource: s,
                  frontMatter: d,
                  recordMap: n,
                  authorDetails: a,
                  prev: c,
                  next: i,
                })
              : (0, o.tZ)('div', {
                  className: 'mt-24 text-center',
                  children: (0, o.BX)(e.Z, {
                    children: [
                      'Under Construction',
                      ' ',
                      (0, o.tZ)('span', {
                        role: 'img',
                        'aria-label': 'roadwork sign',
                        children: '\ud83d\udea7',
                      }),
                    ],
                  }),
                }),
        })
      }
    },
  },
  function (t) {
    t.O(0, [356, 673, 712, 888, 179], function () {
      return (r = 6082), t((t.s = r))
      var r
    })
    var r = t.O()
    _N_E = r
  },
])
